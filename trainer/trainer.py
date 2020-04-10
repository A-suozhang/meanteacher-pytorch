"""Train CIFAR10 with PyTorch."""
from __future__ import print_function

import os
import six
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms

from utils import progress_bar

def _getattr(obj, n):
    if isinstance(obj, torch.nn.DataParallel):
        return getattr(obj.module, n)
    else:
        return getattr(obj, n)
    
def _setattr(obj, n, v):
    if isinstance(obj, torch.nn.DataParallel):
        return setattr(obj.module, n, v)
    else:
        return setattr(obj, n, v)



class Trainer(object):
    NAME = "normal"
    default_cfg = {
        "optimizer_type": "SGD",
        "optimizer": {
            "lr": 0.1,
            "momentum": 0.9,
            "weight_decay": 5e-4
        },
        "lr_schedule": {
            "start": 0.1,
            "boundaries": [125, 200],
            "rate": 0.1
        }
    }
    def __init__(self, net, p_net, trainloader, testloader, savepath, save_every, log, cfg):
        self.net = net
        self.p_net = p_net
        self.trainloader = trainloader[0]
        # self.validloader = trainloader[1]
        # self.ori_trainloader = trainloader[2]
        self.testloader = testloader
        self.savepath = savepath
        self.save_every = save_every
        self.log = log

        self.cfg = copy.deepcopy(self.default_cfg)
        self.cfg.update(cfg)

        self.log("Configuration:\n" + "\n".join(["\t{:10}: {:10}".format(n, str(v)) for n, v in self.cfg.items()]) + "\n")

        self.best_acc = 0.
        self.epoch = 0

        self.start_epoch = 1

        self.multi_head_weight = self.cfg.get("multi_head_weight", None)

    def init(self, device, local_rank=-1,resume=None, pretrain=False):
        self.device = device
        self.local_rank = local_rank
        self.criterion = nn.CrossEntropyLoss()
        self.lr_schedule = self.cfg["lr_schedule"]
        self.optimizer = getattr(optim, self.cfg.get("optimizer_type", "SGD"))(self.net.parameters(), **self.cfg["optimizer"])
        if resume:
            # Load checkpoint.
            self.log("==> Resuming from checkpoint..")
            print("==> Resuming from checkpoint..")
            assert os.path.exists(resume), "Error: no checkpoint directory found!"
            ckpt_path = os.path.join(resume, "ckpt.t7") if os.path.isdir(resume) else resume
            checkpoint = torch.load(ckpt_path,map_location="cpu")
            self.net.load_state_dict(checkpoint["net"], strict=False)
            if not pretrain:
                self.best_acc = checkpoint["acc"]
                self.start_epoch = checkpoint["epoch"]
                if "optimizer" not in checkpoint:
                    self.log("!!! Resume mode: do not found optimizer in {}".format(ckpt_path))
                else:
                    self.optimizer.load_state_dict(checkpoint["optimizer"])
            # self.test(save=False)

    @staticmethod
    def get_schedule_value(cfg, epoch, epochs):
        if isinstance(cfg, dict):
            type_ = cfg.get("type", "exp")
            start_ = cfg["start"]
            if type_ == "cosine": # cosine type
                min_ = cfg.get("min", 0)
                v_ = min_ + (start_ - min_) * (1 + np.cos(np.pi * (epoch - 1) / epochs)) / 2
            else:
                if "boundaries" in cfg:
                    index = np.where(np.array(cfg["boundaries"]) >= epoch)[0]
                    index = index[0] if len(index) else len(cfg["boundaries"])
                elif "every" in cfg:
                    index = (epoch - 1) // cfg["every"]                
                if type_ == "step":
                    v_ = start_ + cfg["step"] * index 
                else: # exp type
                    v_ = start_ * cfg["rate"] ** index
                v_ = min(max(v_, cfg.get("min", -np.inf)), cfg.get("max", np.inf))
        elif isinstance(cfg, (np.float32, float)):
            v_ = cfg
        return v_

    def schedule_var(self, epoch=None):
        epoch = epoch or self.epoch
        v_dct = []
        schedule = self.cfg.get("schedule", {})
        if schedule:
            for name, cfg in six.iteritems(schedule):
                v = self.get_schedule_value(cfg, epoch, self.cfg["epochs"])
                var_ = _getattr(self.net, name)
                if isinstance(var_, torch.Tensor):
                    var_.data[:] = v
                else:
                    _setattr(self.net, name, v)
                v_dct.append((name, v))
            self.log("\t" + "; ".join(["{:10}: {:.4f}".format(n, v)
                                       for n, v in sorted(v_dct, key=lambda x: x[0])]))
         
    def lr_warmup(self, steps = 100, finetune=False):
        if not finetune: 
            target_lr = self.cfg["optimizer"]["lr"]
            if self.lr < target_lr:
                for param_group in self.optimizer.param_groups:
                    self.lr = self.lr + target_lr/(steps)
                    param_group['lr'] = self.lr
        else:
            target_lr=self.cfg["finetune_optimizer"]["lr"]
            if self.finetune_optimizer.param_groups[0]['lr'] < target_lr:
                self.finetune_optimizer.param_groups[0]['lr'] = self.finetune_optimizer.param_groups[0]['lr'] + target_lr/steps
                # print("warmup")
            else:
                # print(self.finetune_optimizer.param_groups[0]['lr'])
                pass

    def schedule_lr(self, epoch=None, lr_schedule=None, optimizer=None):
        epoch = epoch or self.epoch
        if lr_schedule is None:
            if self.lr_schedule is None:
                self.lr = self.optimizer.param_groups[0]["lr"]
                return
            self.lr_schedule["start"] = self.lr_schedule.get("start", self.cfg["optimizer"]["lr"])
            lr_schedule = self.lr_schedule
        self.lr = self.get_schedule_value(lr_schedule, epoch, self.cfg["epochs"])
        if optimizer is None:
            optimizer = self.optimizer
        for p_group in optimizer.param_groups:
            p_group["lr"] = self.lr
        return self.lr

    def _get_loss(self, outputs, targets, multi_outputs=None):
        loss = self.criterion(outputs, targets)
        if self.multi_head_weight is not None and multi_outputs is not None:
            loss += self.multi_head_weight * self.criterion(multi_outputs, targets)
        return loss

    def train(self):
        for epoch in range(self.start_epoch, self.cfg["epochs"] + 1):
            self.epoch = epoch
            self.schedule_lr()
            self.log("\nEpoch {:4d}: lr {:.4f}".format(epoch, self.lr))
            self.schedule_var()
            self.net.train()
            train_loss = 0
            correct = 0
            total = 0
            for batch_idx, (inputs, targets) in enumerate(self.trainloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.p_net(inputs)
                loss = self._get_loss(outputs, targets, getattr(self.net, "logits_multi", None))
                # loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
        
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
                if self.local_rank is not -1:
                    if self.local_rank == 0:
                        progress_bar(batch_idx, len(self.trainloader), "Loss: {:.3f} | Acc: {:.3f} % ({:d}/{:d})"
                                 .format(train_loss/(batch_idx+1), 100.*correct/total, correct, total), ban="Train")
                else:
                    progress_bar(batch_idx, len(self.trainloader), "Loss: {:.3f} | Acc: {:.3f} % ({:d}/{:d})"
                                 .format(train_loss/(batch_idx+1), 100.*correct/total, correct, total), ban="Train")
            self.log("Train: loss: {:.3f} | acc: {:.3f} %"
                             .format(train_loss/len(self.trainloader), 100.*correct/total))
            self.test()

    def test(self, save=True):
        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.testloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.p_net(inputs)
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                if self.local_rank is not -1:
                    if self.local_rank == 0:
                        progress_bar(batch_idx, len(self.testloader),
                                     "Loss: {:.3f} | Acc: {:.3f} % ({:d}/{:d})"\
                                     .format(
                                         test_loss/(batch_idx+1),
                                         100. * correct/total, correct, total), ban="Test")
                else:
                    progress_bar(batch_idx, len(self.testloader),
                                "Loss: {:.3f} | Acc: {:.3f} % ({:d}/{:d})"\
                                .format(
                                    test_loss/(batch_idx+1),
                                    100. * correct/total, correct, total), ban="Test")
        acc = 100.*correct/total
        self.log("Test: loss: {:.3f} | acc: {:.3f} %"
                 .format(test_loss/len(self.testloader), 100.*correct/total))
        if save:
            self.save_if_best(acc)
            self.save_every_epoch(acc)
        return acc

    def save_every_epoch(self, acc):
        if self.save_every is not None:
            if not self.epoch % self.save_every:
                self.log("Saving... acc{} (epoch {})".format(acc, self.epoch))
                self.save(os.path.join(self.savepath, "ckpt_{}.t7".format(self.epoch)), acc, self.epoch)

    def save(self, path, acc, epoch):
        state = {
            "net": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "acc": acc,
            "epoch": epoch,
        }
        torch.save(state, path)

    def _is_new_best(self, acc):
        is_best = acc > self.best_acc
        if is_best:
            self.best_acc = acc
        return is_best

    
    def save_if_best(self, acc):
        if self._is_new_best(acc):
            if self.savepath is not None:
                # Save checkpoint.
                self.log("Saving... acc {} (epoch {})".format(acc, self.epoch))
                state = {
                    "net": self.net.state_dict(),
                    "acc": acc,
                    "epoch": self.epoch,
                }
                torch.save(state, os.path.join(self.savepath, "ckpt.t7"))


