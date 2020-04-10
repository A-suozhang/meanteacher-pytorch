"""Train CIFAR10 with PyTorch."""
from __future__ import print_function

import os
import six
import copy
import math
from tqdm import tqdm

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
from optim import EMAWeightOptimizer

from .trainer import Trainer

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



class SemiTrainer(Trainer):
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
        },
        "th": 0.9,
        "ratio": 3.0,
        "warmup_epochs": 0,
        "teacher_alpha": 0.9921875,

    }
    def __init__(self, tch_net,stu_net, p_tch_net,p_stu_net,  trainloader, testloader, savepath, save_every, log, cfg):
        self.tch_net = tch_net
        for _, param in enumerate(tch_net.parameters()):
            param.requires_grad = False
        self.stu_net = stu_net
        self.p_tch_net = p_tch_net
        self.p_stu_net = p_stu_net
        self.trainloader = trainloader
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
        self.warmup = False

    def init(self, device, local_rank=-1,resume=None, pretrain=False):
        self.device = device
        self.local_rank = local_rank
        # In SemiSupervised Setting , Ignore the "-1" Index（Denoting Non-label）
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1).to(self.device)
        self.lr_schedule = self.cfg["lr_schedule"]
        self.stu_optimizer = getattr(optim, self.cfg.get("optimizer_type", "SGD"))(self.stu_net.parameters(), **self.cfg["stu_optimizer"])
        self.tch_optimizer = EMAWeightOptimizer(self.tch_net, self.stu_net, alpha=self.cfg["teacher_alpha"])
        if resume:
            # Load checkpoint.
            self.log("==> Resuming from checkpoint..")
            print("==> Resuming from checkpoint..")
            assert os.path.exists(resume), "Error: no checkpoint directory found!"
            ckpt_path = os.path.join(resume, "ckpt.t7") if os.path.isdir(resume) else resume
            checkpoint = torch.load(ckpt_path,map_location="cpu")
            self.stu_net.load_state_dict(checkpoint["net"], strict=False)
            self.tch_net.load_state_dict(checkpoint["net"], strict=False)
            if not pretrain:
                self.best_acc = checkpoint["acc"]
                self.start_epoch = checkpoint["epoch"]
                if "optimizer" not in checkpoint:
                    self.log("!!! Resume mode: do not found optimizer in {}".format(ckpt_path))
                else:
                    self.stu_optimizer.load_state_dict(checkpoint["optimizer"])
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
                var_ = _getattr(self.stu_net, name)
                if isinstance(var_, torch.Tensor):
                    var_.data[:] = v
                else:
                    _setattr(self.stu_net, name, v)
                v_dct.append((name, v))
            self.log("\t" + "; ".join(["{:10}: {:.4f}".format(n, v)
                                       for n, v in sorted(v_dct, key=lambda x: x[0])]))
         
    def lr_warmup(self, steps = 100, finetune=False):
        if not finetune: 
            target_lr = self.cfg["optimizer"]["lr"]
            if self.lr < target_lr:
                for param_group in self.stu_optimizer.param_groups:
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
                self.lr = self.stu_optimizer.param_groups[0]["lr"]
                return
            self.lr_schedule["start"] = self.lr_schedule.get("start", self.cfg["optimizer"]["lr"])
            lr_schedule = self.lr_schedule
        self.lr = self.get_schedule_value(lr_schedule, epoch, self.cfg["epochs"])
        if optimizer is None:
            optimizer = self.stu_optimizer
        for p_group in optimizer.param_groups:
            p_group["lr"] = self.lr
        return self.lr

    def _get_loss(self, outputs, targets, multi_outputs=None):
        loss = self.criterion(outputs, targets)
        if self.multi_head_weight is not None and multi_outputs is not None:
            loss += self.multi_head_weight * self.criterion(multi_outputs, targets)
        return loss

    def _get_aug_loss(self, student_confs, teacher_confs, CONFIDENCE_THRESHOLD,teacher_logits, student_logits):
        tecaher_max_conf  = torch.max(teacher_confs,1)[0]   # [0] being the value, [1] as for index
        mean_conf_this_batch = tecaher_max_conf.mean()
        mask_this_batch = (tecaher_max_conf > CONFIDENCE_THRESHOLD).float()            # [0,0,1,0,0,1] - Per Batch Sample
        num_masked_this_batch = mask_this_batch.sum()
        # print ("{} Samples Was Masked".format(num_masked_this_batch.item()))

        # Calc The Augmentation Loss
        conf_diff = teacher_logits - student_logits
        consist_loss = conf_diff*conf_diff

        # If Using BCE loss
        # BinaryCrossEntropy(student_confs, teacher_confs)

        # torch.clamp(teacher_confs.sum(dim=0), min = 1.0)

        consist_loss = consist_loss.mean(dim=1)  # Loss For EverySmaple
        # Apply The Thresholding Mask
        consist_loss = (consist_loss*mask_this_batch).mean()
        
        # If Args.ClassBalance > 0
        avg_conf_per_class = student_confs.mean(dim=0)
        useBCE = True
        # class_balance_loss = getClsBalanceLoss(avg_conf_per_class,float(1/N_CLASS))
        # class_balance_loss = class_balance_loss.sum()*(num_masked_this_batch/teacher_confs.shape[0])

        # aug_loss = consist_loss + CLASS_BALANCE_INDEX*class_balance_loss
        aug_loss = consist_loss 

        return aug_loss, num_masked_this_batch, mean_conf_this_batch



    def train(self):
        for epoch in range(self.start_epoch, self.cfg["epochs"] + 1):
            self.epoch = epoch
            if self.epoch < self.cfg["warmup_epochs"]:
                self.warmup = True
            self.schedule_lr()
            self.log("\nEpoch {:4d}: lr {:.4f}".format(epoch, self.lr))
            self.schedule_var()
            self.stu_net.train()
            self.tch_net.train()
            self.clf_loss = 0
            self.aug_loss = 0
            self.num_masked = 0
            self.mean_conf = 0
            train_loss = 0
            correct = 0
            total = 0
            for batch_idx, ((input_s, input_t), targets) in enumerate(self.trainloader):
                input_s, input_t, targets = input_s.to(self.device), input_t.to(self.device), targets.to(self.device)
                self.stu_optimizer.zero_grad()
                stu_logits = self.p_stu_net(input_s)
                tch_logits = self.p_tch_net(input_t)
                stu_probs = F.softmax(stu_logits, dim=1)
                tch_probs = F.softmax(tch_logits, dim=1)

                self.clf_loss = self._get_loss(stu_logits, targets, getattr(self.stu_net, "logits_multi", None))
                self.aug_loss, self.num_masked, self.mean_conf = self._get_aug_loss(stu_probs, tch_probs, self.cfg["th"], stu_logits, tch_logits)
                if self.warmup:
                    loss = self.clf_loss
                else:
                    loss = self.clf_loss + self.cfg["ratio"]*self.aug_loss
                loss.backward()
                self.stu_optimizer.step()
                self.tch_optimizer.step()
        
                train_loss += loss.item()
                _, predicted = stu_probs.max(1)
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
        self.stu_net.eval()
        self.tch_net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.testloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.p_tch_net(inputs)
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
            "net": self.stu_net.state_dict(),
            "optimizer": self.stu_optimizer.state_dict(),
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
                    "net": self.stu_net.state_dict(),
                    "acc": acc,
                    "epoch": self.epoch,
                }
                torch.save(state, os.path.join(self.savepath, "ckpt.t7"))


