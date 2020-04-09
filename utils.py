# -------------------------
CLASS_BALANCE_INDEX = 0.0
N_CLASS = 10
useBCE = True

import os
import sys
import six
import time
import math

import torch
import torch.nn as nn
import torch.nn.init as init


def BinaryCrossEntropy(pred, tgt):
    inv_tgt = -tgt + 1.0
    inv_pred = -pred + 1.0 + 1e-6
    return -(tgt * torch.log(pred + 1.0e-6) + inv_tgt * torch.log(inv_pred))


def getClsBalanceLoss(pred, target):
    if (useBCE):
        return BinaryCrossEntropy(pred, target)
    else:
        return -torch.log(pred + 1.0e-6)

def GetAugLoss(student_confs, teacher_confs, CONFIDENCE_THRESHOLD):

    # Do The Confidence Thresholding
    tecaher_max_conf  = torch.max(teacher_confs,1)[0]   # [0] being the value, [1] as for index
    mean_conf_this_batch = tecaher_max_conf.mean()
    mask_this_batch = (tecaher_max_conf > CONFIDENCE_THRESHOLD).float()            # [0,0,1,0,0,1] - Per Batch Sample
    num_masked_this_batch = mask_this_batch.sum()
    # print ("{} Samples Was Masked".format(num_masked_this_batch.item()))

    # Calc The Augmentation Loss
    conf_diff = student_confs - teacher_confs
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

def GetAugLossWithLogits(student_confs, teacher_confs, CONFIDENCE_THRESHOLD, teacher_logits, student_logits):

    # Do The Confidence Thresholding
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

# --------------------------------------------------------------------------------------

import numpy as np


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))


def lr_warmup(optimizer, epoch, step_in_epoch, total_steps_in_epoch, lr, LR_RAMP = 1):
    current_lr = optimizer.param_groups[0]['lr']
    epoch = epoch + step_in_epoch / total_steps_in_epoch

    # LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
    current_lr = linear_rampup(epoch, LR_RAMP) * (lr) 

    # # Cosine LR rampdown from https://arxiv.org/abs/1608.03983 (but one cycle only)
    # if args.lr_rampdown_epochs:
    #     assert args.lr_rampdown_epochs >= args.epochs
    #     lr *= ramps.cosine_rampdown(epoch, args.lr_rampdown_epochs)

    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr

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


def schedule(epoch, schedule):
        type_ = schedule["type"]
        if type_ == "value":
            ind = list(np.where(epoch < np.array(schedule["boundary"]))[0])
            if not ind: # if epoch is larger than the last boundary
                ind = len(schedule["boundary"]) - 1
            else:
                ind = ind[0] - 1
            next_v = schedule["value"][ind]
        else:
            min_ = schedule.get("min", -np.inf)
            max_ = schedule.get("max", np.inf)
            start_epoch = schedule.get("start_epoch", 0)
            epoch = epoch - start_epoch
            if epoch <= 0:
                return schedule["start"]
            if "every" in schedule:
                ind = (epoch - 1) // schedule["every"]
            else: # "boundary" in schedule
                ind = list(np.where(epoch < np.array(schedule["boundary"]))[0])
                if not ind: # if epoch is larger than the last boundary
                    ind = len(schedule["boundary"])
                else:
                    ind = ind[0]
            if type_ == "mul":
                next_v = schedule["start"] * schedule["step"] ** ind
            else: # type_ == "add"
                next_v = schedule["start"] + schedule["step"] * ind
            next_v = max(min(next_v, max_), min_)
        return next_v


def lr_decay(optimizer, epoch, lr, rate=5,stop=40):
    """Sets the learning rate to the initial LR decayed by 2 every 10 epochs""" 

    if (epoch < stop):
        lr_to_be = lr * (0.5 ** (epoch // rate))
        for param_group in optimizer.param_groups:
                param_group["lr"] = lr_to_be


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)

TOTAL_BAR_LENGTH = 15.
last_time = time.time()
begin_time = last_time

_disable_progress_bar = int(os.environ.get("DISABLE_PROGRESS_BAR", 0))
if _disable_progress_bar > 0:
    def progress_bar(current, total, msg=None, ban=""):
        pass
else:
    _, term_width = os.popen('stty size', 'r').read().split()
    term_width = int(term_width)

    def progress_bar(current, total, msg=None, ban=""):
        global last_time, begin_time
        if current == 0:
            begin_time = time.time()  # Reset for new bar.
    
        cur_len = int(TOTAL_BAR_LENGTH*current/total)
        rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1
    
        sys.stdout.write(' ['+ban)
        for i in range(cur_len):
            sys.stdout.write('=')
        sys.stdout.write('>')
        for i in range(rest_len):
            sys.stdout.write('.')
        sys.stdout.write(']')
    
        cur_time = time.time()
        step_time = cur_time - last_time
        last_time = cur_time
        tot_time = cur_time - begin_time
    
        L = []
        L.append('  Step: %s' % format_time(step_time))
        L.append(' | Tot: %s' % format_time(tot_time))
        if msg:
            L.append(' | ' + msg)
    
        msg = ''.join(L)
        sys.stdout.write(msg)
        # for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        #     sys.stdout.write(' ')
    
        # Go back to the center of the bar.
        #for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        #    sys.stdout.write('\b')
        sys.stdout.write(' %d/%d ' % (current+1, total))
    
        if current < total-1:
            sys.stdout.write('\r')
        else:
            sys.stdout.write('\n')
        sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

def get_forward(forward):
    def _new_forward(self, *args, **kwargs):
        res = forward(self, *args, **kwargs)
        if not hasattr(self, "o_size"):
            self.o_size = tuple(res.size())
        return res
    return _new_forward

_ALREADY_PATCHED = False
def patch_conv2d_4_size():
    global _ALREADY_PATCHED
    nn.Conv2d.forward = get_forward(nn.Conv2d.forward)
    _ALREADY_PATCHED = True

    
class InfIterator(six.Iterator):
    def __init__(self, iterable):
        self.iterable = iterable
        self.iter_ = None

    def __getattr__(self, name):
        return getattr(self.iterable, name)

    def __len__(self):
        return len(self.iterable)

    def __next__(self):
        if self.iter_ is None:
            self.iter_ = iter(self.iterable)
        try:
            data = next(self.iter_)
        except StopIteration:
            self.iter_ = iter(self.iterable)
            data = next(self.iter_)

        return data

    next = __next__

def get_inf_iterator(iterable):
    return InfIterator(iterable)
# valid_queue = get_inf_iterator(DataLoader(...))
# next(valid_queue)

def get_list_str(lst, format_):
    return "[" + ", ".join([format_.format(item) for item in lst]) + "]"

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


