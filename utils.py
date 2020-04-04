
import torch

CLASS_BALANCE_INDEX = 0.0
N_CLASS = 10
useBCE = True

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


def lr_decay(optimizer, epoch, lr, rate=5,stop=40):
    """Sets the learning rate to the initial LR decayed by 2 every 10 epochs""" 

    if (epoch < stop):
        lr_to_be = lr * (0.5 ** (epoch // rate))
        for param_group in optimizer.param_groups:
                param_group["lr"] = lr_to_be
        # print ("Setting Lr as {}".format(lr_to_be))

    # current_lr = optimizer.param_groups[0]["lr"]
    # if (epoch % 1 == 0):
    #     for param_group in optimizer.param_groups:
    #         param_group["lr"] = current_lr*0.5
    
    # print("LR Decaying... from {} to {}".format(current_lr, optimizer.param_groups[0]['lr']))
    # ? mWould Result In 0.4998-ish LR




# ------------------ For The Fixed Point Expreiment------------------------



# ---------------- Define The Model -------------------

