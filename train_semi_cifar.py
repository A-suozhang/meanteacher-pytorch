
import re
import argparse
import os
import time
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torchvision.datasets
import torchvision.transforms as transforms

from tqdm import tqdm
from tensorboardX import SummaryWriter

from datasets import *
from models.resnet32 import *
from models.net import *
from utils import *
from optim import EMAWeightOptimizer


# ---------- Make Arg --------------
parser = argparse.ArgumentParser(description='PyTorch Cifar10 Semi-Supervised Training')
parser.add_argument('--onlylabel', default=False, type=bool,
                        help='exclude unlabeled examples from the training set')
parser.add_argument('--labeledbatchsize', default=128, type=int,
                    help="labeled examples per minibatch (default: no constrain)")
parser.add_argument('--batch', default=256, type=int,
                    help="Batch Size")
parser.add_argument('--opt', default="sgd",
                    help="Opt Type")
parser.add_argument('--lr', default=5e-3, type=float,
                    help="Learinig rate")
parser.add_argument('--epoch', default=200, type=int,
                    help="The  Training Epochs")
parser.add_argument('--th', default=0.9, type=float,
                    help="The Confidence Threshold")
parser.add_argument('--ratio', default=3.0, type=float,
                    help="The Consist Loss ratio")
parser.add_argument('--gpu', default="m", 
                    help="The GPU")
parser.add_argument('--quan', default=False, type = bool,
                    help="The Quantize In Training")
parser.add_argument('--warmup', default=0,type = int, 
                    help="How Many epoch For WarmUp(Training Only On Labeled Data)")
parser.add_argument('--resume', default = '', type=str,
                    help = 'the path to the checkpoint file')
parser.add_argument('--lrdecay', default = False,
                    help = 'Whether Using Lr Decay')
parser.add_argument('--numlabel', default=4000, type=int,
                    help = 'Num Of Labels.')
args = parser.parse_args()

if (args.numlabel == 4000):
    LABEL_DIR = 'data-local/labels/cifar10/4000_balanced_labels/00.txt'
elif (args.numlabel == 1000):
    LABEL_DIR = 'data-local/labels/cifar10/1000_balanced_labels/00.txt'
else:
    print("Please Choose The Supported Num Label")

DATA_DIR = 'data-local/images/cifar/cifar10/by-image'
BATCH_SIZE = args.batch
LEARINING_RATE = args.lr
AUG_LOSS_INDEX = args.ratio

if (args.numlabel == 4000):
    TEACHRE_ALPHA = 0.9921875
elif(args.numlabel == 1000):
    TEACHRE_ALPHA = 0.96875
CONFIDENCE_THRESHOLD = args.th

if(args.quan):
    print("Detecting Using Qunatized Training,Note That The BITWIDTH is Defined In ./model/net.py")

# ------- Prepare Data -----------

channel_stats = dict(mean=[0.4914, 0.4822, 0.4465],
                        std=[0.2470,  0.2435,  0.2616])

train_transformation = TransformTwice(transforms.Compose([
        RandomTranslateWithReflect(4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ]))

eval_transformation = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(**channel_stats)
])

traindir = os.path.join(DATA_DIR, 'train')
evaldir = os.path.join(DATA_DIR, 'val')

dataset = torchvision.datasets.ImageFolder(traindir, train_transformation)

# Make Labels
with open(LABEL_DIR) as f:
        labels = dict(line.split(' ') for line in f.read().splitlines())
        labeled_idxs, unlabeled_idxs = relabel_dataset(dataset, labels)

if args.onlylabel:  # Only Train On Labeled Data
    sampler = SubsetRandomSampler(labeled_idxs)
    batch_sampler = BatchSampler(sampler, args.batch, drop_last=True)
elif args.labeledbatchsize:   # Train With Both  (Pack The DataSet)
    batch_sampler = TwoStreamBatchSampler(
        unlabeled_idxs, labeled_idxs, args.batch, args.labeledbatchsize)
else:
    assert False, "labeled batch size {}".format(args.labeled_batch_size)

# print(len(labeled_idxs))

# Create DataLoader
train_loader = torch.utils.data.DataLoader(dataset,
                                            batch_sampler=batch_sampler,
                                            num_workers= 4,
                                            pin_memory=True)

eval_loader = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder(evaldir, eval_transformation),
    batch_size=args.batch,
    shuffle=False,
    num_workers=2 * 4,  # Needs images twice as fast
    pin_memory=True,
    drop_last=False)

# Define Writer
writer = SummaryWriter(log_dir = './runs/Semi-Cifar10/' +time.asctime(time.localtime(time.time())))
writer.add_text('Text', "| Hyper Params Are : LR:{} Batch Size:{} Ratio:{} th:{}, Quan: {}".format( args.lr, BATCH_SIZE, args.ratio, args.th, args.quan) + time.asctime(time.localtime(time.time())))

# Define Model
if (args.quan):
    teacher_net = MyNet_fix()
    student_net = MyNet_fix()
else:
    teacher_net = MyNet()
    student_net = MyNet()

for _, param in enumerate(teacher_net.parameters()):
    param.requires_grad = False

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
gpu_list = []
if (DEVICE == 'cuda'):
    if (args.gpu is not "m"):
        gpu_list = args.gpu.split(",")
        DEVICE = DEVICE+":"+gpu_list[0]


teacher_net.to(DEVICE)
student_net.to(DEVICE)


if (args.gpu == "m"):
    teacher_net = torch.nn.DataParallel(teacher_net, device_ids=[0,1,2,3])
    student_net = torch.nn.DataParallel(student_net, device_ids=[0,1,2,3])
elif len(gpu_list) > 1:
    teacher_net = torch.nn.DataParallel(teacher_net, device_ids=[int(d) for d in gpu_list])
    student_net = torch.nn.DataParallel(student_net, device_ids=[int(d) for d in gpu_list])
use_parallel = args.gpu == "m" or len(gpu_list) > 1
cudnn.benchmark = True

# Disable BackProp For Teacher
for _, param in enumerate(teacher_net.parameters()):
    param.requires_grad = False

# Define Optimizer
if (args.opt == "sgd"):
    # student_optimizer = torch.optim.SGD(student_net.parameters(), lr = LEARINING_RATE,momentum = 0.99,weight_decay=5e-4) 
    student_optimizer = torch.optim.SGD(student_net.parameters(), lr = LEARINING_RATE,momentum = 0.9, weight_decay=4e-2, nesterov=True) 
elif(args.opt == "adam"):
    # student_optimizer = torch.optim.Adam(student_net.parameters(), lr = LEARINING_RATE,weight_decay=5e-4) 
    student_optimizer = torch.optim.Adam(student_net.parameters(), lr = LEARINING_RATE) 
else:
    print("Unrecognized Optim Method, Use SGD as default")
    # student_optimizer = torch.optim.SGD(student_net.parameters(), lr = LEARINING_RATE,momentum = 0.99,weight_decay=5e-4) 
    student_optimizer = torch.optim.SGD(student_net.parameters(), lr = LEARINING_RATE,momentum = 0.9, weight_decay=4e-2, nesterov=True) 
# The Teacher's Param Is Changeing According To The Student
teacher_optimizer = EMAWeightOptimizer(teacher_net, student_net, alpha = TEACHRE_ALPHA, quan = args.quan)
criterion = nn.CrossEntropyLoss(ignore_index=-1).to(DEVICE)

if (args.resume):
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(args.resume)
    student_net.load_state_dict(checkpoint['net'])
    student_net.load_state_dict(checkpoint['net1'])
    best_acc = checkpoint['acc']

def _set_quan_method_train():
    if (use_parallel):
        student_net.module.set_fix_method(nfp.FIX_AUTO, method_by_type={"BatchNorm2d_fix": {"weight": nfp.FIX_AUTO, "bias": nfp.FIX_AUTO, "running_mean": nfp.FIX_NONE, "running_var": nfp.FIX_NONE}})
        teacher_net.module.set_fix_method(nfp.FIX_AUTO, method_by_type={"BatchNorm2d_fix": {"weight": nfp.FIX_AUTO, "bias": nfp.FIX_AUTO, "running_mean": nfp.FIX_NONE, "running_var": nfp.FIX_NONE}})
    else:
        student_net.set_fix_method(nfp.FIX_AUTO, method_by_type={"BatchNorm2d_fix": {"weight": nfp.FIX_AUTO, "bias": nfp.FIX_AUTO, "running_mean": nfp.FIX_NONE, "running_var": nfp.FIX_NONE}})
        teacher_net.set_fix_method(nfp.FIX_AUTO, method_by_type={"BatchNorm2d_fix": {"weight": nfp.FIX_AUTO, "bias": nfp.FIX_AUTO, "running_mean": nfp.FIX_NONE, "running_var": nfp.FIX_NONE}})

def _set_quan_method_eval():
    if (use_parallel):
        student_net.module.set_fix_method(nfp.FIX_FIXED, method_by_type={"BatchNorm2d_fix": {"weight": nfp.FIX_FIXED, "bias": nfp.FIX_FIXED, "running_mean": nfp.FIX_AUTO, "running_var": nfp.FIX_AUTO}})
        teacher_net.module.set_fix_method(nfp.FIX_FIXED, method_by_type={"BatchNorm2d_fix": {"weight": nfp.FIX_FIXED, "bias": nfp.FIX_FIXED, "running_mean": nfp.FIX_AUTO, "running_var": nfp.FIX_AUTO}})
    else:
        student_net.set_fix_method(nfp.FIX_FIXED, method_by_type={"BatchNorm2d_fix": {"weight": nfp.FIX_FIXED, "bias": nfp.FIX_FIXED, "running_mean": nfp.FIX_AUTO, "running_var": nfp.FIX_AUTO}})
        teacher_net.set_fix_method(nfp.FIX_FIXED, method_by_type={"BatchNorm2d_fix": {"weight": nfp.FIX_FIXED, "bias": nfp.FIX_FIXED, "running_mean": nfp.FIX_AUTO, "running_var": nfp.FIX_AUTO}})

# ---------------------------------------
def WarmUp(epoch):

    accumulated_clf_loss = 0.0
    accumulated_aug_loss = 0.0
    accumulated_mean_conf = 0.0

    # student_net.module.train()
    # teacher_net.module.train()

    if(args.quan):
        _set_quan_method_train()
    
    student_net.module.train()
    teacher_net.module.train()

    # Load Data & Main Train Loop
    num_of_iter = len(train_loader)
    pbar_train = tqdm(train_loader)

    for i, ((input_s, input_t), label) in enumerate(pbar_train):
            
        lr_warmup(student_optimizer, epoch, i, num_of_iter, args.lr)
        input_s, input_t, label = input_t.to(DEVICE), input_s.to(DEVICE), label.to(DEVICE)

        student_logits_out_t = student_net(input_s)   # The Net Gievs a tuple [BATCH_SIZE*128, BATCH_SIZE*NUM_CLASSES]

        clf_loss = criterion(student_logits_out_t, label) 

        student_optimizer.zero_grad()
        clf_loss.backward()
        student_optimizer.step()

        accumulated_clf_loss += clf_loss

        pbar_train.set_description("| The {} Epoch | Loss {:3f}| ".\
        format(epoch,accumulated_clf_loss/(i+1)))

        writer.add_scalar('WarmUp/Loss',clf_loss,i+epoch*num_of_iter)
        writer.add_scalar('Warmup/lr',student_optimizer.param_groups[0]['lr'], i+epoch*num_of_iter)


def train(epoch):

    global student_optimizer

    if (args.quan):
        lr_decay(student_optimizer, epoch, args.lr, 10)
    else:
        if(args.lrdecay):
            lr_decay(student_optimizer, epoch, args.lr, rate = 10, stop = 50)

    accumulated_clf_loss = 0.0
    accumulated_aug_loss = 0.0
    accumulated_mean_conf = 0.0

    if(args.quan):
        _set_quan_method_train()

    student_net.train()
    teacher_net.train()


    # Load Data & Main Train Loop
    num_of_iter = len(train_loader)
    pbar_train = tqdm(train_loader)

    # iter_train = iter(train_loader)
    # pbar_train = tqdm(range(len(train_loader)))

    # for i in pbar_train:
        # ((input_s,input_t),label) = iter_train.next()

    for i, ((input_s, input_t), label) in enumerate(pbar_train):

        # make sure batch_szie is BATCH_SIZE
        try:
            assert label.shape[0] == BATCH_SIZE
        except AssertionError as er:
            print("Error, The Batch Size is {}".format(label.shape[0]))

        # Only Apply WarmUp At 1st epoch
        if (args.resume == ''):
            if (args.batch > 256):
                if (epoch < 2): # * Could Be Buggy with LR Decay,so only works in rampup epochs
                    lr_warmup(student_optimizer, epoch, i, num_of_iter, args.lr, 2)

        input_s, input_t, label = input_t.to(DEVICE), input_s.to(DEVICE), label.to(DEVICE)


        student_logits_out_t = student_net(input_s)   # The Net Gievs a tuple [BATCH_SIZE*128, BATCH_SIZE*NUM_CLASSES]
        teacher_logits_out_t = teacher_net(input_t) 
        student_prob_out_t = F.softmax(student_logits_out_t, dim=1)
        teacher_prob_out_t = F.softmax(teacher_logits_out_t, dim=1)

        # print("!",label[int(BATCH_SIZE/2):BATCH_SIZE])
        clf_loss = criterion(student_logits_out_t, label)   # The Crossentropy ignores -1, so dont need 
        # aug_loss, num_masked_this_batch, mean_conf_this_batch = GetAugLoss(student_prob_out_t, teacher_prob_out_t, CONFIDENCE_THRESHOLD)
        aug_loss, num_masked_this_batch, mean_conf_this_batch = GetAugLossWithLogits(student_prob_out_t, teacher_prob_out_t, CONFIDENCE_THRESHOLD, student_logits_out_t, teacher_logits_out_t)
        # aug_loss, num_masked_this_batch, mean_conf_this_batch = GetAugLossWithLogits(student_prob_out_t, teacher_prob_out_t, CONFIDENCE_THRESHOLD, student_prob_out_t, teacher_prob_out_t)   # * Use The Prob For Softmax
        final_loss = clf_loss + AUG_LOSS_INDEX*aug_loss

        student_optimizer.zero_grad()
        final_loss.backward()
        student_optimizer.step()
        teacher_optimizer.step()

        accumulated_clf_loss += clf_loss
        accumulated_aug_loss += aug_loss
        accumulated_mean_conf += mean_conf_this_batch

        pbar_train.set_description("| The {} Epoch | LR : {:4f} | Loss {:3f}| Cls Loss {:3f} | Aug Loss {:3f} | {:3f} MeanConf | {:3f}% Was Masked |".\
        format(epoch, student_optimizer.param_groups[0]['lr'], final_loss,accumulated_clf_loss/(i+1), accumulated_aug_loss/(i+1), accumulated_mean_conf/(i+1),100 - 100*(num_masked_this_batch/BATCH_SIZE)))

        writer.add_scalar('Train/Loss',final_loss,i+epoch*num_of_iter)
        writer.add_scalar('Train/lr',student_optimizer.param_groups[0]['lr'], i+epoch*num_of_iter)
        # writer.add_scalar('Train/Clf_Loss', accumulated_clf_loss/(i+1),i+epoch*num_of_iter)
        writer.add_scalar('Train/Clf_Loss', clf_loss,i+epoch*num_of_iter)
        # writer.add_scalar('Train/Aug_Loss',  accumulated_aug_loss/(i+1),i+epoch*num_of_iter)
        writer.add_scalar('Train/Aug_Loss',  aug_loss,i+epoch*num_of_iter)
        writer.add_scalar('Train/Mean_Conf', accumulated_mean_conf/(i+1),i+epoch*num_of_iter)
        writer.add_scalar('Train/Mask_Ratio', 100-100*(num_masked_this_batch/BATCH_SIZE),i+epoch*num_of_iter)

# The Eval
def test(epoch):
    global best_acc
    global test_num

    if(args.quan):
        _set_quan_method_eval()

    student_net.eval()
    teacher_net.eval()

    accumulated_loss = 0.0
    accumulated_acc = 0.0
    correct = 0
    total = 0
    # Test On Source
    
    pbar_val = tqdm(range(int(len(eval_loader))))
    iter_val = iter(eval_loader)

    # dataloader_s1_tqdm = tqdm(testloader.datasets[0])
    # for batch_idx, (inputs, targets) in enumerate(dataloader_s1_tqdm):
    for batch_idx in pbar_val:
        inputs, targets = iter_val.next()
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs = F.softmax(teacher_net(inputs), dim = 1)
        # outputs = F.softmax(student_net(inputs), dim = 1) # !!!
        _,predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        acc_this_batch = 100*(correct/total)
        accumulated_acc += acc_this_batch
        # dataloader_s1_tqdm.set_description("| SOURCE: [{}] | Val Acc Is {:3f}|".format(SOURCE_DOMAIN,accumulated_acc/(1+batch_idx)))
        pbar_val.set_description("| Val Acc Is {:3f}|".format(accumulated_acc/(1+batch_idx)))
    
    test_num = test_num + 1
    writer.add_scalar('Val/Acc',accumulated_acc/(1+batch_idx),test_num)

        # Save The Model
    acc = accumulated_acc/(1+batch_idx)
    if acc > best_acc:
        state = {
            'net': student_net.state_dict(),
            'net1': teacher_net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        if (args.quan):
            torch.save(state, './checkpoint/ckpt_'+'SemiCifarQ'+'_'+str(round(acc,2))+'.pth')
        else:
            torch.save(state, './checkpoint/ckpt_'+'SemiCifar'+'_'+str(round(acc,2))+'.pth')
        best_acc = acc
    writer.add_scalar('Final/Acc',best_acc,epoch)



    # ---- Then Test Float Again ----------
    if(args.quan):
        if (use_parallel):
            student_net.module.set_fix_method(nfp.FIX_NONE)
            teacher_net.module.set_fix_method(nfp.FIX_NONE)
        else:
            student_net.set_fix_method(nfp.FIX_NONE)
            teacher_net.set_fix_method(nfp.FIX_NONE)

        student_net.eval()
        teacher_net.eval()

        accumulated_loss = 0.0
        accumulated_acc = 0.0
        correct = 0
        total = 0
        # Test On Source
        
        pbar_val = tqdm(range(int(len(eval_loader))))
        iter_val = iter(eval_loader)

        # dataloader_s1_tqdm = tqdm(testloader.datasets[0])
        # for batch_idx, (inputs, targets) in enumerate(dataloader_s1_tqdm):
        for batch_idx in pbar_val:
            inputs, targets = iter_val.next()
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = F.softmax(teacher_net(inputs), dim = 1)
            # outputs = F.softmax(student_net(inputs), dim = 1) # !!!
            _,predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            acc_this_batch = 100*(correct/total)
            accumulated_acc += acc_this_batch
            # dataloader_s1_tqdm.set_description("| SOURCE: [{}] | Val Acc Is {:3f}|".format(SOURCE_DOMAIN,accumulated_acc/(1+batch_idx)))
            pbar_val.set_description("| Val Acc(Float) Is {:3f}|".format(accumulated_acc/(1+batch_idx)))
        
        writer.add_scalar('Val/Acc_f',accumulated_acc/(1+batch_idx),test_num)

if __name__ == "__main__":
    best_acc = 0.0
    test_num = 0

    if (args.warmup > 0):
        for i in range(args.warmup):
            WarmUp(i) 
    
    for i in range(args.epoch):
        train(i)
        test(i)
