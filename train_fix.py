 
from __future__ import print_function

import argparse
import os
import shutil
import time
import math
import logging
import ipdb
import oyaml as yaml
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import nics_fix_pt as nfp
import nics_fix_pt.nn_fix as nnf

from tqdm import tqdm
from tensorboardX import SummaryWriter

from datasets import *
from models.net import *
from utils import *

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='PyTorch Cifar10 Semi-Supervised Training')
parser.add_argument('--batch', default=256, type=int,
                    help="Batch Size")
parser.add_argument('--opt', default="sgd",
                    help="Opt Type")
parser.add_argument('--resume', default=None, 
                    help="The Path To The chekpoint To Load")
#parser.add_argument('--lr', default=2e-3, type=float,
parser.add_argument('--lr', default=0.01, type=float,
                    help="Learinig rate")
parser.add_argument('--epoch', default=200, type=int,
                    help="The  Training Epochs")
parser.add_argument('--gpu', default="m", type=str,
                    help="The GPU, m being Multiple")
parser.add_argument('--bit', default=8, type=int,
                    help="The Quantize DATA_WIDTH")
parser.add_argument('--float', default=False,
                    help="Train Float, No Quantize")
parser.add_argument('--weight-decay', default=4e-5,type = float,
                    help="Weight decay.")
parser.add_argument("--float-bn", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="No Fix For BN.")
# parser.add_argument('--float-bn', default=False, action="store_true", help="store_true")
parser.add_argument("--fixgrad", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="No Fix For Grad.")
parser.add_argument("--test", action="store_true", default=False)
parser.add_argument("--fixrunning",type = str2bool, nargs='?',const=True,default=False,help="Choosing set-Model-eval() or _ori()" )
parser.add_argument("--keeplast", type = str2bool, nargs="?",const=True,default= True, help="Whether Only Saving The Last Model")


args = parser.parse_args() # Add The "args = []" to work with the jupyter notebook, but it will disbale args while training

# ipdb.set_trace()

print("-------------------------Configs------------------------")
# print("Configs: bit: {}, fix-bn:{}, fixgrad:{}, Weight Decay:{}\n".format(args.bit, not args.float_bn, args.fixgrad, args.weight_decay))

cfg_dict = vars(args)
cfg_dict['BITWIDTH']=BITWIDTH
cfg_dict['BITWIDTH_GRAD']=BITWIDTH_GRAD

from models.net import _generate_default_fix_cfg
cfg_dict['Range_Method'] = _generate_default_fix_cfg(' ')[' ']['range_method']


for _,i in enumerate(cfg_dict):
    print(i, cfg_dict[i], end = " | ")

print("\n---------------------------------------------------------")


EXPERIMENT_NAME = "Fix-Cifar10"

now = time.asctime(time.localtime(time.time())).replace(' ','-')
if not os.path.isdir('logs/'+EXPERIMENT_NAME+'-'+now):
    os.mkdir('logs/'+EXPERIMENT_NAME+'-'+now)

LOG_DIR = os.path.abspath("./logs/"+EXPERIMENT_NAME+'-'+now) + "/"
print(LOG_DIR)
# Dump The Config File as YAML
with open(LOG_DIR + "config.yaml","w") as f:
    yaml.safe_dump(cfg_dict, f)

# Save The File
file_to_copy  = [sys.argv[0], "./models/net.py"]
for file in file_to_copy:
    shutil.copyfile(file, LOG_DIR + file.split("/")[-1])

logging.basicConfig(level =  logging.DEBUG,
                    format = "%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s \t: %(message)s",
                    filename = LOG_DIR + "log",
                    filemode = "w"
                    )
logger = logging.getLogger("OnceLogger")
logger1 = logging.getLogger("GlobalLogger")

fh = logging.FileHandler('./logs/global.log')
logger1.addHandler(fh)


logger.info("Creating Log File...")
logger1.info("\n\nSaved To: " + LOG_DIR)
logger1.info("Confs: "+str(cfg_dict))
# ipdb.set_trace()

BATCH_SIZE = args.batch
LEARINING_RATE = args.lr
BITWIDTH = args.bit

channel_stats = dict(mean=[0.4914, 0.4822, 0.4465],
                        std=[0.2470,  0.2435,  0.2616])

train_transformation = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])


eval_transformation = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(**channel_stats)
])


writer = SummaryWriter(log_dir = LOG_DIR + 'runs/Fix-Cifar10-' +time.asctime(time.localtime(time.time())))

# writer.add_text('Text', "| Hyper Params Are : LR:{} Batch Size:{} BITWIDTH {} ".format( args.lr, BATCH_SIZE, BITWIDTH) + time.asctime(time.localtime(time.time())))
writer.add_text("Test", str(args))

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
gpu_list = []
if (DEVICE == 'cuda'):
    if (args.gpu is not "m"):
        gpu_list = [int(d) for d in args.gpu.split(",")]
        
    else:
        gpu_list = [0, 1, 2, 3]
torch.cuda.set_device("cuda:{}".format(gpu_list[0]))


# net = MyNet(useMyBN = False, useDropout = True) # ! for 1 Test only
net = MyNet_fix(fix_bn=not args.float_bn, fix_grad=args.fixgrad)

ori_net = net
use_parallel = args.gpu == "m" or len(gpu_list) > 1
if (use_parallel):
    net = torch.nn.DataParallel(net, device_ids=[int(d) for d in gpu_list])

with open(LOG_DIR + "net.log", "w") as f:
    f.write(str(net))


def adjust_learning_rate(optimizer, epoch, interval = 5):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.5 ** (epoch // interval))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def _set_quan_method_none_ori():
    ori_net.set_fix_method(nfp.FIX_NONE)

def _set_quan_method_train_ori():
    ori_net.set_fix_method(nfp.FIX_AUTO)
    ori_net.set_fix_method(nfp.FIX_AUTO)

def _set_quan_method_eval_ori():
    ori_net.set_fix_method(nfp.FIX_FIXED)
    ori_net.set_fix_method(nfp.FIX_FIXED)

def _set_quan_method_train():
    bn_param_method = nfp.FIX_AUTO
    #bn_param_method = nfp.FIX_NONE
    bn_buffer_method = nfp.FIX_NONE
    # ori_net.set_fix_method(nfp.FIX_AUTO, method_by_type={"BatchNorm2d_fix": {"weight": bn_param_method, "bias": bn_param_method, "running_mean": nfp.FIX_NONE, "running_var": nfp.FIX_NONE}})
    ori_net.set_fix_method(nfp.FIX_AUTO, method_by_type={"MyBN_fix": {"weight": bn_param_method, "bias": bn_param_method, "running_mean": nfp.FIX_NONE, "running_var": nfp.FIX_NONE}})
    # print("Setting Running Fix None")
    # ori_net.set_fix_method(nfp.FIX_AUTO, method_by_type={"BatchNorm2d_fix": {"weight": bn_param_method, "bias": bn_param_method, "running_mean": nfp.FIX_NONE, "running_var": nfp.FIX_NONE}})
    ori_net.set_fix_method(nfp.FIX_AUTO, method_by_type={"MyBN_fix": {"weight": bn_param_method, "bias": bn_param_method, "running_mean": nfp.FIX_NONE, "running_var": nfp.FIX_NONE}})

def _set_quan_method_eval():
    bn_param_method = nfp.FIX_FIXED
    bn_buffer_method = nfp.FIX_AUTO
    #bn_param_method = nfp.FIX_NONE
    #bn_buffer_method = nfp.FIX_NONE
    ori_net.set_fix_method(nfp.FIX_FIXED, method_by_type={"MyBN_fix": {"weight": bn_param_method, "bias": bn_param_method, "running_mean": bn_buffer_method, "running_var": bn_buffer_method}})
    ori_net.set_fix_method(nfp.FIX_FIXED, method_by_type={"MyBN_fix": {"weight": bn_param_method, "bias": bn_param_method, "running_mean": bn_buffer_method, "running_var": bn_buffer_method}})

# Log The Fix Configs
if not args.float:
    if not args.fixrunning:
        _set_quan_method_train()

    else:
        _set_quan_method_train_ori()


# print(ori_net.bn1_1.nf_fix_params)

if not args.float:
    d_wa = dict(ori_net.get_fix_configs(data_only = True))
    d_g = dict(ori_net.get_fix_configs(data_only = True, grad = True))
    fix_params_d = {'WEIGHT&ACTIVATION':d_wa, 'GRADIENT':d_g}

    with open(LOG_DIR + "fix_cfg.yaml", "a") as f:
        yaml.safe_dump(fix_params_d ,f)


net.to(DEVICE)
cudnn.benchmark = True


if (args.opt == "sgd"):
    # student_optimizer = torch.optim.SGD(net.parameters(), lr = LEARINING_RATE,momentum = 0.99,weight_decay=5e-4) 
    optimizer = torch.optim.SGD(net.parameters(), lr = LEARINING_RATE,momentum = 0.9, weight_decay=args.weight_decay, nesterov=True) 
elif(args.opt == "adam"):
    # student_optimizer = torch.optim.Adam(net.parameters(), lr = LEARINING_RATE,weight_decay=5e-4) 
    optimizer = torch.optim.Adam(net.parameters(), lr = LEARINING_RATE) 
else:
    print("Unrecognized Optim Method, Use SGD as default")
    # student_optimizer = torch.optim.SGD(net.parameters(), lr = LEARINING_RATE,momentum = 0.99,weight_decay=5e-4) 
    optimizer = torch.optim.SGD(net.parameters(), lr = LEARINING_RATE,momentum = 0.9, weight_decay=args.weight_decay, nesterov=True) 

criterion = nn.CrossEntropyLoss().to(DEVICE) 



# ------------ Loading --------------------

if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint.get("epoch", 0)
            best_prec1 = checkpoint.get("best_prec1", 0)
            net.load_state_dict(checkpoint["net"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            assert os.path.isfile(args.resume)

train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(
        root="~/code/datasets/cifar10",
        train=True,
        transform=train_transformation,
        download=True,
    ),
    batch_size=args.batch,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)

val_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(
        root="~/code/datasets/cifar10",
        train=False,
        transform=eval_transformation,
    ),
    batch_size=args.batch,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

num_of_iter = len(train_loader)


def train(epoch):

    accumulated_clf_loss = 0.0

    if not args.float:
        if not args.fixrunning:
            _set_quan_method_train()
        else:
            _set_quan_method_train_ori()
  
    
    net.train()

    pbar_train = tqdm(train_loader)
    correct = -1
    total = -1

    for i , (inputs, targets) in enumerate(pbar_train):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

       #  ipdb.set_trace()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accumulated_clf_loss += loss

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        pbar_train.set_description("| The {} Epoch | Loss {:3f} | Acc {:3f} | Lr {:3f}".\
            format(epoch, accumulated_clf_loss/(i+1), float(correct) / total, optimizer.param_groups[0]['lr']))

        writer.add_scalar('Train/Loss',loss,i+epoch*num_of_iter)
        writer.add_scalar('Train/lr',optimizer.param_groups[0]['lr'], i+epoch*num_of_iter)

        if not args.float:
            for j in ori_net.fix_state_dict():
                if ("scale" in j):
                    writer.add_scalar('Scale/'+j, ori_net.fix_state_dict()[j], i+epoch*num_of_iter)
            for j in ori_net.fix_state_dict():
                if ("bn" in j and not "scale" in j):
                    writer.add_histogram('hist/'+j, ori_net.fix_state_dict()[j], epoch)

   
        logger.info("| The {}th Epoch | Loss {} | Acc {} | LR {} | ".format(epoch, accumulated_clf_loss/(i+1), float(correct / total)*100, optimizer.param_groups[0]['lr']))


def test(epoch):
    global test_acc
    global best_acc


    if not args.float:
        if not args.fixrunning:
            _set_quan_method_eval()
        else:
            _set_quan_method_eval_ori()

    net.eval()

    accumulated_loss = 0.0
    correct = 0
    total = 0
    pbar_val = tqdm(val_loader)

    for i , (inputs, targets) in enumerate(pbar_val):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs = F.softmax(net(inputs), dim = 1)
        _,predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        pbar_val.set_description("| Val Acc Is {:3f}|".format(float(correct)/total * 100))
    
    writer.add_scalar('Val/Acc',float(correct / total * 100),epoch)
    logger.info(" | The {}th Epoch| Test Acc: {} |  ".format(epoch, float(correct / total)*100))
    logger1.info(" | The {}th Epoch| Test Acc: {} | ".format(epoch, float(correct / total)*100))

    acc = correct / total
    if not (args.keeplast):
        if acc > best_acc:
            state = {
                'net': ori_net.state_dict(),
                'acc': acc,
                'epoch': epoch,
             }
            torch.save(state, LOG_DIR + 'ckpt-FixCifar_'+str(round(100*acc,2))+'.pth')
            best_acc = acc
            writer.add_scalar('Final/Acc',best_acc,epoch)
    else:
        state = {
            'net':ori_net.state_dict(),
            'acc':acc,
            'epoch': epoch,
        }
        torch.save(state, LOG_DIR + 'ckpt-FixCifar' + '.pth')
        best_acc = acc
        writer.add_scalar('Final/Acc',best_acc,epoch)

def unit_test():
    net.eval()
    print("Setting Net As Eval Mode")
    for i in [0,1,2]:
        ori_net.set_fix_method(i)
        dummy_test()
    net.train()
    print("Setting Net As Train Mode")
    for i in [0,1,2]:
        ori_net.set_fix_method(i)
        dummy_test()



def dummy_test(net):

    if not args.float:
        _set_quan_method_eval_ori()


    net.eval()
    accumulated_loss = 0.0
    correct = 0
    total = 0
    pbar_val = tqdm(val_loader)

    # for i in net.fix_state_dict().keys():
    #     if ("running" in i and "scale" in i):
    #         print(i, net.fix_state_dict()[i])
    for n, b in net.named_buffers():
        # if "fp_scale" in n and "running" in n:
        if "fp_scale" in n:
            print(n, b)

    for i , (inputs, targets) in enumerate(pbar_val):
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs = F.softmax(net(inputs), dim = 1)
        _,predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        acc_this_batch = 100*(correct/total)

        pbar_val.set_description("| Val Acc Is {:3f}|".format(float(correct) / total * 100))

    accumulated_loss = 0.0
    correct = 0
    total = 0
    pbar_val = tqdm(val_loader)
    if not args.float:
        _set_quan_method_train_ori()

    for i , (inputs, targets) in enumerate(pbar_val):
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs = F.softmax(net(inputs), dim = 1)
        _,predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        acc_this_batch = 100*(correct/total)

        pbar_val.set_description("| Val Acc Is {:3f}|".format(float(correct) / total * 100))

    # for i in net.fix_state_dict().keys():
    #     if ("running" in i and "scale" in i):
    #         print(i, net.fix_state_dict()[i])
    for n, b in net.named_buffers():
        # if "fp_scale" in n and "running" in n:
        if "fp_scale" in n:
            print(n, b)


if __name__ == "__main__":
    if args.test:
        dummy_test(net)
    else:
        # train
        best_acc = 0.0
        test_num = 0
        for i in range(args.epoch):
            if not args.float:
                adjust_learning_rate(optimizer, i)
            else:
                adjust_learning_rate(optimizer, i, 20)
            train(i)
            test(i)
            if (i%5 == 0):
                # unit_test()
                pass

