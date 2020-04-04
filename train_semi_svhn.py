
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
parser.add_argument('--labeledbatchsize', default=64, type=int,
                    help="labeled examples per minibatch (default: no constrain)")
parser.add_argument('--batch', default=128, type=int,
                    help="Batch Size")
parser.add_argument('--opt', default="sgd",
                    help="Opt Type")
parser.add_argument('--lr', default=1e-2, type=float,
                    help="Learinig rate")
parser.add_argument('--epoch', default=20, type=int,
                    help="The  Training Epochs")
parser.add_argument('--th', default=0.8, type=float,
                    help="The Confidence Threshold")
parser.add_argument('--ratio', default=1.0, type=float,
                    help="The Consist Loss ratio")
parser.add_argument('--gpu', default=0, 
                    help="The GPU")
args = parser.parse_args()


ROOT_DIR = './data'
BATCH_SIZE = args.batch
LEARINING_RATE = args.lr
AUG_LOSS_INDEX = args.ratio
TEACHRE_ALPHA = 0.99
CONFIDENCE_THRESHOLD = args.th

# ------- Prepare Data -----------

channel_stats = dict(mean=[0.5,  0.5,  0.5],
                    std=[ 0.5,  0.5,  0.5])

train_transformation = TransformTwice(transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ]))

eval_transformation = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(**channel_stats)
])

l_train_dataset = SVHN_SEMI(ROOT_DIR, "l_train",train = True)
u_train_dataset = SVHN_SEMI(ROOT_DIR, "u_train", train = True)
val_dataset = SVHN_SEMI(ROOT_DIR, "val", train = False)
test_dataset = SVHN_SEMI(ROOT_DIR,"test", train = False)

l_loader = DataLoader(
    l_train_dataset, args.batch//2, drop_last=True,
    sampler=RandomSampler(len(l_train_dataset), len(u_train_dataset)*args.epoch)
)
u_loader = DataLoader(
    u_train_dataset, args.batch//2, drop_last=True,
    sampler=RandomSampler(len(u_train_dataset), len(u_train_dataset)*args.epoch)
)

iter_l = iter(l_loader)
iter_u = iter(u_loader)


print("labeled data : {}, unlabeled data : {}, training data : {}".format(
    len(l_train_dataset), len(u_train_dataset), len(l_train_dataset)+len(u_train_dataset)))
print("validation data : {}, test data : {}".format(len(val_dataset), len(test_dataset)))



# l_loader = DataLoader(l_train_dataset, batch_size = int(BATCH_SIZE/2), shuffle = True, num_workers = 4, drop_last = True)
# u_loader = DataLoader(u_train_dataset, batch_size = int(BATCH_SIZE/2), shuffle = True, num_workers = 4, drop_last = True)


val_loader = DataLoader(val_dataset, 128, shuffle=False, drop_last=False)
test_loader = DataLoader(test_dataset, 128, shuffle=False, drop_last=False)

# Define Writer
writer = SummaryWriter(log_dir = './runs/Semi-SVHN/' +time.asctime(time.localtime(time.time())))
writer.add_text('Text', "| Hyper Params Are : LR:{} Batch Size:{} Ratio:{} th:{}".format(args.lr, BATCH_SIZE, args.ratio, args.th) + time.asctime(time.localtime(time.time())))

# Define Model
teacher_net = MyNet()
student_net = MyNet()

for _, param in enumerate(teacher_net.parameters()):
    param.requires_grad = False

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if (DEVICE == 'cuda'):
    if (args.gpu is not "m"):
        DEVICE = DEVICE+":"+args.gpu
    

teacher_net.to(DEVICE)
student_net.to(DEVICE)
if (args.gpu == "m"):
    teacher_net = torch.nn.DataParallel(teacher_net, device_ids=[0,1,2,3])
    student_net = torch.nn.DataParallel(student_net, device_ids=[0,1,2,3])
cudnn.benchmark = True


# Disable BackProp For Teacher
for _, param in enumerate(teacher_net.parameters()):
    param.requires_grad = False

# Define Optimizer
if (args.opt == "sgd"):
    # student_optimizer = torch.optim.SGD(student_net.parameters(), lr = LEARINING_RATE,momentum = 0.99,weight_decay=5e-4) 
    student_optimizer = torch.optim.SGD(student_net.parameters(), lr = LEARINING_RATE,momentum = 0.9, weight_decay=2e-4, nesterov=True) 
elif(args.opt == "adam"):
    # student_optimizer = torch.optim.Adam(student_net.parameters(), lr = LEARINING_RATE,weight_decay=5e-4) 
    student_optimizer = torch.optim.Adam(student_net.parameters(), lr = LEARINING_RATE) 
else:
    print("Unrecognized Optim Method, Use SGD as default")
    # student_optimizer = torch.optim.SGD(student_net.parameters(), lr = LEARINING_RATE,momentum = 0.99,weight_decay=5e-4) 
    student_optimizer = torch.optim.SGD(student_net.parameters(), lr = LEARINING_RATE,momentum = 0.9, weight_decay=2e-4, nesterov=True) 
# The Teacher's Param Is Changeing According To The Student
teacher_optimizer = EMAWeightOptimizer(teacher_net, student_net, alpha = TEACHRE_ALPHA)
criterion = nn.CrossEntropyLoss(ignore_index=-1).to(DEVICE)

# ---------------------------------------
def train(epoch):

    accumulated_clf_loss = 0.0
    accumulated_aug_loss = 0.0
    accumulated_mean_conf = 0.0

    student_net.train()
    teacher_net.train()

    # Load Data & Main Train Loop
    num_of_iter = len(u_loader)//args.epoch
    # num_of_iter = len(u_loader)
    pbar_train = tqdm(range(num_of_iter))

    # iter_l = iter(l_loader)
    # iter_u = iter(u_loader)


    # for i in pbar_train:
        # ((input_s,input_t),label) = iter_train.next()

    for i in pbar_train:

        adjust_learning_rate(student_optimizer, epoch, i, num_of_iter, args.lr)

        # try:
        #     l_input, l_target = iter_l.next()
        # except StopIteration:
        #     new_l_loader = DataLoader(l_train_dataset, batch_size = int(BATCH_SIZE/2), shuffle = True, num_workers = 4, drop_last = True)
        #     iter_l = iter(new_l_loader)
        #     l_input, l_target = iter_l.next()
            # print ("Reloading Labeled Set")

        l_input, l_target = iter_l.next() 
        u_input, u_target = iter_u.next()

        input_s = torch.cat([l_input[0], u_input[0]], 0)
        input_t = torch.cat([l_input[1], u_input[1]], 0)
        label = torch.cat([l_target, u_target],0)

        # print(label.eq(-1).sum())

        input_s, input_t, label = input_s.to(DEVICE), input_t.to(DEVICE), label.to(DEVICE)

        student_logits_out_t = student_net(input_s)    # The Net Gievs a tuple [BATCH_SIZE*128, BATCH_SIZE*NUM_CLASSES]
        teacher_logits_out_t = teacher_net(input_t)
        student_prob_out_t = F.softmax(student_logits_out_t, dim=1)
        teacher_prob_out_t = F.softmax(teacher_logits_out_t, dim=1)

        # print("!",label[int(BATCH_SIZE/2):BATCH_SIZE])
        clf_loss = criterion(student_logits_out_t, label)   # The Crossentropy ignores -1, so dont need 
        aug_loss, num_masked_this_batch, mean_conf_this_batch = GetAugLossWithLogits(student_prob_out_t, teacher_prob_out_t, CONFIDENCE_THRESHOLD, student_logits_out_t, teacher_logits_out_t)
        final_loss = clf_loss + AUG_LOSS_INDEX*aug_loss

        student_optimizer.zero_grad()
        final_loss.backward()
        student_optimizer.step()
        teacher_optimizer.step()

        accumulated_clf_loss += clf_loss
        accumulated_aug_loss += aug_loss
        accumulated_mean_conf += mean_conf_this_batch

        pbar_train.set_description("| The {} Epoch | Loss {:3f}| Cls Loss {:3f} | Aug Loss {:3f} | {:3f} MeanConf | {:3f}% Was Masked |".\
        format(epoch, final_loss,accumulated_clf_loss/(i+1), accumulated_aug_loss/(i+1), accumulated_mean_conf/(i+1),100 - 100*(num_masked_this_batch/BATCH_SIZE)))

        writer.add_scalar('Train/Loss',final_loss,i+epoch*num_of_iter)
        writer.add_scalar('Train/lr',student_optimizer.param_groups[0]['lr'], i+epoch*num_of_iter)
        # writer.add_scalar('Train/Clf_Loss', accumulated_clf_loss/(i+1),i+epoch*num_of_iter)
        writer.add_scalar('Train/Clf_Loss', clf_loss,i+epoch*num_of_iter)
        # writer.add_scalar('Train/Aug_Loss',  accumulated_aug_loss/(i+1),i+epoch*num_of_iter)
        writer.add_scalar('Train/Aug_Loss',  aug_loss,i+epoch*num_of_iter)
        writer.add_scalar('Train/Mean_Conf', accumulated_mean_conf/(i+1),i+epoch*num_of_iter)
        writer.add_scalar('Train/Mask_Ratio', 100 - 100*(num_masked_this_batch/BATCH_SIZE),i+epoch*num_of_iter)

# The Eval
def test(epoch):
    global best_acc
    global test_num
    student_net.eval()
    teacher_net.eval()
    accumulated_loss = 0.0
    accumulated_acc = 0.0
    correct = 0
    total = 0
    # Test On Source
    
    pbar_val = tqdm(range(int(len(val_loader))))
    iter_val = iter(val_loader)

    # dataloader_s1_tqdm = tqdm(testloader.datasets[0])
    # for batch_idx, (inputs, targets) in enumerate(dataloader_s1_tqdm):
    for batch_idx in pbar_val:
        inputs, targets = iter_val.next()
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs = F.softmax(teacher_net(inputs), dim = 1)
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
            'tch_net': teacher_net.state_dict(),
            'stu_net': student_net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt_'+'SemiSVHN'+'_'+str(round(acc,2))+'.pth')
        best_acc = acc
    writer.add_scalar('Final/Acc',best_acc,epoch)

if __name__ == "__main__":
    best_acc = 0.0
    test_num = 0
    for i in range(args.epoch):
        train(i)
        test(i)
