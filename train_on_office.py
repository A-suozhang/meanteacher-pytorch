
# ----------------- TODO: List --------------------------
# // Dataset Size Imbalance?
# Haven't get DataParallel (Still Buggy) - BN runnning_mean Not On Same Device
# //  Data Precrocessing(Scale into same size)
# // Data Augumentation
# // Class Balance Scaling (Maybe No Need)
# // Put Inputs On CUDA
# // Add Tensorboard (Maybe)
# Use MyNet
# Test Dropout 
# Test Prunning

import os
import sys
import pickle
import time
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import torch.backends.cudnn as cudnn
from torchvision import transforms
from tqdm import *
import argparse

from models.net import *
# import augmentation # Containde In Salad
from optim import EMAWeightOptimizer
from models.mobilenet import MobileNet
from models.mobilenetv2 import *
from models.vgg import *
from models.resnet import *
from utils import *

# Use Salad
# from salad.datasets import DigitsLoader
# from salad.datasets import MNIST, USPS, SVHN, Synth
from datasets import DigitsLoader

# Log Through Tensorboard
from tensorboardX import SummaryWriter


# Define Args
parser = argparse.ArgumentParser(description='PyTorch Self_Ensemble DA')
parser.add_argument('--source', default="a", help='source domain')
parser.add_argument('--target', default="d", help='target domain')
parser.add_argument('--gpu', default="0", help='Use which GPU')
parser.add_argument('--th', default=0.5, type=float, help='The Confidence Threshold')
parser.add_argument('--epoch', default=10, type=int, help='Num Of Epochs')
parser.add_argument('--lr', default = 1e-3, type=float, help='The Learning Rate')
parser.add_argument('--ratio', default=5, type=float, help="The Ratio between Clf & Con Loss")
parser.add_argument('--opt', default='sgd', help="The Optim Method")
parser.add_argument('--net', default='my', help="The Network Structure")
parser.add_argument('--batch', default=16,type=int, help="The Batch Size")
args = parser.parse_args()

if (args.source == "a"):
    SOURCE_DOMAIN = "amazon"
elif(args.source == "d"):
    SOURCE_DOMAIN = "dslr"
elif(args.source == "w"):
    SOURCE_DOMAIN = "webcam"
elif (args.source == "i"):
    SOURCE_DOMAIN = "imagenet"
elif (args.source == "c"):
    SOURCE_DOMAIN = "caltech"
elif (args.source == "p"):
    SOURCE_DOMAIN = "pascal"
else:
    print(" Not Recognized Source Domain {}, Use Amazon As Default".format(args.source))

if (args.target == "a"):
    TARGET_DOMAIN = "amazon"
elif(args.target == "d"):
    TARGET_DOMAIN = "dslr"
elif(args.target == "w"):
    TARGET_DOMAIN = "webcam"
elif (args.target == "i"):
    TARGET_DOMAIN = "imagenet"
elif (args.target == "c"):
    TARGET_DOMAIN = "caltech"
elif (args.target == "p"):
    TARGET_DOMAIN = "pascal"
else:
    print(" Not Recognized Target Domain {}, Use DSLR As Default".format(args.target))

BATCH_SIZE = args.batch


N_CLASS = 31    
# LEARINING_RATE = 0.0001
LEARINING_RATE = args.lr
# CONFIDENCE_THRESHOLD = 0.97
CONFIDENCE_THRESHOLD = args.th
TEACHRE_ALPHA = 0.99
CLASS_BALANCE_INDEX = 0
AUG_LOSS_INDEX = args.ratio
NUM_OF_EPOCHS = args.epoch



DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if (DEVICE == 'cuda'):
    if (args.gpu is not "m"):
        DEVICE = DEVICE+":"+args.gpu
    
useBCE = True # Use BCE Loss in Class Balance

# writer = SummaryWriter(log_dir = './runs/office/' +args.source+'_'+args.target+'/'+time.asctime(time.localtime(time.time())))
writer = SummaryWriter(log_dir = './runs/office/' +args.source+'_'+args.target+'/'+time.asctime(time.localtime(time.time())))
writer.add_text('Text', " Using [{}] with {} : Transfer From {} To {} | Hyper Params Are : LR:{} Batch Size:{} Ratio:{} th:{}".format(args.net, args.opt, SOURCE_DOMAIN, TARGET_DOMAIN, args.lr, BATCH_SIZE, args.ratio, args.th) + time.asctime(time.localtime(time.time())))


dataset_names = [SOURCE_DOMAIN, TARGET_DOMAIN]

digitloader = DigitsLoader('../datasets/office31/Original_images/', dataset_names, shuffle=True, batch_size = BATCH_SIZE, normalize=True, download=True,num_workers= 4, augment={TARGET_DOMAIN: 2},pin_memory = True)
testloader = DigitsLoader('../datasets/office31/Original_images/',dataset_names, shuffle = True, batch_size = BATCH_SIZE, normalize =True,num_workers = 4,augment_func = None)

teacher_net = torchvision.models.resnet50(pretrained=True)
student_net = torchvision.models.resnet50(pretrained=True)

n_features = teacher_net.fc.in_features


fc0_t = torch.nn.Linear(n_features, n_features)
fc1_t = torch.nn.Linear(n_features, N_CLASS) 

fc0_s = torch.nn.Linear(n_features, n_features)
fc1_s = torch.nn.Linear(n_features, N_CLASS)

teacher_net.fc = torch.nn.Sequential(fc0_t, fc1_t)
student_net.fc = torch.nn.Sequential(fc0_s, fc1_s)           

param_group = []
for k, v in student_net.named_parameters():
            if not k.__contains__('fc'):
                param_group += [{'params': v, 'lr': LEARINING_RATE }]
            else:
                param_group += [{'params': v, 'lr': LEARINING_RATE * 10}]

teacher_net.to(DEVICE)
student_net.to(DEVICE)
if (args.gpu == "m"):
    teacher_net = torch.nn.DataParallel(teacher_net, device_ids=[0,1,2,3])
    student_net = torch.nn.DataParallel(student_net, device_ids=[0,1,2,3])
cudnn.benchmark = True

# Disable BackProp For Teacher
for _, param in enumerate(teacher_net.parameters()):
    param.requires_grad = False

if (args.opt == "sgd"):
    # student_optimizer = torch.optim.SGD(student_net.parameters(), lr = LEARINING_RATE,momentum = 0.99,weight_decay=5e-4) 
    student_optimizer = torch.optim.SGD(student_net.parameters(), lr = LEARINING_RATE,momentum = 0.99) 
elif(args.opt == "adam"):
    # student_optimizer = torch.optim.Adam(student_net.parameters(), lr = LEARINING_RATE,weight_decay=5e-4) 
    student_optimizer = torch.optim.Adam(student_net.parameters(), lr = LEARINING_RATE) 
else:
    print("Unrecognized Optim Method, Use SGD as default")
    # student_optimizer = torch.optim.SGD(student_net.parameters(), lr = LEARINING_RATE,momentum = 0.99,weight_decay=5e-4) 
    student_optimizer = torch.optim.SGD(student_net.parameters(), lr = LEARINING_RATE,momentum = 0.99) 
# The Teacher's Param Is Changeing According To The Student
teacher_optimizer = EMAWeightOptimizer(teacher_net, student_net, alpha = TEACHRE_ALPHA)
criterion = nn.CrossEntropyLoss()

def train(epoch):
    # //  Fix This, NOT very Elegant
    num_of_iter = len(digitloader.datasets[0])
    # num_of_iter = min(len(digitloader.datasets[0]),len(digitloader.datasets[1]))
    pbar_train = tqdm(range(num_of_iter))
    it_s = iter(digitloader.datasets[0])
    it_t = iter(digitloader.datasets[1])
    accumulated_clf_loss = 0.0
    accumulated_aug_loss = 0.0
    accumulated_mean_conf = 0.0

    for i in pbar_train:
        # input_t0, input_t1, label_t1 = it_t.next()
        # input_s, target_s = it_s.next()

        try:
            input_t0, input_t1, label_t1 = it_t.next()
        except StopIteration:
            new_digitloader = DigitsLoader('../datasets/office31/Original_images/', dataset_names, shuffle=True, batch_size = BATCH_SIZE, normalize=True, download=True,num_workers= 4, augment={TARGET_DOMAIN: 2},pin_memory = True)
            it_t = iter(new_digitloader.datasets[1])
            input_t0, input_t1, label_t1 = it_t.next()
            # ! this is not common, but when syn is source, the epoch is too long
            # test(test_num)

        try:
            input_s, target_s = it_s.next()
        except StopIteration:
            new_digitloader = DigitsLoader('../datasets/office31/Original_images/', dataset_names, shuffle=True, batch_size = BATCH_SIZE, normalize=True, download=True,num_workers= 4, augment={TARGET_DOMAIN: 2},pin_memory = True)
            it_s = iter(new_digitloader.datasets[0])
            input_s, target_s = it_s.next()
            # ! this is not common, but when syn is source, the epoch is too long
            # test(test_num)


        # Apply Transform & AUgumentation For Image Here
        # Was Contained In Salad's DigitLoader Implemention
        # input_s, target_s, input_t0, input_t1 = Apply_Transfrom()
        # ---------------------------------------
        input_s, target_s, input_t0, input_t1, label_t1 = input_s.to(DEVICE), target_s.to(DEVICE), input_t0.to(DEVICE), input_t1.to(DEVICE), label_t1.to(DEVICE)

        student_optimizer.zero_grad()
        student_net.train()
        teacher_net.train()

        # Feed Data Into Net
        logits_out_s = student_net(input_s)
        student_logits_out_t = student_net(input_t0)    # The Net Gievs a tuple [BATCH_SIZE*128, BATCH_SIZE*NUM_CLASSES]
        teacher_logits_out_t = teacher_net(input_t1)
        student_prob_out_t = F.softmax(student_logits_out_t, dim=1)
        teacher_prob_out_t = F.softmax(teacher_logits_out_t, dim=1)

        clf_loss = criterion(logits_out_s, target_s)
        aug_loss, num_masked_this_batch, mean_conf_this_batch = GetAugLossWithLogits(student_prob_out_t, teacher_prob_out_t, CONFIDENCE_THRESHOLD, student_logits_out_t, teacher_logits_out_t)
        final_loss = clf_loss + AUG_LOSS_INDEX*aug_loss

        # Do The BackProp
        final_loss.backward()
        student_optimizer.step()
        teacher_optimizer.step()

        accumulated_clf_loss += clf_loss
        accumulated_aug_loss += aug_loss
        accumulated_mean_conf += mean_conf_this_batch

        pbar_train.set_description("| The {} Epoch | Loss {:3f}| Cls Loss {:3f} | Aug Loss {:3f} | {:3f} MeanConf | {:3f}% Has Passed |".\
        format(epoch, final_loss,accumulated_clf_loss/(i+1), accumulated_aug_loss/(i+1), accumulated_mean_conf/(i+1),100*(num_masked_this_batch/BATCH_SIZE)))

        writer.add_scalar('Train/Loss',final_loss,i+epoch*num_of_iter)
        # writer.add_scalar('Train/Clf_Loss', accumulated_clf_loss/(i+1),i+epoch*num_of_iter)
        writer.add_scalar('Train/Clf_Loss', clf_loss,i+epoch*num_of_iter)
        # writer.add_scalar('Train/Aug_Loss',  accumulated_aug_loss/(i+1),i+epoch*num_of_iter)
        writer.add_scalar('Train/Aug_Loss',  aug_loss,i+epoch*num_of_iter)
        writer.add_scalar('Train/Mean_Conf', accumulated_mean_conf/(i+1),i+epoch*num_of_iter)
        writer.add_scalar('Train/Mask_Ratio', 100*(num_masked_this_batch/BATCH_SIZE),i+epoch*num_of_iter)


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
    
    pbar_val = tqdm(range(int(len(testloader.datasets[0]))))
    iter_val = iter(testloader.datasets[0])

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
        pbar_val.set_description("| SOURCE: [{}] | Val Acc Is {:3f}|".format(SOURCE_DOMAIN,accumulated_acc/(1+batch_idx)))
    
    test_num = test_num + 1
    writer.add_scalar('Val/Acc',accumulated_acc/(1+batch_idx),test_num)



    # Test On Target
    student_net.eval()
    teacher_net.eval()
    accumulated_loss = 0.0
    accumulated_acc = 0.0
    correct = 0
    total = 0

    pbar_test = tqdm(range(int(len(testloader.datasets[1]))))
    iter_test = iter(testloader.datasets[1])

    # dataloader_t1_tqdm = tqdm(testloader.datasets[1])
    # for batch_idx, (inputs, targets) in enumerate(dataloader_t1_tqdm):
    for batch_idx in pbar_test:
        inputs, targets = iter_test.next()
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs = F.softmax(teacher_net(inputs), dim = 1)
        _,predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        acc_this_batch = 100*(correct/total)
        accumulated_acc += acc_this_batch
        # dataloader_t1_tqdm.set_description("| TARGET: [{}] | Test Acc Is {:3f}|".format(TARGET_DOMAIN, accumulated_acc/(1+batch_idx)))
        pbar_test.set_description("| TARGET: [{}] | Test Acc Is {:3f}|".format(TARGET_DOMAIN, accumulated_acc/(1+batch_idx)))
    
    test_num = test_num + 1
    writer.add_scalar('Test/Acc',accumulated_acc/(1+batch_idx),test_num)
    
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
        torch.save(state, './checkpoint/ckpt_'+args.source+'_'+args.target+'_'+str(round(acc,2))+'.pth')
        best_acc = acc
    writer.add_scalar('Final/Acc',best_acc,epoch)

if __name__ == "__main__":
# The Main Loop
    # The Training
    best_acc = 0.0
    test_num = 0
    for epoch in range(NUM_OF_EPOCHS):
        train(epoch)
        test(epoch)
        




