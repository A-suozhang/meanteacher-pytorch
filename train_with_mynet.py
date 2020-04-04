
# ----------------- TODO: List --------------------------
# // Dataset Size Imbalance?
# // Haven't get DataParallel (Still Buggy) - BN runnning_mean Not On Same Device
# //  Data Precrocessing(Scale into same size)
# // Data Augumentation
# // Class Balance Scaling (Maybe No Need)
# // Put Inputs On CUDA
# // Add Tensorboard (Maybe)
# // Use MyNet
# // Test Dropout 
# // Test Prunning

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

# import augmentation # Containde In Salad
from optim import EMAWeightOptimizer
from models.mobilenet import MobileNet
from models.mobilenetv2 import *
from models.vgg import *
from models.resnet import *
from models.net import *
from utils import *


# Use Salad
# from salad.datasets import DigitsLoader
# from salad.datasets import MNIST, USPS, SVHN, Synth
from datasets import *

# Log Through Tensorboard
from tensorboardX import SummaryWriter


# Define Args
parser = argparse.ArgumentParser(description='PyTorch Self_Ensemble DA')
parser.add_argument('--source', default="m", help='source domain')
parser.add_argument('--target', default="s", help='target domain')
parser.add_argument('--gpu', default="m", help='Use GPU')
parser.add_argument('--th', default=0.99, type=float, help='The Confidence Threshold')
parser.add_argument('--epoch', default=10, type=int, help='Num Of Epochs')
parser.add_argument('--lr', default = 5e-3, type=float, help='The Learning Rate')
parser.add_argument('--ratio', default=1, type=float, help="The Ratio between Clf & Con Loss")
parser.add_argument('--opt', default='sgd', help="The Optim Method")
parser.add_argument('--net', default='my', help="The Network Structure")
parser.add_argument('--quan', default=False, help="Use Quantize Training")
args = parser.parse_args()

if (args.source == "m"):
    SOURCE_DOMAIN = "mnist"
elif(args.source == "s"):
    SOURCE_DOMAIN = "svhn"
elif(args.source == "u"):
    SOURCE_DOMAIN = "usps"
elif(args.source == "syn"):
    SOURCE_DOMAIN = "synth"
else:
    print(" Not Recognized Source Domain {}, Use Mnist As Default".format(args.source))

if (args.target == "m"):
    TARGET_DOMAIN = "mnist"
elif(args.target == "s"):
    TARGET_DOMAIN = "svhn"
elif(args.target == "u"):
    TARGET_DOMAIN = "usps"
elif(args.target == "syn"):
    TARGET_DOMAIN = "synth"
else:
    print(" Not Recognized Target Domain {}, Use SVHN As Default".format(args.target))

N_CLASS = 10
BATCH_SIZE = 64


# LEARINING_RATE = 0.0001
LEARINING_RATE = args.lr
# CONFIDENCE_THRESHOLD = 0.97
CONFIDENCE_THRESHOLD = args.th
TEACHRE_ALPHA = 0.99
CLASS_BALANCE_INDEX = 0
AUG_LOSS_INDEX = args.ratio
NUM_OF_EPOCHS = args.epoch

useBCE = True # Use BCE Loss in Class Balance
global best_acc



writer = SummaryWriter(log_dir = './runs/' +args.source+'_'+args.target+'/'+time.asctime(time.localtime(time.time())))
writer.add_text('Text', " Using [{}] with {} : Transfer From {} To {} | Hyper Params Are : LR:{} Batch Size:{} Ratio:{} th:{}".format(args.net, args.opt, SOURCE_DOMAIN, TARGET_DOMAIN, args.lr, BATCH_SIZE, args.ratio, args.th) + time.asctime(time.localtime(time.time())))

# -------- Define DataSet ------------s
# Transform Data - Stage 1
# if (SOURCE_DOMAIN == "m"):
#     # dataset_s0 = torchvision.datasets.MNIST(root='./data', train= True, transform=transforms.ToTensor(), download=False)
#     # dataset_s1 = torchvision.datasets.MNIST(root='./data', train= False, transform=transforms.ToTensor(), download=False)
#     # Use DataLoader in salad instead
#     mnist = MNIST('./data/')
# elif(SOURCE_DOMAIN == "s"):
#     svhn = SVHN('./data/')
# elif(SOURCE_DOMAIN == "u"):
#     usps = USPS('./data/')
# elif(SOURCE_DOMAIN == 'sy'):
#     synth = Synth('./data')
# else:
#     print ("Source Domain :{} Not Understood, Using Mnist As Default".format(SOURCE_DOMAIN))


# if (TARGET_DOMAIN == "s"):

#     # dataset_t0 = torchvision.datasets.SVHN(root='./data', transform=transforms.ToTensor(), download=False)
#     # # Load The Training Set, However Not Using The Label
#     # dataset_t1 = torchvision.datasets.SVHN(root='./data', transform=transforms.ToTensor(), download=False)

# # Define DataLoader (Use Salad Instead)
# # dataloader_s0 = torch.utils.data.DataLoader(dataset_s0, shuffle=True, batch_size = 1)
# # dataloader_t0 = torch.utils.data.DataLoader(dataset_t0, shuffle=True, batch_size = 1)
# # dataloader_s1 = torch.utils.data.DataLoader(dataset_s1, shuffle=True, batch_size = 1)
# # dataloader_t1 = torch.utils.data.DataLoader(dataset_t1, shuffle=True, batch_size = 1)

dataset_names = [SOURCE_DOMAIN, TARGET_DOMAIN]

digitloader = DigitsLoader('./data/', dataset_names, shuffle=True, batch_size = BATCH_SIZE, normalize=True, download=True,num_workers= 4, augment={TARGET_DOMAIN: 2},pin_memory = True)
testloader = DigitsLoader('./data',dataset_names, shuffle = True, batch_size = BATCH_SIZE, normalize =True,num_workers = 4,augment_func = None)
# digitloader.datasets is a list of dataloader
# the digitloader[0] returns a list 3 samples - is a pair of sample after different aug & label
# the digitloader[1] returns a list 2 samples - which is the sample drawn & label (Source Domain)


# Define Net
if (args.net == 'vgg'):
    teacher_net = VGG('VGG11')
    student_net = VGG('VGG11')
elif (args.net == 'resnet'):
    teacher_net = ResNet18()
    student_net = ResNet18()
elif (args.net == "mobile"):
    teacher_net = MobileNet()
    student_net = MobileNet()
elif (args.net == "mobilev2"):
    teacher_net = MobileNetV2()
    student_net = MobileNetV2()
else:
    # teacher_net = MyNet()
    # student_net = MyNet()
    teacher_net = DigitNet()
    student_net = DigitNet() 

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if (DEVICE == 'cuda'):
    if (args.gpu is not "m"):
        DEVICE = DEVICE+":"+args.gpu

teacher_net.to(DEVICE)
student_net.to(DEVICE)

# //  Make DataParallel Useful
cudnn.benchmark = True


# Disable BackProp For Teacher
for _, param in enumerate(teacher_net.parameters()):
    param.requires_grad = False


if (args.opt == "sgd"):
    student_optimizer = torch.optim.SGD(student_net.parameters(), lr = LEARINING_RATE,momentum = 0.9,weight_decay=5e-2) 
elif(args.opt == "adam"):
    student_optimizer = torch.optim.Adam(student_net.parameters(), lr = LEARINING_RATE,weight_decay=5e-4) 
else:
    print("Unrecognized Optim Method, Use SGD as default")
    student_optimizer = torch.optim.SGD(student_net.parameters(), lr = LEARINING_RATE,momentum = 0.9,weight_decay=5e-2) 
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
            new_digitloader = DigitsLoader('./data/', dataset_names, shuffle=True, batch_size = BATCH_SIZE, normalize=True, download=True,num_workers= 4, augment={TARGET_DOMAIN: 2},pin_memory = True)
            it_t = iter(new_digitloader.datasets[1])
            input_t0, input_t1, label_t1 = it_t.next()
            # ! this is not common, but when syn is source, the epoch is too long
            test(epoch)  

        try:
            input_s, target_s = it_s.next()
        except StopIteration:
            new_digitloader = DigitsLoader('./data/', dataset_names, shuffle=True, batch_size = BATCH_SIZE, normalize=True, download=True,num_workers= 4, augment={TARGET_DOMAIN: 2},pin_memory = True)
            it_s = iter(new_digitloader.datasets[0])
            input_s, target_s = it_s.next()
            # ! this is not common, but when syn is source, the epoch is too long
            test(epoch)  


        # Apply Transform & AUgumentation For Image Here
        # Was Contained In Salad's DigitLoader Implemention
        # input_s, target_s, input_t0, input_t1 = Apply_Transfrom()
        # ---------------------------------------
        input_s, target_s, input_t0, input_t1, label_t1 = input_s.to(DEVICE), target_s.to(DEVICE), input_t0.to(DEVICE), input_t1.to(DEVICE), label_t1.to(DEVICE)

        
        student_net.train()
        teacher_net.train()

        # Feed Data Into Net
        logits_out_s = student_net(input_s)
        student_logits_out_t = student_net(input_t0)    # The Net Gievs a tuple [BATCH_SIZE*128, BATCH_SIZE*NUM_CLASSES]
        teacher_logits_out_t = teacher_net(input_t1)
        student_prob_out_t = F.softmax(student_logits_out_t, dim=1)
        teacher_prob_out_t = F.softmax(teacher_logits_out_t, dim=1)

        clf_loss = criterion(logits_out_s, target_s)
        # aug_loss, num_masked_this_batch, mean_conf_this_batch = GetAugLoss(student_prob_out_t, teacher_prob_out_t)
        aug_loss, num_masked_this_batch, mean_conf_this_batch = GetAugLossWithLogits(student_prob_out_t, teacher_prob_out_t, CONFIDENCE_THRESHOLD, student_logits_out_t, teacher_logits_out_t)
        final_loss = clf_loss + AUG_LOSS_INDEX*aug_loss

        # Do The BackProp
        student_optimizer.zero_grad()
        final_loss.backward()
        student_optimizer.step()
        teacher_optimizer.step()

        accumulated_clf_loss += clf_loss
        accumulated_aug_loss += aug_loss
        accumulated_mean_conf += mean_conf_this_batch

        pbar_train.set_description("| The {} Epoch | Loss {:3f}| Cls Loss {:3f} | Aug Loss {:3f} | {:3f} MeanConf | {:3f}% Was Masked |".\
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

    if(args.quan):
        if (args.gpu == "m"):
            student_net.module.set_fix_method(nfp.FIX_FIXED)
            teacher_net.module.set_fix_method(nfp.FIX_FIXED)
        else:
            student_net.set_fix_method(nfp.FIX_FIXED)
            teacher_net.set_fix_method(nfp.FIX_FIXED)


    student_net.eval()
    teacher_net.eval()

    accumulated_loss = 0.0
    accumulated_acc = 0.0
    correct = 0
    total = 0
    # Test On Source

    if (args.target == "syn"):
        pbar_val = tqdm(range(int(len(testloader.datasets[0])/10)))
    else:
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
        writer.add_scalar('Val/Acc',accumulated_acc/(1+batch_idx),epoch)



    # Test On Target
    student_net.eval()
    teacher_net.eval()
    accumulated_loss = 0.0
    accumulated_acc = 0.0
    correct = 0
    total = 0

    pbar_test = tqdm(range(int(len(testloader.datasets[1])/10)))
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
        writer.add_scalar('Test/Acc',accumulated_acc/(1+batch_idx),epoch)
    
    # Save The Model
    acc = 100.*correct/total
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
    for epoch in range(NUM_OF_EPOCHS):
        train(epoch)
        test(epoch)
        




