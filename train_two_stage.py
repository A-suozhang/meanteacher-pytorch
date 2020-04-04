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

from net import *
# import augmentation # Containde In Salad
from optim import EMAWeightOptimizer
from models.mobilenet import MobileNet
from models.mobilenetv2 import *
from models.vgg import *
from models.resnet import *

# Use Salad
# from salad.datasets import DigitsLoader
# from salad.datasets import MNIST, USPS, SVHN, Synth
from datasets import DigitsLoader

# Log Through Tensorboard
from tensorboardX import SummaryWriter

# Define Args
parser = argparse.ArgumentParser(description='PyTorch Self_Ensemble DA')
parser.add_argument('--source', default="m", help='source domain')
parser.add_argument('--target', default="s", help='target domain')
parser.add_argument('--gpu', default="0", help='Use which GPU')
parser.add_argument('--th', default=0.97, type=float, help='The Confidence Threshold')
parser.add_argument('--epoch', default=25, type=int, help='Num Of Epochs')
parser.add_argument('--lr', default = 5e-3, type=float, help='The Learning Rate')
parser.add_argument('--ratio', default=1, type=float, help="The Ratio between Clf & Con Loss")
parser.add_argument('--opt', default='sgd', help="The Optim Method")
parser.add_argument('--net', default='my', help="The Network Structure")
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

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if (DEVICE == 'cuda'):
    DEVICE = DEVICE+":"+args.gpu

useBCE = True # Use BCE Loss in Class Balance
global best_acc


writer = SummaryWriter(log_dir = './runs/' +args.source+'/'+time.asctime(time.localtime(time.time())))
writer.add_text('Text', " Using [{}] with {} : Transfer From {} To {} | Hyper Params Are : LR:{} Batch Size:{} Ratio:{} th:{}".format(args.net, args.opt, SOURCE_DOMAIN, TARGET_DOMAIN, args.lr, BATCH_SIZE, args.ratio, args.th) + time.asctime(time.localtime(time.time())))


dataset_names = [SOURCE_DOMAIN, TARGET_DOMAIN]

digitloader = DigitsLoader('./data/', dataset_names, shuffle=True, batch_size = BATCH_SIZE, normalize=True, download=True,num_workers= 4, augment={TARGET_DOMAIN: 2},pin_memory = True)
testloader = DigitsLoader('./data',dataset_names, shuffle = True, batch_size = BATCH_SIZE, normalize =True,num_workers = 4,augment_func = None)

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
    teacher_net = MyNet()
    student_net = MyNet()

teacher_net.to(DEVICE)
student_net.to(DEVICE)

for _, param in enumerate(teacher_net.parameters()):
    param.requires_grad = False

if (args.opt == "sgd"):
    student_optimizer = torch.optim.SGD(student_net.parameters(), lr = LEARINING_RATE,momentum = 0.99,weight_decay=5e-4) 
elif(args.opt == "adam"):
    student_optimizer = torch.optim.Adam(student_net.parameters(), lr = LEARINING_RATE,weight_decay=5e-4) 
else:
    print("Unrecognized Optim Method, Use SGD as default")
    student_optimizer = torch.optim.SGD(student_net.parameters(), lr = LEARINING_RATE,momentum = 0.99,weight_decay=5e-4) 
# The Teacher's Param Is Changeing According To The Student
teacher_optimizer = EMAWeightOptimizer(teacher_net, student_net, alpha = TEACHRE_ALPHA)
criterion = nn.CrossEntropyLoss()


def BinaryCrossEntropy(pred, tgt):
    inv_tgt = -tgt + 1.0
    inv_pred = -pred + 1.0 + 1e-6
    return -(tgt * torch.log(pred + 1.0e-6) + inv_tgt * torch.log(inv_pred))


def getClsBalanceLoss(pred, target):
    if (useBCE):
        return BinaryCrossEntropy(pred, target)
    else:
        return -torch.log(pred + 1.0e-6)

def GetAugLoss(student_confs, teacher_confs):

    # Do The Confidence Thresholding
    tecaher_max_conf  = torch.max(teacher_confs,1)[0]   # [0] being the value, [1] as for index
    mean_conf_this_batch = tecaher_max_conf.mean()
    mask_this_batch = (tecaher_max_conf < CONFIDENCE_THRESHOLD).float()            # [0,0,1,0,0,1] - Per Batch Sample
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
    class_balance_loss = getClsBalanceLoss(avg_conf_per_class,float(1/N_CLASS))
    class_balance_loss = class_balance_loss.sum()*(num_masked_this_batch/BATCH_SIZE)

    aug_loss = consist_loss + CLASS_BALANCE_INDEX*class_balance_loss

    return aug_loss, num_masked_this_batch, mean_conf_this_batch




# Only Training On Source Domain, Traditional Way
def train_on_source(epoch):

    # Only Train Student & Copy It Tp Teacher
    num_of_iter = len(digitloader.datasets[0])
    pbar_train = tqdm(range(num_of_iter))
    it_s = iter(digitloader.datasets[0])    # Get Augmented Source Domain Data( 1 piece)
    accumulated_clf_loss = 0.0

    for i in pbar_train:
        input_s, target_s = it_s.next()

        input_s, target_s = input_s.to(DEVICE), target_s.to(DEVICE)
        student_optimizer.zero_grad()
        
        student_net.train()
        teacher_net.train()

        logits_out_s = student_net(input_s)
        # Let The teacher BN to Fit (Or Teacher Will Perform Bad When Val)
        # Howrver If you want To Domain Adapt, You May wanna comment this to let the target network to fit on a new domain
        # logits_out_t = teacher_net(input_s) 
        clf_loss = criterion(logits_out_s, target_s)

        clf_loss.backward()
        student_optimizer.step()
        teacher_optimizer.step()

        accumulated_clf_loss += clf_loss       

        pbar_train.set_description("| The {} Epoch | Loss {:3f}| Cls Loss {:3f}".\
        format(epoch, clf_loss,accumulated_clf_loss/(i+1)))

        writer.add_scalar('Train/Clf_Loss', clf_loss,i+epoch*num_of_iter)

    # Test On Source Domain
    global best_acc
    student_net.eval()
    teacher_net.eval()
    accumulated_acc_s = 0.0
    accumulated_acc_t = 0.0
    correct_s = 0
    correct_t = 0
    total = 0   

    pbar_val = tqdm(range(num_of_iter))
    iter_val = iter(testloader.datasets[0])  
    for batch_idx in pbar_val:
        inputs, targets = iter_val.next()
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        outputs_s = F.softmax(student_net(inputs), dim = 1)
        outputs_t = F.softmax(teacher_net(inputs), dim = 1)
        _,predicted_s = outputs_s.max(1)
        _,predicted_t = outputs_t.max(1)
        total += targets.size(0)
        correct_s += predicted_s.eq(targets).sum().item()
        correct_t += predicted_t.eq(targets).sum().item()
        acc_this_batch_s = 100*(correct_s/total)
        acc_this_batch_t = 100*(correct_t/total)
        accumulated_acc_s += acc_this_batch_s
        accumulated_acc_t += acc_this_batch_t
        # dataloader_s1_tqdm.set_description("| SOURCE: [{}] | Val Acc Is {:3f}|".format(SOURCE_DOMAIN,accumulated_acc/(1+batch_idx)))
        pbar_val.set_description("| SOURCE: [{}] | Val Acc Is Stu {:3f} TCH {:3f}".format(SOURCE_DOMAIN,acc_this_batch_s,acc_this_batch_t))
        writer.add_scalar('Val/Stu_Acc',accumulated_acc_s/(1+batch_idx),epoch)
        writer.add_scalar('Val/Tch_Acc',accumulated_acc_t/(1+batch_idx),epoch)

    # Save The Best Model
    acc = 100.*correct_t/total
    if acc > best_acc:
        state = {
            'tch_net': teacher_net.state_dict(),
            'stu_net': student_net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt_'+args.source+'.pth')
        best_acc = acc
    writer.add_scalar('Final/Acc',best_acc,epoch)


# Testing On Taregt Domain, While Updating The BN Param Based on Target Domain Data
def test_on_target(epoch):

    accumulated_mean_conf = 0.0

    # Load The Model
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt_'+args.source+'.pth')
    teacher_net.load_state_dict(checkpoint['tch_net'])

    # Val On Target Domain
    num_of_iter = len(digitloader.datasets[1])
    pbar_train = tqdm(range(num_of_iter))
    it_t = iter(testloader.datasets[1])
    teacher_net.eval()
    student_net.eval()
    accumulated_acc = 0.0
    correct = 0
    total = 0

    # Test On Traget Domain
    for i in pbar_train:
        
        inputs, targets = it_t.next()
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        teacher_logits_out_t = teacher_net(inputs)
        student_logits_out_t = student_net(inputs)
        student_prob_out_t = F.softmax(student_logits_out_t, dim=1)
        teacher_prob_out_t = F.softmax(teacher_logits_out_t, dim=1)
        _,predicted = teacher_prob_out_t.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        acc_this_batch = 100*(correct/total)
        accumulated_acc += acc_this_batch

        # clf_loss = criterion(teacher_logits_out_t, targets)
        # clf_loss.backward() # ? Maybe No Need?, cant Tch-net no GRAD
        
        aug_loss, num_masked_this_batch, mean_conf_this_batch = GetAugLoss(student_prob_out_t, teacher_prob_out_t)

        accumulated_mean_conf += mean_conf_this_batch

        pbar_train.set_description("| The {} Epoch | Acc: {:3f} | {:3f} MeanConf | {:3f}% Was Masked |".\
        format(epoch, accumulated_acc/(i+1), accumulated_mean_conf/(i+1), 100*(num_masked_this_batch/BATCH_SIZE)))

        writer.add_scalar('Val/Mean_Conf', accumulated_mean_conf/(i+1),i+epoch*num_of_iter)
    
    
    
    # # Prepare Data On Target Domain
    # pbar_train = tqdm(range(num_of_iter))
    # it_t = iter(digitloader.datasets[1])
    # # Train The  Target Domain (Update the BN running mean/var)
    # teacher_net.train()
    # for i in pbar_train:

    #     input_t0, input_t1, label_t1 = it_t.next()
    #     input_t0, input_t1, label_t1 = input_t0.to(DEVICE), input_t1.to(DEVICE), label_t1.to(DEVICE)

    #     teacher_logits_out_t = teacher_net(input_t1)


    state = {
            'tch_net': teacher_net.state_dict(),
            'stu_net': student_net.state_dict(),
            'epoch': epoch,
        }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt_'+args.source+'.pth')

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
        aug_loss, num_masked_this_batch, mean_conf_this_batch = GetAugLoss(student_prob_out_t, teacher_prob_out_t)
        final_loss = clf_loss + AUG_LOSS_INDEX*aug_loss

        # Do The BackProp
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
    student_net.eval()
    teacher_net.eval()
    accumulated_loss = 0.0
    accumulated_acc = 0.0
    correct = 0
    total = 0
    # Test On Source

    pbar_val = tqdm(range(int(len(testloader.datasets[0])/10)))
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
    # acc = 100.*correct/total
    # if acc > best_acc:
    #     state = {
    #         'tch_net': teacher_net.state_dict(),
    #         'stu_net': student_net.state_dict(),
    #         'acc': acc,
    #         'epoch': epoch,
    #     }
    #     if not os.path.isdir('checkpoint'):
    #         os.mkdir('checkpoint')
    #     torch.save(state, './checkpoint/ckpt_'+args.source+'_'+args.target+'_'+str(round(acc,2))+'.pth')
    #     best_acc = acc
    # writer.add_scalar('Final/Acc',best_acc,epoch)










if __name__ == "__main__":
# The Main Loop
    # The Training
    # best_acc = 0.0
    # for epoch in range(3):
    #     train_on_source(epoch)
    
    checkpoint = torch.load('./checkpoint/ckpt_'+args.source+'.pth')
    student_net.load_state_dict(checkpoint['stu_net'])
    teacher_net.load_state_dict(checkpoint['tch_net'])
    for epoch in range(10):
        train(epoch)
        test(epoch)
