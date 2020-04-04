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
from datasets import DigitsLoader


# Log Through Tensorboard
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='Train On Single Domain Test')
parser.add_argument('--source', default="m", help='domain')
parser.add_argument('--gpu', default="0", help='Use which GPU')
parser.add_argument('--epoch', default=25, type=int, help='Num Of Epochs')
parser.add_argument('--lr', default = 1e-3, type=float, help='The Learning Rate')
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

N_CLASS = 10
BATCH_SIZE = 32
# LEARINING_RATE = 0.0001
LEARINING_RATE = args.lr
NUM_OF_EPOCHS = args.epoch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if (DEVICE == 'cuda'):
    DEVICE = DEVICE+":"+args.gpu

writer = SummaryWriter(log_dir = './runs/uno/' +args.source+'/'+time.asctime(time.localtime(time.time())))
writer.add_text('Text', " Train {} From Scratch".format(SOURCE_DOMAIN,time.asctime(time.localtime(time.time()))))

dataset_names = [SOURCE_DOMAIN]

digitloader = DigitsLoader('./data/', dataset_names, shuffle=True, batch_size = BATCH_SIZE, normalize=True, download=True,num_workers= 4, augment={SOURCE_DOMAIN:1},pin_memory = True)
testloader = DigitsLoader('./data',dataset_names, shuffle = True, batch_size = BATCH_SIZE, normalize =True,num_workers = 4,augment_func = None)

teacher_net = SVHN_MNIST_Model(n_classes = N_CLASS, n_domains = 1)
teacher_net.to(DEVICE)

teacher_optimizer = torch.optim.SGD(teacher_net.parameters(), lr = LEARINING_RATE,momentum = 0.9,weight_decay=4e-5) 
criterion = nn.CrossEntropyLoss()

def train(epoch):
    # //  Fix This, NOT very Elegant
    num_of_iter = len(digitloader.datasets[0])
    pbar_train = tqdm(range(num_of_iter))
    it_s = iter(digitloader.datasets[0])
    accumulated_clf_loss = 0.0
    accumulated_aug_loss = 0.0

    for i in pbar_train:
        input_s, target_s = it_s.next()
        input_s, target_s = input_s.to(DEVICE), target_s.to(DEVICE)

        teacher_net.train()
        teacher_optimizer.zero_grad()

        logits_out_s = teacher_net(input_s,d=0)[1]
        prob_out_s = F.softmax(logits_out_s, dim=1)
        mean_conf_this_batch =  torch.max(prob_out_s,1)[0].mean()
        clf_loss = criterion(logits_out_s, target_s)

        clf_loss.backward()
        teacher_optimizer.step()
        accumulated_clf_loss += clf_loss

        pbar_train.set_description("| The {} Epoch | Loss {:3f}| Mean Conf {:3f}|".format(epoch, accumulated_clf_loss/(i+1), mean_conf_this_batch))

        writer.add_scalar('Train/Clf_Loss', clf_loss,i+epoch*num_of_iter)
        writer.add_scalar('Train/Mean_Conf', mean_conf_this_batch ,i+epoch*num_of_iter)


def test(epoch):
    global best_acc
    teacher_net.eval()

    accumulated_loss = 0.0
    accumulated_acc = 0.0
    correct = 0
    total = 0

    dataloader_s1_tqdm = tqdm(testloader.datasets[0])
    for batch_idx, (inputs, targets) in enumerate(dataloader_s1_tqdm):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs = F.softmax(teacher_net(inputs,d=0)[1], dim = 1)
        _,predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        acc_this_batch = 100*(correct/total)
        accumulated_acc += acc_this_batch
        dataloader_s1_tqdm.set_description("| SOURCE: [{}] | Val Acc Is {:3f}|".format(SOURCE_DOMAIN,accumulated_acc/(1+batch_idx)))
        writer.add_scalar('Val/Acc',accumulated_acc/(1+batch_idx),epoch)

    acc = 100.*correct/total
    if acc > best_acc:
        state = {
            'tch_net': teacher_net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint/uno'):
            os.mkdir('checkpoint/uno')
        torch.save(state, './checkpoint/uno/ckpt_'+args.source+'_'+str(round(acc,2))+'.pth')
        best_acc = acc



if __name__ == "__main__":
# The Main Loop
    # The Training
    best_acc = 0.0
    for epoch in range(NUM_OF_EPOCHS):
        train(epoch)
        test(epoch)

    


