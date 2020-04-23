import torch
from torch import nn
from torch.nn import functional as F

class ConditionalBatchNorm(nn.Module):
    
    def __init__(self, *args, n_domains = 1, bn_func = nn.BatchNorm2d, **kwargs):
        
        super(ConditionalBatchNorm, self).__init__()
        
        self.n_domains = n_domains
        self.layers    = [bn_func(*args, affine=False, **kwargs) for i in range(n_domains)]

        self.bias  = nn.Parameter(torch.zeros(args[0]).view(1,-1,1,1))
        self.scale = nn.Parameter(torch.ones(args[0]).view(1,-1,1,1))
        
    def _apply(self, fn): 
        super(ConditionalBatchNorm, self)._apply(fn)
        for layer in self.layers:
            layer._apply(fn)
        
    def parameters(self, d=0):
        return super().parameters()
        
    def forward(self, x, d):
                
        layer = self.layers[d]
        return layer(x) * self.scale + self.bias 

class SVHN_MNIST_Model(nn.Module):
    
    def __init__(self, n_classes=10, n_domains=2):
        super(SVHN_MNIST_Model, self).__init__()
        
        self.conditional_layers = []
        self.n_domains = n_domains
        
        self.norm = nn.InstanceNorm2d(3, affine=False,
                momentum=0,
                track_running_stats=False)
        
        self.conv1_1 = nn.Conv2d(3, 128, (3, 3), padding=1)
        self.conv1_1_bn = self._batch_norm(128)
        self.conv1_2 = nn.Conv2d(128, 128, (3, 3), padding=1)
        self.conv1_2_bn = self._batch_norm(128)
        self.conv1_3 = nn.Conv2d(128, 128, (3, 3), padding=1)
        self.conv1_3_bn = self._batch_norm(128)
        self.pool1 = nn.MaxPool2d((2, 2))
        # self.drop1 = nn.Dropout()

        self.conv2_1 = nn.Conv2d(128, 256, (3, 3), padding=1)
        self.conv2_1_bn = self._batch_norm(256)
        self.conv2_2 = nn.Conv2d(256, 256, (3, 3), padding=1)
        self.conv2_2_bn = self._batch_norm(256)
        self.conv2_3 = nn.Conv2d(256, 256, (3, 3), padding=1)
        self.conv2_3_bn = self._batch_norm(256)
        self.pool2 = nn.MaxPool2d((2, 2))
        # self.drop2 = nn.Dropout()

        self.conv3_1 = nn.Conv2d(256, 512, (3, 3), padding=0)
        self.conv3_1_bn = self._batch_norm(512)
        self.nin3_2 = nn.Conv2d(512, 256, (1, 1), padding=1)
        self.nin3_2_bn = self._batch_norm(256)
        self.nin3_3 = nn.Conv2d(256, 128, (1, 1), padding=1)
        self.nin3_3_bn = self._batch_norm(128)

        self.fc4 = nn.Linear(128, n_classes)
        
    def _batch_norm(self, *args, **kwargs):
        
        layer = ConditionalBatchNorm(*args, n_domains=self.n_domains, **kwargs)
        self.conditional_layers.append(layer)  
        return layer
    
    def __call__(self, x, d=0):
        
        return self.forward(x, d)
        
    
    def forward(self, x, d=0):
        x = self.norm(x)
        
        x = F.relu(self.conv1_1_bn(self.conv1_1(x), d))
        x = F.relu(self.conv1_2_bn(self.conv1_2(x), d))
        x = self.pool1(F.relu(self.conv1_3_bn(self.conv1_3(x), d)))
        x = self.drop1(x)

        x = F.relu(self.conv2_1_bn(self.conv2_1(x), d))
        x = F.relu(self.conv2_2_bn(self.conv2_2(x), d))
        x = self.pool2(F.relu(self.conv2_3_bn(self.conv2_3(x), d)))
        x = self.drop2(x)

        x = F.relu(self.conv3_1_bn(self.conv3_1(x), d))
        x = F.relu(self.nin3_2_bn(self.nin3_2(x), d))
        x = F.relu(self.nin3_3_bn(self.nin3_3(x), d))

        x = F.avg_pool2d(x, 6)
        z = x = x.view(-1, 128)

        x = self.fc4(x)
        return z, x
    
    def conditional_params(self, d=0):
        for module in self.conditional_layers:
            for p in module.parameters(d):
                yield p

    def parameters(self, d=0, yield_shared=True, yield_conditional=True):

        if yield_shared:
            for param in super(SVHN_MNIST_Model, self).parameters():
                yield param

        if yield_conditional:
            for param in self.conditional_params(d):
                yield param

from .mymodules import *

class MyNet(nn.Module):

    def __init__(self, useMyBN = True, useDropout = False):
        super(MyNet, self).__init__()

        # self.norm = nn.InstanceNorm2d(3, affine=False,
        #         momentum=0,
        #         track_running_stats=False)
        if useMyBN:
            self.conv1_1 = nn.Conv2d(3, 128, (3, 3), padding=1)
            self.conv1_1_bn = MyBN(128)
            self.conv1_2 = nn.Conv2d(128, 128, (3, 3), padding=1)
            self.conv1_2_bn = MyBN(128)
            self.conv1_3 = nn.Conv2d(128, 128, (3, 3), padding=1)
            self.conv1_3_bn = MyBN(128)
            self.pool1 = nn.MaxPool2d((2, 2))
            self.conv2_1 = nn.Conv2d(128, 256, (3, 3), padding=1)
            self.conv2_1_bn = MyBN(256)
            self.conv2_2 = nn.Conv2d(256, 256, (3, 3), padding=1)
            self.conv2_2_bn = MyBN(256)
            self.conv2_3 = nn.Conv2d(256, 256, (3, 3), padding=1)
            self.conv2_3_bn = MyBN(256)
            self.pool2 = nn.MaxPool2d((2, 2))

            self.conv3_1 = nn.Conv2d(256, 512, (3, 3), padding=0)
            self.conv3_1_bn = MyBN(512)
            self.nin3_2 = nn.Conv2d(512, 256, (1, 1), padding=1)
            self.nin3_2_bn = MyBN(256)
            self.nin3_3 = nn.Conv2d(256, 128, (1, 1), padding=1)
            self.nin3_3_bn = MyBN(128)

            self.fc4 = nn.Linear(128, 10)

        else:
            self.conv1_1 = nn.Conv2d(3, 128, (3, 3), padding=1)
            self.conv1_1_bn = nn.BatchNorm2d(128)
            self.conv1_2 = nn.Conv2d(128, 128, (3, 3), padding=1)
            self.conv1_2_bn = nn.BatchNorm2d(128)
            self.conv1_3 = nn.Conv2d(128, 128, (3, 3), padding=1)
            self.conv1_3_bn = nn.BatchNorm2d(128)
            self.pool1 = nn.MaxPool2d((2, 2))

            self.conv2_1 = nn.Conv2d(128, 256, (3, 3), padding=1)
            self.conv2_1_bn = nn.BatchNorm2d(256)
            self.conv2_2 = nn.Conv2d(256, 256, (3, 3), padding=1)
            self.conv2_2_bn = nn.BatchNorm2d(256)
            self.conv2_3 = nn.Conv2d(256, 256, (3, 3), padding=1)
            self.conv2_3_bn = nn.BatchNorm2d(256)
            self.pool2 = nn.MaxPool2d((2, 2))

            self.conv3_1 = nn.Conv2d(256, 512, (3, 3), padding=0)
            self.conv3_1_bn = nn.BatchNorm2d(512)
            self.nin3_2 = nn.Conv2d(512, 256, (1, 1), padding=1)
            self.nin3_2_bn = nn.BatchNorm2d(256)
            self.nin3_3 = nn.Conv2d(256, 128, (1, 1), padding=1)
            self.nin3_3_bn = nn.BatchNorm2d(128)

            self.fc4 = nn.Linear(128, 10)

        if useDropout:
            self.useDropout = useDropout
            self.drop1 = nn.Dropout()
            self.drop2 = nn.Dropout()
        else:
            self.useDropout = False


    def forward(self, x):
        # x = self.norm(x)
        
        x = F.relu(self.conv1_1_bn(self.conv1_1(x)))
        x = F.relu(self.conv1_2_bn(self.conv1_2(x)))
        x = self.pool1(F.relu(self.conv1_3_bn(self.conv1_3(x))))
        if self.useDropout:
            x = self.drop1(x)

        x = F.relu(self.conv2_1_bn(self.conv2_1(x)))
        x = F.relu(self.conv2_2_bn(self.conv2_2(x)))
        x = self.pool2(F.relu(self.conv2_3_bn(self.conv2_3(x))))
        if self.useDropout:
            x = self.drop2(x)


        x = F.relu(self.conv3_1_bn(self.conv3_1(x)))
        x = F.relu(self.nin3_2_bn(self.nin3_2(x)))
        x = F.relu(self.nin3_3_bn(self.nin3_3(x)))

        x = F.avg_pool2d(x, 6)
        z = x = x.view(-1, 128)

        x = self.fc4(x)

        return x

class DigitNet (nn.Module):
    def __init__(self):
        super(DigitNet, self).__init__()

        self.conv1_1 = nn.Conv2d(3, 32, (5, 5))
        self.conv1_1_bn = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d((2, 2))

        self.conv2_1 = nn.Conv2d(32, 64, (3, 3))
        self.conv2_1_bn = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, (3, 3))
        self.conv2_2_bn = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d((2, 2))

        # self.drop1 = nn.Dropout()
        self.fc3 = nn.Linear(1600, 256)
        self.fc4 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1_1_bn(self.conv1_1(x))))

        x = F.relu(self.conv2_1_bn(self.conv2_1(x)))
        x = self.pool2(F.relu(self.conv2_2_bn(self.conv2_2(x))))
        x = x.view(-1, 1600)
        # x = self.drop1(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x






