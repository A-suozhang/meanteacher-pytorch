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


# ----------- The Quantized Training ---------------------------

import nics_fix_pt as nfp
import nics_fix_pt.nn_fix as nnf
import numpy as np


BITWIDTH = 8
BITWIDTH_GRAD = 8

def _generate_default_fix_cfg(names, scale=0, bitwidth=8, method=0):
    return {
        n: {
            "method": torch.autograd.Variable(
                torch.IntTensor(np.array([method])), requires_grad=False
            ),
            "scale": torch.autograd.Variable(
                torch.IntTensor(np.array([scale])), requires_grad=False
            ),
            "bitwidth": torch.autograd.Variable(
                torch.IntTensor(np.array([bitwidth])), requires_grad=False
            ),
            # "range_method": nfp.RangeMethod.RANGE_MAX_TENPERCENT
            # "range_method": nfp.RangeMethod.RANGE_SWEEP
            "range_method": nfp.RangeMethod.RANGE_MAX
        }
        for n in names
    }

# class Sandbox_net(nnf.FixTopModule):

#     def __init__(self, fix_bn = True, fix_grad = True):
#         super(Sandbox_net, self).__init__()


#         self.conv1_fix_params = _generate_default_fix_cfg(
#             ["weight", "bias"], method=1, bitwidth=BITWIDTH
#         )  # BITWIDTH)
#         self.bn1_fix_params = _generate_default_fix_cfg(
#             ["weight", "bias", "running_mean", "running_var"],
#             method=1,
#             bitwidth=BITWIDTH,
#         )
#         self.fix_params = [
#             _generate_default_fix_cfg(["activation"], method=1, bitwidth=BITWIDTH)
#             for _ in range(20)
#         ]
#         self.fc_fix_params = _generate_default_fix_cfg(
#             ["weight", "bias"], method=1, bitwidth=BITWIDTH
#         )  # BITWIDTH)

#         if fix_grad:
#             self.conv1_fix_params_grad = _generate_default_fix_cfg(
#                 ["weight", "bias"], method=1, bitwidth=BITWIDTH_GRAD
#             )  
#             self.bn1_params_grad = _generate_default_fix_cfg(
#                 ["weight", "bias"],
#                 method=1,
#                 bitwidth=BITWIDTH_GRAD,
#             )
#             self.fix_params_grad = [
#                 _generate_default_fix_cfg(["activation"], method=1, bitwidth=BITWIDTH_GRAD)
#                 for _ in range(20)
#             ]

        
        
    # def forward(self, x):
    #     # x = self.norm(x)
    #     x = self.fix0(x)
    #     x = self.fix2(self.bn1_1(F.relu(self.fix1(self.conv1_1(x)))))
#
#from torch.nn.parameter import Parameter
#from torch.nn import init
#from torch.autograd import Variable
#
#class MyBN(nn.Module):
#    def __init__(self, num_features, eps = 1e-5, momentum = 0.1, affine = True, track_running_stats = True):
#        super(MyBN, self).__init__()
#        self.num_features = num_features
#        self.eps = eps
#        self.momentum = momentum
#        self.track_running_stats = track_running_stats
#        self.affine = affine
#        # self.mean = Variable(torch.Tensor(num_features))
#        # self.var = Variable(torch.Tensor(num_features))
#        # self.mean = torch.Tensor(num_features).requires_grad_()
#        # self.var = torch.Tensor(num_features).requires_grad_()
#
#
#        if self.affine:
#            self.weight = Parameter(torch.Tensor(num_features))
#            self.bias = Parameter(torch.Tensor(num_features))
#
#        if self.track_running_stats:
#            self.register_buffer('running_mean', torch.zeros(num_features))
#            self.register_buffer('running_var', torch.zeros(num_features))
#            self.register_buffer('num_batches_tracked', torch.zeros(num_features, dtype = torch.long))
#        self.reset_parameters()
#
#    def reset_running_stats(self):
#        if self.track_running_stats:
#            self.running_mean.zero_()
#            self.running_var.fill_(1)
#            self.num_batches_tracked.zero_()
#
#    def reset_parameters(self):
#        self.reset_running_stats()
#        if self.affine:
#            init.ones_(self.weight)
#            init.zeros_(self.bias)
#
#    def _check_input_dim(self, input):
#        if (input.dim() != 4):
#            raise ValueError("The Input Should Be in [Num_Batch, Channel, W, H]")
#
#    def forward(self, inputs):
#        self._check_input_dim(inputs)
#
#        self.mean = inputs.mean(dim=[0,2,3])
#        mean = self.mean.reshape([1, -1, 1, 1])
#        self.var = ((inputs - mean) ** 2).mean(dim=[0, 2, 3])
#        var = self.var.reshape([1, -1, 1, 1])
#        # self.var = inputs.var(dim=[0,2,3], unbiased = False)
#
#        self.running_mean = self.mean*self.momentum + self.running_mean*(1-self.momentum)
#        self.running_var = self.var*self.momentum + self.running_var*(1-self.momentum)
#
#        self.whitened_inputs = (inputs - mean) / \
#                          torch.sqrt(var + self.eps)
#        output = self.whitened_inputs * self.weight[:, None, None] + self.bias[:, None, None]
#
#        return output
#

from .mymodules import *
from nics_fix_pt import register_fix_module
register_fix_module(MyBN)
import nics_fix_pt.nn_fix as nnf


class MyNet_fix(nnf.FixTopModule):

    def __init__(self, fix=True, fix_bn=True, fix_grad=True):
        super(MyNet_fix, self).__init__()

        print("fix bn: {}; fix grad: {}".format(fix_bn, fix_grad))
        # fix configurations (data/grad) for parameters/buffers
        self.fix_param_cfgs = {}
        self.fix_grad_cfgs = {}
        layers = [("conv1_1", 128, 3), ("bn1_1",), ("conv1_2", 128, 3), ("bn1_2",),
                  ("conv1_3", 128, 3), ("bn1_3",), ("conv2_1", 256, 3), ("bn2_1",),
                  ("conv2_2", 256, 3), ("bn2_2",), ("conv2_3", 256, 3), ("bn2_3",),
                  ("conv3_1", 512, 3), ("bn3_1",), ("nin3_2", 256, 1), ("bn3_2",),
                  ("nin3_3", 128, 1), ("bn3_3",), ("fc4", 10)]
        for layer_cfg in layers:
            name = layer_cfg[0]
            if "bn" in name and not fix_bn:
                continue
            # data fix configv
            if fix:
                self.fix_param_cfgs[name] = _generate_default_fix_cfg(
                    ["weight", "bias", "running_mean", "running_var"] if "bn" in name else ["weight", "bias"],
                    # ["weight", "bias"] if "bn" in name else ["weight", "bias"],
                    method=1, bitwidth=BITWIDTH
                )
            else:
                self.fix_param_cfgs[name] = _generate_default_fix_cfg(
                    ["weight", "bias", "running_mean", "running_var"] if "bn" in name else ["weight", "bias"],
                    # ["weight", "bias"] if "bn" in name else ["weight", "bias"],
                    method=0, bitwidth=BITWIDTH
                )
            if fix_grad:
                # grad fix config
                self.fix_grad_cfgs[name] = _generate_default_fix_cfg(
                    ["weight", "bias"], method=1, bitwidth=BITWIDTH_GRAD
                )

        # fix configurations for activations
        # data fix config
        if fix:
            self.fix_act_cfgs = [
                _generate_default_fix_cfg(["activation"], method=1, bitwidth=BITWIDTH)
                for _ in range(20)
            ]
        else:
            self.fix_act_cfgs = [
                _generate_default_fix_cfg(["activation"], method=0, bitwidth=BITWIDTH)
                for _ in range(20)
            ]

        if fix_grad:
            # grad fix config
            self.fix_act_grad_cfgs = [
                _generate_default_fix_cfg(["activation"], method=1, bitwidth=BITWIDTH_GRAD)
                for _ in range(20)
            ]

        # construct layers
        cin = 3
        for layer_cfg in layers:
            name = layer_cfg[0]
            if "conv" in name or "nin" in name:
                # convolution layers
                cout, kernel_size = layer_cfg[1:]
                layer = nnf.Conv2d_fix(cin, cout,
                                       nf_fix_params=self.fix_param_cfgs[name],
                                       nf_fix_params_grad=self.fix_grad_cfgs[name] if fix_grad else None,
                                       kernel_size=kernel_size,
                                       padding=(kernel_size-1)//2 if name != "conv3_1" else 0)
                cin = cout
            elif "bn" in name:
                # bn layers
                if fix_bn:
                    # layer = nnf.BatchNorm2d_fix(
                    layer = nnf.MyBN_fix(
                        cin,
                        nf_fix_params=self.fix_param_cfgs[name],
                        nf_fix_params_grad=self.fix_grad_cfgs[name] if fix_grad else None)
                else:
                    layer = nn.BatchNorm2d(cin)
            elif "fc" in name:
                # fully-connected layers
                cout = layer_cfg[1]
                layer = nnf.Linear_fix(cin, cout,
                                       nf_fix_params=self.fix_param_cfgs[name],
                                       nf_fix_params_grad=self.fix_grad_cfgs[name] if fix_grad else None)
                cin = cout
            # call setattr
            setattr(self, name, layer)

        for i in range(20):
            setattr(self, "fix" + str(i), nnf.Activation_fix(
                nf_fix_params=self.fix_act_cfgs[i],
                nf_fix_params_grad=self.fix_act_grad_cfgs[i] if fix_grad else None))

        self.pool1 = nn.MaxPool2d((2, 2))
        self.pool2 = nn.MaxPool2d((2, 2))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.fix0(x)
        x = self.fix2(F.relu(self.bn1_1(self.fix1(self.conv1_1(x)))))
        x = self.fix4(F.relu(self.bn1_2(self.fix3(self.conv1_2(x)))))
        x = self.pool1(self.fix6(F.relu(self.bn1_3(self.fix5(self.conv1_3(x))))))
        x = self.fix8(F.relu(self.bn2_1(self.fix7(self.conv2_1(x)))))
        x = self.fix10(F.relu(self.bn2_2(self.fix9(self.conv2_2(x)))))
        x = self.pool2(self.fix12(F.relu(self.bn2_3(self.fix11(self.conv2_3(x))))))
        x = self.fix14(F.relu(self.bn3_1(self.fix13(self.conv3_1(x)))))
        x = self.fix16(F.relu(self.bn3_2(self.fix15(self.nin3_2(x)))))
        x = self.fix18(F.relu(self.bn3_3(self.fix17(self.nin3_3(x)))))
        # ----
        # x = self.fix2(F.relu(self.bn1_1(self.conv1_1(x))))
        # x = self.fix4(F.relu(self.bn1_2(self.conv1_2(x))))
        # x = self.pool1(self.fix6(F.relu(self.bn1_3(self.conv1_3(x)))))
        # x = self.fix8(F.relu(self.bn2_1(self.conv2_1(x))))
        # x = self.fix10(F.relu(self.bn2_2(self.conv2_2(x))))
        # x = self.pool2(self.fix12(F.relu(self.bn2_3(self.conv2_3(x)))))
        # x = self.fix14(F.relu(self.bn3_1(self.conv3_1(x))))
        # x = self.fix16(F.relu(self.bn3_2(self.nin3_2(x))))
        # x = self.fix18(F.relu(self.bn3_3(self.nin3_3(x))))
        # x = F.avg_pool2d(x, 6)
        x = self.avg_pool(x)
        x = x.view(-1, 128)
        x = self.fix19(self.fc4(x))

        return x

# -----------------------------------------------------






