import torch
from torch import nn
from torch.nn import functional as F

# ----------- The Quantized Training ---------------------------

import nics_fix_pt as nfp
import nics_fix_pt.nn_fix as nnf
import numpy as np

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


from models.mymodules import *
from nics_fix_pt import register_fix_module
register_fix_module(MyBN)
import nics_fix_pt.nn_fix as nnf


class MyNet_fix(nnf.FixTopModule):

    def __init__(self, fix=True, fix_bn=True, bitwidths=[8,8,None]):
        '''
        --- The Fixed net ---
        note that here we can only control whether quantize bn or not
        if u want to only fix bn weight&bias but no-fix for runnning values
        use net.set_fix_method() to achieve that
        '''
        super(MyNet_fix, self).__init__()

        print("fix bn: {}; Bitwidths: {}".format(fix_bn, bitwidths))
        fix_grad = bitwidths[2] is not -1
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
            # params fix configv
            if fix:
                self.fix_param_cfgs[name] = _generate_default_fix_cfg(
                    ["weight", "bias", "running_mean", "running_var"] if "bn" in name else ["weight", "bias"],
                    # ["weight", "bias"] if "bn" in name else ["weight", "bias"],
                    method=1, bitwidth=bitwidths[0]
                )
            else:
                self.fix_param_cfgs[name] = _generate_default_fix_cfg(
                    ["weight", "bias", "running_mean", "running_var"] if "bn" in name else ["weight", "bias"],
                    # ["weight", "bias"] if "bn" in name else ["weight", "bias"],
                    method=0, bitwidth=bitwidths[0]
                )

            if bitwidths[2] is not -1:
                # grad fix config
                self.fix_grad_cfgs[name] = _generate_default_fix_cfg(
                    ["weight", "bias"], method=1, bitwidth=bitwidths[2]
                )

        # fix configurations for activations
        if fix:
            self.fix_act_cfgs = [
                _generate_default_fix_cfg(["activation"], method=1, bitwidth=bitwidths[1])
                for _ in range(20)
            ]
        else:
            self.fix_act_cfgs = [
                _generate_default_fix_cfg(["activation"], method=0, bitwidth=bitwidths[1])
                for _ in range(20)
            ]

        if bitwidths[2] is not -1:
            # grad fix config
            self.fix_act_grad_cfgs = [
                _generate_default_fix_cfg(["activation"], method=1, bitwidth=bitwidths[2])
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


def set_fix_mode(net,mode, cfg):
    '''
    0 - Fix Mode None
    1 - Fix Mode Train
    2 - Fix Mode Test
    '''
    if mode == "train":
        if not cfg["fix"]["fix_running"]:
            bn_param_method = nfp.FIX_AUTO
            bn_buffer_method = nfp.FIX_NONE
            net.set_fix_method(nfp.FIX_AUTO, method_by_type={"MyBN_fix": {"weight": bn_param_method, "bias": bn_param_method, "running_mean": nfp.FIX_NONE, "running_var": nfp.FIX_NONE}})
        else: 
            net.set_fix_method(nfp.FIX_AUTO)
    elif mode == "test":
        if not cfg["fix"]["fix_running"]:
            bn_param_method = nfp.FIX_FIXED
            bn_buffer_method = nfp.FIX_AUTO
            net.set_fix_method(nfp.FIX_FIXED, method_by_type={"MyBN_fix": {"weight": bn_param_method, "bias": bn_param_method, "running_mean": nfp.FIX_NONE, "running_var": nfp.FIX_NONE}})
        else:
            net.set_fix_method(nfp.FIX_FIXED)
    elif mode == "none":
        net.set_fix_method(nfp.FIX_NONE)
    else:
        raise Exception("Not Implemented Mode")
    # print("--- Setting the Fix Mode as: {} ---".format(mode))
