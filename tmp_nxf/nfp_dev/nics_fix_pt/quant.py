# -*- coding: utf-8 -*-

from __future__ import print_function

import copy

import numpy as np
import torch

from nics_fix_pt.utils import get_int
from nics_fix_pt.consts import QuantizeMethod, RangeMethod

__all__ = ["quantitize"]


def _do_quantitize(data, scale, bit_width, symmetric=True, stochastic=False):
    scale_f = scale.to(data.device).float()
    bit_width = bit_width.to(data.device)
    tensor_2 = torch.autograd.Variable(torch.FloatTensor([2.0]),
                                       requires_grad=False).to(data.device)
    if symmetric:
        dynamic_range=2*float(scale)
        maximum = float(scale)
        minimum = -maximum
    else:
        # TODO: consider how to feed in range for insymmetric
        pass
    step = torch.tensor(dynamic_range/float(2.**(bit_width)))
    # step = torch.pow(
    #     tensor_2, (scale_f - (bit_width - 1).float())
    # )
    step = step.to(data.device)

    if stochastic:
        output = torch.clamp(StraightThroughStochasticRound.apply(data / step)*step, minimum, maximum)
    else:
        output = torch.clamp(StraightThroughRound.apply(data / step)*step, minimum, maximum)
    # Symmetric Rounding
    # minimum = -float(2. ** (scale.cpu().data.numpy()))
    # maximum = -minimum
    # maximum = -minimum - step
    # TODO: Even if the quantitize cfg is "auto", some overflow may occur,
    #       and maybe cause some problems.
    #       such as maybe weights won't be able to be trained to change scale
    #       if the learning rate is not big enough.
    # Two possible solutions:
    # * Do not minus step at maximum when training on software, this may cause some
    #   small discrepancy between software simulation and actual hardware deployment.
    # * Modify the `new_scale` calculation.
    return (
        output,
        step,
    )


def quantitize_cfg(data, scale, bitwidth, method, range_method=RangeMethod.RANGE_MAX, stochastic=False):
    if (
        not isinstance(method, torch.autograd.Variable)
        and not torch.is_tensor(method)
        and method == QuantizeMethod.FIX_NONE
    ):
        return data, None

    method_v = get_int(method)

    if method_v == QuantizeMethod.FIX_NONE:
        return data, None
    elif method_v == QuantizeMethod.FIX_AUTO:
        EPS = 1e-5
        range_method_v = get_int(range_method)
        if range_method_v == RangeMethod.RANGE_MAX:
            new_scale = torch.pow(2,torch.ceil(
                torch.log(
                    torch.max(
                        torch.max(torch.abs(data)),
                        # torch.tensor(EPS).float().to(data.device),
                        torch.cuda.FloatTensor([1]).fill_(EPS)
                    )
                )/torch.cuda.FloatTensor([1]).fill_(np.log(2.)))
            )
            scale.data = new_scale.data
            return _do_quantitize(data, scale, bitwidth, stochastic=stochastic)

        elif range_method_v == RangeMethod.RANGE_MAX_TENPERCENT:
            # FIXME: Too slow
            scale = torch.pow(2,torch.ceil(
                torch.log(
                    torch.max(
                        # torch.kthvalue(torch.abs(data.view(-1)), 9 * (data.nelement() // 10))[0],
                        torch.topk(torch.abs(data.view(-1)), data.nelement() // 10)[0][-1],
                        # torch.tensor(EPS).float().to(data.device))
                        torch.cuda.FloatTensor(1).fill_(EPS))
                ) / torch.cuda.FloatTensor([1]).fill_(np.log(2.0))
            ))
            return _do_quantitize(data, scale, bitwidth, stochastic=stochastic)

        elif range_method_v == RangeMethod.RANGE_3SIGMA:
            new_scale = torch.ceil(torch.log(new_boundary))
            new_boundary = torch.max(3*torch.std(data)+torch.abs(torch.mean(data)), torch.tensor(EPS).float().to(data.device),)
            new_scale = torch.pow(2,torch.ceil(torch.log(new_boundary) / np.log(2.0)))
            scale = new_scale
            return _do_quantitize(data, scale, bitwidth, stochastic=stochastic)

        elif range_method_v == RangeMethod.RANGE_SWEEP:
            # Iterat through other scale to find the proper scale to minimize error 
            # Noted that the scale is [(MAX - SWEEP),MAX]
            SWEEP = 3
            temp_scale = torch.ceil(torch.log(torch.max(
                torch.max(abs(data)),
                torch.tensor(EPS).float().to(data.device))) / np.log(2.0))
            for i in range(SWEEP):
                errors[i] = torch.abs(_do_quantitize(data, temp_scale-i, bitwidth)[0] - data).sum()
            new_scale = torch.pow(2,temp_scale - errors.argmin())
            scale.data = new_scale
            return _do_quantitize(data, scale, bitwidth, stochastic=stochastic)

        else:
            raise NotImplementedError()

    elif method_v == QuantizeMethod.FIX_FIXED:
        return _do_quantitize(data, scale, bitwidth)

    raise Exception("Quantitize method not legal: {}".format(method_v))


# https://discuss.pytorch.org/t/how-to-override-the-gradients-for-parameters/3417/6
class StraightThroughRound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.round()

    @staticmethod
    def backward(ctx, g):
        return g

class StraightThroughStochasticRound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # The Binary tensor denoting whether ceil or not, closer to ceil means for probabily choose ceil
        # return x.floor() + (torch.rand(x.shape).to(x.device) > x.ceil() - x)*torch.ones(x.shape).to(x.device)
        return x.floor() + (torch.cuda.FloatTensor(x.shape).uniform_() > x.ceil() - x)*torch.cuda.FloatTensor(x.shape).fill_(1.)

    @staticmethod
    def backward(ctx, g):
        return g



class QuantitizeGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, bitwidth, method, range_method=RangeMethod.RANGE_MAX, stochastic=False):
        # FIXME: save the tensor/variables for backward,
        #        maybe should use `ctx.save_for_backward` for standard practice
        # but `save_for_backward` requires scale/bitwidth/method all being of type `Variable`...
        ctx.saved = (scale, bitwidth, method, range_method, stochastic)
        return x

    @staticmethod
    def backward(ctx, g):
        return quantitize_cfg(g, *ctx.saved)[0], None, None, None, None, None


def quantitize(param, fix_cfg={}, fix_grad_cfg={}, kwarg_cfg={}, name=""):
    # fix_cfg/fix_grad_cfg is the configuration saved;
    # kwarg_cfg is the overriding configuration supplied for each `forward` call
    data_cfg = copy.copy(fix_cfg)
    data_cfg.update(kwarg_cfg.get(name + "_fix", {}))
    grad_cfg = copy.copy(fix_grad_cfg)
    grad_cfg.update(kwarg_cfg.get(name + "_grad_fix", {}))
    method = data_cfg.get("method", QuantizeMethod.FIX_NONE)

    step = 0
    # quantitize data
    out_param = param
    if (
        isinstance(method, torch.autograd.Variable)
        or torch.is_tensor(method)
        or method != QuantizeMethod.FIX_NONE
    ):
        out_param, step = quantitize_cfg(
            out_param,
            data_cfg["scale"],
            data_cfg["bitwidth"],
            data_cfg["method"],
            data_cfg.get("range_method", RangeMethod.RANGE_MAX),
            data_cfg.get("stochastic", False),
        )

    # quantitize gradient
    method = grad_cfg.get("method", QuantizeMethod.FIX_NONE)
    if (
        isinstance(method, torch.autograd.Variable)
        or torch.is_tensor(method)
        or method != QuantizeMethod.FIX_NONE
    ):
        out_param = QuantitizeGradient().apply(
            out_param,
            grad_cfg["scale"],
            grad_cfg["bitwidth"],
            grad_cfg["method"],
            grad_cfg.get("range_method", RangeMethod.RANGE_MAX),
            grad_cfg.get("stochastic", False),
        )

    out_param.data_cfg = data_cfg
    out_param.grad_cfg = grad_cfg
    if param is not out_param:
        # avoid memory leaking: old `buffer` tensors could remain referenced unexpectedly
        if hasattr(param, "nfp_actual_data"):
            del param.nfp_actual_data
            del param.data_cfg
            del param.grad_cfg
        out_param.nfp_actual_data = param  # avoid loop ref
    # NOTE: the returned step is data fix stepsize, not gradient fix step size;
    return out_param, step
