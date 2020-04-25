import nics_fix_pt as nfp
from nics_fix_pt.consts import QuantizeMethod, RangeMethod

import torch
import numpy as np

for j in range(1):
    for i in range(1000):
        err = [0.,0.,0.,0.]

        x=10*torch.randn([1000,1000]) + 50
        fp_scale_sym=torch.tensor([x.abs().max()])
        fp_scale_asym=torch.tensor([x.min(),x.max()])
        scale = torch.tensor([torch.pow(2, torch.ceil(torch.log(torch.tensor(x.abs().max())) / np.log(2.0)))])
        bit_width=torch.tensor([3],dtype=torch.int32)
        # print(scale,fp_scale_asym)

# qx = nfp.quant._do_quantize(x,scale,bit_width)

# Test for Stochastic Rounidng
# accu_qx = torch.zeros(x.shape)
# for i in range(20):
#     qx = nfp.quant.quantize_cfg(x,scale,bit_width,method=QuantizeMethod.FIX_AUTO,range_method=RangeMethod.RANGE_MAX,stochastic=True)
#     accu_qx = accu_qx+qx[0]
# 
# accu_qx = accu_qx/20


        qx_0 = nfp.quant._do_quantize(x,fp_scale_sym,bit_width,stochastic=False,symmetric=True)
        qx_1 = nfp.quant._do_quantize(x,fp_scale_asym,bit_width,stochastic=False,symmetric=False)
        qx_2 = nfp.quant._do_quantize(x,scale,bit_width,stochastic=False,symmetric=True)
        qx_3 = nfp.quant._do_quantize(x.cuda(),scale,bit_width,stochastic=True,symmetric=True)
        err[0] = err[0]+(qx_0[0] - x).sum().abs()
        err[1] = err[1]+(qx_1[0] - x).sum().abs()
        err[2] = err[2]+(qx_2[0] - x).sum().abs()
        err[3] = err[3]+(qx_3[0].cpu() - x).sum().abs()
    print(err)
