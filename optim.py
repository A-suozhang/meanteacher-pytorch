COPY_BN_STAS = True
import torch

class EMAWeightOptimizer (object):
    def __init__(self, target_net, source_net, alpha=0.99, quan=False):
        self.target_net = target_net
        self.source_net = source_net
        self.ema_alpha = alpha
        self.quan = quan
        self.target_params = list(target_net.state_dict().values())
        self.source_params = list(source_net.state_dict().values())

        if self.target_net.state_dict().keys() != self.source_net.state_dict().keys():
            raise ValueError('Source and target networks do not have the same state dict keys; do they have different architectures?')

         # Whether Copy The BN Statistics
        for i in self.target_net.state_dict().keys():
            s_pos = i.split('.')[-1]

            if (self.target_net.state_dict()[i].dtype == torch.float32):
                if (COPY_BN_STAS is not True):
                    if(s_pos != "running_mean" and s_pos != "running_var"):
                        self.target_net.state_dict()[i] = self.source_net.state_dict()[i]
                else:
                    self.target_net.state_dict()[i] = self.source_net.state_dict()[i]
            elif (self.target_net.state_dict()[i].dtype == torch.int32):
                if (self.quan):
                    self.target_net.state_dict()[i] = self.source_net.state_dict()[i]

        # if(COPY_BN_STAS):
        #     for i in self.target_net.state_dict().keys():
        #         s_pos = i.split('.')[-1] # The Posix of State_dict Keys "conv1_1.weight"
        #         if (s_pos != "num_batches_tracked"):
        #             self.target_net.state_dict()[i] = self.source_net.state_dict()[i]
        
        # else:
        #     for i in self.target_net.state_dict().keys():
        #         s_pos = i.split('.')[-1] # The Posix of State_dict Keys "conv1_1.weight"
        #         if (s_pos != "running_mean" and s_pos != "running_var" and s_pos != "num_batches_tracked"):
        #             self.target_net.state_dict()[i] = self.source_net.state_dict()[i]


        # * The Old Way Of Copying 
        # for tgt_p, src_p in zip(self.target_params, self.source_params):
        #     # * Dirty Way To Avoid Indexing []
        #     # if(len(src_p.shape) > 0):
        #     #     tgt_p[:] = src_p[:]
        #     tgt_p = src_p



    def step(self):
        one_minus_alpha = 1.0 - self.ema_alpha
        for i in self.target_net.state_dict().keys():
            s_pos = i.split('.')[-1] # The Posix of State_dict Keys "conv1_1.weight"

            if (self.target_net.state_dict()[i].dtype == torch.float32):

                if (COPY_BN_STAS is not True):
                    if(s_pos != "running_mean" and s_pos != "running_var"):
                        (self.target_net.state_dict()[i]).mul_(self.ema_alpha)
                        (self.target_net.state_dict()[i]).add_(self.source_net.state_dict()[i] * (1 - self.ema_alpha))
                else:
                    (self.target_net.state_dict()[i]).mul_(self.ema_alpha)
                    (self.target_net.state_dict()[i]).add_(self.source_net.state_dict()[i] * (1 - self.ema_alpha))

            elif(self.target_net.state_dict()[i].dtype == torch.int32):
                if (self.quan):
                    self.target_net.state_dict()[i] = self.source_net.state_dict()[i]
                    # Copy The Scale Index (which is int32)

            
            
            # if (COPY_BN_STAS):
            #     if (s_pos != "num_batches_tracked"):
            #         # ? If not using inplace method here , the trainig would elapse
            #         (self.target_net.state_dict()[i]).mul_(self.ema_alpha)
            #         (self.target_net.state_dict()[i]).add_(self.source_net.state_dict()[i] * (1 - self.ema_alpha))
            # else:
            #     if (s_pos != "running_mean" and s_pos != "running_var" and s_pos != "num_batches_tracked"):
            #         # ? If not using inplace method here , the trainig would elapse
            #         (self.target_net.state_dict()[i]).mul_(self.ema_alpha)
            #         (self.target_net.state_dict()[i]).add_(self.source_net.state_dict()[i] * (1 - self.ema_alpha))




        # * The Old Method
        # for tgt_p, src_p in zip(self.target_params, self.source_params):
        #     tgt_p.mul_(self.ema_alpha)
        #     tgt_p.add_(src_p * one_minus_alpha)

            # tgt_p = tgt_p.mul(self.ema_alpha)
            # tgt_p = tgt_p.add(src_p * (1.0 - self.ema_alpha))
