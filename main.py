"""train cifar10 with pytorch."""
from __future__ import print_function

import os
import sys
import random
import shutil
import argparse
import logging
from time import *

import yaml

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torchvision
import torchvision.datasets
import torchvision.transforms as transforms


import datasets
from models import *
from trainer import *
# import trainer
from utils import *

def main(argv):

    parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training", prog="main.py")
    parser.add_argument("cfg", help="Available models: ")
    parser.add_argument("--local_rank",default=-1,type=int,help="the rank of this process")
    parser.add_argument("--gpu", default="0", help="gpu ids, seperate by comma")
    parser.add_argument("--save",  required=True,
                        help="store logs/checkpoints under this directory")
    parser.add_argument("--resume", "-r", help="resume from checkpoint")
    parser.add_argument("--pretrain", action="store_true",
                        help="used with `--resume`, regard as a pretrain model, do not recover epoch/best_acc")
    parser.add_argument("--no-cuda", action="store_true", default=False, help="do not use gpu")
    parser.add_argument("--seed", default=None, help="random seed", type=int)
    parser.add_argument("--save-every", default=20, type=int, help="save every N epoch")
    parser.add_argument("--distributed", action="store_true",help="whether to use distributed training")
    parser.add_argument("--dataset-path", default=None, help="dataset path")
    args = parser.parse_args(argv)

    savepath = args.save

    if args.local_rank > -1:
        torch.cuda.set_device(args.local_rank)
    
    if args.distributed: 
        # device = torch.cuda.current_device()
        torch.distributed.init_process_group(backend="nccl")
    else:
        gpus = [int(d) for d in args.gpu.split(",")]
        torch.cuda.set_device(gpus[0])

    if args.seed is not None:
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    if not os.path.isdir(savepath):
        if sys.version_info.major == 2:
            os.makedirs(savepath)
        else:
            os.makedirs(savepath, exist_ok=True)
    # Setup logfile

    # log_format = "%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s \t: %(message)s"
    log_format = "%(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        filemode="w")
    if args.local_rank == 0 or args.local_rank == -1:
        file_handler = logging.FileHandler(os.path.join(savepath, "train.log"))
    else:
        # file_handler = logging.FileHandler("/dev/null")
        file_handler = logging.FileHandler(os.path.join(savepath, "train_{}.log".format(args.local_rank)))
    file_handler.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(file_handler)
    logging.info("CMD: %s", " ".join(sys.argv))

    # Load and backup configuration file
    shutil.copyfile(args.cfg, os.path.join(savepath, "config.yaml"))
    with open(args.cfg) as cfg_f:
        cfg = yaml.load(cfg_f)
    
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    
    if device == "cuda":
        logging.info("Using GPU! Available gpu count: {}".format(torch.cuda.device_count()))
    else:
        logging.info("\033[1;3mWARNING: Using CPU!\033[0m")


    # For Fix cfg
    if cfg["trainer"]["fix"] is not None:

        # Regenerate the default generation cfg
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
                    # "range_method": nfp.RangeMethod.RANGE_MAX_TENPERCENT,
                    # "range_method": nfp.RangeMethod.RANGE_SWEEP
                    "range_method": nfp.RangeMethod.RANGE_MAX,
                    "stochastic": cfg["trainer"]["fix"]["stochastic"],
                    "float_scale": cfg["trainer"]["fix"]["float_scale"],
                    "zero_point": cfg["trainer"]["fix"]["zero_point"],
                }
                for n in names
            }

        # Fix-Net are defined in the fix_utlis.py, it has dependency of _generate_default_fix_cfg
        # so make sure import it after defining the function based-on the config
        import fix_utils
        from fix_utils import set_fix_mode


# ----------------------  Dataset -------------------------

    logging.info("==> Preparing data..")
    if cfg["trainer_type"] == "plain":
        if cfg["trainer"]["dataset"] == "cifar":
            trainloader,validloader, ori_trainloader, testloader, _ = datasets.cifar10(cfg["trainer"].get("train_batch_size",None), cfg["trainer"].get("test_batch_size",None), cfg["trainer"].get("train_transform", None), cfg["trainer"].get("test_transform", None), train_val_split_ratio = None, distributed=args.distributed, root=args.dataset_path)
        elif cfg["trainer"]["dataset"] == "imagenet":
            trainloader,validloader, ori_trainloader, testloader, _ = datasets.imagenet(cfg["trainer"]["train_batch_size"], cfg["trainer"]["test_batch_size"], cfg["trainer"].get("train_transform", None), cfg["trainer"].get("test_transform", None), train_val_split_ratio = None, distributed=args.distributed, path=args.dataset_path)
    elif cfg["trainer_type"] == "semi":
        if cfg["trainer"]["dataset"] == "cifar":
            trainloader, testloader, _ = datasets.semi_cifar10(numlabel=cfg["trainer"]["numlabel"], 
                                                               label_bs=cfg["trainer"].get("label_batch_size",None),
                                                               train_bs=cfg["trainer"].get("train_batch_size",None),
                                                               test_bs=cfg["trainer"].get("test_batch_size",None),
                                                               train_transform=None,
                                                               test_transform=None,
                                                               root=args.dataset_path,
                                                               label_dir=None)
        elif cfg["trainer"]["dataset"] == "svhn":
            trainloader_l,trainloader_u,valloader,testloader = datasets.semi_svhn(
                numlabel=cfg["trainer"]["numlabel"],
                label_bs=cfg["trainer"].get("label_batch_size",None),
                train_bs=cfg["trainer"].get("train_batch_size",None),
                test_bs=cfg["trainer"].get("test_batch_size",None),
                train_transform=None,
                test_transform=None,
                root=args.dataset_path,
                cfg=cfg
            )
    elif cfg["trainer_type"] == "da":
        d_da = {
            # Digits
            "s": "svhn",
            "m": "mnist",
            "syn": "synth",
            "synth-sl": "synth-small",
            "u": "usps",
            # Office 31
            "d": "dslr",
            "w": "webcam",
            "p": "pascal",
            # Caltech
            "c": "Caltech",
            # imgnet
            "i": "imagenet"
        }

        cfg["trainer"]["source"] = d_da[cfg["trainer"]["source"]]
        cfg["trainer"]["target"] = d_da[cfg["trainer"]["target"]]

        if cfg["trainer"]["dataset"] == "digits":
            trainloader, testloader = datasets.da_digits(
                domains = [cfg["trainer"]["source"],cfg["trainer"]["target"]],
                train_bs = cfg["trainer"].get("train_batch_size",None),
                test_bs = cfg["trainer"].get("test_batch_size",None),
                train_transform=None,
                test_transform=None,
                root=args.dataset_path,
                cfg=cfg
            )
    ## Build model

    logging.info("==> Building model..")

    ## ------- Net --------------
    ## -----(Fix Cfg Here)-------
    net_type = cfg["trainer"]["model"]
    if net_type == "vgg":
        net = vgg.VGG("VGG16")
    elif net_type == "convnet":
        if cfg["trainer"]["fix"] is not None:
            net = fix_utils.MyNet_fix(fix=True, fix_bn=cfg["trainer"]["fix"]["fix_bn"], bitwidths=list(cfg["trainer"]["fix"]["bitwidth"].values()),default_f=_generate_default_fix_cfg)
        else:
            net = convnet.MyNet()

    ## ---- Setting the Fix-Mode -------
    if cfg["trainer"]["fix"] is not None:
        set_fix_mode(net,"train",cfg["trainer"])

    # Copy a piece of net for semi training & DA
    if cfg["trainer_type"]=="semi" or cfg["trainer_type"]=="da":
        net_ = type(net)()
        net_.load_state_dict(net.state_dict())
        net_ = net_.to(device)
        if device == "cuda":
            cudnn.benchmark = True
            if args.distributed:
                p_net_ = torch.nn.parallel.DistributedDataParallel(net_, [args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
            else: 
                if len(gpus) > 1:
                    p_net_ = torch.nn.DataParallel(net_, gpus)
                else:
                    p_net_ = net_

    net = net.to(device)
    if device == "cuda":
        cudnn.benchmark = True
        if args.distributed:
            p_net = torch.nn.parallel.DistributedDataParallel(net, [args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
        else: 
            if len(gpus) > 1:
                p_net = torch.nn.DataParallel(net, gpus)
            else:
                p_net = net

    ## Build trainer and train
    if cfg["trainer_type"] == "plain":
        trainer_ = trainer.Trainer(net,p_net,[trainloader,validloader,ori_trainloader],testloader,
                                                               savepath=savepath,
                                                               save_every=args.save_every,
                                                               log=logging.info, cfg=cfg["trainer"])
    elif cfg["trainer_type"] == "semi":
        if cfg["trainer"]["dataset"] == "cifar":
            trainer_ = semi_trainer.SemiTrainer(net, net_, p_net, p_net_,
                                                trainloader, testloader,
                                                savepath=savepath,
                                                save_every=args.save_every, 
                                                 log=logging.info,
                                                 cfg=cfg["trainer"])
        if cfg["trainer"]["dataset"] == "svhn":
            trainer_ = semi_trainer.SemiTrainer(net, net_, p_net, p_net_,
                                                [trainloader_l,trainloader_u], testloader,
                                                savepath=savepath,
                                                save_every=args.save_every, 
                                                 log=logging.info,
                                                 cfg=cfg["trainer"])
    elif cfg["trainer_type"] == "da":
        trainer_ = da_trainer.DATrainer(net, net_, p_net, p_net_,
                                                    trainloader, testloader,
                                                    savepath=savepath,
                                                    save_every=args.save_every, 
                                                     log=logging.info,
                                                     cfg=cfg["trainer"])


    trainer_.init(device=device, local_rank=args.local_rank,resume=args.resume, pretrain=args.pretrain)
    trainer_.train()

    # Default save for plot
    torch.save({"net":trainer_.net.state_dict()}, os.path.join(savepath,'ckpt_final.t7'))

if __name__ == "__main__":
    main(sys.argv[1:])


