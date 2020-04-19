import os
import copy
import yaml
import numpy as np


with open("./da/digits.yaml", "r") as rf:
   base_cfg = yaml.load(rf)

for lambda_ in [1.05,1.1,1.2,1.3]:
    cfg = copy.deepcopy(base_cfg)
    cfg["trainer"]["beta2_grow_ratio"] = float(lambda_) 
    with open("./finetune/beta_{:.3}.yaml".format(lambda_), "w") as wf:
            yaml.dump(cfg, wf)
