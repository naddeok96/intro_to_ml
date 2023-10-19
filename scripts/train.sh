#!/bin/bash

# Run the Python3 command with the specified arguments
python3 train.py \
        --net_cfg_path cfgs/models/fc_small.yaml \
        --train_cfg_path cfgs/train/simple.yaml \
        --wandb_project yaml_style \
        --wandb_entity naddeok