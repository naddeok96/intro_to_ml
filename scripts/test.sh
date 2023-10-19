#!/bin/bash

# Run the Python3 command with the specified arguments
python3 test.py \
        --net_cfg_path cfgs/models/fc_small.yaml \
        --test_cfg_path cfgs/test/simple.yaml \
        --wandb_project yaml_style \
        --wandb_entity naddeok