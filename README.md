# Training Script (`train.py`)

## Overview

The `train.py` script is designed for training Convolutional Neural Networks (CNN) and Feedforward Neural Networks (NN) using the PyTorch framework. It also supports logging metrics to Weights and Biases (wandb). The script reads configurations from YAML files for both the network and training settings.

## Usage

To run the script, use the following command:

```
python train.py -n <network_config.yaml> -t <train_config.yaml> [-p <wandb_project> -e <wandb_entity>]
```

## Configuration Options

### Example Training Configuration

Here is an example of a training configuration YAML file:

```
batch_size: 32
learning_rate: 0.001
epochs: 10
transform_list:
  - name: RandomHorizontalFlip
  - name: RandomCrop
    size: 28
    padding: 4
  - name: ToTensor
  - name: Normalize
    mean: [0.1307]
    std: [0.3081]
optimizer_type: Adam
optimizer_hyperparams:
  betas: [0.9, 0.999]
  eps: 1e-08
  weight_decay: 0
annealing_type: StepLR
annealing_hyperparams:
  step_size: 5
  gamma: 0.1
```

### Example Network Configurations

1. For CNN:

```
model_type: CNN
input_channels: 1
num_classes: 10
conv_layers:
  - { out_channels: 64, kernel_size: 3, stride: 1, padding: 1, batch_norm: True, max_pool: True }
  - { out_channels: 128, kernel_size: 3, stride: 1, padding: 1, batch_norm: False, max_pool: True }
  - { out_channels: 256, kernel_size: 3, stride: 1, padding: 1, batch_norm: True, max_pool: False }
  - { out_channels: 512, kernel_size: 3, stride: 1, padding: 1, batch_norm: False, max_pool: False }
fc_layers:
  - 512
  - 256
  - 128
  - 64
```

2. For NN:

```
model_type: NN
input_features: 784
num_classes: 10
fc_layers:
  - 1024
  - 512
  - 256
  - 128
  - 64
  - 32
```

## Functions

### `get_transforms(transform_cfg)`

- Generates a composition of transformations based on the provided configuration.

### `get_optimizer(optimizer_type, model, learning_rate, optimizer_hyperparams)`

- Initializes and returns the optimizer based on the given parameters.

### `get_scheduler(annealing_type, optimizer, hyperparams)`

- Initializes and returns the learning rate scheduler based on given parameters.

### `train(net_cfg_path, train_cfg_path, wandb_run=None)`

- Main function to train the model based on given configurations.

## Example Code

To train a CNN model with specific configurations and log metrics to a wandb project:

```
python train.py -n cnn_config.yaml -t train_config.yaml -p MyWandbProject -e MyWandbEntity
```
```
