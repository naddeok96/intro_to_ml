```
 _____    _______             _     
|  __ \  |__   __|           | |    
| |__) |   _| | ___  _ __ ___| |__  
|  ___/ | | | |/ _ \| '__/ __| '_ \ 
| |   | |_| | | (_) | | | (__| | | |
|_|    \__, |_|\___/|_|  \___|_| |_|
        __/ |                       
       |___/                        
 _____                  _                           _             
|  __ \                | |                         (_)            
| |  | | ___  ___ _ __ | |     ___  __ _ _ __ _ __  _ _ __   __ _ 
| |  | |/ _ \/ _ \ '_ \| |    / _ \/ _` | '__| '_ \| | '_ \ / _` |
| |__| |  __/  __/ |_) | |___|  __/ (_| | |  | | | | | | | | (_| |
|_____/ \___|\___| .__/|______\___|\__,_|_|  |_| |_|_|_| |_|\__, |
                 | |                                         __/ |
                 |_|                                        |___/ 
 ______                                           _    
|  ____|                                         | |   
| |__ _ __ __ _ _ __ ___   _____      _____  _ __| | __
|  __| '__/ _` | '_ ` _ \ / _ \ \ /\ / / _ \| '__| |/ /
| |  | | | (_| | | | | | |  __/\ V  V / (_) | |  |   < 
|_|  |_|  \__,_|_| |_| |_|\___| \_/\_/ \___/|_|  |_|\_\
                                                      
 ```                                                     

# Neural Network Training and Testing Framework

This framework is designed for training and testing Convolutional Neural Networks (CNN) and Feedforward Neural Networks (NN) using the PyTorch framework. It also supports logging metrics to Weights and Biases (wandb). The framework reads configurations from YAML files for both the network and training/testing settings.

## Table of Contents

1. [Installing Dependencies](#installing-dependencies)
2. [Fully Connected Neural Network (`nn.py`)](#fully-connected-neural-network-nnpy)
3. [Convolutional Neural Network (`cnn.py`)](#convolutional-neural-network-cnnpy)
4. [Training Script (`train.py`)](#training-script-trainpy)
5. [Testing Script (`test.py`)](#testing-script-testpy)
6. [Bash Scripts (`scripts/train.sh` and `scripts/test.sh`)](#bash-scripts-scriptstrainsh-and-scriptstestsh)

---

# Setting Up a Virtual Environment

This guide will walk you through the steps to create a new virtual environment, install the required dependencies specified in the `requirements.txt` file.

## Prerequisites

Make sure you have the following installed on your system:

- Python 3.6 or later
- Git
- W&B account (sign up [here](https://wandb.ai/site))

## Creating a Virtual Environment

1. Clone the repository to your local machine.

   ```bash
   git clone https://github.com/<username>/<repository_name>.git
   ```

2. Navigate to the project directory.
    ```bash
    cd <repository_name>
    ```

3. Create a new virtual environment.
    ```bash
    python3 -m venv venv_name
    ```

4. Activate the virtual environment.
    ```bash
    source venv_name/bin/activate
    ```

## Installing Dependencies

1. Install the dependencies specified in the `requirements.txt` file.
    ```bash
    pip3 install -r requirements.txt
    ```

---

### Fully Connected Neural Network (`nn.py`)

#### Overview

Defines a fully connected neural network (NN) using PyTorch.

#### Configuration Options

The YAML configuration files in the `cfgs/` directory can affect the architecture of the NN:

- `input_features`: Sets the number of input features.
- `num_classes`: Sets the number of output classes.
- `fc_layers`: Specifies the number of neurons in each fully connected layer.

#### Class Definition

- `NN(nn.Module)`

#### Methods

- `__init__(self, cfg_file)`: Constructor that initializes the model based on the YAML configuration file.
- `forward(self, x)`: Forward pass through the network.

---

### Convolutional Neural Network (`cnn.py`)

#### Overview

Defines a Convolutional Neural Network (CNN) using PyTorch.

#### Configuration Options

The YAML configuration files in the `cfgs/` directory can affect the architecture of the CNN:

- `input_channels`: Sets the number of input channels.
- `num_classes`: Sets the number of output classes.
- `conv_layers`: Specifies the parameters for each convolutional layer.
- `fc_layers`: Specifies the number of neurons in each fully connected layer.

#### Class Definition

- `CNN(nn.Module)`

#### Methods

- `__init__(self, cfg_file)`: Constructor that initializes the model based on the YAML configuration file.
- `forward(self, x)`: Forward pass through the network.
- `_get_flatten_size(self)`: Calculates the size of the flattened input for the fully connected layers.

---

### Training Script (`train.py`)

#### Overview

The `train.py` script is designed for training CNNs and NNs. 

#### Usage

To run the script:

```
python train.py -n <network_config.yaml> -t <train_config.yaml> [-p <wandb_project> -e <wandb_entity>]
```

#### Configuration Options

The YAML configuration files in the `cfgs/` directory can affect various aspects of training:

- `batch_size`: Controls the number of samples per batch.
- `learning_rate`: Sets the learning rate for the optimizer.
- `epochs`: Specifies the number of training epochs.
- `optimizer_type`: Chooses the type of optimizer (e.g., Adam, SGD).
- `annealing_type`: Chooses the type of learning rate scheduler.

#### Functions

- `get_transforms(transform_cfg)`: Generates a composition of transformations based on the provided configuration.
- `get_optimizer(optimizer_type, model, learning_rate, optimizer_hyperparams)`: Initializes and returns the optimizer based on the given parameters.
- `get_scheduler(annealing_type, optimizer, hyperparams)`: Initializes and returns the learning rate scheduler based on given parameters.
- `train(net_cfg_path, train_cfg_path, wandb_run=None)`: Main function to train the model based on given configurations.

---

### Testing Script (`test.py`)

#### Overview

The `test.py` script is designed for testing CNNs and NNs.

#### Usage

To run the script:

```
python test.py -n <network_config.yaml> -t <test_config.yaml> [-p <wandb_project> -e <wandb_entity>]
```

#### Configuration Options

The YAML configuration files in the `cfgs/` directory can affect various aspects of testing:

- `batch_size`: Controls the number of samples per batch during testing.
- `transform_list`: Specifies image transformations for preprocessing.

#### Functions

- `get_transforms(transform_cfg)`: Generates a composition of transformations based on the provided configuration.
- `test(model, testloader, wandb_run=None)`: Main function to test the model and optionally log metrics to wandb.
- `run_test(net_cfg_path, test_cfg_path, wandb_run=None)`: Load configurations, initialize the model, and run the test.

---

### Bash Scripts (`scripts/train.sh` and `scripts/test.sh`)

#### Overview

These bash scripts are used to run the training and testing Python scripts, respectively.

#### Usage

To run the training script:

```
bash scripts/train.sh
```

To run the testing script:

```
bash scripts/test.sh
```

#### Script Content

For `train.sh`:

```
#!/bin/bash
python3 train.py \\
        --net_cfg_path cfgs/models/fc_small.yaml \\
        --train_cfg_path cfgs/train/simple.yaml \\
        --wandb_project yaml_style \\
        --wandb_entity naddeok
```

For `test.sh`:

```
#!/bin/bash
python3 test.py \\
        --net_cfg_path cfgs/models/fc_small.yaml \\
        --test_cfg_path cfgs/test/simple.yaml \\
        --wandb_project yaml_style \\
        --wandb_entity naddeok
```
