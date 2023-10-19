'''
 _____          _                 
|_   _| __ __ _(_)_ __   ___ _ __ 
  | || '__/ _` | | '_ \ / _ \ '__|
  | || | | (_| | | | | |  __/ |   
  |_||_|  \__,_|_|_| |_|\___|_|   

Training and optional testing script for neural network models using PyTorch and wandb for logging.

This script provides a way to train Convolutional Neural Networks (CNN) and 
Feedforward Neural Networks (NN) using the PyTorch framework. It reads the
network and training configurations from YAML files and optionally logs metrics
to Weights and Biases (wandb). Additionally, the script can call a test function
to evaluate the model if specified in the training configuration.

Usage:
  python <script_name>.py -n <network_config.yaml> -t <train_config.yaml> [-p <wandb_project> -e <wandb_entity>]

'''
import os
import wandb
import torch
import torchsummary
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import yaml
import argparse
from tqdm import tqdm

from test import test
from models.nn import NN
from models.cnn import CNN 

# Create transformations based on configuration
def get_transforms(transform_cfg):
    '''
    Generate a composition of transformations based on the provided configuration.

    Parameters:
    - transform_cfg (list): List of dictionaries containing transformation configurations

    Returns:
    - transforms.Compose: Composed transformations to be applied to images
    '''
    transform_ops = []
    for op in transform_cfg:
        if op['name'] == 'ToTensor':
            transform_ops.append(transforms.ToTensor())
        elif op['name'] == 'Normalize':
            mean = op.get('mean', [0.5])
            std = op.get('std', [0.5])
            transform_ops.append(transforms.Normalize(mean, std))
        elif op['name'] == 'RandomHorizontalFlip':
            transform_ops.append(transforms.RandomHorizontalFlip())
        elif op['name'] == 'RandomCrop':
            size = op.get('size', 32)
            padding = op.get('padding', 4)
            transform_ops.append(transforms.RandomCrop(size, padding=padding))
    return transforms.Compose(transform_ops)

# Create optimizer based on configuration
def get_optimizer(optimizer_type, model, learning_rate, optimizer_hyperparams):
    '''
    Initialize and return the optimizer based on the given parameters.
    
    Parameters:
    - optimizer_type (str): Type of optimizer ('SGD' or 'Adam')
    - model (nn.Module): The neural network model
    - learning_rate (float): Learning rate for optimization
    - optimizer_hyperparams (dict): Additional hyperparameters for the optimizer

    Returns:
    - optim.Optimizer: Initialized optimizer
    '''
    if optimizer_type == 'SGD':
        momentum = optimizer_hyperparams.get('momentum', 0.9)
        weight_decay = optimizer_hyperparams.get('weight_decay', 0)
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_type == 'Adam':
        betas = optimizer_hyperparams.get('betas', [0.9, 0.999])
        eps = float(optimizer_hyperparams.get('eps', 1e-08))  
        weight_decay = optimizer_hyperparams.get('weight_decay', 0)
        return optim.Adam(model.parameters(), lr=learning_rate, betas=betas, eps=eps, weight_decay=weight_decay)

# Create learning rate scheduler based on configuration
def get_scheduler(annealing_type, optimizer, hyperparams):
    '''
    Initialize and return the learning rate scheduler based on given parameters.
    
    Parameters:
    - annealing_type (str): Type of learning rate annealing ('StepLR' or 'ExponentialLR')
    - optimizer (optim.Optimizer): The optimizer
    - hyperparams (dict): Hyperparameters for the scheduler

    Returns:
    - lr_scheduler._LRScheduler: Initialized learning rate scheduler, or None if not specified
    '''
    if annealing_type == 'StepLR':
        return lr_scheduler.StepLR(optimizer, step_size=hyperparams['step_size'], gamma=hyperparams['gamma'])
    elif annealing_type == 'ExponentialLR':
        return lr_scheduler.ExponentialLR(optimizer, gamma=hyperparams['gamma'])
    else:
        return None

# Fit model to data
def train(net_cfg_path, train_cfg_path, wandb_run=None):
    '''
    Train a CNN model based on given configurations.

    Parameters:
    - net_cfg_path (str): Path to the network configuration YAML file
    - train_cfg_path (str): Path to the training configuration YAML file
    - wandb_run (wandb.wandb_run.Run): Optional. A Weights and Biases (wandb) run object for logging metrics.

    Returns:
    - nn.Module: Trained model
    '''
    # Load the training configuration
    with open(train_cfg_path, 'r') as f:
        training_config = yaml.safe_load(f)

    with open(net_cfg_path, 'r') as f:
        net_config = yaml.safe_load(f)

    # Initialize model
    if net_config['model_type'] == "CNN":
        model = CNN(net_cfg_path)
    elif net_config['model_type'] == "NN":
        model = NN(net_cfg_path)
    else:
        raise ValueError(f"Unknown model_type: {net_config['model_type']}")

    # Summerize dimensions
    torchsummary.summary(model, (1,28,28))

    # Get the transformations to apply on images
    transform = get_transforms(training_config.get('transform_list', [{'name': 'ToTensor'}]))

    # Load and preprocess the train dataset
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=training_config['batch_size'], shuffle=True)

    # Load and preprocess test dataset
    test_every = training_config.get('test_every', None)
    if test_every:  # Load testloader if test_every is specified
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        testloader = DataLoader(testset, batch_size=training_config['batch_size'], shuffle=False)

    # Initialize loss function
    criterion = nn.CrossEntropyLoss()

    # Initialize optimizer
    optimizer_hyperparams = training_config.get('optimizer_hyperparams', {})
    optimizer = get_optimizer(
        training_config.get('optimizer_type', 'SGD'),
        model,
        training_config['learning_rate'],
        optimizer_hyperparams
    )

    # Initialize learning rate scheduler if specified
    scheduler_type = training_config.get('annealing_type', None)
    scheduler_hyperparams = training_config.get('annealing_hyperparams', {})
    scheduler = get_scheduler(scheduler_type, optimizer, scheduler_hyperparams)

    test_every = training_config.get('test_every', None)
    save_every = training_config.get('save_every', None)

    # Create folder for pretrained_weights only if save_every is not None or False
    if save_every:
        if not os.path.exists('pretrained_weights'):
            os.makedirs('pretrained_weights')

    # Epoch loop
    for epoch in tqdm(range(training_config['epochs']), desc='Epoch'):
        # Log metrics to wandb
        if wandb_run:
            wandb_run.log({"Epoch": epoch})

        # Batch Loop
        for i, data in enumerate(tqdm(trainloader, desc='Batch', leave=False), 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Log metrics to wandb
            if wandb_run:
                wandb_run.log({"Loss": loss.item()})
            
        # Call the test script every N epochs if specified
        if test_every and (epoch + 1) % test_every == 0:
            test(model, testloader, wandb_run)
        
        # Save the model every N epochs if specified
        if save_every and (epoch + 1) % save_every == 0:
            save_path = f"pretrained_weights/"
            if wandb_run:
                save_path += f"{wandb_run.project}_{wandb_run.name}_epoch_{epoch+1}.pth"
            else:
                save_path += f"{os.path.basename(net_cfg_path).replace('.yaml', '')}_{os.path.basename(train_cfg_path).replace('.yaml', '')}_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), save_path)

        if scheduler:
            scheduler.step()

            # Log learning rate to wandb
            if wandb_run:
                wandb_run.log({"learning_rate": scheduler.get_last_lr()[0]})
        
    print("Training complete.")
    return model

# Entry point
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a CNN model.')
    parser.add_argument('--net_cfg_path', '-n', type=str, required=True, help='Path to the network configuration YAML file.')
    parser.add_argument('--train_cfg_path', '-t', type=str, required=True, help='Path to the training configuration YAML file.')
    parser.add_argument('--wandb_project', '-p', type=str, required=False, help='Wandb project name.')
    parser.add_argument('--wandb_entity', '-e', type=str, required=False, help='Wandb entity name.')
    
    args = parser.parse_args()

    # Load the training configuration
    with open(args.train_cfg_path, 'r') as f:
        training_config = yaml.safe_load(f)

    with open(args.net_cfg_path, 'r') as f:
        net_config = yaml.safe_load(f)
    
    # Initialize wandb if project and entity are provided
    wandb_run = None
    if args.wandb_project and args.wandb_entity:
        wandb_run = wandb.init(project=args.wandb_project, entity=args.wandb_entity)
        wandb.config.update({"Network Configuration": net_config, "Training Configuration": training_config})
    
    trained_model = train(  net_cfg_path=args.net_cfg_path, 
                            train_cfg_path=args.train_cfg_path,
                            wandb_run=wandb_run
                        )