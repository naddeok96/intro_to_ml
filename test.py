'''
 _____         _            
|_   _|__  ___| |_ ___ _ __ 
  | |/ _ \/ __| __/ _ \ '__|
  | |  __/\__ \ ||  __/ |   
  |_|\___||___/\__\___|_|   
                            
Testing script for neural network models using PyTorch and wandb for logging.

This script provides a way to test Convolutional Neural Networks (CNN) and 
Feedforward Neural Networks (NN) using the PyTorch framework. It reads the
network and testing configurations from YAML files and optionally logs metrics
to Weights and Biases (wandb).

Usage:
  python <script_name>.py -n <network_config.yaml> -t <test_config.yaml> [-p <wandb_project> -e <wandb_entity>]
'''

import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import yaml
import argparse
from tqdm import tqdm

from models.nn import NN  # Custom NN model
from models.cnn import CNN  # Custom CNN model

def get_transforms(transform_cfg):
    '''
    Generate a composition of image transformations based on the provided configuration.

    Parameters:
        transform_cfg (list): List of dictionaries containing transformation configurations.
    
    Returns:
        transforms.Compose: A composition of transformations to be applied to images.
    '''
    transform_ops = []
    for op in transform_cfg:
        if op['name'] == 'ToTensor':
            transform_ops.append(transforms.ToTensor())
        elif op['name'] == 'Normalize':
            mean = op.get('mean', [0.5])
            std = op.get('std', [0.5])
            transform_ops.append(transforms.Normalize(mean, std))
        # Add more transformations here if needed
    return transforms.Compose(transform_ops)

def test(model, testloader, wandb_run=None):
    '''
    Test a PyTorch model and optionally log metrics to wandb.

    Parameters:
        model (torch.nn.Module): PyTorch model to test.
        testloader (DataLoader): DataLoader for the test dataset.
        wandb_run (wandb.wandb_run.Run, optional): A Weights and Biases run object for logging metrics.
    
    Returns:
        float: Test accuracy percentage.
    '''
    criterion = nn.CrossEntropyLoss()  # Loss function
    correct = 0
    total = 0
    test_loss = 0.0

    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():  # Disable gradient calculation
        for i, data in enumerate(tqdm(testloader, desc='Test Batch'), 0):
            inputs, labels = data
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Test accuracy: {accuracy:.2f}%')
    
    # Log metrics to wandb if provided
    if wandb_run:
        wandb_run.log({"Test Loss": test_loss / total, "Test Accuracy": accuracy})

    return accuracy

def run_test(net_cfg_path, test_cfg_path, wandb_run=None):
    '''
    Load configurations, initialize the model, and run the test.

    Parameters:
        net_cfg_path (str): Path to the network configuration YAML file.
        test_cfg_path (str): Path to the testing configuration YAML file.
        wandb_run (wandb.wandb_run.Run, optional): A Weights and Biases run object for logging metrics.
    
    Returns:
        float: Test accuracy percentage.
    '''
    # Load network configuration
    with open(net_cfg_path, 'r') as f:
        net_config = yaml.safe_load(f)
    
    # Load test configuration
    with open(test_cfg_path, 'r') as f:
        test_config = yaml.safe_load(f)

    # Initialize model based on configuration
    if net_config['model_type'] == "CNN":
        model = CNN(net_cfg_path)
    elif net_config['model_type'] == "NN":
        model = NN(net_cfg_path)
    else:
        raise ValueError(f"Unknown model_type: {net_config['model_type']}")
    
    # Get image transformations
    transform = get_transforms(test_config.get('transform_list', [{'name': 'ToTensor'}]))

    # Initialize DataLoader
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=test_config['batch_size'], shuffle=False)

    return test(model, testloader, wandb_run)

# Entry point for the script
if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser(description='Test a CNN or NN model.')
    parser.add_argument('--net_cfg_path', '-n', type=str, required=True, help='Path to the network configuration YAML file.')
    parser.add_argument('--test_cfg_path', '-t', type=str, required=True, help='Path to the testing configuration YAML file.')
    parser.add_argument('--wandb_project', '-p', type=str, required=False, help='Wandb project name (optional).')
    parser.add_argument('--wandb_entity', '-e', type=str, required=False, help='Wandb entity name (optional).')
    
    args = parser.parse_args()

    # Initialize wandb logging if project and entity are provided
    wandb_run = None
    if args.wandb_project and args.wandb_entity:
        wandb_run = wandb.init(project=args.wandb_project, entity=args.wandb_entity)

    run_test(net_cfg_path=args.net_cfg_path, test_cfg_path=args.test_cfg_path, wandb_run=wandb_run)
