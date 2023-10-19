import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

class CNN(nn.Module):
    """
    Convolutional Neural Network class.

    Args:
        cfg_file (str): Path to a YAML configuration file containing model specifications.

    Attributes:
        input_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
        conv_layers (list): List of dictionaries specifying convolutional layers.
        fc_layers (list): List of integers specifying fully connected layers.
        convs (nn.ModuleList): List of convolutional layers.
        fcs (nn.ModuleList): List of fully connected layers.
        flat_size (int): Size of the flattened input.

    Methods:
        forward(x): Forward pass through the network.
        _get_flatten_size(): Calculate the size of the flattened input.

    """

    def __init__(self, cfg_file):
        super(CNN, self).__init__()
        
        with open(cfg_file, 'r') as f:
            config = yaml.safe_load(f)
        
        self.input_channels = config['input_channels']
        self.num_classes = config['num_classes']
        self.conv_layers = config['conv_layers']
        self.fc_layers = config['fc_layers']
        
        self.convs = nn.ModuleList()
        in_channels = self.input_channels
        for layer in self.conv_layers:
            out_channels = layer['out_channels']
            kernel_size = layer['kernel_size']
            stride = layer['stride']
            padding = layer['padding']
            
            self.convs.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
            
            if layer.get('batch_norm', False):
                self.convs.append(nn.BatchNorm2d(out_channels))
            
            self.convs.append(nn.ReLU())
            
            if layer.get('max_pool', False):
                self.convs.append(nn.MaxPool2d(2, 2))
            
            in_channels = out_channels
        
        self.flat_size = self._get_flatten_size()
        
        self.fcs = nn.ModuleList()
        in_features = self.flat_size
        for out_features in self.fc_layers:
            self.fcs.append(nn.Linear(in_features, out_features))
            self.fcs.append(nn.ReLU())
            
            in_features = out_features
        
        self.fcs.append(nn.Linear(in_features, self.num_classes))

        # Load pretrained weights if specified in cfg
        if 'pretrained_weights' in config:
            pretrained_weights_path = config['pretrained_weights']
            if os.path.exists(pretrained_weights_path):
                self.load_state_dict(torch.load(pretrained_weights_path))
                print(f"Loaded pretrained weights from {pretrained_weights_path}")
            else:
                print(f"Warning: Pretrained weights file {pretrained_weights_path} not found.")
    
    def _get_flatten_size(self):
        """
        Calculate the size of the flattened input.
        
        Returns:
            int: Size of the flattened input.
        """
        x = torch.randn(1, self.input_channels, 28, 28)
        for layer in self.convs:
            x = layer(x)
        return x.view(x.size(0), -1).size(1)
    
    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        for layer in self.convs:
            x = layer(x)
        
        x = x.view(x.size(0), -1)
        
        for layer in self.fcs:
            x = layer(x)
        
        return x

if __name__ == '__main__':
    import torchsummary

    input_shape = (1, 28, 28)

    print(125*"*")
    print(125*"*")
    for cfg in ['cfgs/models/cnn_small.yaml', 'cfgs/models/cnn_large.yaml']:
        model = CNN(cfg)
        torchsummary.summary(model, input_shape)

        x = torch.randn(1, 1, 28, 28)
        output = model(x)

        print(output)

        print(125*"*")
        print(125*"*")
