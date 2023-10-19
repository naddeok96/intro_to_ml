import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

class NN(nn.Module):
    """
    Fully Connected Network class.

    Args:
        cfg_file (str): Path to a YAML configuration file containing model specifications.

    Attributes:
        input_features (int): Number of input features.
        num_classes (int): Number of output classes.
        fc_layers (list): List of integers specifying fully connected layers.
        fcs (nn.ModuleList): List of fully connected layers.

    """

    def __init__(self, cfg_file):
        super(NN, self).__init__()
        
        with open(cfg_file, 'r') as f:
            config = yaml.safe_load(f)
        
        self.input_features = config['input_features']
        self.num_classes = config['num_classes']
        self.fc_layers = config['fc_layers']
        
        self.fcs = nn.ModuleList()
        in_features = self.input_features
        for out_features in self.fc_layers:
            self.fcs.append(nn.Linear(in_features, out_features))
            self.fcs.append(nn.ReLU())
            
            in_features = out_features
        
        self.fcs.append(nn.Linear(in_features, self.num_classes))
    
        # Load pretrained weights if specified in cfg
        if config.get('pretrained_weights', None):
            pretrained_weights_path = config['pretrained_weights']
            if os.path.exists(pretrained_weights_path):
                self.load_state_dict(torch.load(pretrained_weights_path))
                print(f"Loaded pretrained weights from {pretrained_weights_path}")
            else:
                print(f"Warning: Pretrained weights file {pretrained_weights_path} not found.")
    
    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = x.view(x.size(0), -1)
        
        for layer in self.fcs:
            x = layer(x)
        
        return x

if __name__ == '__main__':
    import torchsummary

    input_shape = (784,)

    print(125*"*")
    print(125*"*")
    for cfg in ['cfgs/models/fc_small.yaml', 'cfgs/models/fc_large.yaml']:
        model = NN(cfg)
        torchsummary.summary(model, input_shape)

        x = torch.randn(1, 784)
        output = model(x)

        print(output)

        print(125*"*")
        print(125*"*")
