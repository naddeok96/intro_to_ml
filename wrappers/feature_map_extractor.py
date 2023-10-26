'''
 _____          _                    __  __             
|  ___|__  __ _| |_ _   _ _ __ ___  |  \/  | __ _ _ __  
| |_ / _ \/ _` | __| | | | '__/ _ \ | |\/| |/ _` | '_ \ 
|  _|  __/ (_| | |_| |_| | | |  __/ | |  | | (_| | |_) |
|_|  \___|\__,_|\__|\__,_|_|  \___| |_|  |_|\__,_| .__/ 
                                                 |_|    
 _____      _                  _             
| ____|_  _| |_ _ __ __ _  ___| |_ ___  _ __ 
|  _| \ \/ / __| '__/ _` |/ __| __/ _ \| '__|
| |___ >  <| |_| | | (_| | (__| || (_) | |   
|_____/_/\_\\__|_|  \__,_|\___|\__\___/|_|   

'''
import torch
import torch.nn as nn

class FeatureMapHook:
    """
    A wrapper class for a CNN model to capture the flattened layer during forward pass.
    
    Attributes:
        cnn_model (nn.Module): The CNN model to be wrapped.
        flattened_layer (torch.Tensor): The captured flattened layer.
    """
    
    def __init__(self, cnn_model):
        """
        Initialize the FeatureMapHook class.
        
        Args:
            cnn_model (nn.Module): The CNN model to be wrapped.
        """
        self.cnn_model = cnn_model
        # Register forward hook to grab flattened layer
        self.cnn_model.fcs[0].register_forward_hook(self.hook_fn)
        self.flattened_layer = None

    def hook_fn(self, module, input, output):
        """
        Hook function to capture the flattened layer.
        
        Args:
            module (nn.Module): The module that this hook is registered to.
            input (tuple): Input to the module.
            output (torch.Tensor): Output from the module.
        """
        self.flattened_layer = input[0]

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor.
        """
        return self.cnn_model(x)

# Example usage
if __name__ == "__main__":
    from 
    # Replace CNN with your actual CNN model class and cfg_file with your actual YAML config file
    cfg_file = "your_config.yaml"  
    cnn_model = CNN(cfg_file)  # Assuming CNN is a class you've defined elsewhere
    wrapped_model = FeatureMapHook(cnn_model)
    
    # Create a random tensor to simulate input
    x = torch.randn(1, cnn_model.input_channels, 28, 28)
    output = wrapped_model.forward(x)
    
    # Print the captured flattened layer
    print("Flattened layer:", wrapped_model.flattened_layer)
