import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, resnet101


class Backbone(nn.Module):
    def __init__(self, hidden_dim, model="resnet50"):
        """
        The Backbone uses a pre-trained ResNet model with the last layers removed as a feature 
        extractor. A convolutional layer is used to reduce the number of channels.

        Args:
            hidden_dim (int): The number of channels in the output feature map.
            model (str): The ResNet model to use. Options are "resnet50" or "resnet101".
        """
        super().__init__()

        # Load a pre-trained ResNet model
        if model == "resnet50":
            self.resnet = resnet50(pretrained=True)
        elif model == "resnet101":
            self.resnet = resnet101(pretrained=True)
        else:
            raise ValueError("Unsupported model. Choose 'resnet50' or 'resnet101'.")

        # Modify the ResNet model to remove the last two layers (fc and avgpool)
        self.cnn = nn.Sequential(*list(self.resnet.children())[:-2])
        self.channel_reducer = nn.Conv2d(2048, hidden_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """
        Forward pass through the backbone.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W].

        Returns:
            torch.Tensor: Feature maps of shape [B, hidden_dim, H/32, W/32].
        """
        # Pass the input through the ResNet model to get the feature maps
        feats = self.cnn(x) # [B, 2048, H/32, W/32]
        # Reduce the channel dimension
        reduced_feats = self.channel_reducer(feats) # [B, hidden_dim, H/32, W/32]
        return reduced_feats
    

class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, max_len=5000):
        """
        Learned positional encoding for the input features.

        Args:
            hidden_dim (int): The number of channels in the input feature map.
            max_len (int): The maximum length of the positional encoding.
        """
        super().__init__()
        self.pos_embed = nn.Embedding(max_len, hidden_dim) # [max_len, hidden_dim]

    def forward(self, x):
        """
        Retrieve the positional encoding.
        Args:
            x (torch.Tensor): Input tensor of shape [B, hidden_dim, H, W].
        Returns:
            torch.Tensor: Positional encoding to be added to the input tensor.
        """
        # Get the dimensions of the input tensor
        B, _, H, W = x.shape
        # Create the positional encoding tensor
        pos = self.pos_embed.unsqueeze(0).repeat(B, -1, -1) # [B, max_len, hidden_dim]
        
        return



class DETR(nn.Module):
    def __init__(self, num_classes, hidden_dim=256, num_queries=100, model="resnet50"):
        super().__init__()
        self.backbone = Backbone(hidden_dim, model)