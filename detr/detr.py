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
    def __init__(self, hidden_dim, max_len=2500):
        """
        Learned positional encoding for the input features.

        Args:
            hidden_dim (int): The number of channels in the input feature map.
            max_len (int): The maximum length of the positional encoding (min H*W of feature map).
        """
        super().__init__()
        self.pos_embed = nn.Embedding(max_len, hidden_dim) # [max_len, hidden_dim]

    def forward(self, x):
        """
        Retrieve the positional encoding.

        Args:
            x (torch.Tensor): Input tensor of shape [B, hidden_dim, H, W].

        Returns:
            torch.Tensor: Positional encoding [B, H*W, hidden_dim].
        """
        # Get the dimensions of the input tensor
        B, _, H, W = x.shape
        # Check the max length of the positional encoding
        max_len = H * W
        if max_len > self.pos_embed.num_embeddings:
            raise ValueError(f"Input size {max_len} exceeds max_len {self.pos_embed.num_embeddings}.")
        # Create the index tensor for the positional encoding
        pos_idx = torch.arange(H*W, device=x.device).unsqueeze(0).expand(B, H*W)
        # Retrieve the positional encoding
        pos_enc = self.pos_embed(pos_idx) # [B, H*W, hidden_dim]
        return pos_enc
    

class TransformerEncoder(nn.Module):
    def __init__(self, hidden_dim, num_layers=6, nhead=8):
        """
        Transformer encoder for the DETR model.

        Args:
            hidden_dim (int): The number of channels in the input feature map.
            num_layers (int): The number of layers in the transformer encoder.
            nhead (int): The number of attention heads.
        """
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

    def forward(self, x):
        """
        Forward pass through the transformer encoder.

        Args:
            x (torch.Tensor): Input tensor of shape [B, H*W, hidden_dim].

        Returns:
            torch.Tensor: Encoded features of shape [B, H*W, hidden_dim].
        """
        # Pass the input through the transformer encoder
        encoded_feats = self.encoder(x)
        return encoded_feats
    

class TransformerDecoder(nn.Module):
    def __init__(self, hidden_dim, num_queries, num_layers=6, nhead=8):
        """
        Transformer decoder for the DETR model.

        Args:
            hidden_dim (int): The number of channels in the input feature map.
            num_queries (int): The number of queries for the decoder.
            num_layers (int): The number of layers in the transformer decoder.
            nhead (int): The number of attention heads.
        """
        super().__init__()
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.num_queries = num_queries
        self.query_embed = nn.Embedding(self.num_queries, hidden_dim)

    def forward(self, encoded_feats):
        """
        Forward pass through the transformer decoder.

        Args:
            encoded_feats (torch.Tensor): Encoded features from the encoder of shape [B, H*W, hidden_dim].

        Returns:
            torch.Tensor: Decoded features of shape [B, num_queries, hidden_dim].
        """
        # Get the number of batches
        B = encoded_feats.size(0)
        # Create a tensor of query indicies [B, num_queries]
        query_idx = torch.arange(self.num_queries, device=encoded_feats.device).unsqueeze(0).expand(B, self.num_queries)
        # Retrieve the query embeddings
        query_embed = self.query_embed(query_idx) # [B, num_queries, hidden_dim]
        # Pass the query embeddings and encoded features through the transformer decoder
        decoded_feats = self.decoder(query_embed, encoded_feats)
        return decoded_feats
    
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        """
        Standard multi-layer perceptron (MLP). Applies a ReLU activation function after each layer except the last.

        Args:
            input_dim (int): The number of input features.
            hidden_dim (int): The number of hidden features.
            output_dim (int): The number of output features.
            num_layers (int): The number of layers in the MLP.
        """
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for i in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the MLP.

        Args:
            x (torch.Tensor): Input tensor of shape [B, input_dim].

        Returns:
            torch.Tensor: Output tensor of shape [B, output_dim].
        """
        return self.layers(x)
    

class PredictionHeads(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        """
        Prediction heads for the DETR model.

        Args:
            hidden_dim (int): The number of channels in the input feature map.
            num_classes (int): The number of classes for the object detection task.
        """
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)  # +1 for the no-object class
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

    def forward(self, x):
        """
        Forward pass through the prediction heads.

        Args:
            x (torch.Tensor): Input tensor of shape [B, num_queries, hidden_dim].

        Returns:
            class_logits (torch.Tensor): Predicted class logits of shape [B, num_queries, num_classes + 1].
            bbox_preds (torch.Tensor): Predicted bounding box coordinates of shape [B, num_queries, 4].
                The coordinates are normalized to [0, 1] range.
        """
        # Apply layer normalization to the input
        x = self.layer_norm(x)
        # Pass the input through the class and bounding box prediction heads
        class_logits = self.class_embed(x)
        bbox_preds = self.bbox_embed(x)
        # Apply sigmoid to the bounding box predictions to normalize them to [0, 1]
        bbox_preds = torch.sigmoid(bbox_preds)
        return class_logits, bbox_preds

 
class DETR(nn.Module):
    def __init__(self, num_classes, hidden_dim=256, num_queries=100, model="resnet50"):
        """
        DETR model for object detection.

        Args:
            num_classes (int): The number of classes for the object detection task.
            hidden_dim (int): The number of channels in the input feature map.
            num_queries (int): The number of queries for the decoder (max number of objects per image).
            model (str): The ResNet model to use. Options are "resnet50" or "resnet101".
        """
        super().__init__()
        self.backbone = Backbone(hidden_dim, model)
        self.positional_encoding = PositionalEncoding(hidden_dim, max_len=2500)
        self.transformer_encoder = TransformerEncoder(hidden_dim, num_layers=6, nhead=8)
        self.transformer_decoder = TransformerDecoder(hidden_dim, num_queries, num_layers=6, nhead=8)
        self.prediction_heads = PredictionHeads(hidden_dim, num_classes)

    def forward(self, x):
        """
        Forward pass through the DETR model.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W].

        Returns:
            class_logits (torch.Tensor): Predicted class logits of shape [B, num_queries, num_classes + 1].
            bbox_preds (torch.Tensor): Predicted bounding box coordinates of shape [B, num_queries, 4].
                The coordinates are normalized to [0, 1] range.
        """
        # Pass the input through the backbone
        feats = self.backbone(x) # [B, hidden_dim, H/32, W/32]

        # Get the positional encoding for the feature maps
        pos_enc = self.positional_encoding(feats)

        # Reshape the feature maps to [B, H*W, hidden_dim]
        feats_flat = feats.flatten(2).permute(0, 2, 1)  # [B, H*W, hidden_dim]

        # Sum the positional encoding with the feature maps
        feats_pos = feats_flat + pos_enc

        # Pass the feature maps through the transformer encoder
        encoded_feats = self.transformer_encoder(feats_pos)

        # Pass the encoded features through the transformer decoder
        decoded_feats = self.transformer_decoder(encoded_feats)

        # Pass the decoded features through the prediction heads
        class_logits, bbox_preds = self.prediction_heads(decoded_feats)
        # Return the class logits and bounding box predictions
        return class_logits, bbox_preds
    
    def predict(self, x):
        """
        Make predictions using the DETR model.
        This method is a wrapper around the forward method to ensure that no gradients are computed.
        """
        with torch.no_grad():
            return self.forward(x)
        

def demo():
    # Parameters
    num_classes = 90  # Number of object classes (e.g., COCO dataset)
    hidden_dim = 256  # Hidden dimension for the transformer
    num_queries = 100  # Number of queries for the transformer decoder

    # Create a random input tensor with shape [B, C, H, W]
    x = torch.randn(2, 3, 800, 800)

    # Create the DETR model and pass the input through it
    model = DETR(num_classes=num_classes, hidden_dim=hidden_dim, num_queries=num_queries)
    class_logits, bbox_preds = model(x)

    print("Class logits shape:", class_logits.shape)  # Expected: [B, num_queries, num_classes + 1]
    print("Bounding box predictions shape:", bbox_preds.shape)  # Expected: [B, num_queries, 4]

if __name__ == "__main__":
    demo()

