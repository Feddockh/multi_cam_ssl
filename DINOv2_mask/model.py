import torch
import torch.nn as nn
from dinov2.dinov2.models import vision_transformer as vits
from configs import NUM_CLASSES

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
    
class DinoMaskDETR(nn.Module):
    def __init__(self, hidden_dim=256, nheads=8, num_encoder_layers=6, num_decoder_layers=6, num_queries=100):
        super().__init__()
        self.backbone = vits.vit_small(patch_size=14)
        ckpt = torch.load("dinov2_vits14_pretrain.pth", map_location="cpu")
        filtered_ckpt = {k: v for k, v in ckpt.items() if k in self.backbone.state_dict() and v.size() == self.backbone.state_dict()[k].size()}
        self.backbone.load_state_dict(filtered_ckpt, strict=False)
        
        self.input_proj = nn.Conv2d(self.backbone.embed_dim, hidden_dim, kernel_size=1)

        self.transformer = nn.Transformer(
            d_model=hidden_dim, nhead=nheads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers
        )

        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        self.class_embed = nn.Linear(hidden_dim, NUM_CLASSES + 1)  # +1 for "no object"
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        # mask head
        self.mask_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # transformer back to pixel space
        self.pixel_decoder = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        
    def forward(self, samples):
        features = self.backbone.forward_features(samples)
        patch_tokens = features["x_norm_patchtokens"]  # (B, num_tokens, C)

        B, N, C = patch_tokens.shape
        H = W = int(N ** 0.5)
        patch_tokens = patch_tokens.transpose(1, 2).reshape(B, C, H, W)

        features = self.input_proj(patch_tokens)  # (B, hidden_dim, H, W)

        src = features.flatten(2).permute(2, 0, 1)  # (HW, B, hidden_dim)

        memory = self.transformer.encoder(src)  # (HW, B, hidden_dim)

        queries = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)  # (num_queries, B, hidden_dim)
        hs = self.transformer.decoder(queries, memory)  # (num_queries, B, hidden_dim)

        hs_last = hs.transpose(0, 1)  # (B, num_queries, hidden_dim)

        outputs_class = self.class_embed(hs_last)  # (B, num_queries, num_classes)
        outputs_coord = self.bbox_embed(hs_last).sigmoid()  # (B, num_queries, 4)

        mask_features = self.pixel_decoder(features)  # (B, hidden_dim, H, W)

        mask_embed = self.mask_embed(hs_last)  # (B, num_queries, hidden_dim)

        masks = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)  # (B, num_queries, H, W)

        return {
            'pred_logits': outputs_class,
            'pred_boxes': outputs_coord,
            'pred_masks': masks
        }

