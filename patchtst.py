import torch
import torch.nn as nn

class PatchTST(nn.Module):
    def __init__(
        self,
        input_window,
        num_features,
        d_model=64,
        depth=2,
        n_heads=4,
        patch_size=16,
        dropout=0.1
    ):
        super().__init__()

        self.patch_size = patch_size
        self.num_patches = input_window // patch_size

        # Patch embedding
        self.embed = nn.Linear(patch_size * num_features, d_model)

        # Transformer encoder
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=256,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)

        # Head
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        B, T, F = x.shape   # (batch, 256, 5)

        x = x.reshape(B, self.num_patches, self.patch_size * F)
        x = self.embed(x)
        x = self.encoder(x)
        x = x.mean(dim=1)  # average patches

        return self.head(x)
