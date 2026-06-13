import torch
import torch.nn as nn
import config

class SkeletonTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Linear(config.INPUT_DIM, config.D_MODEL)
        self.pos_embed = nn.Parameter(torch.randn(1, config.TARGET_FRAMES, config.D_MODEL))
        encoder = nn.TransformerEncoderLayer(
            d_model=config.D_MODEL, nhead=config.NHEAD,
            dim_feedforward=config.DIM_FEEDFORWARD, dropout=config.DROPOUT,
            batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder, num_layers=config.NUM_LAYERS)
        self.fc = nn.Linear(config.D_MODEL, config.NUM_CLASSES)
    
    def forward(self, x):
        x = self.embedding(x) + self.pos_embed[:, :x.size(1), :]
        x = self.transformer(x).mean(dim=1)
        return self.fc(x)