import torch
import torch.nn as nn

import pytorch_lightning as pl

class Regressor(pl.LightningModule):
    def __init__(self, pc_feat_dim):
        super().__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(2*pc_feat_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2)
        )

        # Rotation prediction head
        self.rot_head = nn.Linear(128, 4)

        # Translation prediction head
        self.trans_head = nn.Linear(128, 3)

    def forward(self, x):
        f = self.fc_layers(x)
        quat = self.rot_head(f)
        quat = quat / torch.norm(quat, p=2, dim=1, keepdim=True)
        trans  = self.trans_head(f)
        trans  = torch.unsqueeze(trans, dim=2)
        return quat, trans
