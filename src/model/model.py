import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from .diffusion import TensorProductScoreModel
from .losses import DiffusionLoss


class BaseModel(nn.Module):
    """
        enc(receptor) -> R^(dxL)
        enc(ligand)  -> R^(dxL)
    """
    def __init__(self, args, params,confidence_mode=False):
        super(BaseModel, self).__init__()

        ######## unpack model parameters
        self.model_type = args.model_type
        self.knn_size = args.knn_size
        self.args = args

        ######## initialize (shared) modules
        # raw encoders
        self.encoder = TensorProductScoreModel(args, params, confidence_mode=confidence_mode)

        self._init()

    def _init(self):
        for name, param in self.named_parameters():
            # NOTE must name parameter "bert"
            if "bert" in name:
                continue
            # bias terms
            if param.dim() == 1:
                nn.init.constant_(param, 0)
            # weight terms
            else:
                nn.init.xavier_normal_(param)

    def forward(self, batch):
        raise Exception("Extend me")

    def dist(self, x, y):
        if len(x.size()) > 1:
            return ((x-y)**2).sum(-1)
        return ((x-y)**2)


class ScoreModel(BaseModel):
    def __init__(self, args, params):
        super(ScoreModel, self).__init__(args, params)
        # loss function
        self.loss = DiffusionLoss(args)

        self._init()

    def forward(self, batch):
        # move graphs to cuda
        tr_pred, rot_pred, tor_pred = self.encoder(batch)

        outputs = {}
        outputs["tr_pred"] = tr_pred
        outputs["rot_pred"] = rot_pred
        outputs["tor_pred"] = tor_pred

        return outputs

    def compute_loss(self, batch, outputs):
        losses = self.loss(batch, outputs)
        return losses

class ConfidenceModel(BaseModel):
    def __init__(self, args, params):
        super(ConfidenceModel, self).__init__(args, params, confidence_mode=True)

        self._init()

    def forward(self, batch):
        # move graphs to cuda
        logits = self.encoder(batch)

        return logits
