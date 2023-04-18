import torch
import torch.nn as nn
import torch.nn.functional as F


def _init(model):
    for name, param in model.named_parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)

