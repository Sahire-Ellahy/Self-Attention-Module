import math
import torch
import torch.nn as nn
import numpy as np

class MyAttentionLayer(nn.Module):
    """ Custom attention layer made after studying "Attention is All you Need" paper and attention mechanism """
    def __init__(self, QKV_size):
        super().__init__()
        self.QKV_size = QKV_size
        weights = torch.randn(QKV_size, QKV_size, requires_grad=True)
        self.weights = nn.Parameter(weights)

    def forward(self, QKV): #Query, Keys, and Values are the same thing and represented by QKV
        KV = QKV.clone() #need to use a copy of QKV; https://stackoverflow.com/questions/55266154/pytorch-preferred-way-to-copy-a-tensor is relevant
        QKV = torch.sum(QKV, dim=0)
        weightedQuery = torch.matmul(self.weights, QKV)
        similarities = torch.matmul(KV, weightedQuery)
        weightedVects = KV * similarities[:, None]
        selfAttVect = torch.sum(weightedVects, dim=0)
        return selfAttVect
