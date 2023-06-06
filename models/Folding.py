import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_and_init_FC_layer(din, dout):
    li = nn.Linear(din, dout)
    # init weights/bias
    nn.init.xavier_uniform_(li.weight.data, gain=nn.init.calculate_gain('relu'))
    li.bias.data.fill_(0.)
    return li


def get_MLP_layers(dims, doLastRelu):
    # dims: (C_in,512,512,3)
    layers = []
    for i in range(1, len(dims)):
        layers.append(get_and_init_FC_layer(dims[i-1], dims[i]))
        if i==len(dims)-1 and not doLastRelu:
            continue
        layers.append(nn.ReLU())
    return layers


class FoldingNetSingle(nn.Module):
    def __init__(self, dims):
        super(FoldingNetSingle, self).__init__()
        self.mlp = PointwiseMLP(dims, doLastRelu=False)

    def forward(self, X):
        return self.mlp.forward(X)


class PointwiseMLP(nn.Sequential):
    '''Nxdin ->Nxd1->Nxd2->...-> Nxdout'''
    def __init__(self, dims, doLastRelu=False):
        layers = get_MLP_layers(dims, doLastRelu)
        super(PointwiseMLP, self).__init__(*layers)


# following foldingnet

class FoldingDecoder(nn.Module):
    def __init__(self, token_dim):
        super(FoldingDecoder, self).__init__()

        self.Fold1 = FoldingNetSingle((token_dim, 512, 512, 3))    # 3MLP
        self.Fold2 = FoldingNetSingle((token_dim+3, 512, 512, 3))  # 3MLP


    def forward(self, features):

        global_features = torch.mean(features, dim=1, keepdim=True)     # [B, 1, C]
        local_features_2fold = features + global_features               # [B, N, C]

        fold_xyz = self.Fold1(local_features_2fold)
        fold_xyz = torch.cat((local_features_2fold, fold_xyz), dim=-1)
        fold_xyz = self.Fold2(fold_xyz)                                 # [B, N, 3]
        
        return fold_xyz