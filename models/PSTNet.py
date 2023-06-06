import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'modules'))

from pst_convolutions import PSTConv
import pointnet2_utils
import utils


class Encoder(nn.Module):
    def __init__(self, 
                radius=1.5, 
                nsamples=3*3):
        super(Encoder, self).__init__()

        self.conv1 =  PSTConv(in_planes=0,
                              mid_planes=45,
                              out_planes=64,
                              spatial_kernel_size=[radius, nsamples],
                              temporal_kernel_size=1,
                              spatial_stride=2,
                              temporal_stride=1,
                              temporal_padding=[0,0],
                              spatial_aggregation="multiplication",
                              spatial_pooling="sum")

        self.conv2a = PSTConv(in_planes=64,
                              mid_planes=96,
                              out_planes=128,
                              spatial_kernel_size=[2*radius, nsamples],
                              temporal_kernel_size=3,
                              spatial_stride=2,
                              temporal_stride=2,
                              temporal_padding=[1,0],
                              spatial_aggregation="multiplication",
                              spatial_pooling="sum")

        self.conv2b = PSTConv(in_planes=128,
                              mid_planes=192,
                              out_planes=256,
                              spatial_kernel_size=[2*radius, nsamples],
                              temporal_kernel_size=3,
                              spatial_stride=1,
                              temporal_stride=1,
                              temporal_padding=[1,1],
                              spatial_aggregation="multiplication",
                              spatial_pooling="sum")

        self.conv3a = PSTConv(in_planes=256,
                              mid_planes=384,
                              out_planes=512,
                              spatial_kernel_size=[2*2*radius, nsamples],
                              temporal_kernel_size=3,
                              spatial_stride=2,
                              temporal_stride=2,
                              temporal_padding=[1,0],
                              spatial_aggregation="multiplication",
                              spatial_pooling="sum")

        self.conv3b = PSTConv(in_planes=512,
                              mid_planes=768,
                              out_planes=1024,
                              spatial_kernel_size=[2*2*radius, nsamples],
                              temporal_kernel_size=3,
                              spatial_stride=1,
                              temporal_stride=1,
                              temporal_padding=[1,1],
                              spatial_aggregation="multiplication",
                              spatial_pooling="sum")

        self.conv4 =  PSTConv(in_planes=1024,
                              mid_planes=1536,
                              out_planes=2048,
                              spatial_kernel_size=[2*2*radius, nsamples],
                              temporal_kernel_size=1,
                              spatial_stride=2,
                              temporal_stride=1,
                              temporal_padding=[0,0],
                              spatial_aggregation="multiplication",
                              spatial_pooling="sum")


    def forward(self, clips_input):

        new_xys, new_features = self.conv1(clips_input, None)
        new_features = F.relu(new_features)

        new_xys, new_features = self.conv2a(new_xys, new_features)
        new_features = F.relu(new_features)

        new_xys, new_features = self.conv2b(new_xys, new_features)
        new_features = F.relu(new_features)

        new_xys, new_features = self.conv3a(new_xys, new_features)
        new_features = F.relu(new_features)

        new_xys, new_features = self.conv3b(new_xys, new_features)
        new_features = F.relu(new_features)

        new_xys, new_features = self.conv4(new_xys, new_features)  # [B*S, L', N, 3] [B*S, L', C, N]

        return new_xys, new_features
