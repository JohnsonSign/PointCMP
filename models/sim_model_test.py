import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import timm
from timm.models.layers import trunc_normal_

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'modules'))

from Folding import *
from PSTNet import *
from transformer import *
from pst_convolutions import PSTConv
from chamfer_distance import ChamferDistance
import pointnet2_utils
import utils


class ContrastiveLearningModel(nn.Module):
    def __init__(self, 
                radius=1.5, 
                nsamples=3*3, 
                representation_dim=1024, 
                num_classes=20,
                temperature=0.1,
                pretraining=True):
        super(ContrastiveLearningModel, self).__init__()

        self.encoder = Encoder(radius=radius, nsamples=nsamples)

        self.pretraining = pretraining

        self.mask_channel = True

        self.token_dim = representation_dim
        self.emb_relu = False
        self.depth = 3
        self.heads = 8
        self.dim_head =128
        self.mlp_dim = 2048

        self.mlp_head = nn.Sequential(
            nn.Linear(self.mlp_dim, self.mlp_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.mlp_dim, self.token_dim)
        )

        self.v2_fc = nn.Linear(representation_dim, representation_dim)
        # self.v2_fc = nn.Linear(representation_dim, 512)


    def forward(self, clips, clipsv2):

        device = clips.get_device()

        # pretraining:
        Batchsize, Sub_clips, L_sub_clip, N_point, C_xyz = clips.shape
        clips = clips.reshape((-1, L_sub_clip, N_point, C_xyz))     
        clipsv2 = clipsv2.reshape((-1, L_sub_clip, N_point, C_xyz))  
        clips_input = torch.cat(tensors=(clips, clipsv2), dim=0)   # [2*B*S, L', N, 3]

        new_xys, new_features = self.encoder(clips_input)

        new_features = new_features.permute(0, 1, 3, 2)    # [B*S, L, N, C] 
        BS2, L_out, N_out, C_out = new_features.shape
        new_features = self.mlp_head(new_features)
        new_features = new_features.reshape((2, Batchsize, Sub_clips, L_out, N_out, self.token_dim)) # [2, B, S, L, N, C]
        assert(L_out==1)

        new_features = torch.squeeze(new_features, dim=-3).contiguous()  # [2, B, S, N, C]

        # for global view
        view1_global = torch.mean(input=new_features, dim=-2, keepdim=False)                 # [2, B, S, C]
        view1_global = torch.max(input=view1_global, dim=-2, keepdim=False)[0]               # [2, B, C]
        view1_global = self.v2_fc(view1_global)
        view1_global = F.normalize(view1_global, dim=-1)                                     # [2, B, C]

        view1_ = view1_global[0]
        view2_ = view1_global[1]

        if self.mask_channel:
            # for masking
            with torch.no_grad():
                new_features_detach = new_features.clone().detach()
                new_features_detach = new_features_detach[0]
                view1_global_detach = torch.mean(input=new_features_detach, dim=-2, keepdim=False)             # [B, S, C]
                view1_global_detach = torch.max(input=view1_global_detach, dim=-2, keepdim=False)[0]           # [B, C]
                view1_global_detach = F.normalize(view1_global_detach, dim=-1)

                new_features_detach_norm = F.normalize(new_features_detach, dim=-1)
                new_features_detach_norm = new_features_detach_norm.reshape((Batchsize, Sub_clips*N_out, self.token_dim)) # [B, S*N, C]

                mask_list = []
                mask_indx = []
                mask_high_sim = torch.ones((Batchsize, 10, Sub_clips, N_out, self.token_dim), dtype=torch.float32).to(device) # [B, S, N, C]
                for bi in range(Batchsize):
                    channel_score = new_features_detach_norm[bi] * view1_global_detach[bi]        # [S*N, C]
                    sort_idx = channel_score.argsort(dim=-1)
                    sort_idx_sort = sort_idx.argsort(dim=-1)
                    sort_idx_sort = sort_idx_sort.sum(dim=0)
                    sort_idx_sort = sort_idx_sort.argsort(dim=-1)
                    high_similarity_idx = sort_idx_sort[int(self.token_dim * 0.8):]
                    c_idx_num = high_similarity_idx.shape[0]
                    mask_high_sim[bi,0,:,:,high_similarity_idx] = 0 # mask high sim to 0

                    for mci in range(1, mask_high_sim.shape[1]):
                        c_index = torch.LongTensor(random.sample(range(c_idx_num), int(c_idx_num*0.95))).to(device)
                        high_sim_idx_random = torch.index_select(high_similarity_idx, 0, c_index)
                        mask_high_sim[bi,mci,:,:,high_sim_idx_random] = 0

            erase_global = new_features_detach.unsqueeze(1) * mask_high_sim
            erase_global = torch.mean(input=erase_global, dim=-2, keepdim=False)   
            erase_global = torch.max(input=erase_global, dim=-2, keepdim=False)[0]  
            erase_global = self.v2_fc(erase_global)
            erase_global = F.normalize(erase_global, dim=-1)     

            return view1_, view2_.detach() erase_global.detach()

        else:
            return view1_, view2_.detach()


