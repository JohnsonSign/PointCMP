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

        if self.pretraining:   
            self.temperature = temperature
            self.token_dim = representation_dim

            # regression following P4Transformer
            self.emb_relu = False
            self.depth = 3
            self.heads = 8
            self.dim_head =128
            self.mlp_dim = 2048

            self.pos_embedding = nn.Conv1d(
                in_channels=4, 
                out_channels=self.token_dim, 
                kernel_size=1, 
                stride=1, 
                padding=0, 
                bias=True
            )
            self.transformer = Transformer(
                self.token_dim, 
                self.depth, 
                self.heads, 
                self.dim_head, 
                self.mlp_dim
            )
            self.mlp_head = nn.Sequential(
                nn.Linear(self.mlp_dim, self.mlp_dim, bias=False),
                nn.BatchNorm1d(self.mlp_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.mlp_dim, self.token_dim)
            )
            self.mask_token = nn.Parameter(torch.randn(1, 1, 1, self.token_dim))
            trunc_normal_(self.mask_token, std=.02)

            self.folding = FoldingDecoder(self.token_dim)

            self.v2_fc = nn.Linear(representation_dim, representation_dim)
            # self.v2_fc = nn.Linear(representation_dim, 512)

            self.criterion_local = torch.nn.CrossEntropyLoss()
            # self.criterion_dist = ChamferDistance()
        else:
            self.fc_out = nn.Linear(2048, num_classes)


    def similarity_aug(self, new_features, Batchsize, Sub_clips, N_out, device):
        with torch.no_grad():
            new_features_detach = new_features.clone().detach()
            view1_global_detach = torch.mean(input=new_features_detach, dim=-2, keepdim=False)      # [B, S, C]
            view1_global_detach = torch.max(input=view1_global_detach, dim=-2, keepdim=False)[0]    # [B, C]
            view1_global_detach = F.normalize(view1_global_detach, dim=-1)

            new_features_detach_norm = F.normalize(new_features_detach, dim=-1)
            new_features_detach_norm = new_features_detach_norm.reshape((Batchsize, Sub_clips*N_out, self.token_dim)) # [B, S*N, C]

            mask_list = []
            mask_indx = []
            mask_high_sim = torch.ones((Batchsize, 10, Sub_clips, N_out, self.token_dim), dtype=torch.float32).to(device) # [B, S, N, C]
            for bi in range(Batchsize):
                # for Token Mask
                token_sim_with_global = torch.matmul(new_features_detach_norm[bi], view1_global_detach[bi])  # [S*N]
                sort_token_idx = token_sim_with_global.argsort(dim=-1)
                high_sim_token_idx = sort_token_idx[int(Sub_clips * N_out * 0.8):]
                high_sim_token_clip_idx = (high_sim_token_idx / N_out).int()

                index = -1
                sub_i_max = -1
                for sub_i in range(Sub_clips):
                    len_sub_i = len(torch.where(high_sim_token_clip_idx==sub_i)[0])
                    if len_sub_i > sub_i_max:
                        sub_i_max = len_sub_i
                        index = sub_i

                mask_indx.append(index)
                mask = torch.zeros(Sub_clips, dtype=torch.float32) # [S]
                mask[index] = 1 # mask
                mask_list.append(mask)

                # for Channel Mask 
                channel_score = new_features_detach_norm[bi] * view1_global_detach[bi]         # [S*N, C]
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

        return mask_indx, mask_list, erase_global


    def forward(self, clips):
        device = clips.get_device()

        if self.pretraining:
            Batchsize, Sub_clips, L_sub_clip, N_point, C_xyz = clips.shape                              # [B, S, L', N, 3]
            clips = clips.reshape((-1, L_sub_clip, N_point, C_xyz))

            new_xys, new_features = self.encoder(clips)
            
            new_features = new_features.permute(0, 1, 3, 2)                                             # [B*S, L, N, C] 
            BS, L_out, N_out, C_ = new_features.shape
            new_features = new_features.reshape((-1, C_))
            new_features = self.mlp_head(new_features)
            new_features = new_features.reshape((BS, L_out, N_out, new_features.shape[-1]))

            BS, L_out, N_out, C_out = new_features.shape
            assert(C_out==self.token_dim)

            new_xys = new_xys.reshape((Batchsize, Sub_clips, L_out, N_out, C_xyz))                      # [B, S, L, N, 3]
            new_features = new_features.reshape((Batchsize, Sub_clips, L_out, N_out, C_out))            # [B, S, L, N, C]
            assert(L_out==1) # By default, only the case where each sub-clip is aggregated into one frame is considered.

            new_xys = torch.squeeze(new_xys, dim=-3).contiguous()                                       # [B, S, N, 3]
            new_features = torch.squeeze(new_features, dim=-3).contiguous()                             # [B, S, N, C]

            view1_global = torch.mean(input=new_features, dim=-2, keepdim=False)   
            view1_global = torch.max(input=view1_global, dim=-2, keepdim=False)[0]     
            view1_global = self.v2_fc(view1_global)
            view1_global = F.normalize(view1_global, dim=-1)                          

            # for masking
            mask_indx, mask_list, erase_global = self.similarity_aug(new_features, Batchsize, Sub_clips, N_out, device)

            # mask tokens
            bool_masked_pos = torch.stack(mask_list).to(device) # [B, S]
            mask_token = self.mask_token.expand(Batchsize, Sub_clips, N_out, self.token_dim)       # [B, S, N, C]

            w = bool_masked_pos.unsqueeze(-1).unsqueeze(-1).type_as(mask_token)                    # [B, S, 1, 1]
            maksed_input_tokens = new_features * (1 - w) + mask_token * w                          # [B, S, N, C]
            maksed_input_tokens = maksed_input_tokens.reshape((Batchsize, Sub_clips*N_out, C_out)) # [B, S*N, C]

            # regression following P4Transformer
            xyzts = []
            xyz_list = torch.split(tensor=new_xys, split_size_or_sections=1, dim=1)                # S*[B, 1, N, 3]
            xyz_list = [torch.squeeze(input=xyz, dim=1).contiguous() for xyz in xyz_list]          # S*[B, N, 3]
            for t, xyz in enumerate(xyz_list):
                # [B, N, 3]
                t = torch.ones((xyz.size()[0], xyz.size()[1], 1), dtype=torch.float32, device=device) * (t+1)
                xyzt = torch.cat(tensors=(xyz, t), dim=2)                                          # [B, N, 4]
                xyzts.append(xyzt)
            xyzts = torch.stack(tensors=xyzts, dim=1)                                              # [B, S, N, 4]

            xyzts = torch.reshape(input=xyzts, shape=(xyzts.shape[0], xyzts.shape[1]*xyzts.shape[2], xyzts.shape[3]))  # [B, S*N, 4]
            xyzts = self.pos_embedding(xyzts.permute(0, 2, 1)).permute(0, 2, 1)                    # [B, S*N, C]

            embedding = xyzts + maksed_input_tokens                                                # [B, S*N, C]

            if self.emb_relu:
                embedding = self.emb_relu(embedding)

            output = self.transformer(embedding)                                                   # [B, S*N, C]
            output = output.reshape((Batchsize, Sub_clips, N_out, self.token_dim))                 # [B, S, N, C]

            regression_global = output.clone().detach()
            regression_global = torch.mean(input=regression_global, dim=-2, keepdim=False)   
            regression_global = torch.max(input=regression_global, dim=-2, keepdim=False)[0] 
            regression_global = self.v2_fc(regression_global)
            regression_global = F.normalize(regression_global, dim=-1) 

            # get labels of local features / xyz
            label_mask_feature = []
            label_mask_feature_neg = []
            label_mask_xyz = []
            mask_local_feature = []
            for bi in range(Batchsize):
                mask_i = mask_indx[bi]
                mask_local_feature.append(output[bi,mask_i,:,:])                                   # [N, C]
                label_mask_xyz.append(new_xys[bi,mask_i,:,:])                                      # [N, 3]
                label_mask_feature.append(new_features[bi,mask_i,:,:])                             # [N, C]

                i_dex = np.arange(Sub_clips)
                i_dex = np.delete(i_dex, mask_i)
                label_mask_feature_neg.append(new_features[bi,i_dex,:,:])                          # [S-1, N, C]

            mask_local_feature = torch.stack(tensors=mask_local_feature, dim=0)                    # [B, N, C]
            label_mask_xyz = torch.stack(tensors=label_mask_xyz, dim=0)                            # [B, N, 3]
            label_mask_feature = torch.stack(tensors=label_mask_feature, dim=0)                    # [B, N, C]
            label_mask_feature_neg = torch.stack(tensors=label_mask_feature_neg, dim=0)            # [B, S-1, N, C]

            # matching
            mask_fold_xyz = self.folding(mask_local_feature)                                       # [B, N, 3]

            dist, idx = pointnet2_utils.three_nn(label_mask_xyz.contiguous(), mask_fold_xyz) # (anchor, neighbor)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            mask_local_feature = mask_local_feature.transpose(1,2)                                 # [B, C, N]
            mask_local_feature = pointnet2_utils.three_interpolate(mask_local_feature.contiguous(), idx, weight) # [B, C, N]
            mask_local_feature = mask_local_feature.transpose(1,2)                                 # [B, N, C]

            # local similarity matrixes
            mask_local_feature = mask_local_feature.reshape((-1,C_out))                            # [B*N, C]
            mask_local_feature = F.normalize(mask_local_feature, dim=-1)

            label_mask_feature = label_mask_feature.reshape((-1,C_out))                            # [B*N, C]
            label_mask_feature = F.normalize(label_mask_feature, dim=-1)

            label_mask_feature_neg = label_mask_feature_neg.reshape((-1,C_out))                    # [B*(S-1)*N, C]
            label_mask_feature_neg = F.normalize(label_mask_feature_neg, dim=-1)
            label_mask_feature = torch.cat((label_mask_feature, label_mask_feature_neg), dim=0)    # [B*S*N, C]

            score_local = torch.matmul(mask_local_feature, label_mask_feature.transpose(0,1))      # [B*N, B*S*N]
            score_local = score_local / self.temperature

            target_sim_local = torch.arange(score_local.size()[0]).to(device)
            loss_local = self.criterion_local(score_local, target_sim_local)

            acc1, acc5 = utils.accuracy(score_local, target_sim_local, topk=(1, 5))

            return loss_local, acc1, acc5, view1_global, erase_global.detach(), regression_global.detach()

        else:
            # # If subclip is used like pre-training.
            # Batchsize, Sub_clips, L_sub_clip, N_point, C_xyz = clips.shape                   # [B, S, L, N, 3]
            # clips = clips.reshape((-1, L_sub_clip, N_point, C_xyz))

            new_xys, new_features = self.encoder(clips)

            # # If subclip is used like pre-training.
            # BS, L_out, C_out, N_out = new_features.shape
            # new_features = new_features.reshape((Batchsize, Sub_clips, L_out, C_out, N_out)) # [B, S, L, C, N]
            # assert(L_out==1) # By default, only the case where each sub-clip is aggregated into one frame is considered.
            # new_features = torch.squeeze(new_features, dim=-3).contiguous()                  # [B, S, C, N]

            output = torch.mean(input=new_features, dim=-1, keepdim=False)                    # [B, S, C]
            output = torch.max(input=output, dim=1, keepdim=False)[0]                         # [B, C]

            # Just for linear probing on MSRAction3D
            output = self.fc_out(output)

            return output
