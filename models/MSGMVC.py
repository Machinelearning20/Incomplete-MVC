import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import argparse
import string 
from box import Box
import yaml
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from util import variance_scaling_init
from autoencoders import Encoder, Decoder
from attention_fusion import AttentionFusion
from cluser_layer import ClusteringLayer
from sklearn.impute import KNNImputer
import numpy as np

class MSGMVC(nn.Module):
    def __init__ (
            self, 
            num_samples, 
            n_clusters, 
            view_shape,
            alpha = 1.0,            
            args = None
        ):
        super().__init__()
        self.view_shape = view_shape
        self.view = len(view_shape)
        self.num_samples = num_samples
        self.n_clusters = n_clusters

        self.alpha = alpha

        self.content_dim = args.content_dim
        self.style_dim = args.style_dim
        
        assert all(v == self.content_dim[0] for v in self.content_dim), "content_dim values not equal"
        assert all(v == self.style_dim[0] for v in self.style_dim), "style_dim values not equal"


        self.z_c_dim = self.content_dim[0]
        self.z_s_dim = self.style_dim[0]
        self.z_dim = self.z_c_dim + self.z_s_dim

        self.unique_center_c  = torch.zeros((self.n_clusters, self.z_dim))
        # self.unique_center = None
        self.args = args
        
        self.encoder_trunk_dim = args.encoder_trunk_dim  # 编码器的主干网络
        self.encoder_head_dim = args.encoder_head_dim
        # 编码器内容风格共用的主干网络
        self.encoder_trunk = nn.ModuleList(
            [Encoder([self.view_shape[i]] + self.encoder_trunk_dim) for i in range(len(self.view_shape))]
        )
        input_dim = [self.encoder_trunk_dim[-1] if len(self.encoder_trunk_dim) > 0 else self.view_shape[i] for i in range(len(self.view_shape))]
        
        # 用于内容的编码器分支
        self.encoder_content = nn.ModuleList(
            [Encoder([input_dim[i]] + self.encoder_head_dim + [self.content_dim[i]]) for i in range(len(self.view_shape))]
        )
        # 用于风格的编码器分支
        self.encoder_style = nn.ModuleList(
            [Encoder([input_dim[i]] + self.encoder_head_dim + [self.style_dim[i]]) for i in range(len(self.view_shape))]
        )
        self.decoder_head_dim = args.encoder_head_dim[::-1]
        self.decoder_trunk_dim = args.encoder_trunk_dim[::-1]

        # output_dim = [self.decoder_trunk_dim[0] if len(self.decoder_trunk_dim) > 0 else [] for i in range(len(self.view_shape))]
        # 用于内容的解码器分支
        self.decoder_content = nn.ModuleList(
            [Decoder([self.content_dim[i]] + self.decoder_head_dim) for i in range(len(self.view_shape))]
        )
        # 用于风格的解码器分支
        self.decoder_style = nn.ModuleList(
            [Decoder([self.style_dim[i]] + self.decoder_head_dim) for i in range(len(self.view_shape))]
        )
        # 解码器内容风格共用的主干网络
        self.decoder_trunk = nn.ModuleList(
            [Decoder([2 * self.decoder_head_dim[-1]] + self.decoder_trunk_dim + [self.view_shape[i]]) for i in range(len(self.view_shape))]
        )
        
        self.content_cluster_layers = nn.ModuleList(
            [ClusteringLayer(self.n_clusters, self.z_c_dim) for i in range(len(self.view_shape))]
        )
        
        self.attention_fusion = AttentionFusion(self.view, self.z_s_dim)
        self.best_indice = {
            'acc': 0.0,
            'nmi': 0.0,
            'ari': 0.0,
            'pur': 0.0,
        }

    @staticmethod
    # @torch.no_grad()
    def filter_no_nan(x_list):
        nan_mask_per_view = [torch.isnan(x_v).any(dim=1) for x_v in x_list]  # 每个 [B]
        nan_mask_global = torch.stack(nan_mask_per_view, dim=0).any(dim=0)   # [B]
        keep_mask = ~nan_mask_global                                         # [B]
        filtered_list = [x_v[keep_mask] for x_v in x_list]
        return filtered_list


    # @staticmethod
    # # @torch.no_grad()
    # def fill_content(z_c):
    #     z_filled_all = []
    #     for z_v in z_c:
    #         # 若视角全为 NaN，全填Nan
    #         if torch.isnan(z_v).all():
    #             # z_filled_all.append(torch.full_like(z_v, float('nan')))
    #             continue
            
    #         valid = ~torch.isnan(z_v)  # [B, D]
    #         cnt = valid.sum(dim=0).clamp_min(1)             # 每列有效样本数
    #         s   = torch.where(valid, z_v, torch.zeros_like(z_v)).sum(dim=0)
    #         mean_d = (s / cnt).detach()                     # [D] stop-grad
    #         mean_d = torch.nan_to_num(mean_d, nan=0.0)

    #         # 对 NaN 元素用均值填充
    #         mean_exp = mean_d.unsqueeze(0).expand_as(z_v)   # [B, D]
    #         z_filled = torch.where(torch.isnan(z_v), mean_exp.detach(), z_v)
    #         z_filled_all.append(z_filled)

    #     return z_filled_all


    @staticmethod
    # @torch.no_grad()
    def fill_content(z_c, eps = 1e-8):
        X = torch.stack(z_c, dim=0)              # [V,B,D]
        V, B, D = X.shape
        device, dtype = X.device, X.dtype

        valid = ~torch.isnan(X)


        X0     = torch.where(valid, X, torch.zeros_like(X))
        cnt_v  = valid.sum(dim=1, keepdim=True).clamp_min(1)         # [V,1,D]
        mu_v   = (X0.sum(dim=1, keepdim=True) / cnt_v).detach()      # [V,1,D]
        var_v  = torch.where(valid, (X - mu_v)**2, torch.zeros_like(X)).sum(dim=1, keepdim=True) / cnt_v
        std_v  = var_v.sqrt().clamp_min(eps).detach()                # [V,1,D]


        Xn = (X - mu_v) / std_v                                      # [V,B,D]

        valid_n = ~torch.isnan(Xn)
        Xn0     = torch.nan_to_num(Xn, 0.0)

        sum_all = Xn0.sum(dim=0, keepdim=True)                       # [1,B,D]
        cnt_all = valid_n.sum(dim=0, keepdim=True).to(dtype)         # [1,B,D]

        sum_others = sum_all - Xn0                                   # [V,B,D]
        cnt_others = cnt_all - valid_n.to(dtype)                     # [V,B,D]

        mean_others = sum_others / cnt_others.clamp_min(1)           # [V,B,D]
        mean_others = torch.where(cnt_others > 0, mean_others,
                                torch.full_like(mean_others, float('nan')))


        Xn_filled = torch.where(torch.isnan(Xn), mean_others.detach(), Xn)


        X_filled = Xn_filled * std_v + mu_v

        return [X_filled[v] for v in range(V)]



    @staticmethod
    # @torch.no_grad()
    def fill_style(z_s, z_c, k, weights: str = "uniform", eps = 1e-8):
        filled_list = []
        V = len(z_s)
        for v in range(V):
            zs = z_s[v]
            zc = z_c[v]
            B, D = zs.shape
            device = zs.device
            dtype  = zs.dtype
            missing_row = torch.isnan(zs).all(dim=1)  # [B]
            present_row = ~missing_row
            if not missing_row.any():
                filled_list.append(zs)
                continue
            # 备选近邻（必须在 z_s[v] 中非缺失）, knn中找到的z_c不能说对应的z_s是缺失的
            candidate_idx = torch.nonzero(present_row, as_tuple=False).squeeze(1)  # [M]
            if candidate_idx.numel() > 0:
                global_mean = torch.nanmean(zs[present_row], dim=0)
            else:

                global_mean = torch.zeros(D, device=device, dtype=dtype)
            # 只对缺失的样本做 KNN
            target_idx = torch.nonzero(missing_row, as_tuple=False).squeeze(1)  # [T]
            zc_cand = zc[candidate_idx].detach()                              # [M, D]
            zs_cand = z_s[v][candidate_idx].detach()   
            with torch.no_grad():
                diff = zc[target_idx].unsqueeze(1) - zc_cand.unsqueeze(0)
                dist = torch.norm(diff, dim=2)  # [T, M]
                if candidate_idx.numel() > 0:
                    self_mask = (target_idx.unsqueeze(1) == candidate_idx.unsqueeze(0))  # [T, M]
                    dist = dist.masked_fill(self_mask, float('inf'))
                # 选 top-k 的索引（若 M < k，则取 M）
                k_eff = min(k, candidate_idx.numel())

                if k_eff == 0:
                    # 没有可用近邻
                    for i in target_idx.tolist():
                            zs[i] = global_mean.detach()
                    filled_list.append(zs)
                    continue

                nn_dist, nn_pos = torch.topk(dist, k=k_eff, largest=False, dim=1)  # [T, k_eff]
                nn_idx = candidate_idx[nn_pos] 

            gathered = zs_cand[nn_pos]  # 注意：nn_pos 是在候选中的位置索引
            if weights == "uniform":
                impute = gathered.mean(dim=1)  # [T, D]
            elif weights == "distance":
                # 权重 = 1 / (dist + eps)
                w = 1.0 / (nn_dist + eps)                  # [T, k_eff]
                w = w / (w.sum(dim=1, keepdim=True) + eps) # 归一化
                impute = (gathered * w.unsqueeze(-1)).sum(dim=1)  # [T, D]
            else:
                raise ValueError("weights 只能是 'uniform' 或 'distance'")

            # 写回
            zs[target_idx] = impute.detach()
            # nan_rows = torch.isnan(zs).any(dim=1)
            # if nan_rows.any():
            #     zs[nan_rows] = torch.where(
            #         torch.isnan(zs[nan_rows]),
            #         global_mean.unsqueeze(0).expand_as(zs[nan_rows]),
            #         zs[nan_rows]
            #     )
            filled_list.append(zs)
        return filled_list


    def _masked_forward(self, mod, x_i):
        """
        只对 x_i 的有效行做 mod(x_i[valid])，无效行输出为 NaN；保持梯度对有效行正常回传。
        x_i: [B, D]  (浮点)
        mod: 任意子模块(Linear/Conv/Encoder等)，输入 [B_valid, D] 输出 [B_valid, *]
        """
        valid = ~torch.isnan(x_i).any(dim=1)          # [B]
        if valid.all():
            return mod(x_i)

        y_valid = mod(x_i[valid])                      # 只算有效行
        # 构造 full 输出并回填
        full = x_i.new_full((x_i.size(0), y_valid.size(1)), float('nan'))
        full[valid] = y_valid
        return full


    def forward(self, x, status = 0):
        # status = 0, 预训练的forward
        z_1 = [self._masked_forward(self.encoder_trunk[i],  x[i])     for i in range(len(x))]
        z_c = [self._masked_forward(self.encoder_content[i], z_1[i])  for i in range(len(x))]  # 内容表示
        z_s = [self._masked_forward(self.encoder_style[i],   z_1[i])  for i in range(len(x))]  # 风格表示
        d_c = [self._masked_forward(self.decoder_content[i], z_c[i])  for i in range(len(x))]
        d_s = [self._masked_forward(self.decoder_style[i],   z_s[i])  for i in range(len(x))]
        d   = [torch.cat([d_c[i], d_s[i]], dim=1)                      for i in range(len(x))]  # 内容 + 风格
        reconstructed_x = [self._masked_forward(self.decoder_trunk[i], d[i]) for i in range(len(x))]
        # 融合内容，风格表示        
        if status == 0: # AE预训练
            return z_c, z_s, reconstructed_x
        elif status == 1: # 完整子集数据集的训练
            cluster_sp_c = [self.content_cluster_layers[i](z_c[i]) for i in range(len(x))]
            return z_c, z_s, reconstructed_x, cluster_sp_c
        else: # 完整+不完整
            # 对z_c，用批次中存在的取平均值填充
            # 对z_s，用knn的均值填充
            # 填充的值不对AE网络有梯度回传，但对后面的聚类回传
            x_filtered = MSGMVC.filter_no_nan(x)
            reconstructed_x_filtered = MSGMVC.filter_no_nan(reconstructed_x)
            z_c_filled = MSGMVC.fill_content(z_c)
            z_s_filled = MSGMVC.fill_style(z_s, z_c_filled, self.args.k)
            cluster_sp_c = [self.content_cluster_layers[i](z_c_filled[i]) for i in range(len(z_c_filled))]
            return x_filtered, z_c_filled, z_s_filled, reconstructed_x_filtered, cluster_sp_c

    def update_best_indice(self, new_indice): 
        for key in self.best_indice.keys():
            if self.best_indice[key] < new_indice[key]:
                self.best_indice = new_indice
                return True
            elif self.best_indice[key] > new_indice[key]:
                return False
        return False
    
    def save_pretrain_model(self):
        torch.save({
            "encoder_trunk": self.encoder_trunk.state_dict(),
            "encoder_content": self.encoder_content.state_dict(),
            "encoder_style": self.encoder_style.state_dict(),
            "decoder_trunk": self.decoder_trunk.state_dict(),
            "decoder_content": self.decoder_content.state_dict(),
            "decoder_style": self.decoder_style.state_dict()
        }, self.args.pretrain_weights)

    def save_complete_sub_model(self):
        torch.save({
            "model": self.state_dict(),
        }, self.args.complete_sub_weights)
    
    def save_final_model(self):
        torch.save({
            "model": self.state_dict(),
        }, self.args.final_weights)
    
    def load_pretrain_model(self, device):
        checkpoint = torch.load(self.args.pretrain_weights, map_location=device)
        self.encoder_trunk.load_state_dict(checkpoint["encoder_trunk"])
        self.encoder_content.load_state_dict(checkpoint["encoder_content"])
        self.encoder_style.load_state_dict(checkpoint["encoder_style"])
        self.decoder_trunk.load_state_dict(checkpoint["decoder_trunk"])
        self.decoder_content.load_state_dict(checkpoint["decoder_content"])
        self.decoder_style.load_state_dict(checkpoint["decoder_style"])
        self.encoder_trunk.to(device)
        self.encoder_content.to(device)
        self.encoder_style.to(device)
        self.decoder_trunk.to(device)
        self.decoder_content.to(device)
        self.decoder_style.to(device)

    def load_complete_sub_model(self, device):
        checkpoint = torch.load(self.args.complete_sub_weights, map_location=device)
        self.load_state_dict(checkpoint["model"])
        self.to(device)
        return self

    def load_final_model(self, device):
        checkpoint = torch.load(self.args.final_weights, map_location=device)
        self.load_state_dict(checkpoint["model"])
        self.to(device)
        self.eval()  
        return self