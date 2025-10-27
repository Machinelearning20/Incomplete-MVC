import numpy as np
from time import time
import Nmetrics
import matplotlib.pyplot as plt
import random
from dataset import InCompleteMultiViewDataset
import yaml
from box import Box
import string
import cupy as cp
from cuml.cluster import KMeans as cuKMeans
import torch
from torch.utils.data import DataLoader
from models.MSGMVC import MSGMVC
import cupy as cp
import torch.nn as nn
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack
import Nmetrics
from util import enhance_distribution, student_distribution
from sklearn.preprocessing import normalize
import torch.nn.functional as F
import math

# 最小化内容，风格的相似性
def rbf(x, sigma2=None, eps=1e-8):
    x2 = (x**2).sum(1, keepdim=True)            # (B,1)
    dist2 = (x2 + x2.T - 2 * (x @ x.T)).clamp_min_(0)

    if sigma2 is None:
        vals = dist2.detach()
        vals = vals[~torch.eye(vals.size(0), dtype=torch.bool, device=vals.device)]
        sigma2 = torch.tensor(1.0, device=x.device, dtype=x.dtype) if vals.numel() == 0 else torch.median(vals)

    K = torch.exp(-dist2 / (2 * sigma2 + eps))
    K = 0.5 * (K + K.T)
    return K

def hsic_loss(z1, z2, eps=1e-8, normalized=False):
    n = z1.size(0)
    if n < 2:
        return z1.new_tensor(0.0, requires_grad=True)

    K, L = rbf(z1), rbf(z2)
    # 中心化（等价于 H@K@H，但更省算）
    Kc = K - K.mean(0, keepdim=True) - K.mean(1, keepdim=True) + K.mean()
    Lc = L - L.mean(0, keepdim=True) - L.mean(1, keepdim=True) + L.mean()

    hsic = (Kc * Lc).sum() / ((n - 1) ** 2)   # 或者用 n**2，看你希望的尺度

    if normalized:
        denom = torch.sqrt((Kc*Kc).sum() * (Lc*Lc).sum() + eps)
        hsic = hsic / (denom + eps)
    return hsic

def contrastive_loss_row(y_true: torch.Tensor,
                           y_pred: torch.Tensor,
                           tau: float = 1,
                           eps: float = 1e-12):
    """
    """
    # 防止概率为0
    P = torch.clamp(y_true, min=eps)
    Q = torch.clamp(y_pred, min=eps)
    P = nn.functional.normalize(P, dim=1)
    Q = nn.functional.normalize(Q, dim=1)
    N = P.size(0)
    targets = torch.arange(N, device=P.device)

    # view1 -> view2
    logits = F.cosine_similarity(P.unsqueeze(1), Q.unsqueeze(0), dim=-1)
    # loss = F.nll_loss(logits, targets, reduction="mean") 
    loss = F.cross_entropy(logits, targets, reduction="mean")
    

    # symmetric
    return loss



def contrastive_loss_column(y_true: torch.Tensor,
                           y_pred: torch.Tensor,
                           tau: float = 1,
                           eps: float = 1e-12):
    # 防止概率为0
    P = y_true.t()
    Q = y_pred.t()
    P = torch.clamp(P, min=eps)
    Q = torch.clamp(Q, min=eps)
    P = P / P.sum(dim=1, keepdim=True)
    Q = Q / Q.sum(dim=1, keepdim=True)
    Q_log = torch.log(Q + eps)
    P_log = torch.log(P + eps)
    N = P.size(0)
    targets = torch.arange(N, device=P.device)

    # view1 -> view2
    logits1 = (P @ Q_log.t())/tau
    # loss = F.nll_loss(logits, targets, reduction="mean") 
    loss1 = F.cross_entropy(logits1, targets, reduction="mean")
    
    # view2 -> view1YTF10
    logits2 = (Q @ P_log.t()) / tau
    loss2 = F.cross_entropy(logits2, targets, reduction="mean")

    # symmetric
    return (loss1 + loss2) / 2


def total_loss(
    x,
    z_c,
    z_s,
    z,
    r_x, 
    cluster_unique_assign_c, 
    cluster_sp_assign_c, 
    cluster_unique_center_a,
    args
):
    eps = 1e-15
    # cluster_unique_assign_log = (cluster_unique_assign + eps).log()
    mse_loss = nn.MSELoss()
    cluster_sp_assign_log = [torch.log(cluster_sp_assign_c[v]) for v in range(len(x))]
    cluster_unique_assign_log = torch.log(cluster_unique_assign_c)
    losses_ae_mse = [mse_loss(x[v], r_x[v]) for v in range(len(x))]
    losses_ae_hsic = [hsic_loss(z_c[v], z_s[v]) for v in range(len(x))]
    # losses_cca = [contrastive_loss_row(cluster_unique_assign, cluster_sp_assign[v]) for v in range(len(x))]
    losses_content_contrastive = [contrastive_loss_column(cluster_unique_assign_c, cluster_sp_assign_c[v]) for v in range(len(x))]
    losses_all_contrastive = contrastive_loss_row(cluster_unique_center_a, z) / len(x)

    # losses_kl = [(kl_loss(cluster_sp_assign_log[v],  cluster_unique_assign) + kl_loss(cluster_unique_assign_log, cluster_unique_assign[v]))/2 for v in range(len(x))]
    w = [args.ae_weight, args.hisc_weight, args.contrastive_weight1, args.contrastive_weight2]
    loss = []
    for v in range(len(x)):
        loss_v = w[0] * losses_ae_mse[v] + w[1] * losses_ae_hsic[v] + w[2] * losses_content_contrastive[v] + w[3] * losses_all_contrastive 
        loss.append(loss_v)
    return loss

