import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionFusion(nn.Module):
    """
    多视角注意力融合（每视角输入形状 B x D）
    - 支持 share_proj（所有视角共享投影）或独立投影
    - 归一化策略：softmax / sum / none
    返回:
      H: (B, D) 融合后的表示
      alpha: (B, V_eff) 有效视角的权重（剔除了全 NaN 的视角）
    """

    def __init__(self, num_views, dim, share_proj=False, normalize="softmax"):
        super().__init__()
        self.normalize = normalize
        self.dim = dim
        self.share_proj = share_proj

        # 投影层：W^v 与 W^{v*}（均 D->D）
        if share_proj:
            self.proj = nn.ModuleList([nn.Linear(dim, dim, bias=False)])
            self.out_proj = nn.ModuleList([nn.Linear(dim, dim, bias=False)])
        else:
            self.proj = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(num_views)])
            self.out_proj = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(num_views)])

        # 门控参数：γ, β（标量）
        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.0))

        # 初始化
        for group in (self.proj, self.out_proj):
            for lin in group:
                nn.init.xavier_uniform_(lin.weight)

    def _gate(self, y: torch.Tensor) -> torch.Tensor:
        # y * sigmoid(γ y + β)
        return y * torch.sigmoid(self.gamma * y + self.beta)

    def _norm_simple(self, s: torch.Tensor) -> torch.Tensor:
        """
        对 (B, V_eff) 未归一化得分 s 做归一化。
        不涉及“无效视角”掩码，因为已在 forward 前剔除。
        """
        if self.normalize == "softmax":
            return F.softmax(s, dim=-1)
        elif self.normalize == "sum":
            denom = s.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            return s / denom
        else:  # "none"
            return s

    def forward(self, Z_list):
        """
        Z_list: list[Tensor], 长度原本为 V；每个张量形状 (B, D)
        处理：剔除“整列都是 NaN”的视角，仅基于有效视角进行融合。
        返回:
          H: (B, D)
          alpha: (B, V_eff)  —— 仅对应有效视角
        """
        if len(Z_list) == 0:
            raise ValueError("Z_list cannot be empty.")

        # 从原始列表拿到形状与设备信息（不依赖数值内容）
        B, D = Z_list[0].shape
        device = Z_list[0].device

        # 1) 计算有效视角索引：至少存在一个有限值（非 NaN/Inf）
        # kept_idx = [i for i, Z in enumerate(Z_list) if torch.isfinite(Z).any().item()]
        kept_idx = [i for i, Z in enumerate(Z_list)]
        V_eff = len(kept_idx)

        # 2) 若全部为 NaN（无有效视角），安全返回零
        # if V_eff == 0:
        #     H = torch.zeros(B, D, device=device)
        #     alpha = torch.zeros(B, 0, device=device)  # 没有有效视角
        #     return H, alpha

        # 3) 按有效索引取子列表
        Z_eff = [Z_list[i] for i in kept_idx]

        # 4) 线性投影（D->D）；share_proj 用 0 号权重，否则对应原始视角权重
        if self.share_proj:
            Zp = [self.proj[0](Z) for Z in Z_eff]         # (B, D) 列表
            Zo = [self.out_proj[0](Z) for Z in Z_eff]     # (B, D) 列表
        else:
            Zp = [self.proj[i](Z_list[i]) for i in kept_idx]
            Zo = [self.out_proj[i](Z_list[i]) for i in kept_idx]

        # 5) 两两打分：Y = Zp Zp^T（样本内积），形状 (B, V_eff, V_eff)
        Zp_stack = torch.stack(Zp, dim=1)  # (B, V_eff, D)
        Y = torch.matmul(Zp_stack, Zp_stack.transpose(1, 2))  # (B, V_eff, V_eff)

        # 6) 去对角 + 门控，聚合得到每视角得分 s: (B, V_eff)
        mask_offdiag = ~torch.eye(V_eff, dtype=torch.bool, device=device)
        Y_gate = self._gate(Y)
        s = (Y_gate.masked_fill(~mask_offdiag, 0.0)).sum(dim=-1)  # (B, V_eff)

        # 7) 归一化 -> alpha: (B, V_eff)
        alpha = self._norm_simple(s)

        # 8) 融合：H = Σ_v alpha_v * Zo_v
        Zo_stack = torch.stack(Zo, dim=1)  # (B, V_eff, D)
        H = (alpha.unsqueeze(-1) * Zo_stack).sum(dim=1)  # (B, D)

        return H, alpha
