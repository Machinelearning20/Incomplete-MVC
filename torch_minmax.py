import torch

class TorchMinMaxScaler:
    def __init__(self, feature_range=(0.0, 1.0), eps=1e-8):
        self.min_ = None
        self.max_ = None
        self.range = feature_range
        self.eps = eps

    def fit(self, x, dim=-2):
        """
        x: 任意维度 tensor，例如 (B, D) 或 (N, V, D)
        dim: 按哪个维度统计 min/max（默认倒数第二维是样本维）
        """
        # 例如 x: (B, D) => dim=0; x: (N, V, D) => dim=0 表示按样本维
        x_safe = torch.where(torch.isnan(x), torch.tensor(float('inf'), device=x.device), x)
        self.min_ = torch.min(x_safe, dim=0, keepdim=True).values

        x_safe = torch.where(torch.isnan(x), torch.tensor(float('-inf'), device=x.device), x)
        self.max_ = torch.max(x_safe, dim=0, keepdim=True).values
        return self

    def transform(self, x):
        scale = self.range[1] - self.range[0]
        return (x - self.min_) / (self.max_ - self.min_ + self.eps) * scale + self.range[0]

    def inverse_transform(self, x_scaled):
        scale = self.range[1] - self.range[0]
        return (x_scaled - self.range[0]) / scale * (self.max_ - self.min_) + self.min_

    def fit_transform(self, x, dim=-2):
        return self.fit(x, dim=dim).transform(x)
