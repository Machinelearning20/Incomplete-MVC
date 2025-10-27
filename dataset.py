import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from load_data import load_data
import numpy as np
import random
from torch_minmax import TorchMinMaxScaler

class InCompleteMultiViewDataset(Dataset):
    def __init__(self, dataset, miss_rate):
        self.x, self.y = load_data(dataset)
        # 转成 float32
        self.x = [torch.as_tensor(xv, dtype=torch.float32) for xv in self.x]
        # 转成无符号整数
        self.y = torch.as_tensor(self.y, dtype=torch.uint8)
        # 构造缺失样本
        self.miss_rate = miss_rate
        self.n_samples = len(self.x[0])
        self.n_views = len(self.x)
        self.mask = torch.ones(self.n_samples, self.n_views, dtype = torch.bool)
        n_missing = int(self.n_samples * self.miss_rate)
        self.missing_indices = random.sample(range(self.n_samples), n_missing)
        # 缺失的数据记为Nan
        for idx in self.missing_indices:
            miss_view = miss_view = random.randint(0, self.n_views - 1)
            self.mask[idx, miss_view] = 0
            self.x[miss_view][idx] = torch.nan
        min_max_scaler = TorchMinMaxScaler()
        self.x = [min_max_scaler.fit_transform(self.x[i]) for i in range(self.n_views)]
    
    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return[self.x[i][idx] for i in range(len(self.x))] , self.mask[idx], self.y[idx], idx
    
    def get_num_clusters(self):
        return len(np.unique(self.y))

    def get_views(self):
        return [self.x[i].shape[1] for i in range(len(self.x))]

class CompleteMultiViewSubDataset(Dataset):
    def __init__(self, all_dataset):
        self.n_views = all_dataset.n_views
        self.n_clusters = all_dataset.get_num_clusters()
        self.view_shape = all_dataset.get_views()
        all_x = all_dataset.x
        all_y = all_dataset.y
        mask = torch.ones(all_dataset.n_samples, dtype=torch.bool)
        mask[all_dataset.missing_indices] = False
        self.x = [all_x[i][mask] for i in range(self.n_views)]
        self.y = all_y[mask]
        self.n_samples = len(self.x[0])
        self.mask = torch.ones(self.n_samples, self.n_views, dtype = torch.bool)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return[self.x[i][idx] for i in range(len(self.x))] , self.mask[idx], self.y[idx], idx
    
    def get_num_clusters(self):
        return self.n_clusters

    def get_views(self):
        return self.view_shape



if __name__ == "__main__":
    data = InCompleteMultiViewDataset(dataset='BDGP')
    print(data.get_views())