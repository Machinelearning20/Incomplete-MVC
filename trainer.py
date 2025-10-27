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
from sklearn.cluster import KMeans as skKMeans
import torch
from torch.utils.data import DataLoader
from models.MSGMVC import MSGMVC
import cupy as cp
import torch.nn as nn
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack
import Nmetrics
from util import enhance_distribution, student_distribution, plot_tsne
from sklearn.preprocessing import normalize
from sklearn.metrics import calinski_harabasz_score,silhouette_score, silhouette_samples
import torch.nn.functional as F
import os
from loss import hsic_loss

def minmax_scale_tensor(x: torch.Tensor, eps=1e-12):
    x_min = x.min(dim=0, keepdim=True).values
    x_max = x.max(dim=0, keepdim=True).values
    return (x - x_min) / (x_max - x_min + eps)


class Trainer():
    def __init__(
        self,
        pre_data_loader,
        complete_sub_data_loader,
        all_data_loader,
        model,
        pre_opt,
        opt,
        loss_fn,
        device,
        args,
    ):
        self.pre_data_loader = pre_data_loader    
        self.complete_sub_data_loader = complete_sub_data_loader
        self.all_data_loader = all_data_loader
        self.complete_sub_dataset = complete_sub_data_loader.dataset
        self.all_dataset = all_data_loader.dataset
        self.model = model
        self.pre_opt = pre_opt
        self.opt = opt
        self.loss_fn = loss_fn
        self.device = device
        self.args = args
        self.n_clusters = self.all_dataset.get_num_clusters()
        self.dims = self.all_dataset.get_views()
        self.views = len(self.dims)
        self.seed = args.seed
    
    def pre_train(self):
        self.model.train()
        pre_train_epoch = self.args.pretrain_epochs   
        crit = nn.MSELoss()
        for i in range(pre_train_epoch):
            loss_sum = [0.0] * self.views
            print(f'epoch: {i + 1}')
            for x, _, _, _ in self.pre_data_loader:
                x = [xi.to(self.device) for xi in x]
                self.pre_opt.zero_grad()    
                z_c, z_s,  reconstructed_x = self.model(x, status = 0)
                if i < self.args.warm_up:
                    losses = [crit(x[v], reconstructed_x[v])  for v in range(len(x))]
                else:
                    losses = [crit(x[v], reconstructed_x[v]) + self.args.hisc_weight * hsic_loss(z_c[v], z_s[v])  for v in range(len(x))]
                loss = sum(losses)
                loss.backward()
                self.pre_opt.step()
                for view in range(self.views):
                    loss_sum[view] += losses[view].item()

            loss_total = sum(loss_sum) / self.views
            loss_sum = [loss_total] + loss_sum
            for view in range(len(loss_sum)):
                loss_sum[view] = loss_sum[view] / len(self.pre_data_loader)
            print(f'loss: {loss_sum}')
            print()
        self.model.save_pretrain_model()

    def extract_features(self, dataset):
        '''
        提取特征，包括内容特征和风格特征
        '''
        content_features_list = [[] for i in range(self.views)]
        style_features_list = [[] for i in range(self.views)]
        data_loader = DataLoader(dataset, batch_size = 1024, shuffle = False, num_workers = 0)
        self.model.eval()
        with torch.no_grad():
            for x, _, _, _ in data_loader:
                x = [x[i].to(self.device) for i in range(len(x))]
                if dataset is self.complete_sub_dataset:
                    z_c, z_s, _ = self.model(x, status = 0)
                else:
                    _, z_c, z_s, _, _ = self.model(x, status = 2)
                for i in range(self.views):
                    content_features_list[i].append(z_c[i].detach())
                    style_features_list[i].append(z_s[i].detach())
        content_features = [torch.cat(f, dim = 0) for f in content_features_list]
        style_features = [torch.cat(f, dim = 0) for f in style_features_list]
        self.model.train()
        return content_features, style_features


    def view_sp_cluster_content(self, dataset):
        y_preds = []
        centers = []
        content_features, style_features = self.extract_features(dataset)
        features = content_features
        if self.args.normalize == 1:
            features = [F.normalize(f, p = 2, dim = 1) for f in features]
        # features = [minmax_scale_tensor(f) for f in features]

        # cuml k-means on gpu 
        # for view in range(self.views):
        #     kmeans = cuKMeans(n_clusters=self.n_clusters, init = 'scalable-k-means++', n_init=100, random_state = self.seed)
        #     t = features[view]
        #     t = t.contiguous().float()
        #     feature_cp = cp.fromDlpack(to_dlpack(t))
        #     y_pred_cp = (kmeans.fit_predict(feature_cp))
        #     y_preds.append(from_dlpack(y_pred_cp.toDlpack()).to(self.device))
        #     cc = kmeans.cluster_centers_
        #     centers.append(from_dlpack(cc.toDlpack()).float().to(self.device))
        

        # sklearn k-means on cpu
        for view in range(self.views):
            kmeans = skKMeans(n_clusters=self.n_clusters, init = 'k-means++', n_init=20, random_state = self.seed)
            t = features[view].contiguous().float()
            x = t.detach().cpu().numpy()
            y_pred_np = kmeans.fit_predict(x)
            y_pred = torch.from_numpy(y_pred_np).to(self.device)
            y_preds.append(y_pred)
            cc = torch.from_numpy(kmeans.cluster_centers_).float().to(self.device)  # (K, D)
            centers.append(cc)

        return y_preds, centers, features

    def unique_cluster(self, dataset, data_range):
        self.model.eval()
        with torch.no_grad():
            y_pred = []
            content_features, style_features = self.extract_features(dataset)
            assert data_range in ['content', 'all'], 'unknown type'

            # 先归一化，再融合
            if self.args.normalize == 1:
                content_features = [F.normalize(f, p = 2, dim = 1) for f in content_features]
                style_features = [F.normalize(f, p = 2, dim = 1) for f in style_features]

            if data_range == 'content' :
                z = sum(content_features) / len(content_features)

            else:
                _, weights = self.model.attention_fusion(style_features)
                weights = weights.T.unsqueeze(-1) #[V, B, 1]
                fusion_content = sum(content_features) / len(content_features)
                fusion_style_stack = torch.stack(style_features, dim=0)
                fusion_style = (weights * fusion_style_stack).sum(dim=0)  # [B, D]
                z = torch.cat([fusion_content, fusion_style], dim=1)
            # z = sum(w * f for w, f in zip(weight, features))
            # z = F.normalize(z, p=2, dim=1)

            # cuml k-means on gpu
            # kmeans = cuKMeans(n_clusters=self.n_clusters, init = 'scalable-k-means++', n_init=100, random_state = self.seed)
            # t = z.contiguous().float()
            # t_cp = cp.fromDlpack(to_dlpack(t))
            # y_pred_cp = (kmeans.fit_predict(t_cp))
            # y_pred.append(from_dlpack(y_pred_cp.toDlpack()).to(self.device))
            # cc = kmeans.cluster_centers_
            # center = from_dlpack(cc.toDlpack()).float().to(self.device)

            # sklearn k-means on cpu
            kmeans = cuKMeans(n_clusters=self.n_clusters, init = 'k-means++', n_init=20, random_state = self.seed)
            t = z.contiguous().float()
            t_np = t.detach().cpu().numpy()
            y_pred_np = (kmeans.fit_predict(t_np))
            y_pred = [torch.from_numpy(y_pred_np).to(self.device)]
            cc = kmeans.cluster_centers_
            center = torch.from_numpy(kmeans.cluster_centers_).float().to(self.device)
            assigned_centroids = center[y_pred_np]

        return y_pred, center, z, assigned_centroids

    def init_sp_cluster_centers(self, centers):
        self.model.eval()
        with torch.no_grad():
            for view in range(self.views):
                self.model.content_cluster_layers[view].clusters.copy_(centers[view])
                # 清空梯度，不会关闭梯度
                self.model.content_cluster_layers[view].clusters.grad = None
    
    def evaluate_sp_cluster(self, y_pred_k, centers, features):
        '''
        评估各个视角的聚类结果
        '''
        # y_pred_k, centers, features = self.view_sp_cluster()
        y_pred_k = [y_pred_k[i].cpu().numpy() for i in range(self.views)]
        y_pred_list = [[] for i in range(self.views)]
        y_true = self.dataset.y.cpu().numpy()
        # self.init_sp_cluster_centers(centers)
        self.model.eval()
        with torch.no_grad():
            for view in range(self.views):
                for i in range(0, len(self.dataset), 1024):
                    features_batch = features[view][i:i+1024]
                    q = self.model.cluster_layers[view](features_batch)
                    q = enhance_distribution(q)
                    y_pred_list[view].append(torch.argmax(q, dim = 1))
        
        y_pred = [torch.cat(f, dim = 0).cpu().numpy() for f in y_pred_list]
        for view in range(self.views):
            acc = Nmetrics.acc(y_true, y_pred[view])
            nmi = Nmetrics.nmi(y_true, y_pred[view])
            ari = Nmetrics.ari(y_true, y_pred[view])
            pur = Nmetrics.pur(y_true, y_pred[view])
            # sil = silhouette_score(features[view].cpu().numpy(), y_pred[view])
            print(f'View: {view + 1}, acc: {acc:.5f}, nmi: {nmi:.5f}, ari: {ari:.5f}, pur: {pur:.5f}')
        self.model.train()
        # print()        

    def evaluate_unique_cluster_views(self, y_pred_k, centers, features):
        '''
        评估各个视角的聚类结果
        '''
        # y_pred_k, centers, features = self.view_sp_cluster()
        y_pred_list = []
        q_list = [[] for i in range(self.views)]
        y_true = self.dataset.y.cpu().numpy()
        # self.init_sp_cluster_centers(centers)
        self.model.eval()
        with torch.no_grad():
            for view in range(self.views):
                for i in range(0, len(self.dataset), 1024):
                    features_batch = features[view][i:i+1024]
                    q = self.model.cluster_layers[view](features_batch)
                    q = enhance_distribution(q)
                    q_list[view].append(q)
        q_stacked = [torch.cat(f, dim = 0) for f in q_list]
        q_stacked = torch.stack(q_stacked, dim=0)   # [v, N, K]
        avg_q = q_stacked.mean(dim = 0)
        y_pred = torch.argmax(avg_q, dim = 1)
        y_pred = y_pred.cpu().numpy()
        acc = Nmetrics.acc(y_true, y_pred)
        nmi = Nmetrics.nmi(y_true, y_pred)
        ari = Nmetrics.ari(y_true, y_pred)
        pur = Nmetrics.pur(y_true, y_pred)
        print(f'Unique: acc: {acc:.5f}, nmi: {nmi:.5f}, ari: {ari:.5f}, pur: {pur:.5f}')
        self.model.train()
        indices = {
            'acc': acc,
            'nmi': nmi,
            'ari': ari,
            'pur': pur
        }
        return indices        

    def evaluate_unique_cluster(self, dataset,features, center):
        '''
        评估各个视角的聚类结果
        '''
        # y_pred_k, center, features = self.unique_cluster()
        # center = torch.tensor(center).to(self.device)
        # y_pred_k = y_pred_k[0].cpu().numpy()
        y_pred_list = []
        y_true = dataset.y.cpu().numpy()
        # self.model.unique_center = center
        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(dataset), 1024):
                features_batch = features[i:i+1024]
                q = student_distribution(features_batch, center)
                q = enhance_distribution(q)
                y_pred_list.append(torch.argmax(q, dim = 1))
        y_pred = torch.cat(y_pred_list, dim = 0).cpu().numpy()
        acc = Nmetrics.acc(y_true, y_pred)
        nmi = Nmetrics.nmi(y_true, y_pred)
        ari = Nmetrics.ari(y_true, y_pred)
        pur = Nmetrics.pur(y_true, y_pred)
        # sil = silhouette_score(features.cpu().numpy(), y_pred)
        self.model.train()
        indices = {
            'acc': acc,
            'nmi': nmi,
            'ari': ari,
            'pur': pur,
        }
        return indices


    def train_complete_sub(self):
        print('-'*30)
        print('Train for complete sub dataset')
        print('-'*30)
        self.model.eval()
        with torch.no_grad():
            y_pred_sp_c, centers_sp_c, features_sp_c =  self.view_sp_cluster_content(self.complete_sub_dataset)
            y_pred_uq_c, centers_uq_c, features_uq_c, _ = self.unique_cluster(self.complete_sub_dataset, 'content')
            self.model.unique_center_c = centers_uq_c
            cluster_unique_assign_c = student_distribution(features_uq_c, self.model.unique_center_c)
            cluster_unique_assign_c = enhance_distribution(cluster_unique_assign_c)

            # 更新公共的聚类质心和每个视角的聚类质心
            self.init_sp_cluster_centers(centers_sp_c)
            # self.evaluate_complete_sp_cluster(y_pred_sp, centers_sp, features_sp)    
            new_indices_c = self.evaluate_unique_cluster(self.complete_sub_dataset, features_uq_c, centers_uq_c)
            # is_updated =  self.model.update_best_indice(new_indices)
            print('Copmplete Content Cluster Performance: ACC=%.5f, NMI=%.5f, ARI=%.5f, PUR = %.5f' % 
                       (new_indices_c['acc'], new_indices_c['nmi'], new_indices_c['ari'], new_indices_c['pur']))
            
            y_pred_uq_a, centers_uq_a, features_uq_a, cluster_unique_center_a = self.unique_cluster(self.complete_sub_dataset, 'all')
            new_indices_a = self.evaluate_unique_cluster(self.complete_sub_dataset, features_uq_a, centers_uq_a)
            print('Complete Content+Style Cluster Performance: ACC=%.5f, NMI=%.5f, ARI=%.5f, PUR = %.5f' % 
                       (new_indices_a['acc'], new_indices_a['nmi'], new_indices_a['ari'], new_indices_a['pur']))

            # if is_updated is True and self.args.save is True:
            #     print('saving model to:', self.args.weights)
            #     self.model.save_model()

        for i in range(self.args.complete_epochs):
            print(f'epoch: {i + 1}')
            self.model.train()
            # 评估一下预训练的结果，同时初始化质心，评估不涉及到后续的训练
            # 评估一下全局的聚类结果，评估不涉及到后续的训练
            # 更新自监督聚类标签
            losses_sum = [0.0] * self.views
            for x, y, _, idx in self.complete_sub_data_loader:
                x = [xi.to(self.device) for xi in x]
                self.opt.zero_grad()    
                z_c, z_s, r_x, cluster_sp_assign_c = self.model(x, status = 1)
                if self.args.normalize == 1:
                    z_c = [F.normalize(f, p = 2, dim = 1) for f in z_c]
                    z_s = [F.normalize(f, p = 2, dim = 1) for f in z_s]
                _, w = self.model.attention_fusion(z_s)
                w = w.T.unsqueeze(-1) #[V, B, 1]
                fusion_content = sum(z_c) / len(z_c)
                fusion_style_stack = torch.stack(z_s, dim=0)
                fusion_style = (w * fusion_style_stack).sum(dim=0)  # [B, D]
                z = torch.cat([fusion_content, fusion_style], dim=1)
                
                # reconstructed_z = self.model.generator(z)
                #with torch.no_grad():
                losses = self.loss_fn(
                    x = x,
                    z_c = z_c,
                    z_s = z_s,
                    z = z,
                    r_x = r_x,
                    cluster_unique_assign_c = cluster_unique_assign_c[idx], 
                    cluster_sp_assign_c = cluster_sp_assign_c,
                    cluster_unique_center_a = cluster_unique_center_a[idx],
                    args = self.args
                )
                loss = sum(losses) / self.views
                for view in range(self.views):
                    losses_sum[view] += losses[view]
                loss.backward()
                self.opt.step()
            
            loss_total = sum(losses_sum) / self.views
            losses_sum = [loss_total] + losses_sum
            for view in range(len(losses_sum)):
                losses_sum[view] = losses_sum[view].item() / len(self.complete_sub_data_loader)
            print(f'loss: {losses_sum}')
            print()
            # 更新公共的聚类质心和每个视角的聚类质心，并且评估结果
            if (i + 1) % self.args.update_interval == 0:
                print('更新聚类质心')
                self.model.eval()
                with torch.no_grad():
                    # y_pred_sp, centers_sp, features_sp =  self.view_sp_cluster()
                    y_pred_uq_c, centers_uq_c, features_uq_c, _ = self.unique_cluster(self.complete_sub_dataset, "content")
                    # self.model.unique_center = self.args.m * self.model.unique_center + (1 - self.args.m) * centers_uq
                    self.model.unique_center_c = centers_uq_c
                    cluster_unique_assign = student_distribution(features_uq_c, centers_uq_c)
                    cluster_unique_assign = enhance_distribution(cluster_unique_assign)
                    # 更新公共的聚类质心和每个视角的聚类质心
                    
                    # self.update_sp_cluster_centers(centers_sp)
                    # self.evaluate_sp_cluster(y_pred_sp, centers_sp, features_sp)    
                    y_pred_uq_a, centers_uq_a, features_uq_a, cluster_unique_center_a = self.unique_cluster(self.complete_sub_dataset, 'all') 
                    new_indices_c = self.evaluate_unique_cluster(self.complete_sub_dataset, features_uq_c, centers_uq_c)
                    # is_updated =  self.model.update_best_indice(new_indices)
                    print('Copmplete Content Cluster Performance: ACC=%.5f, NMI=%.5f, ARI=%.5f, PUR = %.5f' % 
                       (new_indices_c['acc'], new_indices_c['nmi'], new_indices_c['ari'], new_indices_c['pur']))
                    
                    new_indices_a = self.evaluate_unique_cluster(self.complete_sub_dataset, features_uq_a, centers_uq_a)
                    print('Complete Content+Style Cluster Performance: ACC=%.5f, NMI=%.5f, ARI=%.5f, PUR = %.5f' % 
                       (new_indices_a['acc'], new_indices_a['nmi'], new_indices_a['ari'], new_indices_a['pur']))
                    print()
                    # if is_updated is True and self.args.save is True:
                    #     print('saving model to:', self.args.weights)
                    #     self.model.save_model()
        self.model.save_complete_sub_model()

    def train_all(self):
        print()
        print('-'*30)
        print('Train for all dataset')
        print('-'*30)
        self.model = self.model.load_complete_sub_model(self.device)
        self.model.eval()
        with torch.no_grad():
            # y_pred_sp_c, centers_sp_c, features_sp_c =  self.view_sp_cluster_content(self.all_dataset)
            # self.init_sp_cluster_centers(centers_sp_c)
            y_pred_uq_c, centers_uq_c, features_uq_c, _ = self.unique_cluster(self.all_dataset, 'content')
            # # self.model.unique_center_c = centers_uq_c
            cluster_unique_assign_c = student_distribution(features_uq_c, centers_uq_c)
            cluster_unique_assign_c = enhance_distribution(cluster_unique_assign_c)
            
            y_pred_uq_sub, centers_uq_sub, features_uq_sub, _ = self.unique_cluster(self.complete_sub_dataset, 'all')
            new_indices_sub = self.evaluate_unique_cluster(self.complete_sub_dataset, features_uq_sub, centers_uq_sub)
            print('Copmplete Sub Dataset Cluster Performance: ACC=%.5f, NMI=%.5f, ARI=%.5f, PUR = %.5f' % 
                       (new_indices_sub['acc'], new_indices_sub['nmi'], new_indices_sub['ari'], new_indices_sub['pur']))
            
            y_pred_uq_all, centers_uq_all, features_uq_all, cluster_unique_center_all = self.unique_cluster(self.all_dataset, 'all')
            new_indices_all = self.evaluate_unique_cluster(self.all_dataset, features_uq_all, centers_uq_all)
            is_updated =  self.model.update_best_indice(new_indices_all)
            print('All Dataset Cluster Performance: ACC=%.5f, NMI=%.5f, ARI=%.5f, PUR = %.5f' % 
                       (new_indices_all['acc'], new_indices_all['nmi'], new_indices_all['ari'], new_indices_all['pur']))
            if is_updated is True and self.args.save is True:
                print('saving model to:', self.args.final_weights)
                self.model.save_final_model()

        for i in range(self.args.epochs):
            print(f'epoch: {i + 1}')
            self.model.train()
            losses_sum = [0.0] * self.views
            for x, y, _, idx in self.all_data_loader:
                x = [xi.to(self.device) for xi in x]
                self.opt.zero_grad()    
                x_filtered = x
                x_filtered, z_c, z_s, r_x, cluster_sp_assign_c = self.model(x, status = 2)
                if self.args.normalize == 1:
                    z_c = [F.normalize(f, p = 2, dim = 1) for f in z_c]
                    z_s = [F.normalize(f, p = 2, dim = 1) for f in z_s]
                _, w = self.model.attention_fusion(z_s)
                w = w.T.unsqueeze(-1) #[V, B, 1]
                fusion_content = sum(z_c) / len(z_c)
                fusion_style_stack = torch.stack(z_s, dim=0)
                fusion_style = (w * fusion_style_stack).sum(dim=0)  # [B, D]
                z = torch.cat([fusion_content, fusion_style], dim=1)
                
                # reconstructed_z = self.model.generator(z)
                # with torch.no_grad():
                losses = self.loss_fn(
                    x = x_filtered,
                    z_c = z_c,
                    z_s = z_s,
                    z = z,
                    r_x = r_x,
                    cluster_unique_assign_c = cluster_unique_assign_c[idx], 
                    cluster_sp_assign_c = cluster_sp_assign_c,
                    cluster_unique_center_a = cluster_unique_center_all[idx],
                    args = self.args
                )
                loss = sum(losses) / self.views
                for view in range(self.views):
                    losses_sum[view] += losses[view]
                # torch.autograd.set_detect_anomaly(True)
                loss.backward()
                self.opt.step()
            
            loss_total = sum(losses_sum) / self.views
            losses_sum = [loss_total] + losses_sum
            for view in range(len(losses_sum)):
                losses_sum[view] = losses_sum[view].item() / len(self.complete_sub_data_loader)
            print(f'loss: {losses_sum}')
            print()
            # 更新公共的聚类质心和每个视角的聚类质心，并且评估结果

            if (i + 1) % self.args.update_interval == 0:
                print('更新聚类质心')
                self.model.eval()
                with torch.no_grad():
                    # y_pred_sp, centers_sp, features_sp =  self.view_sp_cluster()
                    y_pred_uq_c, centers_uq_c, features_uq_c, _ = self.unique_cluster(self.all_dataset, "content")
                    self.model.unique_center_c = centers_uq_c
                    cluster_unique_assign = student_distribution(features_uq_c, centers_uq_c)
                    cluster_unique_assign = enhance_distribution(cluster_unique_assign)
                    
                    y_pred_uq_sub, centers_uq_sub, features_uq_sub, _ = self.unique_cluster(self.complete_sub_dataset, 'all') 
                    new_indices_sub = self.evaluate_unique_cluster(self.complete_sub_dataset, features_uq_sub, centers_uq_sub)
                    print('Copmplete Sub dataset Cluster Performance: ACC=%.5f, NMI=%.5f, ARI=%.5f, PUR = %.5f' % 
                       (new_indices_sub['acc'], new_indices_sub['nmi'], new_indices_sub['ari'], new_indices_sub['pur']))
                    
                    y_pred_uq_all, centers_uq_all, features_uq_all, cluster_unique_center_all = self.unique_cluster(self.all_dataset, 'all')
                    new_indices_all = self.evaluate_unique_cluster(self.all_dataset, features_uq_all, centers_uq_all)
                    print('All Dataset Cluster Performance: ACC=%.5f, NMI=%.5f, ARI=%.5f, PUR = %.5f' % 
                            (new_indices_all['acc'], new_indices_all['nmi'], new_indices_all['ari'], new_indices_all['pur']))
                    is_updated =  self.model.update_best_indice(new_indices_all)
                    print('Best Indicators: ACC=%.5f, NMI=%.5f, ARI=%.5f, PUR = %.5f' % 
                        (self.model.best_indice['acc'], self.model.best_indice['nmi'],self.model.best_indice['ari'],self.model.best_indice['pur']))
                    if is_updated is True and self.args.save is True:
                        print('saving model to:', self.args.final_weights)
                        self.model.save_final_model()
                    print()
    
    def test(self):
        self.model = self.model.load_final_model(self.device)
        self.model.eval()
        y_pred_uq_all, centers_uq_all, features_uq_all, _ = self.unique_cluster(self.all_dataset, 'all')
        cluster_unique_assign = student_distribution(features_uq_all, centers_uq_all)
        cluster_unique_assign = enhance_distribution(cluster_unique_assign)
        new_indices_all = self.evaluate_unique_cluster(self.all_dataset, features_uq_all, centers_uq_all)
        print('All Dataset Cluster Performance: ACC=%.5f, NMI=%.5f, ARI=%.5f, PUR = %.5f' % 
                (new_indices_all['acc'], new_indices_all['nmi'], new_indices_all['ari'], new_indices_all['pur']))
        # fig_dir = os.path.join(self.args.save_dir, self.args.dataset + '.pdf')
        # plot_tsne(features_uq.cpu().numpy(), y_pred_uq[0].cpu().numpy(), fig_dir, self.args.seed)
        # is_pause = 1