#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 18:11:45 2022

@author: Reza Kakooee
"""

#%%
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms

from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Data as gData
from torch_geometric.loader import DataLoader as gDataLoader

from torch_intermediate_layer_getter import IntermediateLayerGetter as MidGetter


#%%
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()
        
        if downsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
                )
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Sequential()
            
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    
    def forward(self, x):
        shortcut = self.shortcut(x)
        
        x = nn.ReLU()(self.bn1(self.conv1(x)))
        x = nn.ReLU()(self.bn2(self.conv2(x)))
        
        x = nn.ReLU()(x + shortcut)
        
        return x
            
        
        
#%%
class ResNet18(nn.Module):
    def __init__(self, in_channels, resblock, outputs=2*64):
        super().__init__()
        
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            )
        
        self.layer1 = nn.Sequential(
            resblock(64, 64, downsample=False),
            resblock(64, 64, downsample=False)
            )
        
        self.layer2 = nn.Sequential(
            resblock(64, 128, downsample=True),
            resblock(128, 128, downsample=False)
            )
        
        self.layer3 = nn.Sequential(
            resblock(128, 256, downsample=True),
            resblock(256, 256, downsample=False)
            )
        
        self.layer4 = nn.Sequential(
            resblock(256, 512, downsample=True),
            resblock(512, 512, downsample=False)
            )
        
        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(512, outputs)
        
    
    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x)
        x = nn.Flatten()(x) # torch.flatten(x)
        x = self.fc(x)
        
        return  x



# #%%
class ContextModel(nn.Module):
    def __init__(self, net_config):
        nn.Module.__init__(self)
        
        self.net_config = net_config
        
        self.p = 0.0
        
        self.resnet18 = ResNet18(net_config['_in_channels_cnn'], ResBlock, outputs=2*64)
        
        self._cnn = nn.Sequential(
            nn.Conv2d(net_config['_in_channels_cnn'], 16, 3),
            nn.ReLU(),
            nn.Conv2d(16, 32, 5),
            nn.ReLU(),
            nn.Conv2d(32, 256, 11),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(12544, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
            )

        self.gconv1_pd = GCNConv(net_config['dim_pd_features'], 512)
        self.gconv1_pc_old = GCNConv(net_config['dim_pc_features'], 512)
        self.gconv1_fc_old = GCNConv(net_config['dim_fc_features'], 512)
        self.gconv1_pc = GCNConv(net_config['dim_pc_features'], 512)
        self.gconv1_fc = GCNConv(net_config['dim_fc_features'], 512)
        
        self.gconv2_pd = GCNConv(512, 512)
        self.gconv2_pc_old = GCNConv(512, 512)
        self.gconv2_fc_old = GCNConv(512, 512)
        self.gconv2_pc = GCNConv(512, 512)
        self.gconv2_fc = GCNConv(512, 512)
        
        self.gconv3_pd = GCNConv(512, 256)
        self.gconv3_pc_old = GCNConv(512, 256)
        self.gconv3_fc_old = GCNConv(512, 256)
        self.gconv3_pc = GCNConv(512, 256)
        self.gconv3_fc = GCNConv(512, 256)
        
        self.gconv4_pd = GCNConv(256, 256)
        self.gconv4_pc_old = GCNConv(256, 256)
        self.gconv4_fc_old = GCNConv(256, 256)
        self.gconv4_pc = GCNConv(256, 256)
        self.gconv4_fc = GCNConv(256, 256)
        
        self.glinear1_pd = torch.nn.Linear(256, 128)
        self.glinear1_pc_old = torch.nn.Linear(256, 128)
        self.glinear1_fc_old = torch.nn.Linear(256, 128)
        self.glinear1_pc = torch.nn.Linear(256, 128)
        self.glinear1_fc = torch.nn.Linear(256, 128)
        
        self.glinear2_pd = torch.nn.Linear(128, 128)
        self.glinear2_pc_old = torch.nn.Linear(128, 128)
        self.glinear2_fc_old = torch.nn.Linear(128, 128)
        self.glinear2_pc = torch.nn.Linear(128, 128)
        self.glinear2_fc = torch.nn.Linear(128, 128)
        
                
        self.glinear_before_end = torch.nn.Linear(5*128, 4*128)
        
        self.glinear_end = torch.nn.Linear(4*128, 2*64)
        
        self._reward_head = nn.Sequential(
            nn.Linear(in_features=4*64, out_features=4*64),
            nn.Tanh(),
            # nn.ReLU(),
            # nn.Dropout(p=self.p),
            nn.Linear(in_features=4*64, out_features=2*64),
            nn.Tanh(),
            # # nn.ReLU(),
            nn.Dropout(p=self.p),
            nn.Linear(in_features=2*64, out_features=64),
            nn.Tanh(),
            # # nn.ReLU(),
            # # nn.Dropout(p=self.p),
            nn.Linear(in_features=64, out_features=32),
            nn.Tanh(),
            # # nn.ReLU(),
            # nn.Dropout(p=self.p),
            # nn.Linear(in_features=32, out_features=16),
            # nn.Tanh(),
            # # nn.ReLU(),
            # nn.Linear(in_features=16, out_features=8),
            # nn.Tanh(),
            # # nn.ReLU(),
            nn.Linear(in_features=32, out_features=1),
            )
    
    
    def _pd_gcn(self, gdata):
        x, edge_index = gdata.x, gdata.edge_index

        x = self.gconv1_pd(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.p, training=self.training)
        
        x = self.gconv2_pd(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.p, training=self.training)
        
        x = self.gconv3_pd(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.p, training=self.training)
        
        x = self.gconv4_pd(x, edge_index)
        # x = x.to(torch.float)
        
        x = global_mean_pool(x, gdata.batch) # to get graph embeddings
        
        x = self.glinear1_pd(x)
        x = F.dropout(x, p=self.p, training=self.training)
        
        x = self.glinear2_pd(x)
        
        return x
    
    
    def _pc_gcn_old(self, gdata):
        x, edge_index = gdata.x, gdata.edge_index

        x = self.gconv1_pc_old(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.p, training=self.training)
        
        x = self.gconv2_pc_old(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.p, training=self.training)
        
        x = self.gconv3_pc_old(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.p, training=self.training)
        
        x = self.gconv4_pc_old(x, edge_index)
        # x = x.to(torch.float)
        
        x = global_mean_pool(x, gdata.batch) # to get graph embeddings
        
        x = self.glinear1_pc_old(x)
        x = F.dropout(x, p=self.p, training=self.training)
        
        x = self.glinear2_pc_old(x)
        
        return x
    
    
    def _fc_gcn_old(self, gdata):
        x, edge_index = gdata.x, gdata.edge_index

        x = self.gconv1_fc_old(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.p, training=self.training)
        
        x = self.gconv2_fc_old(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.p, training=self.training)
        
        x = self.gconv3_fc_old(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.p, training=self.training)
        
        x = self.gconv4_fc_old(x, edge_index)
        # x = x.to(torch.float)
        
        x = global_mean_pool(x, gdata.batch) # to get graph embeddings
        
        x = self.glinear1_fc_old(x)
        x = F.dropout(x, p=self.p, training=self.training)
        
        x = self.glinear2_fc_old(x)
        
        return x
        
    
    def _pc_gcn(self, gdata):
        x, edge_index = gdata.x, gdata.edge_index

        x = self.gconv1_pc(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.p, training=self.training)
        
        x = self.gconv2_pc(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.p, training=self.training)
        
        x = self.gconv3_pc(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.p, training=self.training)
        
        x = self.gconv4_pc(x, edge_index)
        # x = x.to(torch.float)
        
        x = global_mean_pool(x, gdata.batch) # to get graph embeddings
        
        x = self.glinear1_pc(x)
        x = F.dropout(x, p=self.p, training=self.training)
        
        x = self.glinear2_pc(x)
        
        return x
    
    
    def _fc_gcn(self, gdata):
        x, edge_index = gdata.x, gdata.edge_index

        x = self.gconv1_fc(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.p, training=self.training)
        
        x = self.gconv2_fc(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.p, training=self.training)
        
        x = self.gconv3_fc(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.p, training=self.training)
        
        x = self.gconv4_fc(x, edge_index)
        # x = x.to(torch.float)
        
        x = global_mean_pool(x, gdata.batch) # to get graph embeddings
        
        x = self.glinear1_fc(x)
        x = F.dropout(x, p=self.p, training=self.training)
        
        x = self.glinear2_fc(x)
        
        return x



    def forward(self, inputs):
        image = inputs['image']
        pd_graph = inputs['pd_graph']
        pc_graph_old = inputs['pc_graph_old']
        fc_graph_old = inputs['fc_graph_old']
        pc_graph = inputs['pc_graph']
        fc_graph = inputs['fc_graph']
        
        cnn_out = self.resnet18(image)#.permute(0, 3, 1, 2)

        gcn_pd_out = self._pd_gcn(pd_graph)
        gcn_pc_out_old = self._pc_gcn_old(pc_graph_old)
        gcn_fc_out_old = self._fc_gcn_old(fc_graph_old)
        gcn_pc_out = self._pc_gcn(pc_graph)
        gcn_fc_out = self._fc_gcn(fc_graph)
        
        gcn_outs = torch.cat((gcn_pd_out, gcn_pc_out_old, gcn_fc_out_old, gcn_pc_out, gcn_fc_out), -1)
        
        gcn_out = self.glinear_before_end(gcn_outs)
        
        gcn_out = self.glinear_end(gcn_out)
        gcn_out = F.relu(gcn_out)

        x = torch.cat((cnn_out, gcn_out), -1)

                
        # print(f"before reward: {self._get_stats(x)}")
        x = self._reward_head(x)
        # print(f"after reward: {self._get_stats(x)}")
        
        x = x.flatten()
        
        return x
    

    def _get_stats(self, x):
            stats = {'mean': list(x.mean().detach().cpu().numpy().flatten())[0], 
                        'std': list(x.std().detach().cpu().numpy().flatten())[0], 
                        'min': list(x.min().detach().cpu().numpy().flatten())[0], 
                        'max': list(x.max().detach().cpu().numpy().flatten())[0]}
            return stats