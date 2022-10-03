#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 16:28:59 2022

@author: RK
"""
# %%
import os
import ast
import json
import numpy as np
import pandas as pd
from collections import defaultdict

import torch
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from torch_geometric.data import InMemoryDataset, download_url

from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

from gym_floorplan.envs.fenv_config import LaserWallConfig


# %%
class GraphDatasetMaker:
    def __init__(self, csv_files, batch_size=16):
        self.csv_files = csv_files
        self.batch_size = batch_size
        self.data_list = self._data_to_graph_data()
    
    
    def _str_to_list(self, x):
        # print(f"in_x: {x}")
        x = x.replace('.0', '')
        x = ast.literal_eval(x)
        # print(f"out_x: {x}")
        # for xx in x:
        #     for xxx in xx:
        #         if xxx == 0:
        #             print(f"--------------------- {x}")
        return x
        
    
    def _shift_edge_by_one(self, e):
        # print(f"~~~~~~~ in_e: {e}")
        e = np.array(e, dtype=np.int32)-1
        # print(f"+++++++ out_e: {e}")
        return e
    
    def _data_to_graph_data(self):
        data_list = []
        for i, csv_file in enumerate(self.csv_files):
            df = pd.read_csv(csv_file)
            df['desired_edge_list'] = df['desired_edge_list'].apply(lambda x: self._str_to_list(x))  
            df['desired_edge_list'] = df['desired_edge_list'].apply(lambda x: self._shift_edge_by_one(x))
            
            df['areas'] = df['areas'].apply(ast.literal_eval)  
            
            if i == 0:
                self.df = df
            else:
                self.df = pd.concat([self.df, df], axis=0)
            
            for idx, row in df.iterrows():
                # print(f"@@@@@@@@@@@ row['desired_edge_list']: {row['desired_edge_list']}")
                edge_index = torch.tensor(row['desired_edge_list'], dtype=torch.long)
                # print(f"############## edge_list: {edge_index}")
                edge_index = edge_index.t().contiguous()
                # print(f"************** edge_list: {edge_index}")
                mask_numbers = int(row['mask_numbers'])
                x = torch.tensor([[
                    a] for a in row['areas'][mask_numbers:]], dtype=torch.float)
                # print(f"len(x): {len(x)}")
                n_walls = int(row['n_walls'])+1
                # print(f"n_walls: {n_walls}")
                assert n_walls == len(x)
                y = torch.tensor([row['reward']], dtype=torch.float) 
                data_list.append(Data(edge_index=edge_index, x=x, y=y))
                
                
        # edge_index1 = torch.tensor([[0, 1, 1, 2],
        #                             [1, 0, 2, 1]], dtype=torch.long)

        # edge_index2 = torch.tensor([[0, 1, 1, 2 ,0 ,1],
        #                             [1, 0, 2, 1 ,0 ,1]], dtype=torch.long)

        # #  Nodes and characteristics of each node ： from 0 Node No 
        # X = torch.tensor([[-1], [0], [1]], dtype=torch.float)
        # #  Label of each node ： from 0 Node No - Two types of 0,1
        # Y = torch.tensor([0, 1, 0], dtype=torch.float)

        # #  establish data data 
        # data1 = Data(x=X, edge_index=edge_index1, y=Y)
        # data2 = Data(x=X, edge_index=edge_index2, y=Y)

        # #  take data Put in datalist
        # data_list = [data1,data2]
        
        return data_list
    
    
    def data_loader(self):
        loader = DataLoader(self.data_list, batch_size=self.batch_size, shuffle=False)
        return loader
    

# %%
class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data1.pt']

    def download(self):
        pass

    def process(self):
        print('============================================= processing ...')
        # Read data into huge `Data` list.
        data_list = GraphDatasetMaker(csv_files, batch_size).data_list

        if self.pre_filter is not None:
            print("------------------------------ pre_filtering ....")
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            print("------------------------------ pre_transforming ....")
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        # torch.save((data, slices), self.processed_paths[0])
        
        
# %%
class GCNReg(torch.nn.Module):
    def __init__(self, num_node_features):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, 32)
        self.conv2 = GCNConv(32, 16)
        self.linear = torch.nn.Linear(16, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        # x = x.to(torch.float)
        x = global_mean_pool(x, data.batch) # to get graph embeddings
        x = self.linear(x)

        return x


# %%
def train():
    model.train()
    for i, data in enumerate(train_dloader):
        # print(f"----------------------------------------- i : {i}")
        # print(f"data: {data}")
        # print(f"edge_index: {data.edge_index}")
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y.to(device))
        loss.backward()
        optimizer.step()
    return loss


def evaluate(loader):
        model.eval()
        preds = []
        true_vals = []
    
        with torch.no_grad():
            c = 0
            for data in valid_dloader:
                data = data.to(device)
                pred = model(data).detach().cpu().numpy()
                
                true_val = data.y.detach().cpu().numpy()
                
                preds.append(pred)
                true_vals.append(true_val)
                c+=1
                
        preds = np.hstack(preds)
        true_vals = np.hstack(true_vals)
        return criterion(torch.tensor(true_vals), torch.tensor(preds)).detach().cpu().numpy()
    
    
# %%    
if __name__ == '__main__':
    fenv_config = LaserWallConfig().get_config()
    root_dir = os.path.normpath('/home/rdbt/ETHZ/dbt_python/housing_design_making-general-env/')
    storage_dir = os.path.join(root_dir, 'agents_floorplan/storage')
    generated_plans_dir = f"{storage_dir}/generated_plans"
    files = os.listdir(generated_plans_dir)
    
    csv_files = [f"{generated_plans_dir}/{f}" for f in files if f.endswith('.csv')]
    batch_size = 1
    gmk = GraphDatasetMaker(csv_files, batch_size)
    # data = train_dloader.dataset
    
    root = os.path.join(os.getcwd(), 'storage')
    dataset = MyOwnDataset(root)
    
    dataset = dataset.shuffle()
    train_dataset = dataset[:100]
    valid_dataset = dataset[100:]

    train_dloader = DataLoader(train_dataset, batch_size=batch_size)
    valid_dloader = DataLoader(valid_dataset, batch_size=batch_size)        
    
    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
    model = GCNReg(num_node_features=1)
    model = model.to(device)
    
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    
    for epoch in range(2):
        loss = train()
        val_mse = evaluate(valid_dloader)
        # train_acc = evaluate(train_loader)
        # val_acc = evaluate(val_loader)    
        # test_acc = evaluate(test_loader)
        # print('Epoch: {:03d}, Loss: {:.5f}, Train Auc: {:.5f}, Val Auc: {:.5f}, Test Auc: {:.5f}'.
        #       format(epoch, loss, train_acc, val_acc, test_acc))
        
        print(f'Epoch: {epoch:03d}, Loss: {loss:8.5f}, ValMSE: {val_mse:.5f}') # print(f"val = {v:{WIDTH}.{PRECISION}{TYPE}}")
