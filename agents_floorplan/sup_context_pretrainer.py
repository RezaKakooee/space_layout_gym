#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 12:21:10 2022

@author: RK
"""

#%%
import os
import time
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm

wandb_flag = False

if wandb_flag:
    import wandb
    wandb.init(project="housing_design_supervised")


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

from gym_floorplan.envs.observation.context_model import ContextModel

#%%
class ToTensor(object):
    def __call__(self, sample):
        image, reward = sample['image'], sample['reward']
        image = image.transpose((2, 0, 1))
        transformed_sample = {'image': torch.from_numpy(image),
                              'reward': torch.tensor(reward)}
        return transformed_sample


class Normalize(object):    
    def __call__(self, sample):
        image, reward = sample['image'].float(), sample['reward']
        normalized_sample = {'image': transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))(image),
                             'reward': reward}
        return normalized_sample


transform = transforms.Compose([
                                ToTensor(), 
                                # Normalize(),
                                ])
    
    
#%%
class CreateImageDataset(Dataset):
    def __init__(self, context_paths, transform=None):
        self.context_paths = context_paths
        self.transform = transform
    
    
    def __len__(self):
        return len(self.context_paths)
    
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        context = np.load(self.context_paths[idx], allow_pickle=True).tolist()
        
        image_old = context['plan_canvas_arr_old']
        image = context['plan_canvas_arr']

        image = np.concatenate((image_old, image), axis=2)

        reward = context['reward']

        image = np.kron(image, np.ones((4, 4, 1)))

        # image = image.astype(np.uint8)

        # image = image.resize((100, 100))
        # image = ImageOps.grayscale(image)
        # image = np.array(image)
        # image = image.astype(float)
        # image = image/255.0
        # image = np.expand_dims(image, axis=2)
        
        sample = {'image': image, 'reward': reward}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    

        
#%%
class CreateGraphDataset:
    def __init__(self, context_paths=None, batch_size=1):
        self.context_paths = context_paths
        self.batch_size = batch_size

        if context_paths:
            self.pd_data_list, self.pc_data_list_old, self.fc_data_list_old, self.pc_data_list, self.fc_data_list = self._numpy_data_to_graph_data()
    

    def get_data_list(self, gname):
        if gname == 'pd':
            return self.pd_data_list
        elif gname == 'pc':
            return self.pc_data_list
        elif gname == 'fc':
            return self.fc_data_list
        else:
            raise ValueError(f'{gname} is an invalid gname')


    def __shift_edge_by_one(self, e):
        e = np.array(e, dtype=np.int32)-1
        return e

    
    def convert_to_graph(self, context):
        
        graph_data_numpy_old = context['graph_data_numpy_old']
        graph_features_numpy_old = graph_data_numpy_old['graph_features_numpy']
        graph_edge_list_numpy_old = graph_data_numpy_old['graph_edge_list_numpy']

        graph_data_numpy = context['graph_data_numpy']
        graph_features_numpy = graph_data_numpy['graph_features_numpy']
        graph_edge_list_numpy = graph_data_numpy['graph_edge_list_numpy']

        partially_desired_graph_features_dict_numpy = graph_features_numpy['partially_desired_graph_features_dict_numpy']

        partially_current_graph_features_dict_numpy_old = graph_features_numpy_old['partially_current_graph_features_dict_numpy']
        fully_current_graph_features_dict_numpy_old = graph_features_numpy_old['fully_current_graph_features_dict_numpy']

        partially_current_graph_features_dict_numpy = graph_features_numpy['partially_current_graph_features_dict_numpy']
        fully_current_graph_features_dict_numpy = graph_features_numpy['fully_current_graph_features_dict_numpy']


        pd_edge_list = self.__shift_edge_by_one(graph_edge_list_numpy['partially_desired_graph_edge_list_numpy'])

        pc_edge_list_old = self.__shift_edge_by_one(graph_edge_list_numpy_old['partially_current_graph_edge_list_numpy'])
        fc_edge_list_old = self.__shift_edge_by_one(graph_edge_list_numpy_old['fully_current_graph_edge_list_numpy'])

        pc_edge_list = self.__shift_edge_by_one(graph_edge_list_numpy['partially_current_graph_edge_list_numpy'])
        fc_edge_list = self.__shift_edge_by_one(graph_edge_list_numpy['fully_current_graph_edge_list_numpy'])
        

        if len(pc_edge_list_old) == 0:
            pc_edge_list_old = [[0, 0]]
        if len(fc_edge_list_old) == 0:
            fc_edge_list_old = [[0, 0]]

        if len(pc_edge_list) == 0:
            pc_edge_list = [[0, 0]]
        if len(fc_edge_list) == 0:
            fc_edge_list = [[0, 0]]

                
        room_names = partially_desired_graph_features_dict_numpy.keys()
        dim_pd_features = dim_pc_features = 4
        dim_fc_features = 18
        n_nodes = n_rooms = len(room_names)

        pd_features = np.zeros((n_nodes, dim_pd_features))

        pc_features_old = np.zeros((n_nodes, dim_pc_features))
        fc_features_old = np.zeros((n_nodes, dim_fc_features))

        pc_features = np.zeros((n_nodes, dim_pc_features))
        fc_features = np.zeros((n_nodes, dim_fc_features))


        for i, room_name in enumerate(room_names):
            pd_features[i, :] = list(itertools.chain(list(partially_desired_graph_features_dict_numpy[room_name].values())))

            pc_features_old[i, :] = list(itertools.chain(list(partially_current_graph_features_dict_numpy_old[room_name].values())))
            fc_feats_old = fully_current_graph_features_dict_numpy_old[room_name].values()
            fc_feat_vec_old = []
            for fs in fc_feats_old:
                if isinstance(fs, list):
                    for f in fs:
                        fc_feat_vec_old.append(f)
                else:
                    fc_feat_vec_old.append(fs)
            fc_features_old[i, :] = fc_feat_vec_old

            pc_features[i, :] = list(itertools.chain(list(partially_current_graph_features_dict_numpy[room_name].values())))
            fc_feats = fully_current_graph_features_dict_numpy[room_name].values()
            fc_feat_vec = []
            for fs in fc_feats:
                if isinstance(fs, list):
                    for f in fs:
                        fc_feat_vec.append(f)
                else:
                    fc_feat_vec.append(fs)
            fc_features[i, :] = fc_feat_vec

            
        pd_edge_index = torch.tensor(pd_edge_list, dtype=torch.long)

        pc_edge_index_old = torch.tensor(pc_edge_list_old, dtype=torch.long)
        fc_edge_index_old = torch.tensor(fc_edge_list_old, dtype=torch.long)

        pc_edge_index = torch.tensor(pc_edge_list, dtype=torch.long)
        fc_edge_index = torch.tensor(fc_edge_list, dtype=torch.long)

    
        pd_edge_index = pd_edge_index.t().contiguous()
        
        pc_edge_index_old = pc_edge_index_old.t().contiguous()
        fc_edge_index_old = fc_edge_index_old.t().contiguous()

        pc_edge_index = pc_edge_index.t().contiguous()
        fc_edge_index = fc_edge_index.t().contiguous()

    
        xpd = torch.tensor(pd_features, dtype=torch.float)

        xpc_old = torch.tensor(pc_features_old, dtype=torch.float)
        xfc_old = torch.tensor(fc_features_old, dtype=torch.float)

        xpc = torch.tensor(pc_features, dtype=torch.float)
        xfc = torch.tensor(fc_features, dtype=torch.float)

    
        ypd = ypc_old = yfc_old = ypc = yfc = torch.tensor([], dtype=torch.float) 
    
        
        pd_data = gData(edge_index=pd_edge_index, x=xpd, y=ypd)

        pc_data_old = gData(edge_index=pc_edge_index_old, x=xpc_old, y=ypc_old)
        fc_data_old = gData(edge_index=fc_edge_index_old, x=xfc_old, y=yfc_old)

        pc_data = gData(edge_index=pc_edge_index, x=xpc, y=ypc)
        fc_data = gData(edge_index=fc_edge_index, x=xfc, y=yfc)


        assert pd_data.x.shape == pc_data.x.shape == (n_nodes, dim_pd_features)
        assert fc_data.x.shape == (n_nodes, dim_fc_features)
        
        return pd_data, pc_data_old, fc_data_old, pc_data, fc_data
        

    def _numpy_data_to_graph_data(self):
        pd_data_list = []

        pc_data_list_old = []
        fc_data_list_old = []

        pc_data_list = []
        fc_data_list = []


        for i, context_path, in enumerate(self.context_paths):
            context = np.load(context_path, allow_pickle=True).tolist()
            
            pd_data, pc_data_old, fc_data_old, pc_data, fc_data = self.convert_to_graph(context)

            pd_data_list.append(pd_data)

            pc_data_list_old.append(pc_data_old)
            fc_data_list_old.append(fc_data_old)

            pc_data_list.append(pc_data)
            fc_data_list.append(fc_data)


        return pd_data_list, pc_data_list_old, fc_data_list_old, pc_data_list, fc_data_list
    


#%%
class CreateContextDataset(Dataset):
    def __init__(self, root_dir, transform=None, batch_size=1, small_size=None):
        self.root_dir = root_dir
        self.transform = transform
        self.batch_size = batch_size
        
        self.storage_dir = os.path.join(root_dir, 'agents_floorplan/storage')
        self.context_dir = os.path.join(self.storage_dir, "offline_datasets/inference_files/context_data_dict")
        self.train_context_dir = os.path.join(self.context_dir, 'train')
        self.valid_context_dir = os.path.join(self.context_dir, 'valid')
        
        if small_size:
            self.train_context_names = os.listdir(self.train_context_dir)[:small_size]
            self.valid_context_names = os.listdir(self.valid_context_dir)[:small_size]#[:int(small_size/10)]
        else:
            self.train_context_names = os.listdir(self.train_context_dir)
            self.valid_context_names = os.listdir(self.valid_context_dir)

        np.random.seed(42)
        np.random.shuffle(self.train_context_names)
        np.random.shuffle(self.valid_context_names)
        
        self.train_context_paths = [os.path.join(self.train_context_dir, context_name) for context_name in self.train_context_names]
        self.valid_context_paths = [os.path.join(self.valid_context_dir, context_name) for context_name in self.valid_context_names]
        
        self.num_trains = len(self.train_context_names)
        self.num_valids = len(self.valid_context_names)
        self.num_context = self.num_trains + self.num_valids 
        
    
    def create_dataset(self):
        self.train_image_ds = CreateImageDataset(self.train_context_paths, transform=self.transform)
        self.valid_image_ds = CreateImageDataset(self.valid_context_paths, transform=self.transform)
        
        self.train_graph_ds = CreateGraphDataset(self.train_context_paths, self.batch_size)
        self.valid_graph_ds = CreateGraphDataset(self.valid_context_paths, self.batch_size)
        

    def make_loader(self):
        self.image_train_dloader = DataLoader(self.train_image_ds, batch_size=self.batch_size, shuffle=False)
        self.image_valid_dloader = DataLoader(self.valid_image_ds, batch_size=self.batch_size, shuffle=False)


        self.pd_graph_train_dloader = gDataLoader(self.train_graph_ds.pd_data_list, batch_size=self.batch_size, shuffle=False)

        self.pc_graph_train_dloader_old = gDataLoader(self.train_graph_ds.pc_data_list_old, batch_size=self.batch_size, shuffle=False)
        self.fc_graph_train_dloader_old = gDataLoader(self.train_graph_ds.fc_data_list_old, batch_size=self.batch_size, shuffle=False)

        self.pc_graph_train_dloader = gDataLoader(self.train_graph_ds.pc_data_list, batch_size=self.batch_size, shuffle=False)
        self.fc_graph_train_dloader = gDataLoader(self.train_graph_ds.fc_data_list, batch_size=self.batch_size, shuffle=False)

        
        self.pd_graph_valid_dloader = gDataLoader(self.valid_graph_ds.pd_data_list, batch_size=self.batch_size, shuffle=False)

        self.pc_graph_valid_dloader_old = gDataLoader(self.valid_graph_ds.pc_data_list_old, batch_size=self.batch_size, shuffle=False)
        self.fc_graph_valid_dloader_old = gDataLoader(self.valid_graph_ds.fc_data_list_old, batch_size=self.batch_size, shuffle=False)

        self.pc_graph_valid_dloader = gDataLoader(self.valid_graph_ds.pc_data_list, batch_size=self.batch_size, shuffle=False)
        self.fc_graph_valid_dloader = gDataLoader(self.valid_graph_ds.fc_data_list, batch_size=self.batch_size, shuffle=False)


#%%
class Teacher:
    def __init__(self, data_loader, model, device):
        self.data_loader = data_loader
        self.model = model
        
        self.device = device
        self.model = self.model.to(self.device)
        
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5, weight_decay=5e-4)
    
    
    def train(self):
        self.model.train()
        for i, (image_reward, pd_graph, pc_graph_old, fc_graph_old, pc_graph, fc_graph) in enumerate(zip(data_loader.image_train_dloader,
                                                                                                         data_loader.pd_graph_train_dloader,
                                                                                                         data_loader.pc_graph_train_dloader_old,
                                                                                                         data_loader.fc_graph_train_dloader_old,
                                                                                                         data_loader.pc_graph_train_dloader,
                                                                                                         data_loader.fc_graph_train_dloader)):
            
            image = image_reward['image'].float()
            reward = image_reward['reward']
            
            assert (pd_graph.y.detach().cpu().numpy() == pc_graph.y.detach().cpu().numpy()).all()
            assert (pc_graph.y.detach().cpu().numpy() == fc_graph.y.detach().cpu().numpy()).all()
        
            image = image.to(self.device)
            reward = reward.to(self.device)


            pd_graph = pd_graph.to(self.device)

            pc_graph_old = pc_graph_old.to(self.device)
            fc_graph_old = fc_graph_old.to(self.device)

            pc_graph = pc_graph.to(self.device)
            fc_graph = fc_graph.to(self.device)

            
            inputs = {'image': image, 
                      'pd_graph': pd_graph, 
                      'pc_graph_old': pc_graph_old, 
                      'fc_graph_old': fc_graph_old,
                      'pc_graph': pc_graph, 
                      'fc_graph': fc_graph}
            

            self.optimizer.zero_grad()
            
            predicted_reward = self.model(inputs)
            
            loss = self.criterion(predicted_reward, reward.float())
            loss.backward()
            
            self.optimizer.step()
            
            if wandb_flag:  
                wandb.log({"train_loss": loss})
            
        return loss
        
    def evaluate(self):
        self.model.eval()
        preds = []
        true_vals = []
    
        with torch.no_grad():
            c = 0
            for i, (image_reward, pd_graph, pc_graph_old, fc_graph_old, pc_graph, fc_graph) in enumerate(zip(data_loader.image_valid_dloader,
                                                                                                              data_loader.pd_graph_valid_dloader,
                                                                                                              data_loader.pc_graph_valid_dloader_old,
                                                                                                              data_loader.fc_graph_valid_dloader_old,
                                                                                                              data_loader.pc_graph_valid_dloader,
                                                                                                              data_loader.fc_graph_valid_dloader)):
                
            # for i, (image_reward, pd_graph, pc_graph_old, fc_graph_old, pc_graph, fc_graph) in enumerate(zip(data_loader.image_train_dloader,
            #                                                                                              data_loader.pd_graph_train_dloader,
            #                                                                                              data_loader.pc_graph_train_dloader_old,
            #                                                                                              data_loader.fc_graph_train_dloader_old,
            #                                                                                              data_loader.pc_graph_train_dloader,
            #                                                                                              data_loader.fc_graph_train_dloader)):
                image = image_reward['image'].float()
                reward = image_reward['reward']
                
                assert (pd_graph.y.detach().cpu().numpy() == pc_graph.y.detach().cpu().numpy()).all()
                assert (pc_graph.y.detach().cpu().numpy() == fc_graph.y.detach().cpu().numpy()).all()
            
                image = image.to(self.device)
                reward = reward.to(self.device)

                pd_graph = pd_graph.to(self.device)

                pc_graph_old = pc_graph_old.to(self.device)
                fc_graph_old = fc_graph_old.to(self.device)

                pc_graph = pc_graph.to(self.device)
                fc_graph = fc_graph.to(self.device)

                
                inputs = {'image': image, 
                          'pd_graph': pd_graph, 
                          'pc_graph_old': pc_graph_old, 
                          'fc_graph_old': fc_graph_old,
                          'pc_graph': pc_graph, 
                          'fc_graph': fc_graph}

       
                pred = self.model(inputs).detach().cpu().numpy()
                
                true_val = reward.detach().cpu().numpy()

                # print(f"True_Reward: {true_val}")
                # print(f"Predicted_Reward: {pred}")
                preds.append(pred)
                true_vals.append(true_val)
                c+=1
                
        preds = np.hstack(preds)
        true_vals = np.hstack(true_vals)
        mse = self.criterion(torch.tensor(true_vals), torch.tensor(preds)).detach().cpu().numpy()
        
        if wandb_flag:
            wandb.log({"val_mse": mse})
        
        return mse, preds, true_vals
        
    
    def get_embeding(self, inputs):
        self.model.eval()
        with torch.no_grad():
            return_layers = {
                '_reward_head.3': 'emb',
            }
            mid_getter = MidGetter(self.model, return_layers=return_layers, keep_output=True)
            mid_outputs, model_output = mid_getter(inputs)
        
        return mid_outputs['emb'].detach().cpu().numpy().flatten()
        
        
#%%
if __name__ == '__main__':
    start_time = time.time()
    
    
    root_dir = os.path.normpath('/home/rdbt/ETHZ/dbt_python/housing_design_making-general-env/')
    current_dir = os.getcwd()
    model_dir = os.path.join(current_dir, 'storage/offline_datasets')
    model_path = os.path.join(current_dir, 'storage/offline_datasets/model.pt')
    
    
    print('- _ - _ - _ - _ Create dataset ...')
    t0 = time.time()
    data_loader = CreateContextDataset(root_dir, transform, batch_size=16, small_size=10)
    data_loader.create_dataset()
    data_loader.make_loader()
    print(f'Time to create the dataset: {time.time() - t0}\n')
    

    print('- _ - _ - _ - _ Modeling ...')
    
    net_config = {
        '_in_channels_cnn': 4,
        'dim_pd_features': 4,
        'dim_pc_features': 4,
        'dim_fc_features': 18,
        }
    
    model = ContextModel(net_config)

    model = torch.load(model_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    teacher = Teacher(data_loader, model, device)

    for epoch in range(10, 20):
        ts = time.time()
        loss = teacher.train()
        val_mse, preds, true_vals = teacher.evaluate()
       
        print(f'\nEpoch: {epoch:03d}, Loss: {loss:8.5f} , ValMSE: {val_mse:.5f}, Duration: {time.time()-ts:8.5f}')
    
        # if (epoch+1) % 10 == 0: 
        #     rew_df = pd.DataFrame({'true_reward': true_vals, 'predicted_reward': preds})
        #     rew_df.to_csv('rew_df.csv', index=False)
        
        torch.save(teacher.model, os.path.join(model_dir, f'model_{epoch:02}.pt'))
        rew_df = pd.DataFrame({'true_reward': true_vals, 'predicted_reward': preds})
        rew_df.to_csv('rew_df.csv', index=False)

    torch.save(teacher.model, model_path)
    
    
    
    # end_time = time.time()
    # print(f"\n\nElapsed time: {end_time - start_time}")
