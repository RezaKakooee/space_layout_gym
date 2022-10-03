#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 01:50:13 2022

@author: RK
"""

#%%
import os
import umap
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from six.moves import xrange
from scipy.signal import savgol_filter

import torch
torch.cuda.empty_cache()
torch.cuda.memory_summary(device=None, abbreviated=False)

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

from torchvision.utils import make_grid
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from vqvae_create_dataset import CreateDataset, ToTensor
from vqvae_model import Model

# from torch_intermediate_layer_getter import IntermediateLayerGetter as MidGetter



#%%
def plot_loss(train_res_recon_error, train_res_perplexity, window_length=11):
    train_res_recon_error_smooth = savgol_filter(train_res_recon_error, window_length, 7)
    train_res_perplexity_smooth = savgol_filter(train_res_perplexity, window_length, 7)
    
    f = plt.figure(figsize=(16,8))
    ax = f.add_subplot(1,2,1)
    ax.plot(train_res_recon_error_smooth)
    ax.set_yscale('log')
    ax.set_title('Smoothed NMSE.')
    ax.set_xlabel('iteration')
    
    ax = f.add_subplot(1,2,2)
    ax.plot(train_res_perplexity_smooth)
    ax.set_title('Smoothed Average codebook usage (perplexity).')
    ax.set_xlabel('iteration')
    

def show(img):
    fig = plt.figure()
    npimg = img.numpy()
    fig = plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    

def plot_umap(model):
    fig = plt.figure()
    proj = umap.UMAP(n_neighbors=3,
                     min_dist=0.1,
                     metric='cosine').fit_transform(model._vq_vae._embedding.weight.data.cpu())
    
    plt.scatter(proj[:,0], proj[:,1], alpha=0.3)
    
#%%
if __name__ == '__main__':
    dataset_name = "CIFAR_"
    phase = 'train_'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    n_epochs = 10
    batch_size = 16
    num_training_updates = 15000
    
    num_hiddens = 128
    num_residual_hiddens = 32
    num_residual_layers = 2
    
    embedding_dim = 64
    num_embeddings = 512
    
    commitment_cost = 0.25
    
    decay = 0.99
    
    learning_rate = 1e-3
    
    if dataset_name == 'CIFAR':
        training_data = datasets.CIFAR10(root="data", train=True, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
                                  ]))

        validation_data = datasets.CIFAR10(root="data", train=False, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
                                  ]))
        
        data_variance = np.var(training_data.train_data / 255.0)

    
        training_loader = DataLoader(training_data, 
                             batch_size=batch_size, 
                             shuffle=True,
                             pin_memory=True)
        
        validation_loader = DataLoader(validation_data,
                               batch_size=32,
                               shuffle=True,
                               pin_memory=True)
    
    else:
        root_dir = os.path.normpath('/home/rdbt/ETHZ/dbt_python/housing_design_making-general-env/')
        storage_dir = os.path.join(root_dir, 'agents_floorplan/storage')
        # image_dir = os.path.normpath("/home/rdbt/ETHZ/dbt_python/housing_design_making-general-env/agents_floorplan/storage/offline_datasets/inference_files/")
        image_dir = os.path.join(storage_dir, "offline_datasets/inference_files/obs_grayscale")
        train_image_dir = os.path.join(image_dir, 'train')
        valid_image_dir = os.path.join(image_dir, 'valid')
        
        
        ###
        train_image_names = os.listdir(train_image_dir)
        valid_image_names = os.listdir(valid_image_dir)
        
        train_image_paths = [os.path.join(train_image_dir, image_name) for image_name in train_image_names]
        valid_image_paths = [os.path.join(valid_image_dir, image_name) for image_name in valid_image_names]
        
        num_trains = len(train_image_names)
        num_valids = len(valid_image_names)
        num_images = num_trains + num_valids 
        
        train_labels = [1]*num_trains
        valid_labels = [1]*num_valids
        
        # num_images = len(image_names)
        # num_trains = int(0.8*num_images)
        # num_valids = num_images - num_trains
        # train_image_paths = image_paths[:num_trains]
        # train_labels = [1]*num_trains
        # valid_image_paths = image_paths[num_trains:]
        # valid_labels = [1]*num_valids
    
    
        # train_dataset = datasets.ImageFolder(image_dir, transform=transforms.Compose([ToTensor()]))
        # valid_dataset = datasets.ImageFolder(valid_image_dir, transform=transforms.Compose([ToTensor()]))
    
    
        train_img_ds = CreateDataset(train_image_paths, train_labels, 
                                     transform=transforms.Compose([ToTensor()]))
            
        valid_img_ds = CreateDataset(valid_image_paths, valid_labels, 
                                     transform=transforms.Compose([ToTensor()]))
    
        data_variance = 0.06 # np.var(training_data.train_data / 255.0)

    
    
        training_loader = DataLoader(train_img_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
        
        validation_loader = DataLoader(valid_img_ds, batch_size=batch_size, shuffle=True, pin_memory=True)



    ### Modeling
    if phase == 'train':
        model = Model(num_hiddens, num_residual_layers, num_residual_hiddens,
                      num_embeddings, embedding_dim, 
                      commitment_cost, decay).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)
        
        model.train()
        
        train_res_recon_error_vec = []
        train_res_perplexity_vec = []
        for epoch in tqdm(range(n_epochs)): 
            train_res_recon_error = []
            train_res_perplexity = []
        
            # for i in xrange(num_training_updates):
            #     (data, _) = next(iter(training_loader))
            #     data = data.to(device)
            #     optimizer.zero_grad()
            #     vq_loss, data_recon, perplexity = model(data)
        
            for i, data in enumerate(training_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data['image'].float(), data['label']
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                vq_loss, data_recon, perplexity = model(inputs)
                
                recon_error = F.mse_loss(data_recon, inputs) / data_variance
                loss = recon_error + vq_loss
                loss.backward()
            
                optimizer.step()
                
                train_res_recon_error.append(recon_error.item())
                train_res_perplexity.append(perplexity.item())
            
                if (i+1) % 100 == 0:
                    print('%d iterations' % (i+1))
                    print('recon_error: %.3f' % np.mean(train_res_recon_error[-100:]))
                    print('perplexity: %.3f' % np.mean(train_res_perplexity[-100:]))
                    print()
        
        
        
            train_res_recon_error_vec.extend(train_res_recon_error)
            train_res_perplexity_vec.extend(train_res_perplexity)
            
            
            ### loss    
            plot_loss(train_res_recon_error, train_res_perplexity)
            
            
            ### View Reconstructions"""
            model.eval()
            
            if dataset_name == 'CIFAR':
                (valid_originals, _) = next(iter(validation_loader))
            else:
                valid_originals = next(iter(validation_loader))['image']
            
            valid_originals = valid_originals.to(device)
            
            vq_output_eval = model._pre_vq_conv(model._encoder(valid_originals.float()))
            _, valid_quantize, _, _ = model._vq_vae(vq_output_eval)
            valid_reconstructions = model._decoder(valid_quantize)
            
            if dataset_name == 'CIFAR':
                (train_originals, _) = next(iter(training_loader))
            else:
                train_originals = next(iter(training_loader))['image']
                
            train_originals = train_originals.to(device)
            _, train_reconstructions, _, _ = model._vq_vae(train_originals.float())
            
            show(make_grid(valid_reconstructions.cpu().data)+0.5, )
            
            show(make_grid(valid_originals.cpu()+0.5))
            
            
            ### UMAP
            plot_umap(model)
        
        plot_loss(train_res_recon_error_vec, train_res_perplexity_vec)
        
    
        torch.save(model.state_dict(), f'{storage_dir}/offline_datasets/vqvae/model.pt')
        
    else:
        model = Model(num_hiddens, num_residual_layers, num_residual_hiddens,
                      num_embeddings, embedding_dim, 
                      commitment_cost, decay).to(device)
        model.load_state_dict(torch.load(f'{storage_dir}/offline_datasets/vqvae/model.pt'))
        model.eval()
        
        
        
        # return_layers = {
        #     '_vq_vae': 'emb',
        # }
        # mid_getter = MidGetter(model, return_layers=return_layers, keep_output=True)
        # mid_outputs, model_output = mid_getter(valid_originals.float())
        
        
        
        
        # features = {}
        # def get_features(name):
        #     def hook(model, input, output):
        #         features[name] = output.detach()
        #     return hook
        # model._vq_vae._embedding.register_forward_hook(get_features('_embedding'))
        
        
        if dataset_name == 'CIFAR':
            (valid_originals, _) = next(iter(validation_loader))
        else:
            valid_originals = next(iter(validation_loader))['image']
        
        valid_originals = valid_originals.to(device)
        
        _, valid_quantize, q, en = model._vq_vae(valid_originals.float())
            
