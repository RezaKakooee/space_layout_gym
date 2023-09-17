#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 18:13:08 2023

@author: rdbt
"""


#%%
import torch
import torch.nn as nn
from torchvision import models
# torch.autograd.set_detect_anomaly(True)




#%%
class FcFeatureEncoder(nn.Module):
    def __init__(self, feature_input_dim:int=4096, feature_hidden_dim:int=256, feature_output_dim:int=128):
        super(FcFeatureEncoder, self).__init__()
        
        self.feature_input_dim = feature_input_dim
        self.feature_hidden_dim = feature_hidden_dim
        self.feature_output_dim = feature_output_dim
       
        self.feature_net_inp_layer = nn.Sequential(
            nn.Linear(self.feature_input_dim, 2*self.feature_hidden_dim),
            nn.ReLU(),
            )
        
        self.feature_net_1 = nn.Sequential(
            nn.Linear(2*self.feature_hidden_dim, 2*self.feature_hidden_dim),
            )
            
            
        self.feature_net_2 = nn.Sequential(    
            nn.Linear(2*self.feature_hidden_dim, self.feature_hidden_dim),
            nn.ReLU(),
            )
    
        self.feature_net_3 = nn.Sequential(
            nn.Linear(self.feature_hidden_dim, self.feature_output_dim),
            )
        
        
    
    def forward(self, wall_state_vector):
        inp = self.feature_net_inp_layer(wall_state_vector)
        
        feature_net_1_out = self.feature_net_1(inp)
        feature_net_1_out += inp
        feature_net_1_out = nn.ReLU()(feature_net_1_out)
        
        feature_net_2_out = self.feature_net_2(feature_net_1_out)
        
        feature_net_3_out = self.feature_net_3(feature_net_2_out)
        feature_net_3_out += feature_net_2_out
        feature_net_out = nn.ReLU()(feature_net_3_out)
        
        # feature_net_out = self.feature_net_4(feature_net_3_out)
        return feature_net_out
    
    
    
    
#%%
class CnnFeatureEncoder(nn.Module):
    def __init__(self, feature_input_dim:int=256, feature_output_dim:int=256, cnn_scaling_factor:int=2):
        super(CnnFeatureEncoder, self).__init__()
        
        self.feature_input_dim = feature_input_dim
        self.feature_output_dim = feature_output_dim
        
        self.dropout_prob = 0.5
        
        self.feature_net_inp_layer = nn.Conv2d(self.feature_input_dim, 16, kernel_size=5, stride=1, padding=2) 
        
        # stride_2 = cnn_scaling_factor
        self.feature_net_1 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=0),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(p=self.dropout_prob),
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=0),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            )
        
        self.feature_net_2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=0),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(p=self.dropout_prob),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=0),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(p=self.dropout_prob),
            )
        
        self.feature_net_3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(p=self.dropout_prob),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0),
            # nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(p=self.dropout_prob),
            )
        
        self.flatten = nn.Sequential(
            nn.Flatten(),
            )
        
        self.feature_net_out_layer = nn.Sequential(
            nn.Linear(512, self.feature_output_dim),
            nn.ReLU(),
            nn.Dropout2d(p=self.dropout_prob),
            )
        
        
    def forward(self, plan_img):
        inp = self.feature_net_inp_layer(plan_img)
        features = self.feature_net_1(inp)
        features = self.feature_net_2(features)
        features = self.feature_net_3(features)
        features = self.flatten(features)
        feature_net_out = self.feature_net_out_layer(features)
        return feature_net_out
    
    

   
#%%
class PlanDescriptionEncoder(nn.Module):
    def __init__(self, plan_desc_one_hot_input_dim:int=30, plan_desc_continous_input_dim:int=16, plan_desc_hidden_dim:int=16, plan_desc_output_dim:int=32):
        super(PlanDescriptionEncoder, self).__init__()
        
        self.plan_desc_one_hot_input_dim = plan_desc_one_hot_input_dim
        self.plan_desc_continous_input_dim = plan_desc_continous_input_dim
        self.plan_desc_hidden_dim = plan_desc_hidden_dim
        self.plan_desc_output_dim = plan_desc_output_dim 
        
        self.plan_desc_net_one_hot_inp_layer = nn.Sequential(
            nn.Linear(plan_desc_one_hot_input_dim, plan_desc_hidden_dim),
            nn.ReLU()
            )
        self.plan_desc_net_continous_inp_layer = nn.Sequential(
            nn.Linear(plan_desc_continous_input_dim, plan_desc_hidden_dim),
            nn.ReLU()
            )
        
        self.plan_desc_net_1 = nn.Sequential(
            nn.Linear(plan_desc_hidden_dim*2, plan_desc_output_dim),
            nn.ReLU(),
            nn.Linear(plan_desc_output_dim, plan_desc_output_dim),
            )
        
        self.plan_desc_net_2 = nn.Sequential(
            nn.Linear(plan_desc_output_dim, plan_desc_output_dim),
            nn.ReLU(),
            nn.Linear(plan_desc_output_dim, plan_desc_output_dim),
            )
    
    
    
    def forward(self, plan_desc_state_vec):
        inp_oh = self.plan_desc_net_one_hot_inp_layer(plan_desc_state_vec[:, :self.plan_desc_one_hot_input_dim]) #16
        inp_co = self.plan_desc_net_continous_inp_layer(plan_desc_state_vec[:, self.plan_desc_one_hot_input_dim:]) #16
        inp = torch.cat((inp_oh, inp_co), 1) #32
        
        plan_desc_net_out_1 = self.plan_desc_net_1(inp) #32
        plan_desc_net_out_1 += inp
        plan_desc_net_out_1 = nn.ReLU()(plan_desc_net_out_1)
        
        plan_desc_net_out_2 = self.plan_desc_net_2(plan_desc_net_out_1) #32
        plan_desc_net_out_2 += plan_desc_net_out_1
        plan_desc_net_out = nn.ReLU()(plan_desc_net_out_2)
        
        return plan_desc_net_out 
    
    
    
    
#%%
class AreaEncoder(nn.Module):
    def __init__(self, area_input_dim:int=36, area_hidden_dim:int=32, area_output_dim:int=32):
        super(AreaEncoder, self).__init__()
        
        self.area_input_dim = area_input_dim
        self.area_hidden_dim = area_hidden_dim
        self.area_output_dim = area_output_dim 
        
        self.area_net_inp_layer = nn.Sequential(
            nn.Linear(area_input_dim, area_hidden_dim),
            nn.ReLU()
            )
        
        self.area_net_1 = nn.Sequential(
            nn.Linear(area_hidden_dim, area_hidden_dim),
            nn.ReLU(),
            nn.Linear(area_hidden_dim, area_hidden_dim),
            )
        
        self.area_net_2 = nn.Sequential(
            nn.Linear(area_hidden_dim, area_hidden_dim),
            nn.ReLU(),
            nn.Linear(area_hidden_dim, area_output_dim),
            )
    
    
    
    def forward(self, area_state_vec):
        inp = self.area_net_inp_layer(area_state_vec) #32
        
        area_net_out_1 = self.area_net_1(inp) #32
        area_net_out_1 += inp #32
        area_net_out_1 = nn.ReLU()(area_net_out_1)
        
        area_net_out_2 = self.area_net_2(area_net_out_1) #32
        area_net_out_2 += area_net_out_1 #32
        area_net_out = nn.ReLU()(area_net_out_2)
        
        return area_net_out 
    



#%%
class ProportionEncoder(nn.Module):
    def __init__(self, proportion_input_dim:int=18, proportion_hidden_dim:int=16, proportion_output_dim:int=16):
        super(ProportionEncoder, self).__init__()
        
        self.proportion_input_dim = proportion_input_dim
        self.proportion_hidden_dim = proportion_hidden_dim
        self.proportion_output_dim = proportion_output_dim 
        
        self.proportion_net_inp_layer = nn.Sequential(
            nn.Linear(proportion_input_dim, proportion_hidden_dim),
            nn.ReLU()
            )
        
        self.proportion_net_1 = nn.Sequential(
            nn.Linear(proportion_hidden_dim, proportion_hidden_dim),
            nn.ReLU(),
            nn.Linear(proportion_hidden_dim, proportion_hidden_dim),
            )
        
        self.proportion_net_2 = nn.Sequential(
            nn.Linear(proportion_hidden_dim, proportion_hidden_dim),
            nn.ReLU(),
            nn.Linear(proportion_hidden_dim, proportion_output_dim),
            )
    
    
    
    def forward(self, proportion_state_vec):
        inp = self.proportion_net_inp_layer(proportion_state_vec) #16
        
        proportion_net_out_1 = self.proportion_net_1(inp) #16
        proportion_net_out_1 += inp #16
        proportion_net_out_1 = nn.ReLU()(proportion_net_out_1)
        
        proportion_net_out_2 = self.proportion_net_2(proportion_net_out_1) #16
        proportion_net_out_2 += proportion_net_out_1 #16
        proportion_net_out = nn.ReLU()(proportion_net_out_2)
        
        return proportion_net_out 
    
    
    

#%%
class EdgeEncoder(nn.Module):
    def __init__(self, edge_input_dim:int=81, edge_hidden_dim:int=64, edge_output_dim:int=64):
        super(EdgeEncoder, self).__init__()
        
        self.edge_input_dim = edge_input_dim
        self.edge_hidden_dim = edge_hidden_dim
        self.edge_output_dim = edge_output_dim
        
        self.edge_net_inp_layer = nn.Sequential(
            nn.Linear(edge_input_dim, edge_hidden_dim),
            nn.ReLU()
            )
        
        self.edge_net_1 = nn.Sequential(
            nn.Linear(edge_hidden_dim, edge_hidden_dim),
            nn.ReLU(),
            nn.Linear(edge_hidden_dim, edge_hidden_dim),
            )
        
        self.edge_net_2 = nn.Sequential(
            nn.Linear(edge_hidden_dim, edge_hidden_dim),
            nn.ReLU(),
            nn.Linear(edge_hidden_dim, edge_output_dim),
            )
    
    
    
    def forward(self, edge_state_vec):
        inp = self.edge_net_inp_layer(edge_state_vec) #64
        
        edge_net_out_1 = self.edge_net_1(inp) #64
        edge_net_out_1 += inp #64
        edge_net_out_1 = nn.ReLU()(edge_net_out_1)
        
        edge_net_out_2 = self.edge_net_2(edge_net_out_1) #64
        edge_net_out_2 += inp #64
        edge_net_out = nn.ReLU()(edge_net_out_2)
        
        return edge_net_out  




#%%
class MetaEncoder(nn.Module):
    def __init__(self, meta_input_dim:int=144, meta_hidden_dim:int=128, meta_output_dim:int=128):
        super(MetaEncoder, self).__init__()
        
        self.meta_input_dim = meta_input_dim
        self.meta_hidden_dim = meta_hidden_dim
        self.meta_output_dim = meta_output_dim
        
        self.dropout_prob = 0.5
        
        self.meta_net_inp_layer = nn.Sequential(
            nn.Linear(meta_input_dim, meta_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_prob),
            )
        
        self.meta_net_1 = nn.Sequential(
            nn.Linear(meta_hidden_dim, meta_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_prob),
            nn.Linear(meta_hidden_dim, meta_hidden_dim),
            )
        
        self.meta_net_2 = nn.Sequential(
            nn.Linear(meta_hidden_dim, meta_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_prob),
            nn.Linear(meta_hidden_dim, meta_hidden_dim),
            )
        
        
        
    def forward(self, plan_desc_net_out, area_net_out, proportion_net_out, edge_net_out):
        pape = torch.cat((plan_desc_net_out, area_net_out, proportion_net_out, edge_net_out), 1)
        inp = self.meta_net_inp_layer(pape) #128
        
        meta_net_out_1 = self.meta_net_1(inp) #128
        meta_net_out_1 += inp #128
        meta_net_out_1 = nn.ReLU()(meta_net_out_1)
        
        meta_net_out_2 = self.meta_net_2(meta_net_out_1) #128
        meta_net_out_2 += meta_net_out_1 #128
        meta_net_out = nn.ReLU()(meta_net_out_2)
        
        return meta_net_out
    
    
    

#%%
class FeatureAndMetaEncoder(nn.Module):
    def __init__(self, feature_output_dim:int=256, meta_output_dim:int=128, latent_hidden_dim:int=256, latent_out_dim:int=256, fenv_config:dict={}):
        super(FeatureAndMetaEncoder, self).__init__()
        
        self.feature_output_dim = feature_output_dim
        self.meta_output_dim = meta_output_dim
        self.latent_hidden_dim = latent_hidden_dim
        self.latent_out_dim = latent_out_dim
        
        self.fenv_config = fenv_config
        
        self.dropout_prob = 0.5
        
        self.feat_mata_concat_layer = nn.Sequential(
            nn.Linear(feature_output_dim+meta_output_dim, latent_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_prob),
            )
        
        self.feat_mata_output_layer_1 = nn.Sequential(
            nn.Linear(latent_hidden_dim, latent_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_prob),
            nn.Linear(latent_hidden_dim, latent_out_dim),
            )
        
        self.feat_mata_output_layer_2 = nn.Sequential(
            nn.Linear(latent_hidden_dim, latent_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_prob),
            nn.Linear(latent_hidden_dim, latent_out_dim),
            )
        
        
        
    def forward(self, feature_net_out, meta_net_out):
        latent_in = torch.cat((feature_net_out, meta_net_out), 1) #256
        inp = self.feat_mata_concat_layer(latent_in) #256
        
        latent_out_1 = self.feat_mata_output_layer_1(inp) #256
        latent_out_1 += inp
        latent_out_1 = nn.ReLU()(latent_out_1)
        
        latent_out_2 = self.feat_mata_output_layer_2(latent_out_1) #256
        latent_out_2 += latent_out_1
        latent_out = nn.ReLU()(latent_out_2)
        
        return latent_out
    



#%%
class MetaFcEncoder(nn.Module):
    def __init__(self, fenv_config):
        super(MetaFcEncoder, self).__init__()
        
        self.fenv_config = fenv_config
        
        self.feature_input_dim = self.fenv_config['len_feature_state_vec']
        self.feature_hidden_dim = 128
        self.feature_output_dim = 128
        
        self.len_plan_state_vec = self.fenv_config['len_plan_state_vec']
        self.len_plan_area_state_vec = self.fenv_config['len_plan_area_state_vec']
        self.len_plan_area_prop_state_vec = self.fenv_config['len_plan_area_prop_state_vec']
        
        self.plan_desc_one_hot_input_dim = self.fenv_config['len_plan_state_vec_one_hot']
        self.plan_desc_continous_input_dim = self.fenv_config['len_plan_state_vec_continous']
        self.plan_desc_hidden_dim = 16
        self.plan_desc_output_dim = 32
        
        self.area_input_dim = self.fenv_config['len_area_state_vec']
        self.area_hidden_dim = 32
        self.area_output_dim = 32
        
        self.proportion_input_dim = self.fenv_config['len_proportion_state_vec']
        self.proportion_hidden_dim = 16
        self.proportion_output_dim = 16
        
        self.edge_input_dim = self.fenv_config['len_adjacency_state_vec']
        self.edge_hidden_dim = 64
        self.edge_output_dim = 64
        
        self.meta_input_dim = self.plan_desc_output_dim + self.area_output_dim + self.proportion_output_dim + self.edge_output_dim
        self.meta_hidden_dim = 128
        self.meta_output_dim = 128
        
        self.latent_hidden_dim = self.feature_output_dim + self.meta_output_dim
        self.latent_output_dim = 256
        
        self.linear_feature_net = FcFeatureEncoder(self.feature_input_dim, self.feature_hidden_dim, self.feature_output_dim)
        self.plan_desc_encoder_net = PlanDescriptionEncoder(self.plan_desc_one_hot_input_dim, self.plan_desc_continous_input_dim, self.plan_desc_hidden_dim, self.plan_desc_output_dim)
        self.area_encoder_net = AreaEncoder(self.area_input_dim, self.area_hidden_dim, self.area_output_dim)
        self.proportion_encoder_net = ProportionEncoder(self.proportion_input_dim, self.proportion_hidden_dim, self.proportion_output_dim)
        self.edge_encoder_net = EdgeEncoder(self.edge_input_dim, self.edge_hidden_dim, self.edge_output_dim)
        self.meta_encoder_net = MetaEncoder(self.meta_input_dim, self.meta_hidden_dim, self.meta_output_dim)
        self.feature_and_meta_net = FeatureAndMetaEncoder(self.feature_output_dim, self.meta_output_dim,
                                                          self.latent_hidden_dim, self.latent_output_dim,
                                                          self.fenv_config)
        

    
    def forward(self, real_obs):
        feature_net_out = self.linear_feature_net(real_obs['observation_fc'])
        plan_desc_net_out = self.plan_desc_encoder_net(real_obs['observation_meta'][:, :self.len_plan_state_vec])
        area_net_out = self.area_encoder_net(real_obs['observation_meta'][:, self.len_plan_state_vec:self.len_plan_area_state_vec])
        proportion_net_out= self.proportion_encoder_net(real_obs['observation_meta'][:, self.len_plan_area_state_vec:self.len_plan_area_prop_state_vec])
        edge_net_out = self.edge_encoder_net(real_obs['observation_meta'][:, self.len_plan_area_prop_state_vec:])
        meta_out = self.meta_encoder_net(plan_desc_net_out, area_net_out, proportion_net_out, edge_net_out)
        latent_out = self.feature_and_meta_net(feature_net_out, meta_out)
        return latent_out
    
    


#%%
class TinyMetaFcEncoder(nn.Module):
    def __init__(self, fenv_config):
        super(TinyMetaFcEncoder, self).__init__()
        
        self.fenv_config = fenv_config
        
        self.len_plan_state_vec = self.fenv_config['len_plan_state_vec']
        self.len_plan_area_state_vec = self.fenv_config['len_plan_area_state_vec']
        self.len_plan_area_prop_state_vec = self.fenv_config['len_plan_area_prop_state_vec']
        
        self.feature_input_dim = self.fenv_config['len_feature_state_vec']
        self.feature_hidden_dim = 256
        self.feature_output_dim = 128
        
        self.plan_desc_input_dim = 54
        self.plan_desc_output_dim = 54
        
        self.area_input_dim = 54
        self.area_output_dim = 54
        
        self.proportion_input_dim = 18
        self.proportion_output_dim = 18
        
        self.edge_input_dim = 81
        self.edge_output_dim = 81
        
        self.pape_input_dim = 207
        self.pape_output_dim = 207
        
        self.meta_input_dim = 207
        self.meta_output_dim = 128
        
        self.latent_hidden_dim = 256
        self.latent_output_dim = 256
        
        
        self.feature_encoder_inp_layers = nn.Sequential(
            nn.Linear(self.feature_input_dim, self.feature_hidden_dim),
            nn.Tanh(),
            nn.Linear(self.feature_hidden_dim, self.feature_output_dim),
            )

        self.feature_encoder = nn.Sequential(
            nn.Linear(self.feature_output_dim, self.feature_output_dim),
            nn.Tanh(),
            )
        
        
        self.plan_encoder = nn.Linear(self.plan_desc_input_dim, self.plan_desc_output_dim)
        
        self.area_encoder = nn.Linear(self.area_input_dim, self.area_output_dim)
        
        self.proportion_encoder = nn.Linear(self.proportion_input_dim, self.proportion_output_dim)
        
        self.edge_encoder = nn.Linear(self.edge_input_dim, self.edge_output_dim)
        
        self.pape_encoder = nn.Linear(self.pape_input_dim, self.pape_output_dim)
        
        self.meta_encoder = nn.Linear(self.meta_input_dim, self.meta_output_dim)
        
        self.feat_meta_encoder = nn.Linear(self.latent_hidden_dim, self.latent_output_dim)

        
    
    def forward(self, real_obs):
        feat_in = self.feature_encoder_inp_layers(real_obs['observation_fc'])
        feat = self.feature_encoder(feat_in)
        
        # Instead of using in-place addition, use addition with a new tensor
        feat_sum = feat + feat_in
        feat_tanh = nn.Tanh()(feat_sum)
        
        plan = self.plan_encoder(real_obs['observation_meta'][:, :self.len_plan_state_vec])
        area = self.area_encoder(real_obs['observation_meta'][:, self.len_plan_state_vec:self.len_plan_area_state_vec])
        prop = self.proportion_encoder(real_obs['observation_meta'][:, self.len_plan_area_state_vec:self.len_plan_area_prop_state_vec])
        edge = self.edge_encoder(real_obs['observation_meta'][:, self.len_plan_area_prop_state_vec:])
        
        pape_in = torch.cat((plan, area, prop, edge), 1)
        pape = self.pape_encoder(pape_in)
        
        # Similarly, avoid in-place addition here
        pape_sum = pape + pape_in
        pape_tanh = nn.Tanh()(pape_sum)
        
        meta = self.meta_encoder(pape_tanh)
        meta_tanh = nn.Tanh()(meta)
        
        fm_in = torch.cat((feat_tanh, meta_tanh), 1)
        latent = self.feat_meta_encoder(fm_in)
        
        # And here as well
        latent_sum = latent + fm_in
        out = nn.Tanh()(latent_sum)
        return out
    

#%%
class MetaCnnEncoder(nn.Module):
    def __init__(self, fenv_config):
        super(MetaCnnEncoder, self).__init__()
        
        self.fenv_config = fenv_config
        
        self.cnn_scaling_factor = self.fenv_config['cnn_scaling_factor']
        self.feature_input_dim = self.fenv_config['n_channels']
        
        self.feature_hidden_dim = 128
        self.feature_output_dim = 128
        
        self.len_plan_state_vec = self.fenv_config['len_plan_state_vec']
        self.len_plan_area_state_vec = self.fenv_config['len_plan_area_state_vec']
        self.len_plan_area_prop_state_vec = self.fenv_config['len_plan_area_prop_state_vec']
        
        self.plan_desc_one_hot_input_dim = self.fenv_config['len_plan_state_vec_one_hot']
        self.plan_desc_continous_input_dim = self.fenv_config['len_plan_state_vec_continous']
        self.plan_desc_hidden_dim = 16
        self.plan_desc_output_dim = 32
        
        self.area_input_dim = self.fenv_config['len_area_state_vec']
        self.area_hidden_dim = 32
        self.area_output_dim = 32
        
        self.proportion_input_dim = self.fenv_config['len_proportion_state_vec']
        self.proportion_hidden_dim = 16
        self.proportion_output_dim = 16
        
        self.edge_input_dim = self.fenv_config['len_adjacency_state_vec']
        self.edge_hidden_dim = 64
        self.edge_output_dim = 64
        
        self.meta_input_dim = self.plan_desc_output_dim + self.area_output_dim + self.proportion_output_dim + self.edge_output_dim
        self.meta_hidden_dim = 128
        self.meta_output_dim = 128
        
        self.latent_hidden_dim = self.feature_output_dim + self.meta_output_dim
        self.latent_output_dim = 256
        
        self.cnn_feature_net = CnnFeatureEncoder(self.feature_input_dim, self.feature_output_dim, self.cnn_scaling_factor)
        self.plan_desc_encoder_net = PlanDescriptionEncoder(self.plan_desc_one_hot_input_dim, self.plan_desc_continous_input_dim, self.plan_desc_hidden_dim, self.plan_desc_output_dim)
        self.area_encoder_net = AreaEncoder(self.area_input_dim, self.area_hidden_dim, self.area_output_dim)
        self.proportion_encoder_net = ProportionEncoder(self.proportion_input_dim, self.proportion_hidden_dim, self.proportion_output_dim)
        self.edge_encoder_net = EdgeEncoder(self.edge_input_dim, self.edge_hidden_dim, self.edge_output_dim)
        self.meta_encoder_net = MetaEncoder(self.meta_input_dim, self.meta_hidden_dim, self.meta_output_dim)
        self.feature_and_meta_net = FeatureAndMetaEncoder(self.feature_output_dim, self.meta_output_dim,
                                                          self.latent_hidden_dim, self.latent_output_dim,
                                                          self.fenv_config)

    
    def forward(self, real_obs):
        feature_net_out = self.cnn_feature_net(real_obs['observation_cnn'])
        plan_desc_net_out = self.plan_desc_encoder_net(real_obs['observation_meta'][:, :self.len_plan_state_vec])
        area_net_out = self.area_encoder_net(real_obs['observation_meta'][:, self.len_plan_state_vec:self.len_plan_area_state_vec])
        proportion_net_out= self.proportion_encoder_net(real_obs['observation_meta'][:, self.len_plan_area_state_vec:self.len_plan_area_prop_state_vec])
        edge_net_out = self.edge_encoder_net(real_obs['observation_meta'][:, self.len_plan_area_prop_state_vec:])
        meta_out = self.meta_encoder_net(plan_desc_net_out, area_net_out, proportion_net_out, edge_net_out)
        latent_out = self.feature_and_meta_net(feature_net_out, meta_out)
        return latent_out
    
    
    

#%% 
class MetaCnnResEncoder(nn.Module):
    def __init__(self, fenv_config):
        super(MetaCnnResEncoder, self).__init__()
        
        self.fenv_config = fenv_config
        
        self.cnn_scaling_factor = self.fenv_config['cnn_scaling_factor']
        self.feature_input_dim = self.fenv_config['n_channels']
        
        self.feature_hidden_dim = 128
        self.feature_output_dim = 128
        
        self.len_plan_state_vec = self.fenv_config['len_plan_state_vec']
        self.len_plan_area_state_vec = self.fenv_config['len_plan_area_state_vec']
        self.len_plan_area_prop_state_vec = self.fenv_config['len_plan_area_prop_state_vec']
        
        self.plan_desc_one_hot_input_dim = self.fenv_config['len_plan_state_vec_one_hot']
        self.plan_desc_continous_input_dim = self.fenv_config['len_plan_state_vec_continous']
        self.plan_desc_hidden_dim = 16
        self.plan_desc_output_dim = 32
        
        self.area_input_dim = self.fenv_config['len_area_state_vec']
        self.area_hidden_dim = 32
        self.area_output_dim = 32
        
        self.proportion_input_dim = self.fenv_config['len_proportion_state_vec']
        self.proportion_hidden_dim = 16
        self.proportion_output_dim = 16
        
        self.edge_input_dim = self.fenv_config['len_adjacency_state_vec']
        self.edge_hidden_dim = 64
        self.edge_output_dim = 64
        
        self.meta_input_dim = self.plan_desc_output_dim + self.area_output_dim + self.proportion_output_dim + self.edge_output_dim
        self.meta_hidden_dim = 128
        self.meta_output_dim = 128
        
        self.latent_hidden_dim = self.feature_output_dim + self.meta_output_dim
        self.latent_output_dim = 256
        
        self.resnet_feature_encoder = models.resnet50(pretrained=self.fenv_config['resnet_pretrained_flag'])
        if self.fenv_config['n_channels'] == 1: self.resnet_feature_encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        num_ftrs = self.resnet_feature_encoder.fc.in_features
        self.resnet_feature_encoder.fc = nn.Sequential(
                nn.Linear(num_ftrs, self.feature_hidden_dim),
                nn.ReLU(),
                nn.Linear(self.feature_hidden_dim, self.feature_hidden_dim))
        
        self.plan_desc_encoder_net = PlanDescriptionEncoder(self.plan_desc_one_hot_input_dim, self.plan_desc_continous_input_dim, self.plan_desc_hidden_dim, self.plan_desc_output_dim)
        self.area_encoder_net = AreaEncoder(self.area_input_dim, self.area_hidden_dim, self.area_output_dim)
        self.proportion_encoder_net = ProportionEncoder(self.proportion_input_dim, self.proportion_hidden_dim, self.proportion_output_dim)
        self.edge_encoder_net = EdgeEncoder(self.edge_input_dim, self.edge_hidden_dim, self.edge_output_dim)
        self.meta_encoder_net = MetaEncoder(self.meta_input_dim, self.meta_hidden_dim, self.meta_output_dim)
        self.feature_and_meta_net = FeatureAndMetaEncoder(self.feature_output_dim, self.meta_output_dim,
                                                          self.latent_hidden_dim, self.latent_output_dim,
                                                          self.fenv_config)

    
    def forward(self, real_obs):
        feature_net_out = self.resnet_feature_encoder(real_obs['observation_cnn'])
        plan_desc_net_out = self.plan_desc_encoder_net(real_obs['observation_meta'][:, :self.len_plan_state_vec])
        area_net_out = self.area_encoder_net(real_obs['observation_meta'][:, self.len_plan_state_vec:self.len_plan_area_state_vec])
        proportion_net_out= self.proportion_encoder_net(real_obs['observation_meta'][:, self.len_plan_area_state_vec:self.len_plan_area_prop_state_vec])
        edge_net_out = self.edge_encoder_net(real_obs['observation_meta'][:, self.len_plan_area_prop_state_vec:])
        meta_out = self.meta_encoder_net(plan_desc_net_out, area_net_out, proportion_net_out, edge_net_out)
        latent_out = self.feature_and_meta_net(feature_net_out, meta_out)
        return latent_out
    
  
    
#%%
class ResidualBlock(nn.Module):
    def __init__(self, channel_num, stride):
        super(ResidualBlock, self).__init__()
        
        self.channel_num = channel_num
        self.stride = stride
        
        self.res_block = nn.Sequential(
            nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=self.stride, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
        )
    
    
    
    def forward(self, x):
        res = self.res_block(x)
        x = x + res
        return x




#%%
class MetaCnnResidual(nn.Module):
    def __init__(self, in_channels, feature_hidden_dim):
        super(MetaCnnResidual, self).__init__()

        self.inp_layer = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            )
        
        self._residual_block_1 = ResidualBlock(32, stride=1)
        
        self.downsampler_1 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=5),
                nn.BatchNorm2d(64)
                )
        
        self._residual_block_2 = ResidualBlock(64, stride=1)
        
        self.downsampler_2 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=5),
                nn.BatchNorm2d(128)
                )
        
        self._residual_block_3 = ResidualBlock(128, stride=1)
        
        self.downsampler_3 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, stride=2),
                nn.BatchNorm2d(256)
                )
        


    def forward(self, x_):
        x = self.inp_layer(x_)
        x = self._residual_block_1(x)
        x = self.downsampler_1(x)
        x = self._residual_block_2(x)
        x = self.downsampler_2(x)
        x = self._residual_block_3(x)
        x = self.downsampler_3(x)
        return x
   
    
   
    
#%%
class MetaCnnResidualEncoder(nn.Module):
    def __init__(self, fenv_config):
        super(MetaCnnResidualEncoder, self).__init__()
        
        self.fenv_config = fenv_config
        
        self.cnn_scaling_factor = self.fenv_config['cnn_scaling_factor']
        self.feature_input_dim = self.fenv_config['n_channels']
        
        self.feature_hidden_dim = 128
        self.feature_output_dim = 128
        
        self.len_plan_state_vec = self.fenv_config['len_plan_state_vec']
        self.len_plan_area_state_vec = self.fenv_config['len_plan_area_state_vec']
        self.len_plan_area_prop_state_vec = self.fenv_config['len_plan_area_prop_state_vec']
        
        self.plan_desc_one_hot_input_dim = self.fenv_config['len_plan_state_vec_one_hot']
        self.plan_desc_continous_input_dim = self.fenv_config['len_plan_state_vec_continous']
        self.plan_desc_hidden_dim = 16
        self.plan_desc_output_dim = 32
        
        self.area_input_dim = self.fenv_config['len_area_state_vec']
        self.area_hidden_dim = 32
        self.area_output_dim = 32
        
        self.proportion_input_dim = self.fenv_config['len_proportion_state_vec']
        self.proportion_hidden_dim = 16
        self.proportion_output_dim = 16
        
        self.edge_input_dim = self.fenv_config['len_adjacency_state_vec']
        self.edge_hidden_dim = 64
        self.edge_output_dim = 64
        
        self.meta_input_dim = self.plan_desc_output_dim + self.area_output_dim + self.proportion_output_dim + self.edge_output_dim
        self.meta_hidden_dim = 128
        self.meta_output_dim = 128
        
        self.latent_hidden_dim = self.feature_output_dim + self.meta_output_dim
        self.latent_output_dim = 256
        
        
        self.feature_net_inp_layer = nn.Conv2d(self.fenv_config['n_channels'], 16, kernel_size=5, stride=1, padding=2) # nn.Linear(obs_dim, self.feature_hidden_dim)# nn.Linear(self.fenv_config['len_feature_state_vec'], 256)#nn.Linear(obs_dim, 256)
        
        self.res_encoder = MetaCnnResidual(16, self.feature_hidden_dim)

        self.feature_hidden_layer = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            )
        
        K = 2304 if self.fenv_config['cnn_scaling_factor'] == 1 else 4096
        self.feature_out_layer = nn.Sequential(
            nn.Linear(K , 128),
            nn.ReLU(inplace=True)
            )
        
        self.plan_desc_encoder_net = PlanDescriptionEncoder(self.plan_desc_one_hot_input_dim, self.plan_desc_continous_input_dim, self.plan_desc_hidden_dim, self.plan_desc_output_dim)
        self.area_encoder_net = AreaEncoder(self.area_input_dim, self.area_hidden_dim, self.area_output_dim)
        self.proportion_encoder_net = ProportionEncoder(self.proportion_input_dim, self.proportion_hidden_dim, self.proportion_output_dim)
        self.edge_encoder_net = EdgeEncoder(self.edge_input_dim, self.edge_hidden_dim, self.edge_output_dim)
        self.meta_encoder_net = MetaEncoder(self.meta_input_dim, self.meta_hidden_dim, self.meta_output_dim)
        self.feature_and_meta_net = FeatureAndMetaEncoder(self.feature_output_dim, self.meta_output_dim,
                                                          self.latent_hidden_dim, self.latent_output_dim,
                                                          self.fenv_config)
        

    
    def forward(self, real_obs):
        plan_img = real_obs['observation_cnn']
        res_inp = self.feature_net_inp_layer(plan_img)
        res_out = self.res_encoder(res_inp)
        hidden_out = self.feature_hidden_layer(res_out)
        feature_net_out = self.feature_out_layer(hidden_out)
        
        plan_desc_net_out = self.plan_desc_encoder_net(real_obs['observation_meta'][:, :self.len_plan_state_vec])
        area_net_out = self.area_encoder_net(real_obs['observation_meta'][:, self.len_plan_state_vec:self.len_plan_area_state_vec])
        proportion_net_out= self.proportion_encoder_net(real_obs['observation_meta'][:, self.len_plan_area_state_vec:self.len_plan_area_prop_state_vec])
        edge_net_out = self.edge_encoder_net(real_obs['observation_meta'][:, self.len_plan_area_prop_state_vec:])
        meta_out = self.meta_encoder_net(plan_desc_net_out, area_net_out, proportion_net_out, edge_net_out)
        latent_out = self.feature_and_meta_net(feature_net_out, meta_out)
        
        return latent_out
