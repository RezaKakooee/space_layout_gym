# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 00:53:46 2024

@author: Reza Kakooee
"""


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvmodels

from ray.rllib.models.torch.misc import SlimFC, SlimConv2d, normc_initializer


def create_activation(name='relu'):
    if name == "relu":
        return nn.ReLU()
    elif name == "elu":
        return nn.ELU()
    else:
        raise ValueError(f"Unsupported activation function: {name}")
        
def get_linear_layer(name='nn.Linear'):
    if name == 'SlimFC':
        return SlimFC
    else:
        return nn.Linear
    
def get_conv2d_layer(name='nn.Conv2d'):
    if name == 'SlimConv2d':
        return SlimConv2d
    else:
        return nn.Conv2d
    
LinearLayer = get_linear_layer()

Conv2dLayer = get_conv2d_layer()

###############################################################################
### Context Encoder ###########################################################
###############################################################################

### Plan Description ##########################################################
class PlanDescriptionEmbEncoder(nn.Module):
    def __init__(self, cfg):
        super(PlanDescriptionEmbEncoder, self).__init__()

        self.cfg = cfg
        self.hidden_dim = self.cfg['plan_desc_embedding_dim'] * 7
        
        # Embeddings
        self.masked_corners_embedding = nn.Embedding(num_embeddings=self.cfg['maximum_num_masked_rooms']+1, embedding_dim=self.cfg['plan_desc_embedding_dim']) # 4
        self.blocked_facades_embedding = nn.Embedding(num_embeddings=self.cfg['num_of_facades']+1, embedding_dim=self.cfg['plan_desc_embedding_dim']) # 4
        self.entrance_directions_embedding = nn.Embedding(num_embeddings=self.cfg['num_of_facades']+1, embedding_dim=self.cfg['plan_desc_embedding_dim']) # 4
        self.required_rooms_embedding = nn.Embedding(num_embeddings=self.cfg['maximum_num_real_rooms']+1, embedding_dim=self.cfg['plan_desc_embedding_dim']) # 9

        self.masked_corners_lw_linear = LinearLayer(self.cfg['maximum_num_masked_rooms']*2, self.cfg['plan_desc_embedding_dim'])  # 8
        self.entrance_coordinates_linear = LinearLayer(self.cfg['num_entrance_coords'], self.cfg['plan_desc_embedding_dim'])  # 8
        self.masked_corners_facade_areas_linear = LinearLayer(self.cfg['maximum_num_masked_rooms'] + self.cfg['num_of_facades'], self.cfg['plan_desc_embedding_dim']) # 8

        self.post_concat_normalizer = nn.LayerNorm(self.hidden_dim)

        # Processing path with Dropout
        self.mlp = nn.Sequential(
            LinearLayer(self.hidden_dim, self.hidden_dim),
            create_activation(self.cfg['activation_fn_name']),
            LinearLayer(self.hidden_dim, self.hidden_dim),
            nn.Dropout(p=self.cfg['plan_desc_drop']), 
        )

        self.last_linear = LinearLayer(self.hidden_dim, self.cfg['plan_desc_output_dim']) 

        self.last_normalizer = nn.LayerNorm(self.cfg['plan_desc_output_dim']) # 32


    def forward(self, desc): # 45
        desc_corners = (desc[:, :4]).long()
        desc_facades = (desc[:, 4:8]).long()
        desc_entrance = (desc[:, 8:12]).long()
        desc_rooms = (desc[:, 12:21]).long()
        desc_corners_lw = desc[:, 21:29]
        desc_entrance_coords = desc[:, 29:37]
        desc_corners_facade_areas = desc[:, 37:]

        corner_embeds = self.masked_corners_embedding(desc_corners).mean(dim=1, keepdim=False)
        facade_embeds = self.blocked_facades_embedding(desc_facades).mean(dim=1, keepdim=False)
        entrance_embeds = self.entrance_directions_embedding(desc_entrance).mean(dim=1, keepdim=False)
        room_embeds = self.required_rooms_embedding(desc_rooms).mean(dim=1, keepdim=False)

        masked_corner_lw_embeds = self.masked_corners_lw_linear(desc_corners_lw)
        entrance_coords_embeds = self.entrance_coordinates_linear(desc_entrance_coords)
        masked_corners_facade_areas_embeds = self.masked_corners_facade_areas_linear(desc_corners_facade_areas)

        embeds = torch.cat([corner_embeds, facade_embeds, entrance_embeds, room_embeds, 
                            masked_corner_lw_embeds, entrance_coords_embeds, masked_corners_facade_areas_embeds], dim=-1)

        mlp_x = self.post_concat_normalizer(embeds)
        mlp_y = self.mlp(mlp_x)
        skip = mlp_x + mlp_y
        out = self.last_normalizer(self.last_linear(skip))
        return out # 32
    


class PlanDescriptionLinEncoder(nn.Module):
    def __init__(self, cfg):
        super(PlanDescriptionLinEncoder, self).__init__()

        self.cfg = cfg
        
        # Embeddings
        self.masked_corners_embedding = LinearLayer(self.cfg['maximum_num_masked_rooms'], self.cfg['plan_desc_embedding_dim']) # 16
        self.blocked_facades_embedding = LinearLayer(self.cfg['num_of_facades'], self.cfg['plan_desc_embedding_dim']) # 16
        self.entrance_directions_embedding = LinearLayer(self.cfg['num_of_facades'], self.cfg['plan_desc_embedding_dim']) # 16
        self.required_rooms_embedding = LinearLayer(self.cfg['maximum_num_real_rooms'], self.cfg['plan_desc_embedding_dim']) # 16

        self.masked_corners_lw_linear = LinearLayer(self.cfg['maximum_num_masked_rooms']*2, self.cfg['plan_desc_embedding_dim']*2)  # 16*2
        self.entrance_coordinates_linear = LinearLayer(self.cfg['num_entrance_coords'], self.cfg['plan_desc_embedding_dim']*2)  # 16*2
        self.masked_corners_facade_areas_linear = LinearLayer(self.cfg['maximum_num_masked_rooms'] + self.cfg['num_of_facades'], self.cfg['plan_desc_embedding_dim']*2) # 16*2

        self.concatenated_vector_size = self.cfg['plan_desc_embedding_dim'] * 10
        self.post_concat_normalizer = nn.LayerNorm(self.concatenated_vector_size)

        self.pre_mlp_linear = LinearLayer(self.concatenated_vector_size, self.cfg['plan_desc_hidden_dim'])

        # Processing path with Dropout
        self.mlp = nn.Sequential(
            LinearLayer(self.cfg['plan_desc_hidden_dim'], self.cfg['plan_desc_hidden_dim']),
            create_activation(self.cfg['activation_fn_name']),
            LinearLayer(self.cfg['plan_desc_hidden_dim'], self.cfg['plan_desc_hidden_dim']),
            nn.Dropout(p=self.cfg['plan_desc_drop']), 
        )

        self.last_linear = LinearLayer(self.cfg['plan_desc_hidden_dim'], self.cfg['plan_desc_output_dim']) 

        self.last_normalizer = nn.LayerNorm(self.cfg['plan_desc_output_dim']) # 128


    def forward(self, desc): # 45
        desc_corners = (desc[:, :4])#.long()
        desc_facades = (desc[:, 4:8])#.long()
        desc_entrance = (desc[:, 8:12])#.long()
        desc_rooms = (desc[:, 12:21])#.long()
        desc_corners_lw = desc[:, 21:29]
        desc_entrance_coords = desc[:, 29:37]
        desc_corners_facade_areas = desc[:, 37:]

        corner_embeds = self.masked_corners_embedding(desc_corners)
        facade_embeds = self.blocked_facades_embedding(desc_facades)
        entrance_embeds = self.entrance_directions_embedding(desc_entrance)
        room_embeds = self.required_rooms_embedding(desc_rooms)

        masked_corner_lw_embeds = self.masked_corners_lw_linear(desc_corners_lw)
        entrance_coords_embeds = self.entrance_coordinates_linear(desc_entrance_coords)
        masked_corners_facade_areas_embeds = self.masked_corners_facade_areas_linear(desc_corners_facade_areas)

        embeds = torch.cat([corner_embeds, facade_embeds, entrance_embeds, room_embeds, 
                            masked_corner_lw_embeds, entrance_coords_embeds, masked_corners_facade_areas_embeds], dim=-1)

        embeds = self.post_concat_normalizer(embeds)
        mlp_x = self.pre_mlp_linear(embeds)
        mlp_y = self.mlp(mlp_x)
        skip = mlp_x + mlp_y
        out = self.last_normalizer(self.last_linear(skip))
        return out # 32


### Geometry Encoder ##########################################################
class AreaProportionEncoder(nn.Module):
    def __init__(self, cfg):
        super(AreaProportionEncoder, self).__init__()
        
        self.cfg = cfg

        self.area_embeddings = LinearLayer(self.cfg['area_input_dim'], self.cfg['area_embedding_dim']) # 64
        self.proportion_embeddings = LinearLayer(self.cfg['prop_input_dim'], self.cfg['prop_embedding_dim']) # 64

        self.concatenated_vector_size = self.cfg['area_embedding_dim'] + self.cfg['prop_embedding_dim'] # 64*2
        self.post_concat_normalizer = nn.LayerNorm(self.concatenated_vector_size)

        self.pre_mlp_linear = LinearLayer(self.concatenated_vector_size, self.cfg['area_prop_hidden_dim'])
        
        self.mlp = nn.Sequential(
            LinearLayer(self.cfg['area_prop_hidden_dim'], self.cfg['area_prop_hidden_dim']),
            create_activation(self.cfg['activation_fn_name']),
            LinearLayer(self.cfg['area_prop_hidden_dim'], self.cfg['area_prop_hidden_dim']),
            nn.Dropout(p=self.cfg['area_prop_drop']), 
        )
        
        self.last_linear = LinearLayer(self.cfg['area_prop_hidden_dim'], self.cfg['area_prop_output_dim'])
        self.last_normalizer = nn.LayerNorm(self.cfg['area_prop_output_dim'])
   
    
    def forward(self, area_state_vec, proportion_state_vec): # 36
        area_embeds = self.area_embeddings(area_state_vec) 
        proportion_embeds = self.proportion_embeddings(proportion_state_vec)
        embeds = torch.cat([area_embeds, proportion_embeds], dim=-1)
        embeds = self.post_concat_normalizer(embeds)
        mlp_x = self.pre_mlp_linear(embeds)
        mlp_y = self.mlp(mlp_x)
        skip = mlp_x + mlp_y
        out = self.last_normalizer(self.last_linear(skip)) 
        return out # 32


### Topology Encoder ##########################################################
class EdgeEmbEncoder(nn.Module):
    def __init__(self, cfg):
        super(EdgeEmbEncoder, self).__init__()

        self.cfg = cfg
        self.hidden_dim = self.cfg['maximum_num_real_rooms'] * self.cfg['edge_embedding_dim'] # 72
        
        # Create a separate embedding layer for each room
        self.desired_room_embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=self.cfg['max_rr_connections_per_room'] + 1, 
                         embedding_dim=self.cfg['edge_embedding_dim']) for _ in range(self.cfg['maximum_num_real_rooms'])
        ])

        self.achieved_room_embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=self.cfg['max_rr_connections_per_room'] + 1, 
                         embedding_dim=self.cfg['edge_embedding_dim']) for _ in range(self.cfg['maximum_num_real_rooms'])
        ])
        
        self.achieved_facade_embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=self.cfg['max_rf_connections_per_room'] + 1, 
                         embedding_dim=self.cfg['edge_embedding_dim']) for _ in range(self.cfg['maximum_num_real_rooms'])
        ])

        # self.desired facade []
        self.post_concat_normalizer = nn.LayerNorm(2*self.hidden_dim)
        
        self.post_concat_linear = LinearLayer(2*self.hidden_dim, self.hidden_dim)
        
        self.mlp = nn.Sequential(
            LinearLayer(self.hidden_dim, self.hidden_dim),
            create_activation(self.cfg['activation_fn_name']),
            LinearLayer(self.hidden_dim, self.hidden_dim),
            nn.Dropout(p=self.cfg['edge_drop']), 
        )
        
        self.last_linear = LinearLayer(self.hidden_dim, self.cfg['edge_output_dim'])
        self.last_normalizer = nn.LayerNorm(self.cfg['edge_output_dim'])


    def forward(self, edge_state_vec): # 171
        desired_room_connections = (edge_state_vec[:, :self.cfg['maximum_num_real_rooms']**2]).reshape(-1, self.cfg['maximum_num_real_rooms'], self.cfg['maximum_num_real_rooms'])
        achieved_room_connections = (edge_state_vec[:, self.cfg['maximum_num_real_rooms']**2:2*self.cfg['maximum_num_real_rooms']**2]).reshape(-1, self.cfg['maximum_num_real_rooms'], self.cfg['maximum_num_real_rooms'])
        # edge_state_vec.shape[-1]-self.cfg['maximum_num_real_rooms']
        achieved_room_facade_connections = (edge_state_vec[:, 2*self.cfg['maximum_num_real_rooms']**2:]).reshape(-1, self.cfg['maximum_num_real_rooms'], self.cfg['num_of_facades'])
        
        rr_aggregated_embeddings = []
        for i in range(self.cfg['maximum_num_real_rooms']):
            desired_emb = self.desired_room_embeddings[i]((desired_room_connections[:, i]).long())
            desired_agg_emb = torch.mean(desired_emb, dim=1)
        
            achieved_emb = self.achieved_room_embeddings[i]((achieved_room_connections[:, i]).long())
            achieved_agg_emb = torch.mean(achieved_emb, dim=1)
            
            room_emb = desired_agg_emb + achieved_agg_emb 
            rr_aggregated_embeddings.append(room_emb)
            
        rf_aggregated_embeddings = []
        for i in range(self.cfg['maximum_num_real_rooms']):
            fachieved_emb = self.achieved_facade_embeddings[i]((achieved_room_facade_connections[:, i]).long())
            fachieved_agg_emb = torch.mean(fachieved_emb, dim=1)
            
            rf_aggregated_embeddings.append(fachieved_agg_emb)
            
        rr_embeds = torch.cat(rr_aggregated_embeddings, dim=1).to(edge_state_vec.device) # 32*72 st: 23=batch_size, 72 = 9*8=n_rooms*n_emd
        rf_embeds = torch.cat(rf_aggregated_embeddings, dim=1).to(edge_state_vec.device)
        rrf_embeds = torch.cat([rr_embeds, rf_embeds], dim=1)
        rrf_embeds = self.post_concat_normalizer(rrf_embeds)
        mlp_x = self.post_concat_linear(rrf_embeds)
        mlp_y = self.mlp(mlp_x)
        skip = mlp_x + mlp_y
        out = self.last_normalizer(self.last_linear(skip)) 
        return out # 64



class EdgeLinEncoder(nn.Module):
    def __init__(self, cfg):
        super(EdgeLinEncoder, self).__init__()

        self.cfg = cfg
        
        # Create a separate embedding layer for each room
        self.desired_room_embeddings = nn.ModuleList([
            LinearLayer(self.cfg['max_rr_connections_per_room'], 
                      self.cfg['edge_embedding_dim']) for _ in range(self.cfg['maximum_num_real_rooms'])
        ])

        self.achieved_room_embeddings = nn.ModuleList([
            LinearLayer(self.cfg['max_rr_connections_per_room'], 
                         self.cfg['edge_embedding_dim']) for _ in range(self.cfg['maximum_num_real_rooms'])
        ])
        
        self.achieved_facade_embeddings = nn.ModuleList([
            LinearLayer(self.cfg['max_rf_connections_per_room'], 
                         self.cfg['edge_embedding_dim']) for _ in range(self.cfg['maximum_num_real_rooms'])
        ])

        self.concatenated_vector_size = self.cfg['maximum_num_real_rooms'] * self.cfg['edge_embedding_dim'] * 2  # 9*16*2
        self.post_concat_normalizer = nn.LayerNorm(self.concatenated_vector_size)
        
        self.pre_mlp_linear = LinearLayer(self.concatenated_vector_size, self.cfg['edge_hidden_dim'])
        
        self.mlp = nn.Sequential(
            LinearLayer(self.cfg['edge_hidden_dim'], self.cfg['edge_hidden_dim']),
            create_activation(self.cfg['activation_fn_name']),
            LinearLayer(self.cfg['edge_hidden_dim'], self.cfg['edge_hidden_dim']),
            nn.Dropout(p=self.cfg['edge_drop']), 
        )
        
        self.last_linear = LinearLayer(self.cfg['edge_hidden_dim'], self.cfg['edge_output_dim'])
        self.last_normalizer = nn.LayerNorm(self.cfg['edge_output_dim'])


    def forward(self, edge_state_vec): # 171
        desired_room_connections = (edge_state_vec[:, :self.cfg['maximum_num_real_rooms']**2]).reshape(-1, self.cfg['maximum_num_real_rooms'], self.cfg['maximum_num_real_rooms'])
        achieved_room_connections = (edge_state_vec[:, self.cfg['maximum_num_real_rooms']**2:2*self.cfg['maximum_num_real_rooms']**2]).reshape(-1, self.cfg['maximum_num_real_rooms'], self.cfg['maximum_num_real_rooms'])
        # edge_state_vec.shape[-1]-self.cfg['maximum_num_real_rooms']
        achieved_room_facade_connections = (edge_state_vec[:, 2*self.cfg['maximum_num_real_rooms']**2:]).reshape(-1, self.cfg['maximum_num_real_rooms'], self.cfg['num_of_facades'])
        
        rr_aggregated_embeddings = []
        for i in range(self.cfg['maximum_num_real_rooms']):
            desired_emb = self.desired_room_embeddings[i]((desired_room_connections[:, i]))
            # desired_emb = torch.mean(desired_emb, dim=1)
        
            achieved_emb = self.achieved_room_embeddings[i]((achieved_room_connections[:, i]))
            # achieved_emb = torch.mean(achieved_emb, dim=1)
            
            room_emb = desired_emb + achieved_emb 
            rr_aggregated_embeddings.append(room_emb)
            
        rf_aggregated_embeddings = []
        for i in range(self.cfg['maximum_num_real_rooms']):
            fachieved_emb = self.achieved_facade_embeddings[i]((achieved_room_facade_connections[:, i]))
            # fachieved_emb = torch.mean(fachieved_emb, dim=1)
            
            rf_aggregated_embeddings.append(fachieved_emb)
            
        rr_embeds = torch.cat(rr_aggregated_embeddings, dim=1).to(edge_state_vec.device) # 32*72 st: 23=batch_size, 72 = 9*8=n_rooms*n_emd
        rf_embeds = torch.cat(rf_aggregated_embeddings, dim=1).to(edge_state_vec.device)
        rrf_embeds = torch.cat([rr_embeds, rf_embeds], dim=1)
        rrf_embeds = self.post_concat_normalizer(rrf_embeds)
        mlp_x = self.pre_mlp_linear(rrf_embeds)
        mlp_y = self.mlp(mlp_x)
        skip = mlp_x + mlp_y
        out = self.last_normalizer(self.last_linear(skip)) 
        return out # 64

    
### Topology and Topology Encoder #############################################
class GeometryTopologyEncoder(nn.Module):
    def __init__(self, cfg):
        super(GeometryTopologyEncoder, self).__init__()
        
        self.cfg = cfg

        self.pape_linear = LinearLayer(self.cfg['meta_input_dim'], self.cfg['meta_hidden_dim'])
        self.pre_mlp_normalizer = nn.LayerNorm(self.cfg['meta_hidden_dim'])
        self.mlp = nn.Sequential(
            LinearLayer(self.cfg['meta_hidden_dim'], self.cfg['meta_hidden_dim']),
            create_activation(self.cfg['activation_fn_name']),
            LinearLayer(self.cfg['meta_hidden_dim'], self.cfg['meta_hidden_dim']),
            nn.Dropout(p=self.cfg['meta_drop']), 
        )

        self.last_linear = LinearLayer(self.cfg['meta_hidden_dim'], self.cfg['meta_output_dim'])
        self.last_normalizer = nn.LayerNorm(self.cfg['meta_output_dim'])


    def forward(self, plan_desc_encoder_feature, area_prop_encoder_feature, edge_encoder_feature): # 128
        pape = torch.cat((plan_desc_encoder_feature, area_prop_encoder_feature, edge_encoder_feature), 1) # 32, 32, 64
        embeds = self.pape_linear(pape)
        mlp_x = self.pre_mlp_normalizer(embeds)
        mlp_y = self.mlp(mlp_x)
        skip = mlp_x + mlp_y
        out = self.last_normalizer(self.last_linear(skip)) 
        return out # 64
    

### Context Encoder #############################################
class ContextEncoder(nn.Module):
    def __init__(self, cfg):
        super(ContextEncoder, self).__init__()
        self.cfg = cfg       

        self.plan_desc_encoder = PlanDescriptionEmbEncoder(self.cfg) if self.cfg['encoding_type'] == 'EMB' else PlanDescriptionLinEncoder(self.cfg)
        self.area_proportion_encoder = AreaProportionEncoder(self.cfg)
        self.edge_encoder = EdgeEmbEncoder(self.cfg) if self.cfg['encoding_type'] == 'EMB' else EdgeLinEncoder(self.cfg)
        self.geom_topo_encoder = GeometryTopologyEncoder(self.cfg)
        
    
    def forward(self, obs_meta): # 253
        plan_desc = obs_meta[:, :self.cfg['len_plan_state_vec']]
        area_state_vec = obs_meta[:, self.cfg['len_plan_state_vec']:self.cfg['len_plan_area_state_vec']]
        proportion_state_vec = obs_meta[:, self.cfg['len_plan_area_state_vec']:self.cfg['len_plan_area_prop_state_vec']]
        edge_state_vec = obs_meta[:, self.cfg['len_plan_area_prop_state_vec']:]
        
        plan_desc_encoder_feature = self.plan_desc_encoder(plan_desc.float())
        area_prop_encoder_feature = self.area_proportion_encoder(area_state_vec, proportion_state_vec)
        edge_encoder_feature = self.edge_encoder(edge_state_vec)
        geom_topo_encoder_feature = self.geom_topo_encoder(plan_desc_encoder_feature, area_prop_encoder_feature, edge_encoder_feature)

        return geom_topo_encoder_feature # 64
    

###############################################################################
### Image Encoder #############################################################
###############################################################################

### TinyCnnAcEncoder ##########################################################
class TinyCnnEncoder_dose_not_work_use_average_pooling(nn.Module):
    def __init__(self, cfg):
        super(TinyCnnEncoder, self).__init__()
        
        self.cfg = cfg
        
        in_ch = self.cfg['n_channels']
        cnn_scaling_factor = self.cfg['cnn_scaling_factor']
        
        self.in_channels = [in_ch, 64, 128]
        self.out_channels = [64, 128, 256]
        self.kernels = [5, 5, 6]
        self.strides = [2, 2, 1]
        self.paddings = [2, 2, 0]
        self.activation_fn_name = 'elu'

        self.cnn = nn.Sequential(
            Conv2dLayer(self.in_channels[0], self.out_channels[0], self.kernels[0], self.strides[0], self.paddings[0]),
            create_activation(self.activation_fn_name),
            Conv2dLayer(self.in_channels[1], self.out_channels[1], self.kernels[1], self.strides[1], self.paddings[1]),
            create_activation(self.activation_fn_name),
            Conv2dLayer(self.in_channels[2], self.out_channels[2], self.kernels[2], self.strides[2], self.paddings[2]),
            create_activation(self.activation_fn_name),
            # nn.Flatten(),
        )
        
        with torch.no_grad():
            tensor_size = (1, 23*cnn_scaling_factor, 23*cnn_scaling_factor)
            x = torch.as_tensor(np.random.rand(*tensor_size)).float()
            z = self.cnn(x)
            n_flatten = z.shape[1]
            
        self.last_layer = nn.Flatten() if n_flatten == 1 else nn.Sequential(nn.Flatten(), nn.Linear(n_flatten**2, 1), create_activation(self.activation_fn_name))
        
        with torch.no_grad():
            y = self.last_layer(z)
            assert y.shape == torch.Size([256, 1]), "output shape has to be 256*1"

    def forward(self, obs_cnn):
        if obs_cnn.shape[3] in [1, 3]: 
            obs_cnn = obs_cnn.permute(0, 3, 1, 2)
        else:
            assert obs_cnn.shape[1] in [1, 3], 'The second dimension of the observation_cnn should be 1 or 3 when feeding to the CNN encoder.'

        if obs_cnn.dtype != 'float':
            obs_cnn = obs_cnn.float()
        encoded_obs = self.cnn(obs_cnn)
        encoded_obs = self.last_layer(encoded_obs)
        return encoded_obs


### TinyCnnAcEncoder ##########################################################
class TinyCnnEncoder(nn.Module):
    def __init__(self, cfg):
        super(TinyCnnEncoder, self).__init__()
        
        self.cfg = cfg
        
        in_ch = self.cfg['n_channels']
        cnn_scaling_factor = self.cfg['cnn_scaling_factor']
        
        self.in_channels = [in_ch, 64, 128]
        self.out_channels = [64, 128, 256]
        self.kernels = [5, 5, 6]
        self.strides = [2, 2, 1]
        self.paddings = [2, 2, 0]
        self.activation_fn_name = 'elu'

        self.cnn = nn.Sequential(
            Conv2dLayer(self.in_channels[0], self.out_channels[0], self.kernels[0], self.strides[0], self.paddings[0]),
            create_activation(self.activation_fn_name),
            Conv2dLayer(self.in_channels[1], self.out_channels[1], self.kernels[1], self.strides[1], self.paddings[1]),
            create_activation(self.activation_fn_name),
            Conv2dLayer(self.in_channels[2], self.out_channels[2], self.kernels[2], self.strides[2], self.paddings[2]),
            create_activation(self.activation_fn_name),
        )
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        

    def forward(self, obs_cnn):
        if obs_cnn.shape[3] in [1, 3]: 
            obs_cnn = obs_cnn.permute(0, 3, 1, 2)
        else:
            assert obs_cnn.shape[1] in [1, 3], 'The second dimension of the observation_cnn should be 1 or 3 when feeding to the CNN encoder.'

        if obs_cnn.dtype != 'float':
            obs_cnn = obs_cnn.float()
        encoded_obs = self.cnn(obs_cnn)
        encoded_obs = self.global_avg_pool(encoded_obs)
        encoded_obs = encoded_obs.view(encoded_obs.size(0), -1)
        return encoded_obs
    
### Custom Residual Cnn Encoder ###############################################
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        # First convolutional layer
        self.conv1 = Conv2dLayer(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # Second convolutional layer
        self.conv2 = Conv2dLayer(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                Conv2dLayer(in_channels, self.expansion * out_channels, 1, stride),
                nn.BatchNorm2d(self.expansion * out_channels)
            )


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out



class MiniResidualCnnEncoder(nn.Module):
    def __init__(self, cfg):
        super(MiniResidualCnnEncoder, self).__init__()
        
        self.cfg = cfg
        block = BasicBlock
        self.in_channels = 64

        self.num_blocks = self.cfg.get('num_blocks', [2, 2, 2, 2])

        self.conv1 = Conv2dLayer(self.cfg['n_channels'], 64, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, self.num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, self.num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, self.num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 256, self.num_blocks[3], stride=2)
        self.linear = LinearLayer(256*block.expansion, self.cfg['image_output_dim'])


    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)


    def forward(self, x): # (1, 23, 23)
        out = self.bn1(self.conv1(x))
        out = F.relu(out) if self.cfg['activation_fn_name'] == 'ReLU' else F.elu(out) 
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        if x.shape[-1] == 46:
            out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = F.dropout(out, p=self.cfg['mini_res_drop'], training=self.training)
        out = self.linear(out)
        return out # 64

    def forward_print(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        print("Post conv1 and BN1:", x.min().item(), x.max().item())
        x = F.relu(x)

        x = self.layer1(x)
        print("Post layer1:", x.min().item(), x.max().item())
        x = self.layer2(x)
        print("Post layer2:", x.min().item(), x.max().item())
        x = self.layer3(x)
        print("Post layer3:", x.min().item(), x.max().item())

        if x.shape[-1] == 46:
            x = self.layer4(x)
            print("Post layer4:", x.min().item(), x.max().item())

        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = F.dropout(x, p=self.cfg['mini_res_drop'], training=self.training)
        x = self.linear(x)
        print("Final output:", x.min().item(), x.max().item())
        return x


### Official ResNet Cnn Encoder ###############################################
class ResNetXCnnEncoder(nn.Module):
    def __init__(self, cfg):
        super(ResNetXCnnEncoder, self).__init__()

        self.cfg = cfg

        self.resnet_feature_encoder = tvmodels.resnet50(pretrained=self.cfg['load_resnet_from_pretrained_weights'])
        if self.cfg['n_channels'] == 1: self.resnet_feature_encoder.conv1 = Conv2dLayer(1, 64, 7, 2, 3)

        num_ftrs = self.resnet_feature_encoder.fc.in_features
        self.resnet_feature_encoder.fc = nn.Sequential(
                LinearLayer(num_ftrs, self.cfg['feature_hidden_dim']),
                create_activation(self.cfg['activation_fn_name']),
                LinearLayer(self.cfg['feature_hidden_dim'], self.cfg['resnetx_output_dim']))
        

    def forward(self, x):
        out = self.resnet_feature_encoder(x)
        return out


###############################################################################
### FeatureAndContextEncoder ##################################################
###############################################################################

class ContextAndFeatureEncoder(nn.Module):
    def __init__(self, cfg):
        super(ContextAndFeatureEncoder, self).__init__()
        
        self.cfg = cfg
        self.feature_context_embeddings = LinearLayer(self.cfg['meta_output_dim']+self.cfg['image_output_dim'], self.cfg['latent_hidden_dim'])

        self.pre_mlp_normalizer = nn.LayerNorm(self.cfg['latent_hidden_dim'])

        self.mlp = nn.Sequential(
            LinearLayer(self.cfg['latent_hidden_dim'], self.cfg['latent_hidden_dim']),
            create_activation(self.cfg['activation_fn_name']),
            LinearLayer(self.cfg['latent_hidden_dim'], self.cfg['latent_hidden_dim']),
            nn.Dropout(p=self.cfg['img_meta_drop']), 
        )
        
        self.last_linear = LinearLayer(self.cfg['latent_hidden_dim'], self.cfg['latent_output_dim'])
        self.last_normalizer = nn.LayerNorm(self.cfg['latent_output_dim'])

        
    def forward(self, cotext_emb, feature_emd): # 64, 64
        latent_in = torch.cat((cotext_emb, feature_emd), 1) # 128
        embeds = self.feature_context_embeddings(latent_in)
        mlp_x = self.pre_mlp_normalizer(embeds)
        mlp_y = self.mlp(mlp_x)
        skip = mlp_x + mlp_y
        out = self.last_normalizer(self.last_linear(skip)) 
        return out
    
    
    
###############################################################################
### EncoderNet ################################################################
###############################################################################

class MetaCnnEncoder(nn.Module):
    def __init__(self, cfg):
        super(MetaCnnEncoder, self).__init__()
        
        self.cfg = cfg
        
        self.context_encoder = ContextEncoder(self.cfg)
        
        if self.cfg['image_encoder_type'] == 'TinyCnnEncoder':
            self.feature_encoder = TinyCnnEncoder(self.cfg)
        elif self.cfg['image_encoder_type'] == 'MiniResidualCnnEncoder':
            self.feature_encoder = MiniResidualCnnEncoder(self.cfg)
        elif self.cfg['image_encoder_type'] == 'ResNetXCnnEncoder':
            self.feature_encoder = ResNetXCnnEncoder(self.cfg)
        else:
            raise ValueError(f"Unsupported image_encoder_type: {self.cfg['image_encoder_type']}")
        
        self.context_and_feature_encoder = ContextAndFeatureEncoder(self.cfg)

    

    def forward(self, real_obs):
        obs_cnn = real_obs['observation_cnn']
        obs_meta = real_obs['observation_meta']

        if obs_cnn.shape[3] in [1, 3]: 
            obs_cnn = obs_cnn.permute(0, 3, 1, 2)
        else:
            assert obs_cnn.shape[1] in [1, 3], 'The second dimension of the observation_cnn should be 1 or 3 when feeding to the CNN encoder.'

        cotext_emb = self.context_encoder(obs_meta)   
        feature_emd = self.feature_encoder(obs_cnn)

        out = self.context_and_feature_encoder(cotext_emb, feature_emd)
        return out


class GetTrainedModelAsEncoder(nn.Module):
    def __init__(self, cfg):
        super(GetTrainedModelAsEncoder, self).__init__()
        self.cfg = cfg
        self.load_pretrained_model()
        

    def load_pretrained_model(self):
        self.pretrained_model = torch.load(self.cfg['pretrained_model_path'])
        for param in self.pretrained_model.parameters():
            param.requires_grad = False
        # for param in self.pretrained_model._actor_head.parameters():
        #     param.requires_grad = False
        # for param in self.pretrained_model._critic_head.parameters():
        #     param.requires_grad = False
        

    def get_encoder(self):
        return self.pretrained_model.encoder


    def get_embeddings(self, inp):
        with torch.no_grad():
            emb = self.get_encoder(inp)
        return emb
        
    
    def get_actor_head(self):
        return self.pretrained_model._actor_head
    

    def get_critic_head(self):
        return self.pretrained_model._critic_head
        

class MetaCnnNetPreTrainedEncoder(nn.Module):
    def __init__(self, cfg):
        nn.Module.__init__(self)
        self.cfg = cfg
        
        self.pretrained = GetTrainedModelAsEncoder(self.cfg)
        self.encoder = self.pretrained.get_encoder()
        

    def forward(self, real_obs):
        return self.encoder(real_obs)
        
    
    
# p = '/home/rdbt/ETHZ/dbt_python/housing_design/storage_nobackup/rlb_agents_storage/tunner/Prj__2024_04_10_1800__rlb__bc2ppo__2nd_paper/Scn__2024_04_10_1805__PTM__ZSLR__BC/model/modelsd.pt'   

# class MiniResidualCnnEncoder(nn.Module):
#     def __init__(self, cfg):
#         super(MiniResidualCnnEncoder, self).__init__()
        
#         self.cfg = cfg
#         self.image_net = ResNetXCnnEncoder(self.cfg) if self.cfg['image_encoder_type'] == 'ResNetXCnnEncoder' else MiniResidualCnnEncoder(self.cfg)
            

#     def forward(self, real_obs):
#         obs_cnn = real_obs

#         if obs_cnn.shape[3] in [1, 3]: 
#             obs_cnn = obs_cnn.permute(0, 3, 1, 2)
#         else:
#             assert obs_cnn.shape[1] in [1, 3], 'The second dimension of the observation_cnn should be 1 or 3 when feeding to the CNN encoder.'

#         image_emd = self.image_net(obs_cnn)

#         return image_emd