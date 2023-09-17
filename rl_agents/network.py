#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 14:48:00 2023

@author: Reza Kakooee
"""
import math 

import torch
import torch.nn as nn
import torch.nn.functional as F

from gym_floorplan.envs.observation.plan_encoders import MetaFcEncoder, MetaCnnEncoder, MetaCnnResnetEncoder, MetaCnnResidualEncoder




#%%
class TinyLinearNet(nn.Module):
    def __init__(self, obs_dim:int, act_dim:int, hidden_dim:int=128):
        super(LinearNet, self).__init__()

        self.linear1 = nn.Linear(obs_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.feature_network = nn.Sequential(
            nn.Tanh(),
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            )
        
        self._actor_head = nn.Sequential(
            nn.Linear(hidden_dim, act_dim),
            )
        
        self._critic_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            )

        
        
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.feature_network(obs)
        logits = self._actor_head(features)
        value = self._critic_head(features)
        return logits, value
    
    
    
#%%
class LinearNet(nn.Module):
    def __init__(self, obs_dim:int, act_dim:int, hidden_dim:int=128):
        super(LinearNet, self).__init__()

        self.linear1 = nn.Linear(obs_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.feature_network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            )
        
        self._actor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim)) 
        
        self._critic_head = nn.Sequential(
            nn.Linear(hidden_dim, 1))

        
        
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.feature_network(obs)
        logits = self._actor_head(features)
        value = self._critic_head(features)
        return logits, value
    
    
    
#%%
class CnnNet(nn.Module):
    def __init__(self, obs_dim:int, act_dim:int, fenv_config):
        super(CnnNet, self).__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.fenv_config = fenv_config
        
        self.actor_hidden_dim = 256
        self.critic_hidden_dim = 256
        
        self.feature_net_inp_layer = nn.Conv2d(self.fenv_config['n_channels'], 16, kernel_size=5, stride=1, padding='same') 
        
        stride_2 = self.fenv_config['cnn_scaling_factor']
        self.feature_net_1 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=5, stride=1, padding='same'),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=stride_2, padding=1),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            )
        
        self.feature_net_2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding='same'),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=1),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            )
        
        self.feature_net_3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding='same'),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0),
            # nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten(),
            )
        
        self.feature_net_out_layer = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU()
            )
        
        
        self._actor_head = nn.Sequential(
            nn.Linear(256, self.actor_hidden_dim),
            # nn.BatchNorm1d(self.actor_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.actor_hidden_dim, act_dim)) 
        
        self._critic_head = nn.Sequential(
            nn.Linear(self.critic_hidden_dim, self.critic_hidden_dim),
            # nn.BatchNorm1d(self.critic_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.critic_hidden_dim, 1))

        
        
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.feature_net_inp_layer(obs)
        features = self.feature_net_1(features)
        features = self.feature_net_2(features)
        features = self.feature_net_3(features)
        features = self.feature_net_out_layer(features)
        logits = self._actor_head(features)
        value = self._critic_head(features)
        return logits, value
    
    

#%%
class MetaFc(nn.Module):
    def __init__(self, obs_dim, act_dim, fenv_config):
        super(MetaFc, self).__init__()
        
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.fenv_config = fenv_config
        
        self.actor_hidden_dim = 256
        self.critic_hidden_dim = 256
        
        self.encoder = MetaFcEncoder(fenv_config)
        
        self._actor_head = nn.Sequential(
            nn.Linear(self.actor_hidden_dim, self.actor_hidden_dim),
            nn.BatchNorm1d(self.actor_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.actor_hidden_dim, act_dim)) 
        
        self._critic_head = nn.Sequential(
            nn.Linear(self.critic_hidden_dim, self.critic_hidden_dim),
            nn.BatchNorm1d(self.critic_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.critic_hidden_dim, 1))
        
    
    
    def forward(self, real_obs):
        actor_critic_inp = self.encoder(real_obs)
        logits = self._actor_head(actor_critic_inp)
        value = self._critic_head(actor_critic_inp)
        value = value.reshape(-1)
        return logits, value
    
    
    

#%%
class MetaCnn(nn.Module):
    def __init__(self, obs_dim, act_dim, fenv_config, name="MetaFc"):
        super(MetaCnn, self).__init__()
        
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.fenv_config = fenv_config
        self.name = name
        
        self.actor_hidden_dim = 256
        self.critic_hidden_dim = 256
        
        self.fenv_config = fenv_config
        
        self.encoder = MetaCnnEncoder(fenv_config)        
        
        self._actor_head = nn.Sequential(
            nn.Linear(self.actor_hidden_dim, self.actor_hidden_dim),
            nn.BatchNorm1d(self.actor_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.actor_hidden_dim, self.act_dim)) 
        
        
        self._critic_head = nn.Sequential(
            nn.Linear(self.critic_hidden_dim, self.critic_hidden_dim),
            nn.BatchNorm1d(self.critic_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.critic_hidden_dim, 1))
        
   
    
    def forward(self, real_obs):
        actor_critic_inp = self.encoder(real_obs)
        logits = self._actor_head(actor_critic_inp)
        value = self._critic_head(actor_critic_inp)
        value = value.reshape(-1)
        return logits, value



#%%
class MetaCnnResnet(nn.Module):
    def __init__(self, obs_dim, act_dim, fenv_config, name="MetaFc"):
        super(MetaCnnResnet, self).__init__()
        
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.fenv_config = fenv_config
        self.name = name
        
        self.actor_hidden_dim = 256
        self.critic_hidden_dim = 256
        
        self.fenv_config = fenv_config
        
        self.encoder = MetaCnnResnetEncoder(fenv_config)        
        
        self._actor_head = nn.Sequential(
            nn.Linear(self.actor_hidden_dim, self.actor_hidden_dim),
            nn.BatchNorm1d(self.actor_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.actor_hidden_dim, self.act_dim)) 
        
        
        self._critic_head = nn.Sequential(
            nn.Linear(self.critic_hidden_dim, self.critic_hidden_dim),
            nn.BatchNorm1d(self.critic_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.critic_hidden_dim, 1))
        
   
    
    def forward(self, real_obs):
        actor_critic_inp = self.encoder(real_obs)
        logits = self._actor_head(actor_critic_inp)
        value = self._critic_head(actor_critic_inp)
        value = value.reshape(-1)
        return logits, value
    
    

#%%
class MetaCnnResidual(nn.Module):
    def __init__(self, obs_dim, act_dim, fenv_config, name="MetaFc"):
        super(MetaCnnResidual, self).__init__()
        
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.fenv_config = fenv_config
        self.name = name
        
        self.actor_hidden_dim = 256
        self.critic_hidden_dim = 256
        
        self.fenv_config = fenv_config
        
        self.encoder = MetaCnnResidualEncoder(fenv_config)        
        
        self._actor_head = nn.Sequential(
            nn.Linear(self.actor_hidden_dim, self.actor_hidden_dim),
            nn.BatchNorm1d(self.actor_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.actor_hidden_dim, self.act_dim)) 
        
        
        self._critic_head = nn.Sequential(
            nn.Linear(self.critic_hidden_dim, self.critic_hidden_dim),
            nn.BatchNorm1d(self.critic_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.critic_hidden_dim, 1))
        
   
    
    def forward(self, real_obs):
        actor_critic_inp = self.encoder(real_obs)
        logits = self._actor_head(actor_critic_inp)
        value = self._critic_head(actor_critic_inp)
        value = value.reshape(-1)
        return logits, value
    


#%%
class DqNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int=128):
        super(DqNet, self).__init__()
        
        self.linear1 = nn.Linear(obs_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, act_dim)
        
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.linear1(obs))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x
    
    
#%%
# class DqCnnNet(nn.Module):
#     def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int=128):
#         super(DqCnnNet, self).__init__()
        
#         self.conv1 = nn.Conv2d(obs_dim, hidden_dim)
#         self.conv2 = nn.Linear(hidden_dim, hidden_dim)
#         self.linear3 = nn.Linear(hidden_dim, act_dim)
        
    
#     def forward(self, obs: torch.Tensor) -> torch.Tensor:
#         x = F.relu(self.linear1(obs))
#         x = F.relu(self.linear2(x))
#         x = self.linear3(x)
#         return x
    
#%%
class NoisyLinear(nn.Module):
    def __init__(self, in_features:int, out_features:int, std_init:float=0.5):
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.Tensor(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()
        
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return F.linear(
            x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )
    
    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

        
    def reset_noise(self):
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        
        self.bias_epsilon.copy_(epsilon_out)
        
        
    @staticmethod
    def scale_noise(size:int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())
    


#%%
class NoisyDqNet(nn.Module):
    def __init__(self, obs_dim:int, act_dim:int, hidden_dim:int=128):
        super(NoisyDqNet, self).__init__()
        
        self.feature = nn.Linear(obs_dim, hidden_dim)
        self.noisy_linear1 = NoisyLinear(hidden_dim, hidden_dim)
        self.noisy_linear2 = NoisyLinear(hidden_dim, act_dim)
        
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.feature(obs))
        x = F.relu(self.noisy_linear1(x))
        x = self.noisy_linear2(x)
        return x
   
    
    def reset_noise(self):
       self.noisy_linear1.reset_noise()
       self.noisy_linear2.reset_noise()
       
    
#%%
class C51Net(nn.Module):
    def __init__(self, obs_dim: int, 
                       act_dim: int, 
                       hidden_dim: int,
                       atom_size:int, 
                       support:torch.Tensor):
        super(C51Net, self).__init__()
        
        self.act_dim = act_dim
        self.atom_size = atom_size
        self.support = support
        
        self.layers = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, act_dim * atom_size)
        )
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dist = self.dist(x)
        q = torch.sum(dist * self.support, dim=2)
        return q


    def dist(self, x):
        x = self.layers(x)
        
        q_atoms = x.view(-1, self.act_dim, self.atom_size)
        
        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)
        return dist
    
    
    
#%%
class RainbowNet(nn.Module):
    def __init__(self, obs_dim:int,
                       act_dim:int,
                       hidden_dim:int,
                       atom_size:int,
                       support:int):
        super(RainbowNet, self).__init__()
        
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.atom_size = atom_size
        self.support = support
        
        self.input_layer = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU()
            )
        
        self.feature_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
            )

        self.advantage_hidden_layer = NoisyLinear(hidden_dim, hidden_dim)
        self.advantage_layer = NoisyLinear(hidden_dim, act_dim * atom_size)
        
        self.value_hidden_layer = NoisyLinear(hidden_dim, hidden_dim)
        self.value_layer = NoisyLinear(hidden_dim, atom_size)

        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dist = self.dist(x)
        q = torch.sum(dist * self.support, dim=2)
        return q
    
    
    def dist(self, x:torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        feature = self.feature_layers(x)
        
        adv_hid = F.relu(self.advantage_hidden_layer(feature))
        val_hid = F.relu(self.value_hidden_layer(feature))
        
        advantage = self.advantage_layer(adv_hid).view(-1, self.act_dim, self.atom_size)
        value = self.value_layer(val_hid).view(-1, 1, self.atom_size)
        
        q_atom = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        dist = F.softmax(q_atom, dim=-1)
        dist = dist.clamp(min=1e-3)
        
        return dist
    
    
    def reset_noise(self):
        self.advantage_hidden_layer.reset_noise()
        self.advantage_layer.reset_noise()
        self.value_hidden_layer.reset_noise()
        self.value_layer.reset_noise()
        
    