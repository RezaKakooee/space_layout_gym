#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 01:06:51 2023

@author: Reza Kakooee
"""

import torch
from datetime import datetime


#%%
class PgConfig:
    def __init__(self, env_name, agent_name, phase):
        self.env_name = env_name
        self.agent_name = agent_name
        self.phase = phase
        
        self.gamma = 0.99
        self.net_lr = 1e-3
        self.hidden_dim = 64
        self.clip_coef = 0.1
        self.vf_coef = 0.5
        self.clip_critic_loss_flag = True
        self.entropy_coef = 0.01
        self.gae_lambda = 0.95
        self.anneal_lr = True
        
        self.load_model_flag = False
        self.save_model_flag = True
        
        self.checkpoint_freq = self.verbos_freq = 10
        self.log_freq = 1
        
        self.batch_size = 4096
        self.mini_batch_size = 128
        self.n_updates_per_batch = 5
        
        self.total_timesteps = 5_000_000
        self.total_episodes = 500_000
        self.total_batchs = 5_000
        self.n_interacts = 1_000
        
        self.n_updates = self.n_batches = self.total_timesteps // self.batch_size
        
        self.max_n_episodes_per_batch = 100
        self.max_timesteps_per_episode = 100
        
        self.n_goal_visits = 10
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        
        self.n_workers = self.n_envs = 1
        
        

    def get_config(self):
        return self.__dict__
    


#%%
if __name__ == '__main__':
    self = PgConfig('PPO', 'DOWL-v0', 'train')