# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 01:04:57 2023

@author: Reza Kakooee
"""


import torch
from datetime import datetime



#%%
class DqnConfig:
    def __init__(self, env_name, agent_name, phase):
        self.env_name = env_name
        self.agent_name = agent_name
        self.phase = phase
        
        self.gamma = 0.99
        self.q_net_lr = 1e-3
        self.batch_size = 128
        self.memory_capacity = 10000
         
        self.hidden_dim = 128
         
        self.target_update_type = 'hard'
        self.hard_freq = 100
        self.soft_tau = 0.01
         
        self.use_double = True
        self.use_per = False
        self.use_dueling = False
        self.use_noisy = False
        self.use_categorical = False
        self.use_n_step = False
         
        self.network_name = 'DqNet'# 'RainbowNet' NoisyDqNet DqNet C51Net
         
        self.alpha = 0.2
        self.beta = 0.6
        self.prior_eps = 1e-6
         
        self.atom_size = 51
        self.v_min = 0.0
        self.v_max = 200.0
         
        self.n_step = 3
         
        self.epsilon_decay = 1 / 20000
        self.min_epsilon = 0.1
        self.max_epsilon = 1.0
         
        self.n_episodes = 100
        self.n_iters = 10_000
         
         
        self.load_model_flag = False
        self.save_model_flag = True
        self.checkpoint_freq = self.verbos_freq = 10
        self.log_freq = 10
        # self.model_path = 'model/model_weights.pth'
        
        # self.batch_size = 4096
        # self.mini_batch_size = 128
        # self.n_updates_per_batch = 5
        
        # self.total_timesteps = 5_000
        # self.total_episodes = 500
        # self.total_batchs = 5
        # self.n_interacts = 1
        
        # self.n_updates = self.n_batches = self.total_timesteps // self.batch_size
        
        # self.max_n_episodes_per_batch = 100
        # self.max_timesteps_per_episode = 100
        
        # self.n_goal_visits = 10
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.n_workers = self.n_envs = 2
        
        

    def get_config(self):
        return self.__dict__
    


#%%
if __name__ == '__main__':
    self = DqnConfig('DOWL-v0', 'DQN', 'train')

