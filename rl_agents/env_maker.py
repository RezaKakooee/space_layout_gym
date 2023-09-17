#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 13:01:10 2023

@author: Reza Kakooee
"""

import gymnasium as gym 



#%%
class EnvMaker:
    def __init__(self, fenv_config):
        self.env_config = fenv_config
        self.env_name = fenv_config['env_name']
        
        
        
    def make(self):
        if self.env_name == 'DOLW-v0':
            if self.env_config['phase'] == 'train':
                env = gym.vector.make("DOLW-v0", num_envs=self.env_config['n_envs'], env_config=self.env_config)
            else:
                env = gym.make("DOLW-v0", env_config=self.env_config)
        else:
            if self.env_config['phase'] == 'train':
                env = gym.vector.make("CartPole-v0", num_envs=self.env_config['n_envs'])
            else:
                env = gym.make("CartPole-v0")
                
            env.env_name = 'gym'
        return env