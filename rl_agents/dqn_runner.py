#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 13:08:40 2023

@author: Reza Kakooee
"""

import os
import time

import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter

from gym_floorplan.envs.fenv_config import LaserWallConfig
from env_maker import EnvMaker

from dqn_config import DqnConfig
from dqn import DQN
from dqn_trainer import DQNTrainer



#%%
class DANRunner:
    def __init__(self, env_config, agent_config):
        self.env_config = env_config
        self.agent_config = agent_config
        
        self.env = EnvMaker(env_config).make()
        self._set_seed()
        
        self.writer = SummaryWriter("logs/{run_name}")
        self.writer.add_text("env_config", "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in env_config.items()])))
        self.writer.add_text("agent_config", "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in agent_config.items()])))


    
    def _set_seed(self, seed=41):
        np.random.seed(seed)
        
        torch.manual_seed(seed)
        if torch.backends.cudnn.enabled:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            
        # self.env.seed(seed)
        
    
    
    def run(self):
        self.dqn_agent = DQN(self.env, self.agent_config)
        
        if self.agent_config['load_model_flag']:
            print("Loading the model ...")
            self.dqn_agent.load_model(self.agent_config['model_path'])   
            
        self.dqn_trainer = DQNTrainer(self.env, self.dqn_agent, self.agent_config, self.writer)
        self.dqn_trainer.train()
        
        if self.agent_config['save_model_flag']:
            print("Saving the model ...")
            self.dqn_agent.save_model(self.agent_config['model_path'])
        
        print("Closing the tb-writer ...")
        self.writer.close()
        
    
    
#%%
if __name__ == '__main__':
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_name = "CartPole-v0" # CartPole-v0 DOLW
    agent_name = 'DQN'
    phase = 'train'
    
    print(f'The env is: {env_name}')
    print(f'The device is: {device}')
    
    if env_name == 'DOLW-v0':
        env_config = LaserWallConfig().get_config()
    else:
        env_config = {
            'env_name': "CartPole-v0",
            'phase': 'test',
            }
    
    agent_config = DqnConfig(env_name, agent_name, phase).get_config()
    
    env_config['n_envs'] = agent_config['n_envs']
    self = DANRunner(env_config, agent_config)
    self.run()
    
    end_time = time.time()
    print(f"Elapsed time is: {end_time - start_time}")
    
    
    