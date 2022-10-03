#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 13:32:45 2021

@author: RK
"""

# %% Imports
import os

import gym
import numpy as np
from abc import ABC

import torch
from torch import nn

import ray
from ray import tune

from ray.rllib.agents import ppo, dqn
from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.rllib.agents.dqn.dqn import DQNTrainer

from ray.tune.logger import pretty_print
from ray.rllib.utils.torch_ops import FLOAT_MIN

from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC


from model import (SimpleFc, 
                   SimpleCnn, 
                   SimpleFcCnn, 
                   SimpleActionMaskFc, 
                
                   MySimpleFc, 
                   MySimpleCnn,
                   MySimpleConv, # for testing high resolution
                   MySimpleFcCnn,
                   MySimpleGnnCnn,
                   MySimpleActionMaskFc,
                )


# %% Env class
class SimpleEnv(gym.Env):
    def __init__(self, env_config={'env_name': 'simple_cnn_env'}):
        self.env_name = env_config['env_name']
        self.n_actions = 3
        self.n_states = 5
        self.action_space = gym.spaces.Discrete(self.n_actions)
        observation_space_fc = gym.spaces.Box(0.0,#*np.ones(self.n_states, dtype=float),
                                              1.0,#*np.ones(self.n_states, dtype=float),
                                              shape=(self.n_states,), 
                                              dtype=float)
        
        observation_space_cnn = gym.spaces.Box(low=0, 
                                               high=255,
                                               shape=(21, 21, 3),
                                               dtype=np.uint8)
        if self.env_name == 'simple_fc_env':
            self.observation_space = observation_space_fc
        elif self.env_name == 'simple_cnn_env':
            self.observation_space = observation_space_cnn
        elif self.env_name == 'simple_mixed_obs_env':
            self.observation_space = gym.spaces.Tuple((observation_space_fc, 
                                                observation_space_cnn))
        elif self.env_name == 'simple_action_maksing_env':
            self.observation_space = gym.spaces.Dict({
                                    'action_mask': gym.spaces.Box(low=0,
                                                                  high=1, 
                                                                  shape=(self.n_actions,), 
                                                                  dtype=float),
                                    'action_avail': gym.spaces.Box(low=0,
                                                                    high=1, 
                                                                    shape=(self.n_actions,), 
                                                                    dtype=float),
                                    'real_obs': observation_space_fc,
                })
            
        self.initial_observation = self.reset()
        

    def reset(self):
        observation_fc = np.random.rand(1, self.n_states)[0]
        observation_cnn = np.random.rand(21, 21, 3).astype(np.uint8)
        if self.env_name == 'simple_fc_env':
            observation = observation_fc
        elif self.env_name == 'simple_cnn_env':
            observation = observation_cnn
        elif self.env_name == 'simple_mixed_obs_env':
            observation = (observation_fc, observation_cnn)
        elif self.env_name == 'simple_action_maksing_env':
            observation = {
                "action_mask": np.ones(self.n_actions, dtype=np.int16), #nothing to mask in the beginning
                "action_avail": np.ones(self.n_actions, dtype=np.int16),
                "real_obs": observation_fc}
        self.timestep = 0
        return observation


    def _get_action(self, obs):
        action_mask = obs['action_mask']
        return np.random.choice(np.arange(self.n_actions), size=1, replace=False, p=np.array(action_mask)/(sum(action_mask)))[0] #np.random.randint(10)
    
    
    def _mask_action_randomly(self):
        action_mask_1 = list(np.random.choice([0, 1], self.n_actions-1)) # randomly mask some actions
        random_indx = np.random.randint(self.n_actions-1)
        action_mask = action_mask_1[:random_indx] + [1] + action_mask_1[random_indx:]
        return action_mask
        
    
    def _update_obs(self, action):
        observation_fc = np.random.rand(1, self.n_states)[0]
        observation_cnn = np.random.rand(21, 21, 3).astype(np.uint8)
        if self.env_name == 'simple_fc_env':
            observation = observation_fc
        elif self.env_name == 'simple_cnn_env':
            observation = observation_cnn
        elif self.env_name == 'simple_mixed_obs_env':
            observation = (observation_fc, observation_cnn)
        elif self.env_name == 'simple_action_maksing_env':
            observation = {
                "action_mask": self._mask_action_randomly(), #nothing to mask in the beginning
                "action_avail": np.ones(self.n_actions, dtype=np.int16),
                "real_obs": observation_fc}
        
        return observation
        
    
    def _take_action(self, action):
        next_observation = self._update_obs(action)
        done = False if self.timestep <=3 else True
        reward = 1 if done else 0
        return next_observation, reward, done
        

    def step(self, action):
        self.timestep += 1
        observation, reward, done = self._take_action(action)
        return observation, reward, done, {}        


    def seed(self, seed: int = None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
    
 # %% Env registeration   
def _register_env(env_name):
    if 'simple' in env_name:
        # from simple_env import SimpleEnv
        
        env_config = {'env_name': env_name}
        env = SimpleEnv(env_config)
        
        ray.tune.register_env(env_name, lambda config: SimpleEnv(env_config))
        
    elif env_name == 'master_env':
        from gym_floorplan.envs.fenv_config import LaserWallConfig
        from gym_floorplan.envs.master_env import MasterEnv
    
        fenv_config = LaserWallConfig().get_config()
        env = MasterEnv(fenv_config)
        
        ray.tune.register_env(env_name, lambda config: MasterEnv(fenv_config))

    
    
# %% Main
if __name__ == "__main__":
    env_name = 'master_env' # master_env, simple_fc_env simple_cnn_env simple_mixed_obs_env simple_action_maksing_env
    agent_name = 'dqn'
    learner_name = 'trainer' # trainer tunner  random
    model_sourse =  'MyCustomModel' # 'MyModel'
    net_arch = 'Fc'
    model_name = 'MySimpleActionMaskFc' #  'SimpleActionMaskFcModel' # MySimpleConv MySimpleFc
     
    action_mask_flag = True # True if 'Mask' in custom_model_name else False
    
    n_actions = 1458 if env_name == 'master_env' else 3
    
    if learner_name == 'random':
        env = SimpleEnv(env_config={'env_name': env_name})
        obs = env.reset()
        while True:
            if action_mask_flag:
                action = env._get_action(obs)
            else:
                action = env.action_space.sample()
            obs, rew, done, _ = env.step(action)
            if env.timestep == 10:
                print(f"obs:\n{obs}")
                print('Done!')
                break
    else:
        ray.init(local_mode=True)
        _register_env(env_name)
    
        _config = {
                "env": env_name,  # or "corridor" if registered above
                "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
                "num_workers": 1,  # parallelismmaster_env
                "framework": 'torch',
                "lr":1e-3,
                # "model": {"dim": 21,
                #           "conv_filters": [[16, [3, 3], 1],
                #                           [32, [5, 5], 2],
                #                           [512, [11, 11], 1]],
                #     }
            }
        
        if model_sourse == 'MyCustomModel':
            if action_mask_flag:
                ModelCatalog.register_custom_model("CustomModel", MySimpleActionMaskFc)
                # _config.update({
                #     "model": {
                #         "custom_model": "CustomModel",
                #         "vf_share_layers": True,
                #         'fcnet_hiddens': [256, n_actions]
                #     },
                # })
            elif net_arch == 'Cnn':
                ModelCatalog.register_custom_model("CustomModel", MySimpleCnn)
                _config.update({
                    "model": {
                        "custom_model": "CustomModel",
                        "vf_share_layers": True,
                    },
                })
                
            elif net_arch == 'Fc':
                ModelCatalog.register_custom_model("CustomModel", MySimpleFc)
                _config.update({
                    "model": {
                        "custom_model": "CustomModel",
                        # "vf_share_layers": True,
                        # 'fcnet_hiddens': [256, n_actions]
                    },
                })
                
        elif model_sourse == 'RllibCustom':
            _config.update({
                    "model": {
                        "dim": 21,
                        "conv_filters": [
                          [16, [3, 3], 1], # [Channel, [Kernel, Kernel], Stride]]
                          [32, [5, 5], 2],
                          [512, [11, 11], 1]
                              ],  
                        },
                    })
        
        if agent_name == 'ppo':
            config = ppo.DEFAULT_CONFIG.copy()
            config.update(_config)
            trainer = ppo.PPOTrainer(config=config, env=env_name) # this is for only train
            agent = PPOTrainer # this is for only tune
        elif agent_name == 'dqn':
            config = dqn.DEFAULT_CONFIG.copy()
            config.update(_config)
            trainer = dqn.DQNTrainer(config=config, env=env_name)  # this is for only train
            agent = DQNTrainer  # this is for only tune
        
        
        ## training/tunning
        stop = {"training_iteration": 2,
                "timesteps_total": 100,
                "episode_reward_mean": 6}
        
        simple_storage_dir = "storage/simples/simple"
        
        if learner_name == 'trainer':
            for _ in range(stop['training_iteration']):
                result = trainer.train()
                print(pretty_print(result))
                if result["timesteps_total"] >= stop['timesteps_total'] or \
                        result["episode_reward_mean"] >= stop['episode_reward_mean']:
                    break
                
        elif learner_name == 'tunner':
            results = tune.run(agent, 
                               local_dir=simple_storage_dir,
                               config=config, 
                               stop=stop,
                               checkpoint_freq=10,
                               checkpoint_at_end=True,
                               )
    
        ray.shutdown()