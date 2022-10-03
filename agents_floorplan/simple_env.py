#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 14:35:43 2022

@author: RK
"""

# %%
import gym
import numpy as np

# %%
class SimpleEnv(gym.Env):
    def __init__(self, env_config={'env_name': 'simple_cnn_env'}):
        self.env_name = env_config['env_name']
        self.action_space = gym.spaces.Discrete(2)
        observation_space_fc = gym.spaces.Box(0.0*np.ones(2, dtype=float),
                                              1.0*np.ones(2, dtype=float),
                                              shape=(2,), 
                                              dtype=np.float)
        
        observation_space_cnn = gym.spaces.Box(low=0, 
                                               high=255,
                                               shape=(21, 21, 3),
                                               dtype=np.uint8)
        if self.env_name == 'simple_fc_env':
            self.observation_space = observation_space_fc
        elif self.env_name == 'simple_cnn_env':
            self.observation_space = observation_space_cnn
        elif self.env_name == 'simple_fccnn_env':
            self.observation_space = gym.spaces.Tuple((observation_space_fc, 
                                                observation_space_cnn))
        self.initial_observation = self.reset()
        

    def reset(self):
        if self.env_name == 'simple_fc_env':
            observation = np.array([0, 1])
        elif self.env_name == 'simple_cnn_env':
            observation = np.zeros((21, 21, 3), dtype=np.uint8)
        elif self.env_name == 'simple_fccnn_env':
            observation = (np.array([0, 1]), np.zeros((21, 21, 3), dtype=np.uint8))
        self.timestep = 0
        return observation


    def _get_action(self, obs):
        return np.random.randint(self.action_space.n)
    
    
    def _take_action(self, action):
        next_observation = self.initial_observation
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
