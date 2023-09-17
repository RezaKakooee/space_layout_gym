# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 01:18:33 2021

@author: Reza Kakooee
"""

#%%
import gymnasium as gym

from gym_floorplan.base_env.action.base_actoin import BaseAction

#%%
class Action(BaseAction):
    def __init__(self, fenv_config:dict={}):
        super().__init__()
        self.fenv_config = fenv_config
        
    @property
    def action_space(self):
        self._action_space = gym.spaces.Discrete(self.fenv_config['n_actions'])
        return self._action_space
        