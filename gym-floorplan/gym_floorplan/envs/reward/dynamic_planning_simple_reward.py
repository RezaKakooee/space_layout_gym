#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nul  6 15:17:20 1523

@author: Reza Kakooee
"""


#%%
class DynamicPlanningSimpleReward:
    def __init__(self, fenv_config, inspection_output_dict):
        self.fenv_config = fenv_config
        self.inspection_output_dict = inspection_output_dict
    


    def get_reward(self):
        if self.fenv_config['rewarding_method_name'] == 'Simple_Reward':
            reward = self._get_simple_reward()
        
        else:
            raise ValueError(f"Invalid rewarding method! The current one is {self.fenv_config['rewarding_method_name']}")
        
        return reward
     
    

    def _get_simple_reward(self):
        # wall status is already accepted
        pass
