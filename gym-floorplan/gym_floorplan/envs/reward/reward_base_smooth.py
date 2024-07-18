#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 17:26:02 2023

@author: Reza Kakooee
"""

import numpy as np
import gym_floorplan.envs.reward.reward_utils as reward_utils




#%%
class RewardBaseSmooth:
    def __init__(self, fenv_config, plan_data_dict, active_wall_name, active_wall_status, done, inspection_output_dict):
        self.fenv_config = fenv_config
        self.plan_data_dict = plan_data_dict
        self.active_wall_name = active_wall_name
        self.active_wall_status = active_wall_status
        self.done = done
        self.inspection_output_dict = inspection_output_dict
        
        
    
    def get_reward(self):
        reward = self._get_smooth_reward()
        return reward
        

        
    def _get_smooth_reward(self):
        if not self.done:
            if self.active_wall_status == 'accepted':
                reward = 0
            else:
                reward = -1
        else:
            if self.active_wall_status == 'badly_stopped':
                reward = self.fenv_config['negative_badly_stop_reward']
            
            elif self.active_wall_status == 'well_finished':
                n_bins = self.inspection_output_dict['topology']['n_desired_connections'] + 1
                edge_range = np.linspace(self.fenv_config['edge_diff_min'], self.fenv_config['edge_diff_max'], n_bins)
                
                reward_range = self.reward_fn(edge_range)
                
                reward = reward_range[self.inspection_output_dict['topology']['n_missed_connections']]
                reward = np.clip(reward, 1, self.fenv_config['positive_final_reward'])
    
            else:
                raise ValueError(f'Invalid active_wall_status. The current one is: {self.active_wall_status}')
    
        return reward
    
    
    
    def reward_fn(self, edge_range):
        x = edge_range
        x_start = self.fenv_config['edge_diff_min'] # 0
        x_end = self.fenv_config['edge_diff_max'] # 30
        y_start = 1
        y_end = self.fenv_config['positive_final_reward'] # 1000
        
        
        if 'Smooth_Linear_Reward' in self.fenv_config['rewarding_method_name']:
            reward_range = reward_utils.linear(x, x_start, x_end, y_start, y_end)
            
        elif 'Smooth_Quad_Reward' in self.fenv_config['rewarding_method_name']:
            reward_range = reward_utils.quadratic(x, x_start, x_end, y_start, y_end)
        
        elif 'Smooth_Log_Reward' in self.fenv_config['rewarding_method_name']:
            reward_range = reward_utils.logarithmic(x, x_start, x_end, y_start, y_end)
            
        elif 'Smooth_Exp_Reward' in self.fenv_config['rewarding_method_name']:
            reward_range = reward_utils.exponential(x, x_start, x_end, y_start, y_end)
        
        else:
            raise ValueError(f"Invalid rewarding method. The current one is: {self.fenv_config['rewarding_method_name']}")
            
        return reward_range