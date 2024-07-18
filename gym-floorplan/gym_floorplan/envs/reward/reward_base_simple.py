#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 17:17:20 2023

@author: Reza Kakooee
"""


#%%
class RewardBaseSimple:
    def __init__(self, fenv_config, active_wall_status, done, inspection_output_dict):
        self.fenv_config = fenv_config
        self.active_wall_status = active_wall_status
        self.done = done
        self.inspection_output_dict = inspection_output_dict
    


    def get_reward(self):
        if self.fenv_config['rewarding_method_name'] == 'Constrain_Satisfaction':
            reward = self._get_constraint_satisfaction_reward()
        
        elif self.fenv_config['rewarding_method_name'] == 'Binary_Reward':
            reward = self._get_binary_reward()
        
        elif self.fenv_config['rewarding_method_name'] == 'Simple_Reward':
            reward = self._get_simple_reward()
        
        else:
            raise ValueError(f"Invalid rewarding method! The current one is {self.fenv_config['rewarding_method_name']}")
        
        return reward
   
        
    
    def _get_constraint_satisfaction_reward(self):
        if not self.done:
            if self.active_wall_status == 'accepted':
                reward = 0
            else:
                reward = -1
        else:
            if self.active_wall_status == 'badly_stopped':
                reward = -1
            elif self.active_wall_status == 'well_finished':
                reward = self.fenv_config['positive_done_reward']
            else:
                raise ValueError(f"Invalid active_wall_status! The current one is {self.fenv_config['active_wall_status']}")
        return reward



    def _get_binary_reward(self):
        if not self.done:
            if self.active_wall_status == 'accepted':
                reward = 0
            else:
                reward = -1
        else:
            if self.active_wall_status == 'badly_stopped':
                reward = -1 * self.fenv_config['positive_final_reward']# -1
            elif self.active_wall_status == 'well_finished':
                n_missed_connections = self.inspection_output_dict['topology']['n_missed_connections']
                if n_missed_connections == 0:
                    reward = self.fenv_config['positive_final_reward']
                else:
                    reward = 0
            else:
                raise ValueError(f"Invalid active_wall_status! The current one is {self.fenv_config['active_wall_status']}")
            
        return reward
    
    

    def _get_simple_reward(self):
        if not self.done:
            if self.active_wall_status == 'accepted':
                reward = 0
            else:
                reward = -1
        else:
            if self.active_wall_status == 'badly_stopped':
                reward = self.fenv_config['negative_badly_stop_reward']
            
            elif self.active_wall_status == 'well_finished':
                n_missed_connections = self.inspection_output_dict['topology']['n_missed_connections']
                reward = 1 * self.fenv_config['positive_final_reward'] - 50 * n_missed_connections 
            else:
                raise ValueError(f"Invalid active_wall_status! The current one is {self.fenv_config['active_wall_status']}")
    
        return reward
