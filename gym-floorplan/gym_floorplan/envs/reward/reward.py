# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 01:19:28 2021

@author: RK
"""

# %%

import copy
import numpy as np

from gym_floorplan.base_env.reward.base_reward import BaseReward

# %%

class Reward(BaseReward):
    def __init__(self, fenv_config:dict={}):
        super().__init__()
        self.fenv_config = fenv_config


    def reward(self, active_wall_name, active_wall_status, 
                     plan_data_dict, done):
        
        if self.fenv_config['reward_shaping_flag']:
            _reward = self._get_shaped_reward(active_wall_name, active_wall_status, 
                                              plan_data_dict, done)
        else:
            _reward = self._get_basic_reward(active_wall_status, done)
        
        return _reward


    def _get_shaped_reward(self, active_wall_name, active_wall_status, 
                                 plan_data_dict, done):
        if done:
            _reward = self.fenv_config['positive_done_reward']
        else:
            if active_wall_status == "accepted":
                delta_areas = plan_data_dict['delta_areas']
                room_name = f"room_{int(active_wall_name.split('_')[1])}"
                delta_area = delta_areas[room_name]
                _reward = self.fenv_config['positive_action_reward'] - abs(delta_area) 
                
            elif active_wall_status == "rejected_by_area":
                _reward = self.fenv_config['negative_wrong_area_reward']
                
            elif active_wall_status == "rejected_by_proportion":
                _reward = self.fenv_config['negative_wrong_area_reward']
                
            elif active_wall_status == "rejected_by_both_area_and_proportion":
                _reward = self.fenv_config['negative_wrong_area_reward']
                
            elif active_wall_status == "rejected_by_room":
                _reward = self.fenv_config['negative_rejected_by_room_reward']
                
            elif active_wall_status == "rejected_by_canvas":
                _reward = self.fenv_config['negative_rejected_by_canvas_reward']
            else:
                raise ValueError("Invalid active_wall_status")
            
        return _reward
     
                
    def _get_basic_reward(self, active_wall_status=None, done=False):
        if done:
            _reward = self.fenv_config['positive_done_reward']
        else:
            if active_wall_status == "accepted":
                _reward = self.fenv_config['negative_action_reward']
                
            elif active_wall_status == "wrong_area":
                _reward = self.fenv_config['negative_wrong_area_reward']
                
            elif active_wall_status == "rejected_by_room":
                _reward = self.fenv_config['negative_rejected_by_room_reward']
                
            elif active_wall_status == "rejected_by_canvas":
                _reward = self.fenv_config['negative_rejected_by_canvas_reward']
            else:
                raise ValueError("Invalid active_wall_status")
        return _reward