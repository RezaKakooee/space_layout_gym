#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 14:20:19 2021

@author: Reza Kakooee
"""

# %%
import numpy as np


# %%  
class RewardCalculator:
    def __init__(self, fenv_config=None, active_room_dict=None, active_wall_status=None):
        self.fenv_config = fenv_config
        self.areas_dict = active_room_dict['room_area']
        self.delta_areas_dict = active_room_dict['delta_room_area']
        self.room_names = list(self.areas_dict.keys())
        
    def get_reward(self, time_step=None):
        self.time_step = time_step
        
        # rewards = self._get_rooms_reward()
        
        global_reward = self._get_global_reward()
        
        done = self._check_done_based_on_global_reward(global_reward)
        
        reward = global_reward/1.0
        
        return reward, done
        
        
    def _get_rooms_reward(self):
        def __reward_calculator_of_current_room(condition, abs_delta_area_of_current_wall):
            if condition:
                if self.fenv_config['reward_increment_within_interval_flag']:
                    a_rew = self.fenv_config['desired_local_reward'] -  abs_delta_area_of_current_wall
                else:
                    a_rew = self.fenv_config['desired_local_reward']
            else:
                if self.fenv_config['reward_decremental_flag']:
                    a_rew = self.fenv_config['bad_local_reward'] - self.time_step
                elif self.fenv_config['reward_shaping_flag']:
                    a_rew = self.fenv_config['bad_local_reward'] - abs_delta_area_of_current_wall
                else:
                    a_rew = self.fenv_config['bad_local_reward']
            ## This is the reward I give to the current agent
            return a_rew
        
        room_reward = {}    
        for room_name in self.room_names:
            if self.fenv_config['rewarding_method_name'] == 'Min_Threshold':
                delta_area = self.delta_areas_dict[room_name]
                condition = delta_area >= 0 # we need the delta itself for the condition, and abs(delta) for the decremental reward
                abs_delta_area = np.abs(delta_area)
                
            elif self.fenv_config['rewarding_method_name'] == 'MinMax_Threshold':
                abs_delta_area = np.abs(self.delta_areas_dict[room_name]) # here for both condition and decremental reward we need abs(delta)
                condition = abs_delta_area <= self.fenv_config['area_tolerance'] # it does matter if it is above or below the threshold, the reward is negative anyway
                
            a_rew = __reward_calculator_of_current_room(condition, abs_delta_area)
            room_reward.update({room_name: a_rew})
        return room_reward
            
    
    def _get_global_reward(self):
        ## Only for setting the done variable I calculate global_reward
        def __reward_calculator_of_laser_man(condition):
            if condition:
                global_reward = self.fenv_config['desired_global_reward']
            else:
                global_reward = self.fenv_config['bad_global_reward']
            return global_reward
    
        if self.fenv_config['rewarding_method_name'] == 'Min_Threshold':
            areas_list = list(self.areas_dict.values())
            areas_config_list = list(self.fenv_config['areas_config'].values())
            condition = np.all( np.array(areas_list) >= np.array(areas_config_list) )
            global_reward = __reward_calculator_of_laser_man(condition)
            
        elif self.fenv_config['rewarding_method_name'] == 'MinMax_Threshold':
            abs_delta_areas_list_1 = np.abs(list(self.delta_areas_dict.values())[:-1])
            condition = np.all(abs_delta_areas_list_1 <= self.fenv_config['area_tolerance'])
            global_reward = __reward_calculator_of_laser_man(condition)

        return global_reward
        

    def _check_done_based_on_global_reward(self, global_reward):
        if global_reward == self.fenv_config['desired_global_reward']:
            done = True
        else:
            done = False
        return done
 
