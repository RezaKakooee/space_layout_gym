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



    def reward(self, plan_data_dict, 
                     active_wall_name, active_wall_status, 
                     time_step, done):
        
        if self.fenv_config['basic_reward_flag']:
            _reward = self._get_basic_reward(active_wall_status, 
                                             done)
            
        elif self.fenv_config['only_final_reward_flag']:
            _reward = self._get_final_reward(plan_data_dict, 
                                             active_wall_name, active_wall_status, 
                                             time_step, done)
            
        elif self.fenv_config['only_final_reward_simple_flag']:
            _reward = self._get_final_reward_simple(plan_data_dict, 
                                             active_wall_name, active_wall_status, 
                                             time_step, done)
            
        elif self.fenv_config['reward_shaping_flag']:
            _reward = self._get_shaped_reward(plan_data_dict,
                                              active_wall_name, active_wall_status, 
                                              done)
            
        elif self.fenv_config['binary_reward_flag']:
            _reward = self._get_binary_reward(active_wall_status, done)
        
        else:
            raise ValueError('Invalid rewarding method!')
        
        return _reward

    

    def _get_binary_reward(self, active_wall_status, done):
        if done:
            if active_wall_status == 'badly_stopped':
                _reward = -100
            elif active_wall_status == 'well_finished':
                _reward = 100
            else:
                raise ValueError('Invalid active_wall_status')
            
        else:
            _reward = -1
        
        return _reward
        


    def _get_final_reward(self, plan_data_dict, 
                                active_wall_name, active_wall_status, 
                                time_step, done):
        
        if done:
            
            sum_areas_diff = 0
            if self.fenv_config['is_area_considered'] and self.fenv_config['area_diff_in_reward_flag']: 
                sum_areas_diff = sum([abs(delta) for delta in plan_data_dict['delta_areas'].values()])
                mean_areas_diff = np.mean([abs(delta) for delta in plan_data_dict['delta_areas'].values()])
        
            sum_proportion_diff = 0
            if self.fenv_config['is_proportion_considered'] and self.fenv_config['proportion_diff_in_reward_flag']:
                room_names = plan_data_dict['rooms_dict'].keys()
                delta_aspect_ratio = np.around([plan_data_dict['rooms_dict'][room_name]['delta_aspect_ratio'] 
                                          for room_name in list(room_names)[plan_data_dict['mask_numbers']:]], decimals=1)
                sum_proportion_diff = sum(delta_aspect_ratio)   
        
            sum_graph_diff = 0 
            if self.fenv_config['is_adjacency_considered']:
                for edge in plan_data_dict['desired_edge_list']:
                    if edge not in plan_data_dict['edge_list']:
                        sum_graph_diff += 1
                        
                mean_graph_diff = sum_graph_diff / (len(plan_data_dict['desired_edge_list'])+1)
                    
            nested_rooms = set(plan_data_dict['delta_areas'].keys())
            missed_rooms = set(plan_data_dict['desired_areas'].keys()) - nested_rooms
            missed_areas = sum([ plan_data_dict['desired_areas'][room_name] for room_name in missed_rooms])
                    
            if active_wall_status == 'badly_stopped':
                _reward = -missed_areas - 1*sum_areas_diff - 10*sum_graph_diff
            
            elif active_wall_status == 'well_finished':
                _reward = 0.1*self.fenv_config['stop_time_step'] - 1*sum_areas_diff - 10*sum_graph_diff
                
            else:
                raise ValueError('Invalid active_wall_status')
      
        else:
            if active_wall_status == 'accepted':
                active_room_name = f"room_{int(active_wall_name.split('_')[1])}"
                delta_area = abs(plan_data_dict['delta_areas'][active_room_name])
                _reward = -1 * delta_area
                
            else:
                _reward = len(plan_data_dict['wall_types']) - plan_data_dict['n_walls'] - 2*self.fenv_config['area_tolerance']
            
        return _reward
        
    
    
    def _get_final_reward_simple(self, plan_data_dict, 
                                active_wall_name, active_wall_status, 
                                time_step, done):
        
        if done:
            
            sum_areas_diff = 0
            if self.fenv_config['is_area_considered'] and self.fenv_config['area_diff_in_reward_flag']: 
                sum_areas_diff = sum([abs(delta) for delta in plan_data_dict['delta_areas'].values()])
                mean_areas_diff = np.mean([abs(delta) for delta in plan_data_dict['delta_areas'].values()])
        
            sum_proportion_diff = 0
            if self.fenv_config['is_proportion_considered'] and self.fenv_config['proportion_diff_in_reward_flag']:
                room_names = plan_data_dict['rooms_dict'].keys()
                delta_aspect_ratio = np.around([plan_data_dict['rooms_dict'][room_name]['delta_aspect_ratio'] 
                                          for room_name in list(room_names)[plan_data_dict['mask_numbers']:]], decimals=1)
                sum_proportion_diff = sum(delta_aspect_ratio)   
        
            sum_graph_diff = 0 
            if self.fenv_config['is_adjacency_considered']:
                for edge in plan_data_dict['desired_edge_list']:
                    if edge not in plan_data_dict['edge_list']:
                        sum_graph_diff += 1
                        
                mean_graph_diff = sum_graph_diff / (len(plan_data_dict['desired_edge_list'])+1)
                    
            nested_rooms = set(plan_data_dict['delta_areas'].keys())
            missed_rooms = set(plan_data_dict['desired_areas'].keys()) - nested_rooms
            missed_areas = sum([ plan_data_dict['desired_areas'][room_name] for room_name in missed_rooms])
                    
            if active_wall_status == 'badly_stopped':
                _reward = -0.2*self.fenv_config['stop_time_step'] #-time_step#2*self.fenv_config['positive_final_reward'] #time_step # here time_step is eaual to self.fenv_config['stop_time_step']
            
            elif active_wall_status == 'well_finished':
                # _reward = 0.01*self.fenv_config['stop_time_step'] - 1*mean_areas_diff - 10*mean_graph_diff # -1*sum_proportion_diff - time_step
                # print(sum_graph_diff)
                # print(plan_data_dict['desired_edge_list'])
                _reward = 0.2*self.fenv_config['stop_time_step'] - 10*sum_graph_diff # - 1*sum_areas_diff
                # _reward = _reward ** 0.5
                
            else:
                raise ValueError('Invalid active_wall_status')
      
        else:
            _reward = -1
            
        return _reward
    
    
        
    def _get_shaped_reward(self, plan_data_dict,
                                 active_wall_name, active_wall_status, 
                                 done):
        if done:
            _reward = self.fenv_config['positive_done_reward']
            
        else:
            if active_wall_status == "accepted":
                delta_areas = plan_data_dict['delta_areas']
                room_name = f"room_{int(active_wall_name.split('_')[1])}"
                delta_area = delta_areas[room_name]
                _reward = self.fenv_config['positive_action_reward'] - abs(delta_area) 
                
            elif active_wall_status == "rejected_by_canvas":
                _reward = self.fenv_config['negative_rejected_by_canvas_reward']
                
            elif active_wall_status == "rejected_by_room":
                _reward = self.fenv_config['negative_rejected_by_room_reward']
                
            elif active_wall_status == "rejected_by_area":
                _reward = self.fenv_config['negative_wrong_area_reward']
                
            elif active_wall_status == "rejected_by_proportion":
                _reward = self.fenv_config['negative_wrong_area_reward']
                
            elif active_wall_status == "rejected_by_both_area_and_proportion":
                _reward = self.fenv_config['negative_wrong_area_reward']
            
            elif active_wall_status == "rejected_by_other_blocked_cells":
                _reward = self.fenv_config['negative_wrong_blocked_cells_reward']
            
            elif active_wall_status == "rejected_by_odd_anchor_coord":
                 _reward = self.fenv_config['negative_wrong_odd_anchor_coord']
                
            else:
                raise ValueError("Invalid active_wall_status")
            
        return _reward
     
                
     
    def _get_basic_reward(self, active_wall_status, done):
        if done:
            _reward = self.fenv_config['positive_done_reward']
        else:
            if active_wall_status == "accepted":
                _reward = self.fenv_config['positive_action_reward']
                
            elif active_wall_status == "rejected_by_canvas":
                _reward = self.fenv_config['negative_rejected_by_canvas_reward']
                
            elif active_wall_status == "rejected_by_room":
                _reward = self.fenv_config['negative_rejected_by_room_reward']
                
            elif active_wall_status == "rejected_by_area":
                _reward = self.fenv_config['negative_wrong_area_reward']
                
            elif active_wall_status == "rejected_by_proportion":
                _reward = self.fenv_config['negative_wrong_proportion_reward']
                
            elif active_wall_status == "rejected_by_both_area_and_proportion":
                _reward = self.fenv_config['negative_wrong_area_and_proportion_reward']
                
            elif active_wall_status == "rejected_by_other_blocked_cells":
                _reward = self.fenv_config['negative_wrong_blocked_cells_reward']
                
            elif active_wall_status == "rejected_by_missing_room":
                 _reward = self.fenv_config['negative_wrong_odd_anchor_coord']
            
            elif active_wall_status == "rejected_by_odd_anchor_coord":
                 _reward = self.fenv_config['negative_wrong_odd_anchor_coord']
                 
            else:
                raise ValueError("Invalid active_wall_status")
                
        return _reward