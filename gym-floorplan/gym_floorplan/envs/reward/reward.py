# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 01:19:28 2021

@author: Reza Kakooee
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
                     ep_time_step, done):
        
        self.plan_data_dict = plan_data_dict
        self.active_wall_name = active_wall_name
        self.active_wall_status = active_wall_status
        self.ep_time_step = ep_time_step
        self.done = done
        
        if self.fenv_config['rewarding_method_name'] == 'Constrain_Satisfaction':
            _reward = self._get_constrain_satisfaction_reward()
        
        if self.fenv_config['rewarding_method_name'] == 'Very_Binary_Reward':
            _reward = self._get_very_binary_reward()
            
        elif self.fenv_config['rewarding_method_name'] == 'Binary_Reward':
            _reward = self._get_binary_reward()
            
        elif self.fenv_config['rewarding_method_name'] == 'Simple_Reward':
            _reward = self.get_simple_reward()
            
        elif self.fenv_config['rewarding_method_name'] in ['Simple_Linear_Reward', 'Simple_Quad_Reward', 'Simple_Exp_Reward']:
            _reward = self.get_simple_fn_reward()
            
        elif self.fenv_config['rewarding_method_name'] in ['Smooth_Linear_Reward', 'Smooth_Quad_Reward', 'Smooth_Log_Reward', 'Smooth_Exp_Reward']:
            _reward = self._get_smooth_offset_reward()
            
        elif self.fenv_config['rewarding_method_name'] == 'Detailed_Reward':
            _reward = self._get_detailed_reward() 
            
        else:
            raise ValueError('Invalid rewarding method!')
        
        return _reward

    

    def _get_constrain_satisfaction_reward(self):
        if not self.done:
            if self.active_wall_status == 'accepted':
                _reward = 0
            else:
                _reward = -1
        else:
            if self.active_wall_status == 'badly_stopped':
                _reward = -1
            elif self.active_wall_status == 'well_finished':
                _reward = self.fenv_config['positive_done_reward']
            else:
                raise ValueError('Invalid active_wall_status')
        return _reward
        

    
    def _get_very_binary_reward(self):
        if not self.done:
            _reward = -1
        else:
            if self.active_wall_status == 'badly_stopped':
                _reward = -1 
            
            elif self.active_wall_status == 'well_finished':
                reward_measurements_dict = self._calculate_distance_to_goal()
                
                if reward_measurements_dict['sum_adj_diff'] == 0:
                    _reward = self.fenv_config['positive_done_reward']
                else:
                    _reward = -1
            else:
                raise ValueError('Invalid active_wall_status')
            
        return _reward
        
    
    
    def _get_binary_reward(self):
        if not self.done:
            if self.active_wall_status == 'accepted':
                _reward = 0
            else:
                _reward = -1
        else:
            if self.active_wall_status == 'badly_stopped':
                _reward = -1
            
            elif self.active_wall_status == 'well_finished':
                reward_measurements_dict = self._calculate_distance_to_goal()
                
                if reward_measurements_dict['sum_adj_diff'] == 0:
                    _reward = self.fenv_config['positive_done_reward']
                else:
                    _reward = 0
            else:
                raise ValueError('Invalid active_wall_status')
            
        return _reward
        
    
    
    def get_simple_reward(self):
        if not self.done:
            if self.active_wall_status == 'accepted':
                _reward = 0
            else:
                _reward = -1
        else:
            if self.active_wall_status == 'badly_stopped':
                _reward = self.fenv_config['negative_badly_stop_reward']
            
            elif self.active_wall_status == 'well_finished':
                reward_measurements_dict = self._calculate_distance_to_goal()
                _reward = 1 * self.fenv_config['positive_final_reward'] \
                         -50 * reward_measurements_dict['sum_adj_diff'] 

            else:
                raise ValueError('Invalid active_wall_status')

        return _reward
    
    
    
    def get_simple_fn_reward(self):
        if not self.done:
            if self.active_wall_status == 'accepted':
                _reward = 0
            else:
                _reward = -1
        else:
            if self.active_wall_status == 'badly_stopped':
                _reward = self.fenv_config['negative_badly_stop_reward']
            
            elif self.active_wall_status == 'well_finished':
                reward_measurements_dict = self._calculate_distance_to_goal()

                n_bins = reward_measurements_dict['n_desired_adjacencies'] + 1
                edge_range = np.linspace(self.fenv_config['edge_diff_min'], self.fenv_config['edge_diff_max'], n_bins)
                
                if self.fenv_config['rewarding_method_name'] == 'Simple_Linear_Reward':
                    reward_range = self.fenv_config['positive_final_reward'] + 1 - self.fenv_config['linear_reward_coeff'] * edge_range
                    
                elif self.fenv_config['rewarding_method_name'] == 'Simple_Quad_Reward':
                    reward_range = self.fenv_config['positive_final_reward'] - self.fenv_config['quad_reward_coeff'] * edge_range**2 
                
                elif self.fenv_config['rewarding_method_name'] == 'Simple_Exp_Reward':
                    reward_range = np.exp( (self.fenv_config['positive_final_reward'] - 
                                            self.fenv_config['exp_reward_coeff']*edge_range ) / self.fenv_config['exp_reward_temperature']  ).astype(int)
                
                else:
                    raise ValueError('Invalid rewarding method')
                
                _reward = reward_range[reward_measurements_dict['sum_adj_diff']]
                _reward = np.clip(_reward, 1, self.fenv_config['positive_final_reward'])
                # _reward *= self.fenv_config['last_good_reward_scalar']
                
                # if _reward < self.fenv_config['last_good_reward_threshold']:
                #     _reward = self.fenv_config['last_good_reward_low_val'] = 100
                # else:
                #     _reward *= self.fenv_config['last_good_reward_high_coeff']

            else:
                raise ValueError('Invalid active_wall_status')

        return _reward
        
    

    def _get_detailed_reward(self):
        if not self.done:
            if self.active_wall_status == 'accepted':
                _reward = 0
            else:
                _reward = -1
        else:
            reward_measurements_dict = self._calculate_distance_to_goal()
            if self.active_wall_status == 'badly_stopped':
                _reward = -1 * reward_measurements_dict['missed_areas'] \
                          - 1 * reward_measurements_dict['sum_areas_diff'] \
                          - 10 * reward_measurements_dict['sum_adj_diff']
            
            elif self.active_wall_status == 'well_finished':
                _reward = 1 * self.fenv_config['positive_done_reward'] \
                          - 1 * reward_measurements_dict['sum_areas_diff'] \
                          - 10 * reward_measurements_dict['sum_adj_diff']
                         
                # _reward = 0.01 * self.fenv_config['positive_final_reward'] - \
                #              1 * reward_measurements_dict['mean_areas_diff'] -  \
                #             10 * reward_measurements_dict['mean_adj_diff'] - \
                #              1 * reward_measurements_dict['sum_proportion_diff'] - \
                #                  self.ep_time_step
            else:
                raise ValueError('Invalid active_wall_status')
            
        return _reward
    
   
       
    def _get_smooth_offset_reward(self):
        if not self.done:
            if self.active_wall_status == 'accepted':
                _reward = 0
            else:
                _reward = -1
        else:
            if self.active_wall_status == 'badly_stopped':
                _reward = self.fenv_config['negative_badly_stop_reward']
            
            elif self.active_wall_status == 'well_finished':
                reward_measurements_dict = self._calculate_distance_to_goal()
                
                n_bins = reward_measurements_dict['n_desired_adjacencies'] + 1
                edge_range = np.linspace(self.fenv_config['edge_diff_min'], self.fenv_config['edge_diff_max'], n_bins)
                
                if self.fenv_config['rewarding_method_name'] == 'Smooth_Linear_Reward':
                    reward_range = self.get_linear_reward_ragne(edge_range)
                    
                elif self.fenv_config['rewarding_method_name'] == 'Smooth_Quad_Reward':
                    reward_range = self.get_quadratic_reward_ragne(edge_range)
                
                elif self.fenv_config['rewarding_method_name'] == 'Smooth_Log_Reward':
                    reward_range = self.get_logarithmic_reward_ragne(edge_range)
                    
                elif self.fenv_config['rewarding_method_name'] == 'Smooth_Exp_Reward':
                    reward_range = self.get_exponential_reward_ragne(edge_range)
                
                else:
                    raise ValueError('Invalid rewarding method')
                
                _reward = reward_range[reward_measurements_dict['sum_adj_diff']]
                _reward = np.clip(_reward, 1, self.fenv_config['positive_final_reward'])
    
            else:
                raise ValueError('Invalid active_wall_status')
    
        return _reward
    
    
    
    
    def get_linear_reward_ragne(self, x):
        return self.linear(
            x,
            self.fenv_config['edge_diff_min'],
            self.fenv_config['edge_diff_max'],
            1,
            self.fenv_config['positive_final_reward']
            )
    
    
    
    def get_quadratic_reward_ragne(self, x):
        return self.quadratic(
            x,
            self.fenv_config['edge_diff_min'],
            self.fenv_config['edge_diff_max'],
            1,
            self.fenv_config['positive_final_reward']
            )
    
    
    
    def get_logarithmic_reward_ragne(self, x):
        return self.logarithmic(
            x,
            self.fenv_config['edge_diff_min'],
            self.fenv_config['edge_diff_max'],
            1,
            self.fenv_config['positive_final_reward']
            )
    
    
    
    def get_exponential_reward_ragne(self, x):
        return self.exponential(
            x,
            self.fenv_config['edge_diff_min'],
            self.fenv_config['edge_diff_max'],
            1,
            self.fenv_config['positive_final_reward']
            )
    
    
    @staticmethod
    def linear(x, x_start, x_end, y_start, y_end):
        return ((y_start-y_end)/(x_end-x_start))*(x - x_start) + y_end
    
    @staticmethod
    def quadratic(x, x_start, x_end, y_start, y_end):
        return y_end - ((y_end-y_start)/((x_end-x_start)**2)) * ((x-x_start)**2)
    
    @staticmethod
    def logarithmic(x, x_start, x_end, y_start, y_end):
        return ((y_start - y_end) / np.log(x_end+1 - x_start)) * np.log(x+1 - x_start) + y_end
    
    @staticmethod
    def exponential(x, x_start, x_end, y_start, y_end):
        return y_end * ((y_start/y_end)**((x - x_start)/(x_end - x_start)))
    
    
    def _calculate_distance_to_goalv1(self):
        sum_areas_diff = 0
        mean_areas_diff = 0
        sum_proportion_diff = 0
        sum_adj_diff = 0
        adj_diff_ratio = 0
        adj_performance = 0
        n_desired_adjacencies = 1
        
        if self.fenv_config['is_adjacency_considered'] and self.fenv_config['adaptive_window']:
            n_desired_adjacencies = ( len(self.plan_data_dict['edge_list_room_desired']) +
                                      len(self.plan_data_dict['edge_list_facade_desired_str']) + # this must be 1 when adaptive_room
                                      len(self.plan_data_dict['edge_color_data_dict_facade']['adaptive_rooms'])-1 ) # living room do not need to be counted here
            
            missed_room_connections = self.plan_data_dict['edge_color_data_dict_room']['red']
            cor_liv_edge = [self.plan_data_dict['corridor_id'], self.plan_data_dict['living_room_id']]
            liv_cor_edge = [self.plan_data_dict['living_room_id'], self.plan_data_dict['corridor_id']]
            if ( cor_liv_edge in missed_room_connections or 
                 liv_cor_edge in missed_room_connections):
                n_missed_room_connections = (len(missed_room_connections) - 1) + self.fenv_config['weight_for_missing_corridor_living_room_connection']
            else:
                n_missed_room_connections = len(missed_room_connections)
                
            blind_rooms = self.plan_data_dict['edge_color_data_dict_facade']['blind_rooms']
            n_blind_rooms = len(blind_rooms)-1 if self.plan_data_dict['living_room_id'] in blind_rooms else len(blind_rooms)

            sum_adj_diff = n_missed_room_connections + n_blind_rooms
            
            adj_diff_ratio = sum_adj_diff / n_desired_adjacencies
            adj_performance = 1 - adj_diff_ratio
            
        else:
            raise NotImplementedError('For now we always have adaptive rooms')
                
        nested_rooms = set(self.plan_data_dict['areas_delta'].keys())
        missed_rooms = set(self.plan_data_dict['areas_desired'].keys()) - nested_rooms
        missed_areas = sum([self.plan_data_dict['areas_desired'][room_name] for room_name in missed_rooms])
        
        reward_measurements_dict = {'sum_areas_diff': sum_areas_diff,
                                    'missed_areas': missed_areas,
                                    'mean_areas_diff': mean_areas_diff,
                                    'sum_proportion_diff': sum_proportion_diff,
                                    'sum_adj_diff': sum_adj_diff,
                                    'adj_diff_ratio': adj_diff_ratio,
                                    'adj_performance': adj_performance,
                                    'n_desired_adjacencies': n_desired_adjacencies}
        return reward_measurements_dict
    
    
    
    
    def _calculate_distance_to_goal(self):
        sum_areas_diff = 0
        mean_areas_diff = 0
        if self.fenv_config['is_area_considered'] and self.fenv_config['area_diff_in_reward_flag']: 
            areas_delta_abs = np.abs(list(self.plan_data_dict['areas_delta'].values()))
            sum_areas_diff = np.sum(areas_delta_abs)
            mean_areas_diff = np.mean(areas_delta_abs)
    
    
        sum_proportion_diff = 0
        if self.fenv_config['is_proportion_considered'] and self.fenv_config['proportion_diff_in_reward_flag']:
            room_names = self.plan_data_dict['areas_delta'].keys() # TODO need to be re-checked
            delta_aspect_ratio = np.around([self.plan_data_dict['rooms_dict'][room_name]['delta_aspect_ratio'] for room_name in room_names], decimals=1)
            sum_proportion_diff = sum(delta_aspect_ratio)   
    
        
        sum_adj_diff, adj_diff_ratio = 0, 0
        n_desired_adjacencies = 1
        if self.fenv_config['is_adjacency_considered']:
            n_desired_adjacencies = (len(self.plan_data_dict['edge_list_room_desired']) + 
                                     len(self.plan_data_dict['edge_list_facade_desired_str']) + 
                                     len(self.plan_data_dict['edge_list_entrance_desired_str']*self.fenv_config['weight_for_entrance_edge_diff_in_reward'])
                                     )
            
            room_edge_diff = len(self.plan_data_dict['edge_color_data_dict_room']['red'])
            facade_edge_diff = len(self.plan_data_dict['edge_color_data_dict_facade']['red'])
            entrance_edge_diff = len(self.plan_data_dict['edge_color_data_dict_entrance']['red'])
            sum_adj_diff = (room_edge_diff + 
                              facade_edge_diff + 
                              entrance_edge_diff*self.fenv_config['weight_for_entrance_edge_diff_in_reward'])
            
            if self.fenv_config['adaptive_window']:
                n_desired_adjacencies += len(self.plan_data_dict['edge_color_data_dict_facade']['adaptive_rooms'])
                sum_adj_diff += len(self.plan_data_dict['edge_color_data_dict_facade']['blind_rooms'])
                        
            adj_diff_ratio = sum_adj_diff / n_desired_adjacencies
            adj_performance = 1 - adj_diff_ratio
                
        nested_rooms = set(self.plan_data_dict['areas_delta'].keys())
        missed_rooms = set(self.plan_data_dict['areas_desired'].keys()) - nested_rooms
        missed_areas = sum([self.plan_data_dict['areas_desired'][room_name] for room_name in missed_rooms])
        
        reward_measurements_dict = {'sum_areas_diff': sum_areas_diff,
                                    'missed_areas': missed_areas,
                                    'mean_areas_diff': mean_areas_diff,
                                    'sum_proportion_diff': sum_proportion_diff,
                                    'sum_adj_diff': sum_adj_diff,
                                    'adj_diff_ratio': adj_diff_ratio,
                                    'adj_performance': adj_performance,
                                    'n_desired_adjacencies': n_desired_adjacencies}
        self.reward_measurements_dict = reward_measurements_dict
        return reward_measurements_dict