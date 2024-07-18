#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 17:28:45 2023

@author: Reza Kakooee
"""

import os
import inspect
import numpy as np

import gym_floorplan.envs.reward.reward_utils as reward_utils




#%%
class RewardZcSmooth:
    def __init__(self, fenv_config, plan_data_dict, active_wall_name, active_wall_status, done, inspection_output_dict):
        self.fenv_config = fenv_config
        self.plan_data_dict = plan_data_dict
        # if active_wall_status == 'well_finished':
        #     path = '/home/rdbt/ETHZ/dbt_python/housing_design/rlb_agents/plan_data_dict__reward_zc_smooth.py_RewardZcSmooth__get_fnorm_reward_value.npy'
        #     self.plan_data_dict = np.load(path, allow_pickle=True).tolist()
        self.active_wall_name = active_wall_name
        self.active_wall_status = active_wall_status
        self.done = done
        self.inspection_output_dict = inspection_output_dict
        
        
        
    def get_reward(self):
        if 'DLin' in self.fenv_config['rewarding_method_name']:
            reward = self._get_direct_linear_reward_fn()
        elif 'Perc' in self.fenv_config['rewarding_method_name']:
          reward = self._get_percentage_reward_fn()
        elif 'FNorm' in self.fenv_config['rewarding_method_name']:
            reward = self._get_fnorm_reward_fn()
        else:
            reward = self._get_zc_smooth_reward_fn()
        return reward
    


    def _get_fnorm_reward_fn(self):
        if not self.done:
            if self.active_wall_status == 'accepted':
                if self.fenv_config['zc_ignore_immidiate_reward'] == 1:
                    reward = 0
                else:
                    room_name = f"room_{int(self.active_wall_name.split('_')[1])}"
                    delta_area = self.inspection_output_dict['geometry'][room_name]['delta_area']
                    delta_aspect_ratio = self.inspection_output_dict['geometry'][room_name]['delta_aspect_ratio']

                    delta_area_norm = self._fnorm_normalize_delta(delta_area, name='area')
                    delta_aspect_ratio_norm = self._fnorm_normalize_delta(delta_aspect_ratio, name='aspect_ratio')
    
                    wa, wp = self.fenv_config['fnorm_intra_episode_wa'], self.fenv_config['fnorm_intra_episode_wp']
                    delta_geom_norm = (wa * delta_area_norm + wp * delta_aspect_ratio_norm) / (wa + wp)

                    delta_geom_norm = 0.9
                    reward = self._get_fnorm_reward_value(delta_geom_norm, terminal_state=False)
                
            else:
                reward = self.fenv_config['fnorm_non_accepted_negative_reward']
                
        else:
            if self.active_wall_status == 'badly_stopped':
                reward = self.fenv_config['fnorm_negative_badly_stop_reward']
            
            elif self.active_wall_status == 'well_finished':
                delta_area_list = []
                delta_aspect_ratio_list = []
                for room_name in self.inspection_output_dict['geometry'].keys():
                    delta_area = self.inspection_output_dict['geometry'][room_name]['delta_area']
                    delta_aspect_ratio = self.inspection_output_dict['geometry'][room_name]['delta_aspect_ratio']
                    delta_area_list.append(delta_area)
                    delta_aspect_ratio_list.append(delta_aspect_ratio)

                delta_area_mean = np.mean(delta_area_list)
                delta_aspect_ratio_mean = np.mean(delta_aspect_ratio_list)

                delta_area_std = np.std(delta_area_list)
                delta_aspect_ratio_std = np.std(delta_aspect_ratio_list)
                
                delta_area_mean_norm = self._fnorm_normalize_delta(delta_area_mean, name='area', terminal_state=True)
                delta_aspect_ratio_mean_norm = self._fnorm_normalize_delta(delta_aspect_ratio_mean, name='aspect_ratio', terminal_state=True)
                delta_area_std_norm = self._fnorm_normalize_delta(delta_area_std, name='area', terminal_state=True)
                delta_aspect_ratio_std_norm = self._fnorm_normalize_delta(delta_aspect_ratio_std, name='aspect_ratio', terminal_state=True)

                delta_edge_list = self.inspection_output_dict['topology']['n_missed_connections']
                delta_edge_list_norm = self._fnorm_normalize_delta(delta_edge_list, name='edge', terminal_state=True)

                wa_m, wa_s, wp_m, wp_s, we, wl = (self.fenv_config['fnorm_terminal_state_wa_mean'], 
                                                  self.fenv_config['fnorm_terminal_state_wa_std'],
                                                  self.fenv_config['fnorm_terminal_state_wp_mean'],
                                                  self.fenv_config['fnorm_terminal_state_wp_std'],
                                                  self.fenv_config['fnorm_terminal_state_we_mean'],
                                                  self.fenv_config['fnorm_terminal_state_wl'])
                
                delta_geom_topo_norm = ( (wa_m * delta_area_mean_norm + wa_s * delta_aspect_ratio_mean_norm +
                                            wp_m * delta_area_std_norm + wp_s * delta_aspect_ratio_std_norm +
                                            we * delta_edge_list_norm +
                                            wl * self.plan_data_dict['ep_time_step'] / self.fenv_config['stop_ep_time_step']
                                            ) /
                                            (wa_m + wa_s + wp_m + wp_s + we + wl)
                )
                    
                delta_geom_topo_norm = 0.9
                reward = self._get_fnorm_reward_value(delta_geom_topo_norm, terminal_state=True)
                
            else:
                np.save(f"plan_data_dict__{os.path.basename(__file__)}_{self.__class__.__name__}_{inspect.currentframe().f_code.co_name}.npy", self.plan_data_dict)
                raise ValueError(f"Invalid active_wall_status. active_wall_status: {self.active_wall_status}, plan_id: {self.plan_data_dict['plan_id']}")
                
        return reward



    def _fnorm_normalize_delta(self, x, name, terminal_state=False):
        if name == 'area':
            if x <= self.fenv_config['delta_area_tolerance_for_fnorm']:
                return 0
            shift = self.fenv_config['area_delta_shift']
            scalar = self.fenv_config['area_delta_scalar']
            power = self.fenv_config['area_delta_power']
        elif name == 'aspect_ratio':
            if x <= self.fenv_config['delta_aspect_ratios_tolerance_for_fnorm']:
                return 0
            shift = self.fenv_config['aspect_ratio_delta_shift']
            scalar = self.fenv_config['aspect_ratio_delta_scalar']
            power = self.fenv_config['aspect_ratio_delta_power']
        elif name == 'edge':
            if np.isclose(x, 0):
                return 0
            shift = self.fenv_config['edge_delta_shift']
            scalar = self.fenv_config['edge_delta_scalar']
            power = self.fenv_config['edge_delta_power']
        else:
            raise ValueError('Invalid name for metric. The current name is: {name}')
        
        x = self.sigmoid_normalizer(x, shift, scalar, power)

        return x



    def sigmoid_normalizer(self, x, shift, scalar, power):
        return ( 1 / (1 + np.exp(-scalar * (x - shift))) ) ** power



    def _fnorm_delta_scalar(self, x, name):
        if name == 'area':
           x = x / self.fenv_config['fnorm_area_factor']
        elif name == 'aspect_ratio':
            x = x / self.fenv_config['fnorm_aspect_ratio_factor']
        elif name == 'edge':
            x = x / self.fenv_config['fnorm_edge_factor']
        else:
            raise ValueError('Invalid name for metric. The current name is: {name}')
        return x



    def _get_fnorm_reward_value(self, x, terminal_state=False):
        x_start = self.fenv_config['fnorm_delta_inf']
        x_end = self.fenv_config['fnorm_delta_sup']
        y_start = self.fenv_config['fnorm_reward_bottom']
        y_end = self.fenv_config['fnorm_reward_up']
        k = self.fenv_config['fnorm_reward_concavity_factor']
        
        if terminal_state:
            y_start *= self.fenv_config['reward_vertical_scalar']
            y_end *= self.fenv_config['reward_vertical_scalar']
            
        reward_fn = self._get_reward_fn()
        reward = reward_fn(x, x_start, x_end, y_start, y_end, k)
        if (reward < y_start) or (reward > y_end):
            np.save(f"reward__{os.path.basename(__file__)}_{self.__class__.__name__}_{inspect.currentframe().f_code.co_name}.npy", reward)
            np.save(f"plan_data_dict__{os.path.basename(__file__)}_{self.__class__.__name__}_{inspect.currentframe().f_code.co_name}.npy", self.plan_data_dict)
            message = f"""
            Reward is out of range. 
            reward: {reward}, 
            actions_accepted: {self.plan_data_dict['actions_accepted']}, 
            plan_id: {self.plan_data_dict['plan_id']},

            """
            raise ValueError(message)
        return reward



    def _get_percentage_reward_fn(self):
        if not self.done:
            if self.active_wall_status == 'accepted':
                if self.fenv_config['zc_ignore_immidiate_reward'] == 1:
                    reward = 0
                else:
                    room_name = f"room_{int(self.active_wall_name.split('_')[1])}"
                    reward_area = self._calculate_percentage_reward(self.inspection_output_dict['geometry'][room_name]['achieved_area'], 
                                                                    self.inspection_output_dict['geometry'][room_name]['desired_area'])
                    reward_aspect_ratio = self._calculate_percentage_reward(self.inspection_output_dict['geometry'][room_name]['achieved_aspect_ratio'],
                                                                             self.inspection_output_dict['geometry'][room_name]['desired_aspect_ratio'])
                    wa, wp = self.fenv_config['zc_intra_episode_wa'], self.fenv_config['zc_intra_episode_wp']
                    reward =  (wa * reward_area + wp * reward_aspect_ratio) / (wa + wp)
            else:
                reward = self.fenv_config['non_accepted_negative_reward']
        
        else:
            if self.active_wall_status == 'badly_stopped':
                reward = self.fenv_config['negative_badly_stop_reward']
            
            elif self.active_wall_status == 'well_finished':
                if self.fenv_config['zc_ignore_immidiate_reward'] == 1:
                    reward_area_list = []
                    reward_aspect_ratio_list = []
                    for room_name in self.inspection_output_dict['geometry'].keys():
                        reward_area = self._calculate_percentage_reward(self.inspection_output_dict['geometry'][room_name]['achieved_area'], 
                                                                        self.inspection_output_dict['geometry'][room_name]['desired_area'],
                                                                        name='area')
                        reward_aspect_ratio = self._calculate_percentage_reward(self.inspection_output_dict['geometry'][room_name]['achieved_aspect_ratio'],
                                                                                self.inspection_output_dict['geometry'][room_name]['desired_aspect_ratio'],
                                                                                name='aspect_ratio')
                        reward_area_list.append(reward_area)
                        reward_aspect_ratio_list.append(reward_aspect_ratio)
                    reward_area_sum = np.sum(reward_area_list)
                    reward_aspect_ratio_sum = np.sum(reward_aspect_ratio_list)
                    
                    reward_edge = self._calculate_percentage_reward(self.inspection_output_dict['topology']['n_desired_connections'],
                                                                    self.inspection_output_dict['topology']['n_nicely_achieved_connections'],
                                                                    name='edge')

                    wa_sum, wp_sum, we = (self.fenv_config['zc_terminal_state_wa_sum'], 
                                          self.fenv_config['zc_terminal_state_wp_sum'],
                                          self.fenv_config['zc_terminal_state_we'])
                    
                    reward = ( wa_sum * reward_area_sum + 
                               wp_sum * reward_aspect_ratio_sum + 
                               we * reward_edge / (wa_sum + wp_sum + we) )
                    
                else:
                    active_room_name = f"room_{int(self.active_wall_name.split('_')[1])}"
                    active_reward_area = self._calculate_percentage_reward(self.inspection_output_dict['geometry'][active_room_name]['achieved_area'],
                                                                           self.inspection_output_dict['geometry'][active_room_name]['desired_area'],
                                                                           name='area')
                    active_reward_aspect_ratio = self._calculate_percentage_reward(self.inspection_output_dict['geometry'][active_room_name]['achieved_aspect_ratio'],
                                                                                   self.inspection_output_dict['geometry'][active_room_name]['desired_aspect_ratio'],
                                                                                   name='aspect_ratio')
                    
                    last_room_name = self.plan_data_dict['last_room']['last_room_name']
                    last_reward_area = self._calculate_percentage_reward(self.inspection_output_dict['geometry'][last_room_name]['achieved_area'],
                                                                         self.inspection_output_dict['geometry'][last_room_name]['desired_area'],
                                                                         name='area')
                    last_reward_aspect_ratio = self._calculate_percentage_reward(self.inspection_output_dict['geometry'][last_room_name]['achieved_aspect_ratio'],
                                                                                 self.inspection_output_dict['geometry'][last_room_name]['desired_aspect_ratio'],
                                                                                 name='aspect_ratio')

                    reward_area = (active_reward_area + last_reward_area)
                    reward_aspect_ratio = (active_reward_aspect_ratio + last_reward_aspect_ratio)
                    
                    reward_edge = self._calculate_percentage_reward(self.inspection_output_dict['topology']['n_desired_connections'],
                                                                    self.inspection_output_dict['topology']['n_nicely_achieved_connections'],
                                                                    name='edge')

                    wa_sum, wp_sum, we = self.fenv_config['zc_terminal_state_wa_sum'], self.fenv_config['zc_terminal_state_wp_sum'], self.fenv_config['zc_terminal_state_we']                
                    reward = ( wa_sum * reward_area + wp_sum * reward_aspect_ratio + we * reward_edge ) / (wa_sum + wp_sum + we)
                
            else:
                np.save(f"plan_data_dict__{os.path.basename(__file__)}_{self.__class__.__name__}_{inspect.currentframe().f_code.co_name}.npy", self.plan_data_dict)
                raise ValueError(f"Invalid active_wall_status. active_wall_status: {self.active_wall_status}, plan_id: {self.plan_data_dict['plan_id']}")
                
        return reward
        


    def _calculate_percentage_reward(self, acheived, desired, name='area'):
        rew = -abs( (acheived / desired) - 1)
        if name == 'area':
            rew /= self.fenv_config['area_delta_sup']
        elif name == 'aspect_ratio':
            rew /= self.fenv_config['aspect_ratio_delta_sup']
        elif name == 'edge':
            rew = rew
        return rew
    


    def _get_direct_linear_reward_fn(self):
        if not self.done:
            if self.active_wall_status == 'accepted':
                if self.fenv_config['zc_ignore_immidiate_reward'] == 1:
                    reward = 0
                else:
                    room_name = f"room_{int(self.active_wall_name.split('_')[1])}"
                    reward_area = self._measure_direct_linear_reward(self.inspection_output_dict['geometry'][room_name]['delta_area'], name='area')
                    reward_aspect_ratio = self._measure_direct_linear_reward(self.inspection_output_dict['geometry'][room_name]['delta_aspect_ratio'], name='aspect_ratio')
                    wa, wp = self.fenv_config['zc_intra_episode_wa'], self.fenv_config['zc_intra_episode_wp']
                    reward =  (wa * reward_area + wp * reward_aspect_ratio) / (wa + wp)
            else:
                reward = self.fenv_config['non_accepted_negative_reward']
        
        else:
            if self.active_wall_status == 'badly_stopped':
                reward = self.fenv_config['negative_badly_stop_reward']
            
            elif self.active_wall_status == 'well_finished':
                if self.fenv_config['zc_ignore_immidiate_reward'] == 1:
                    delta_area_list = []
                    delta_aspect_ratio_list = []
                    for room_name in self.inspection_output_dict['geometry'].keys():
                        delta_area = self.inspection_output_dict['geometry'][room_name]['delta_area']
                        delta_aspect_ratio = self.inspection_output_dict['geometry'][room_name]['delta_aspect_ratio']
                        delta_area_list.append(delta_area)
                        delta_aspect_ratio_list.append(delta_aspect_ratio)

                    delta_area_mean = np.mean(delta_area_list)
                    delta_aspect_ratio_mean = np.mean(delta_aspect_ratio_list)

                    delta_area_std = np.std(delta_area_list)
                    delta_aspect_ratio_std = np.std(delta_aspect_ratio_list)
                    
                    reward_area_mean = self._measure_direct_linear_reward(delta_area_mean, name='area')
                    reward_aspect_ratio_mean = self._measure_direct_linear_reward(delta_aspect_ratio_mean, name='aspect_ratio')
                    reward_area_std = self._measure_direct_linear_reward(delta_area_std, name='area')
                    reward_aspect_ratio_std = self._measure_direct_linear_reward(delta_aspect_ratio_std, name='aspect_ratio')

                    reward_edge = self._measure_direct_linear_reward(self.inspection_output_dict['topology']['n_missed_connections'], name='edge')

                    wa_m, wa_s, wp_m, wp_s, we = (self.fenv_config['zc_terminal_state_wa_mean'], 
                                                  self.fenv_config['zc_terminal_state_wa_std'],
                                                  self.fenv_config['zc_terminal_state_wp_mean'],
                                                  self.fenv_config['zc_terminal_state_wp_std'],
                                                  self.fenv_config['zc_terminal_state_we_mean'])
                    
                    reward = ( wa_m * reward_area_mean + 
                               wa_s * reward_area_std + 
                               wp_m * reward_aspect_ratio_mean + 
                               wp_s * reward_aspect_ratio_std + 
                               we * reward_edge / (wa_m + wa_s + wp_m + wp_s + we) )
                    
                else:
                    active_room_name = f"room_{int(self.active_wall_name.split('_')[1])}"
                    active_reward_area = self._measure_direct_linear_reward(self.inspection_output_dict['geometry'][active_room_name]['delta_area'], name='area')
                    active_reward_aspect_ratio = self._measure_direct_linear_reward(self.inspection_output_dict['geometry'][active_room_name]['delta_aspect_ratio'], name='aspect_ratio')
                    
                    last_room_name = self.plan_data_dict['last_room']['last_room_name']
                    last_reward_area = self._measure_direct_linear_reward(self.inspection_output_dict['geometry'][last_room_name]['delta_area'], name='area')
                    last_reward_aspect_ratio = self._measure_direct_linear_reward(self.inspection_output_dict['geometry'][last_room_name]['delta_aspect_ratio'], name='aspect_ratio')
                    
                    reward_area = (active_reward_area + last_reward_area) / 2
                    reward_aspect_ratio = (active_reward_aspect_ratio + last_reward_aspect_ratio) / 2
                    
                    reward_edge = self._measure_direct_linear_reward(self.inspection_output_dict['topology']['n_missed_connections'], name='edge')

                    wa, wp, we = self.fenv_config['zc_end_episode_wa'], self.fenv_config['zc_end_episode_wp'], self.fenv_config['zc_end_episode_we']                
                    reward = ( wa * reward_area + wp * reward_aspect_ratio + we * reward_edge ) / (wa + wp + we)

        return reward
    
    

    def _measure_direct_linear_reward(self, x, name):
        if name == 'area':
            den = self.fenv_config['area_delta_sup']  
        elif name == 'aspect_ratio':
            den = self.fenv_config['aspect_ratio_delta_sup']
        elif name == 'edge':
            den = self.fenv_config['edge_delta_sup']
        reward_fn = reward_utils.direct_linear
        reward = reward_fn(x, den)
        return reward
        
    
    
    def _get_zc_smooth_reward_fn(self):
        if not self.done:
            if self.active_wall_status == 'accepted':
                if self.fenv_config['zc_ignore_immidiate_reward'] == 1:
                    reward = 0
                else:
                    room_name = f"room_{int(self.active_wall_name.split('_')[1])}"
                    delta_area = self.inspection_output_dict['geometry'][room_name]['delta_area']
                    delta_aspect_ratio = self.inspection_output_dict['geometry'][room_name]['delta_aspect_ratio']

                    reward_area = self._get_reward_per_fn(delta_area, name='area')
                    reward_aspect_ratio = self._get_reward_per_fn(delta_aspect_ratio, name='aspect_ratio')
    
                    wa, wp = self.fenv_config['zc_intra_episode_wa'], self.fenv_config['zc_intra_episode_wp']
                    reward =  (wa * reward_area + wp * reward_aspect_ratio) / (wa + wp)
                
            else:
                reward = self.fenv_config['non_accepted_negative_reward']
                
        else:
            if self.active_wall_status == 'badly_stopped':
                reward = self.fenv_config['negative_badly_stop_reward']
            
            elif self.active_wall_status == 'well_finished':
                delta_area_list = []
                delta_aspect_ratio_list = []
                for room_name in self.inspection_output_dict['geometry'].keys():
                    delta_area = self.inspection_output_dict['geometry'][room_name]['delta_area']
                    delta_aspect_ratio = self.inspection_output_dict['geometry'][room_name]['delta_aspect_ratio']
                    delta_area_list.append(delta_area)
                    delta_aspect_ratio_list.append(delta_aspect_ratio)

                delta_area_mean = np.mean(delta_area_list)
                delta_aspect_ratio_mean = np.mean(delta_aspect_ratio_list)

                delta_area_std = np.std(delta_area_list)
                delta_aspect_ratio_std = np.std(delta_aspect_ratio_list)
                
                reward_area_mean = self._get_reward_per_fn(delta_area_mean, name='area', terminal_state=True)
                reward_aspect_ratio_mean = self._get_reward_per_fn(delta_aspect_ratio_mean, name='aspect_ratio', terminal_state=True)
                reward_area_std = self._get_reward_per_fn(delta_area_std, name='area', terminal_state=True)
                reward_aspect_ratio_std = self._get_reward_per_fn(delta_aspect_ratio_std, name='aspect_ratio', terminal_state=True)

                delta_edge_list = self.inspection_output_dict['topology']['n_missed_connections']
                reward_edge = self._get_reward_per_fn(delta_edge_list, name='edge', terminal_state=True)

                wa_m, wa_s, wp_m, wp_s, we = (self.fenv_config['zc_terminal_state_wa_mean'], 
                                              self.fenv_config['zc_terminal_state_wa_std'],
                                              self.fenv_config['zc_terminal_state_wp_mean'],
                                              self.fenv_config['zc_terminal_state_wp_std'],
                                              self.fenv_config['zc_terminal_state_we_mean'])
                
                reward = ( wa_m * reward_area_mean + 
                           wa_s * reward_area_std + 
                           wp_m * reward_aspect_ratio_mean + 
                           wp_s * reward_aspect_ratio_std + 
                           we * reward_edge / (wa_m + wa_s + wp_m + wp_s + we) )
                
                reward = np.clip(reward, self.fenv_config['zc_reward_bottom']*self.fenv_config['reward_vertical_scalar'], self.fenv_config['reward_vertical_scalar']*self.fenv_config['zc_reward_up'])
                reward *= self.fenv_config['zc_reward_terminal_state_factor']
                if delta_edge_list == 0:
                    reward += self.fenv_config['reward_vertical_scalar']
                
            else:
                np.save(f"plan_data_dict__{os.path.basename(__file__)}_{self.__class__.__name__}_{inspect.currentframe().f_code.co_name}.npy", self.plan_data_dict)
                raise ValueError(f"Invalid active_wall_status. active_wall_status: {self.active_wall_status}, plan_id: {self.plan_data_dict['plan_id']}")
                
        return reward
    


    def _get_reward_per_fn(self, x, name, terminal_state=False):
        if name == 'area':
            x_start = self.fenv_config['area_delta_inf']
            x_end = self.fenv_config['area_delta_sup']
        elif name == 'aspect_ratio':
            x_start = self.fenv_config['aspect_ratio_delta_inf']
            x_end = self.fenv_config['aspect_ratio_delta_sup']
        elif name == 'edge':
            x_start = self.fenv_config['edge_delta_inf']
            x_end = self.fenv_config['edge_delta_sup']
        else:
            raise ValueError('Invalid name')

        y_start = self.fenv_config['zc_reward_bottom']
        y_end = self.fenv_config['zc_reward_up']

        if terminal_state:
            y_start *= self.fenv_config['reward_vertical_scalar']
            y_end *= self.fenv_config['reward_vertical_scalar']

        reward_fn = self._get_reward_fn()
        reward = reward_fn(x, x_start, x_end, y_start, y_end)
        if (reward < y_start) or (reward > y_end):
            np.save(f"reward__{os.path.basename(__file__)}_{self.__class__.__name__}_{inspect.currentframe().f_code.co_name}_1.npy", reward)
            np.save(f"plan_data_dict__{os.path.basename(__file__)}_{self.__class__.__name__}_{inspect.currentframe().f_code.co_name}_1.npy", self.plan_data_dict)
            message = f"""
            Reward is out of range: 
            reward: {reward}, 
            actions_accepted: {self.plan_data_dict['actions_accepted']}, 
            plan_id: {self.plan_data_dict['plan_id']},

            """
            raise ValueError(message)
        
        return reward
          


    def _get_reward_fn(self):
        if 'Smooth_Linear_Reward' in self.fenv_config['rewarding_method_name']:
            reward_fn = reward_utils.linear
            
        elif 'Smooth_Quad_Reward' in self.fenv_config['rewarding_method_name']:
            reward_fn = reward_utils.quadratic
        
        elif 'Smooth_Log_Reward' in self.fenv_config['rewarding_method_name']:
            reward_fn = reward_utils.logarithmic
            
        elif 'Smooth_Exp_Reward' in self.fenv_config['rewarding_method_name']:
            reward_fn = reward_utils.exponential
            
        elif 'Smooth_FNorm_Reward' in self.fenv_config['rewarding_method_name']:
            reward_fn = reward_utils.logarithmic

        else:
            raise ValueError(f"Invalid rewarding method name. The current name is: {self.fenv_config['rewarding_method_name']}")
       
        return reward_fn
    
    
    
