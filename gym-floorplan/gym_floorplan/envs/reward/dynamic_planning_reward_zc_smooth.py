#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 21:15:59 2024

@author: Reza Kakooee
"""

import os
import csv
import inspect
import numpy as np
from datetime import datetime

import gym_floorplan.envs.reward.reward_utils as reward_utils




#%%
class DynamicPlanningRewardZcSmooth:
    def __init__(self, fenv_config):
        self.fenv_config = fenv_config

        self.collect_stats_flag = True
        self.stats_max_len = 10_000
        self.dyp_rew_stats = []
        if 'results_dir' in self.fenv_config.keys():
            writable_dir = os.path.join(self.fenv_config['results_dir'], 'dyp_rew_stats') 
        else:
            writable_dir = os.path.join(self.fenv_config['rnd_agents_storage_dir'], f"{self.fenv_config['scenario_name']}/dyp_rew_stats")
        os.makedirs(writable_dir, exist_ok=True)

        self.dyp_simple_min_max_reward_falg = self.fenv_config.get('dyp_simple_min_max_reward_falg', False)
        self.dyp_min_achieved_area_to_accept = self.fenv_config.get('dyp_min_achieved_area_to_accept', 64)
        self.dyp_max_achieved_area_to_accept = self.fenv_config.get('dyp_max_achieved_area_to_accept', 256)
        self.dyp_max_achieved_aspect_ratio_to_accept = self.fenv_config.get('dyp_max_achieved_aspect_ratio_to_accept', 4)

        self.dyp_geom_condition_features_default = [
            'min_achieved_area',
            'max_achieved_area',
            'max_achieved_aspect_ratio', 
            'delta_area_mean', 
            'delta_area_std',
            'delta_aspect_ratio_mean', 
            'delta_aspect_ratio_std'
        ]
        self.dyp_geom_condition_features = self.fenv_config.get('dyp_geom_condition_features', self.dyp_geom_condition_features_default)

        self.well_finished_condition_features_default = [
            'delta_area_mean',
            'delta_aspect_ratio_mean',
            'delta_edge_list',
        ]
        self.dyp_well_finished_condition_features = self.fenv_config.get('dyp_well_finished_condition_features', self.well_finished_condition_features_default)
       

    
    def store_stats(self, stats):
        # Append the new stats to dyp_rew_stats
        self.dyp_rew_stats.append(stats)
        # print(f"len(self.dyp_rew_stats): {len(self.dyp_rew_stats)}")

        # Check if the length of dyp_rew_stats exceeds the threshold
        if len(self.dyp_rew_stats) >= self.stats_max_len:
            # Generate a timestamped filename
            timestamp = datetime.now().strftime('%Y_%m_%d_%H%M')
            filename = f"rew_stats_{timestamp}.csv"
            filepath = os.path.join(self.fenv_config['results_dir'], 'dyp_rew_stats', filename)

            # Write the stats to the CSV file
            with open(filepath, mode='w', newline='') as file:
                writer = csv.writer(file)
                # Write header
                writer.writerow([
                    'delta_area_mean', 'delta_area_std', 'delta_area_max',
                    'delta_aspect_ratio_mean', 'delta_aspect_ratio_std', 'delta_aspect_ratio_max',
                    'delta_edge_list',
                    'reward_area_mean', 'reward_area_std', 'reward_area_max',
                    'reward_aspect_ratio_mean', 'reward_aspect_ratio_std', 'reward_aspect_ratio_max',
                    'reward_edge',
                    'is_geom_condition_violated', 'dyp_lvroom_entrance_topo_condition_violation_negative_reward',
                    'is_topo_condition_violated', 'well_finished_condition', 
                    'reward',
                    'min_achieved_area', 'max_achieved_area', 'max_achieved_aspect_ratio',
                ])
                # Write the collected stats
                writer.writerows(self.dyp_rew_stats)

            # Clear the stats after writing to disk
            self.dyp_rew_stats.clear()


            
    def get_reward(self, plan_data_dict, active_wall_status, inspection_output_dict):
        self.plan_data_dict = plan_data_dict
        self.active_wall_status = active_wall_status
        self.inspection_output_dict = inspection_output_dict

        self.area_weight = {}
        total_desired_area = sum(list(self.plan_data_dict['areas_desired'].values()))
        for room_name in self.inspection_output_dict['geometry'].keys():
            if room_name != f"room_{self.fenv_config['lvroom_id']}":
                desired_area = self.plan_data_dict['areas_desired'][room_name]
                self.area_weight[room_name] = 1 - desired_area / total_desired_area

        # if self.dyp_simple_min_max_reward_falg: # no longer needed as 
        #     reward, well_finished_condition = self._get_simple_min_max_reward()
        # else:
        reward, well_finished_condition = self._get_zc_smooth_reward_fn()
        return reward, well_finished_condition
    
    
    
    def _get_min_max_geom(self):
        achieved_areas = []
        achieved_aspect_ratios = []
        for room_name in self.inspection_output_dict['geometry'].keys():
            if room_name != 'room_11':
                achieved_areas.append(self.inspection_output_dict['geometry'][room_name]['achieved_area'])
                achieved_aspect_ratios.append(self.inspection_output_dict['geometry'][room_name]['achieved_aspect_ratio'])
        min_achieved_area = min(achieved_areas)
        max_achieved_area = max(achieved_areas)
        max_achieved_aspect_ratio = max(achieved_aspect_ratios)
        return min_achieved_area, max_achieved_area, max_achieved_aspect_ratio
    
    

    def _get_misfit_stats(self):
        delta_area_dict = {}
        delta_aspect_ratio_dict = {}
        for room_name in self.inspection_output_dict['geometry'].keys():
            if room_name != f"room_{self.fenv_config['lvroom_id']}":
                delta_area = self.inspection_output_dict['geometry'][room_name]['delta_area']
                delta_aspect_ratio = self.inspection_output_dict['geometry'][room_name]['delta_aspect_ratio']
                delta_area_dict[room_name] = delta_area
                delta_aspect_ratio_dict[room_name] = delta_aspect_ratio
            
        # Normalize the weights to ensure they sum to 1
        normalized_weights = {room_name: w / sum(self.area_weight.values()) for room_name, w in self.area_weight.items()}

        # Calculate the weighted mean with normalized weights
        delta_area_mean = np.sum(
            [delta_area_dict[room_name] * normalized_weights[room_name] for room_name in delta_area_dict.keys()]
        )

        delta_area_std = np.std(list(delta_area_dict.values()))
        delta_area_max = np.max(list(delta_area_dict.values()))

        delta_aspect_ratio_mean = np.mean(list(delta_aspect_ratio_dict.values()))
        delta_aspect_ratio_std = np.std(list(delta_aspect_ratio_dict.values()))
        delta_aspect_ratio_max = np.max(list(delta_aspect_ratio_dict.values()))

        delta_edge_list = self.inspection_output_dict['topology']['n_missed_connections']

        return abs(delta_area_mean), abs(delta_area_std), abs(delta_area_max), abs(delta_aspect_ratio_mean), abs(delta_aspect_ratio_std), abs(delta_aspect_ratio_max), abs(delta_edge_list)

 

    def _map_misfit_to_reward(self, delta_area_mean, delta_area_std, delta_area_max, delta_aspect_ratio_mean, delta_aspect_ratio_std, delta_aspect_ratio_max, delta_edge_list):
        reward_area_mean = self._get_reward_per_fn(delta_area_mean, name='area', terminal_state=True)
        reward_area_std = self._get_reward_per_fn(delta_area_std, name='area', terminal_state=True)
        reward_area_max = self._get_reward_per_fn(delta_area_max, name='area', terminal_state=True)

        reward_aspect_ratio_mean = self._get_reward_per_fn(delta_aspect_ratio_mean, name='aspect_ratio', terminal_state=True)
        reward_aspect_ratio_std = self._get_reward_per_fn(delta_aspect_ratio_std, name='aspect_ratio', terminal_state=True)
        reward_aspect_ratio_max = self._get_reward_per_fn(delta_aspect_ratio_max, name='aspect_ratio', terminal_state=True)

        reward_edge = self._get_reward_per_fn(delta_edge_list, name='edge', terminal_state=True)

        wa_m, wa_s, wp_m, wp_s, we = (self.fenv_config['zc_terminal_state_wa_mean'], 
                                      self.fenv_config['zc_terminal_state_wa_std'],
                                      self.fenv_config['zc_terminal_state_wp_mean'],
                                      self.fenv_config['zc_terminal_state_wp_std'],
                                      self.fenv_config['zc_terminal_state_we_mean'])
        
        # reward = ( (wa_m+wa_s) * reward_area_max + 
        #            (wp_m+wp_s) * reward_aspect_ratio_max + 
        #             we * reward_edge / (wa_m + wa_s + wp_m + wp_s + we) )
        
        reward = ( wa_m * reward_area_mean + 
                   wa_s * reward_area_std +
                   wp_m * reward_aspect_ratio_mean +
                   wp_s * reward_aspect_ratio_std + 
                   we   * reward_edge / (wa_m + wa_s + wp_m + wp_s + we) )
        
        return reward_area_mean, reward_area_std, reward_area_max, reward_aspect_ratio_mean, reward_aspect_ratio_std, reward_aspect_ratio_max, reward_edge, reward


    
    def _get_zc_smooth_reward_fn(self):
        min_achieved_area, max_achieved_area, max_achieved_aspect_ratio = self._get_min_max_geom()
        
        ( delta_area_mean, delta_area_std, delta_area_max, 
          delta_aspect_ratio_mean, delta_aspect_ratio_std, delta_aspect_ratio_max, 
          delta_edge_list ) = self._get_misfit_stats()
        
        ( reward_area_mean, reward_area_std, reward_area_max, 
          reward_aspect_ratio_mean, reward_aspect_ratio_std, reward_aspect_ratio_max, 
          reward_edge, reward) = self._map_misfit_to_reward(
                delta_area_mean, delta_area_std, delta_area_max, 
                delta_aspect_ratio_mean, delta_aspect_ratio_std, delta_aspect_ratio_max, 
                delta_edge_list)

        # Define a dictionary mapping features to their condition checks
        condition_checks = {
            'min_achieved_area': lambda: min_achieved_area < self.dyp_min_achieved_area_to_accept,
            'max_achieved_area': lambda: max_achieved_area > self.dyp_max_achieved_area_to_accept,
            'max_achieved_aspect_ratio': lambda: max_achieved_aspect_ratio > self.dyp_max_achieved_aspect_ratio_to_accept,
            'delta_area_mean': lambda: delta_area_mean > self.fenv_config['dyp_area_tolerance'],
            'delta_area_std': lambda: delta_area_std > self.fenv_config['dyp_area_tolerance'] / 2,
            'delta_aspect_ratio_mean': lambda: delta_aspect_ratio_mean > self.fenv_config['dyp_aspect_ratios_tolerance'],
            'delta_aspect_ratio_std': lambda: delta_aspect_ratio_std > self.fenv_config['dyp_aspect_ratios_tolerance'] / 2
        }

        if self.fenv_config['dyp_check_geom_violation']:
            is_geom_condition_violated = any(
                condition_checks[feature]() 
                for feature in self.dyp_geom_condition_features 
                if feature in condition_checks
            )
        else:
            is_geom_condition_violated = False
            
        if self.fenv_config['dyp_check_lvroom_entrance_topo_violation']:
            is_lvroom_entrance_topo_condition_violated = (
                ['d', 11] not in self.plan_data_dict['edge_color_data_dict_entrance']['green']
            ) 
        else:
            is_lvroom_entrance_topo_condition_violated = False

        if is_lvroom_entrance_topo_condition_violated:
            raise ValueError('dyp_lvroom_entrance_topo_condition_violation_negative_reward must be always False')

        if self.fenv_config['dyp_check_topo_violation']:
            is_topo_condition_violated = (
                delta_edge_list > self.fenv_config['dyp_edge_tolerance']
            )
        else:
            is_topo_condition_violated = False

        if is_geom_condition_violated:
            reward = self.fenv_config['dyp_geom_violation_negative_reward'] 
            well_finished_condition = False
            
        elif is_lvroom_entrance_topo_condition_violated:
            reward = self.fenv_config['dyp_lvroom_entrance_topo_condition_violation_negative_reward']
            well_finished_condition = False

        elif is_topo_condition_violated:
            reward = self.fenv_config['dyp_topo_violation_negative_reward']
            well_finished_condition = False

        else:
            # Define a dictionary mapping features to their condition checks for well_finished
            well_finished_checks = {
                'delta_area_mean': lambda: delta_area_mean <= self.fenv_config['dyp_well_finished_condition']['delta_area_threshold'],
                'delta_aspect_ratio_mean': lambda: delta_aspect_ratio_mean <= self.fenv_config['dyp_well_finished_condition']['delta_aspect_ratio_threshold'],
                'delta_edge_list': lambda: delta_edge_list <= self.fenv_config['dyp_well_finished_condition']['delta_edge_threshold']
            }
            well_finished_condition = all(
                well_finished_checks[feature]()
                for feature in self.dyp_well_finished_condition_features
                if feature in well_finished_checks
            )

            if well_finished_condition: # I added delta_edge_list to this condition on Aug 16, 2024
                reward += self.fenv_config['dyp_reward_positive_constant_terminal']
                if delta_edge_list == 0:
                    reward += (2 * self.fenv_config['dyp_bonus_reward'])
                elif delta_edge_list <= self.fenv_config['dyp_well_finished_condition_relaxation_factor_for_edge']:
                    if delta_edge_list == 1:
                        reward += (1.75 * self.fenv_config['dyp_bonus_reward'])
                    elif delta_edge_list == 2:
                        reward += (1.5 * self.fenv_config['dyp_bonus_reward'])
                    elif delta_edge_list == 3:
                        reward += (1.25 * self.fenv_config['dyp_bonus_reward'])
                    elif delta_edge_list == 4:
                        reward += (1 * self.fenv_config['dyp_bonus_reward'])
                    else:
                        reward = reward
                else:
                    reward = reward # (1 * self.fenv_config['dyp_bonus_reward'])
            else:
                if self.fenv_config['dyp_activate_simple_reward']:
                    reward = self.fenv_config['dyp_simple_negative_reward']

        # Store stats for debugging
        stats = [
            delta_area_mean, delta_area_std, delta_area_max,
            delta_aspect_ratio_mean, delta_aspect_ratio_std, delta_aspect_ratio_max,
            delta_edge_list,
            reward_area_mean, reward_area_std, reward_area_max,
            reward_aspect_ratio_mean, reward_aspect_ratio_std, reward_aspect_ratio_max,
            reward_edge,
            is_geom_condition_violated, is_lvroom_entrance_topo_condition_violated,
            is_topo_condition_violated, well_finished_condition, 
            reward,
            min_achieved_area, max_achieved_area, max_achieved_aspect_ratio,
        ]
        self.store_stats(stats)
        return reward, well_finished_condition
    
    
    
    def _get_simple_min_max_reward(self):
        min_achieved_area, max_achieved_area, max_achieved_aspect_ratio = self._get_min_max_geom()
        
        ( delta_area_mean, delta_area_std, delta_area_max, 
          delta_aspect_ratio_mean, delta_aspect_ratio_std, delta_aspect_ratio_max, 
          delta_edge_list ) = self._get_misfit_stats()
        
        ( reward_area_mean, reward_area_std, reward_area_max, 
          reward_aspect_ratio_mean, reward_aspect_ratio_std, reward_aspect_ratio_max, 
          reward_edge, reward) = self._map_misfit_to_reward(
                delta_area_mean, delta_area_std, delta_area_max, 
                delta_aspect_ratio_mean, delta_aspect_ratio_std, delta_aspect_ratio_max, 
                delta_edge_list)
        
        if self.fenv_config['dyp_check_geom_violation']:
            is_geom_condition_violated = (
                min_achieved_area < self.dyp_min_achieved_area_to_accept or
                max_achieved_area > self.dyp_max_achieved_area_to_accept or
                max_achieved_aspect_ratio > self.dyp_max_achieved_aspect_ratio_to_accept
            )
        else:
            is_geom_condition_violated = False

        if self.fenv_config['dyp_check_lvroom_entrance_topo_violation']:
            is_lvroom_entrance_topo_condition_violated = (
                ['d', 11] not in self.plan_data_dict['edge_color_data_dict_entrance']['green']
            ) 
        else:
            is_lvroom_entrance_topo_condition_violated = False

        if is_lvroom_entrance_topo_condition_violated:
            raise ValueError('dyp_lvroom_entrance_topo_condition_violation_negative_reward must be always False')

        if self.fenv_config['dyp_check_topo_violation']:
            is_topo_condition_violated = (
                delta_edge_list > self.fenv_config['dyp_edge_tolerance']
            )
        else:
            is_topo_condition_violated = False


        if is_geom_condition_violated:
            reward = self.fenv_config['dyp_geom_violation_negative_reward'] 
            well_finished_condition = False
            
        elif is_lvroom_entrance_topo_condition_violated:
            reward = self.fenv_config['dyp_lvroom_entrance_topo_condition_violation_negative_reward']
            well_finished_condition = False

        elif is_topo_condition_violated:
            reward = self.fenv_config['dyp_topo_violation_negative_reward']
            well_finished_condition = False

        else:
            well_finished_condition = (
                delta_edge_list <= self.fenv_config['dyp_well_finished_condition']['delta_edge_threshold']
            )

            if well_finished_condition: # I added delta_edge_list to this condition on Aug 16, 2024
                reward += self.fenv_config['dyp_reward_positive_constant_terminal']
                if delta_edge_list == 0:
                    reward += (2 * self.fenv_config['dyp_bonus_reward'])
                elif delta_edge_list <= self.fenv_config['dyp_well_finished_condition_relaxation_factor_for_edge']:
                    if delta_edge_list == 1:
                        reward += (1.75 * self.fenv_config['dyp_bonus_reward'])
                    elif delta_edge_list == 2:
                        reward += (1.5 * self.fenv_config['dyp_bonus_reward'])
                    elif delta_edge_list == 3:
                        reward += (1.25 * self.fenv_config['dyp_bonus_reward'])
                    elif delta_edge_list == 4:
                        reward += (1 * self.fenv_config['dyp_bonus_reward'])
                    else:
                        reward = reward
                else:
                    reward = reward # (1 * self.fenv_config['dyp_bonus_reward'])
            else:
                if self.fenv_config['dyp_activate_simple_reward']:
                    reward = self.fenv_config['dyp_simple_negative_reward']

        # Store stats for debugging
        stats = [
            delta_area_mean, delta_area_std, delta_area_max,
            delta_aspect_ratio_mean, delta_aspect_ratio_std, delta_aspect_ratio_max,
            delta_edge_list,
            reward_area_mean, reward_area_std, reward_area_max,
            reward_aspect_ratio_mean, reward_aspect_ratio_std, reward_aspect_ratio_max,
            reward_edge,
            is_geom_condition_violated, is_lvroom_entrance_topo_condition_violated,
            is_topo_condition_violated, well_finished_condition, 
            reward,
            min_achieved_area, max_achieved_area, max_achieved_aspect_ratio,
        ]
        self.store_stats(stats)
        
        return reward, well_finished_condition
        


    def _get_reward_per_fn(self, x, name, terminal_state=False):
        if name == 'area':
            x_start = self.fenv_config['dyp_x_start']
            x_end = self.fenv_config['dyp_x_end_area']
            y_start = self.fenv_config['dyp_y_start']
            y_end = self.fenv_config['dyp_y_end']
        elif name == 'aspect_ratio':
            x_start = self.fenv_config['dyp_x_start']
            x_end = self.fenv_config['dyp_x_end_aspect_ratio']
            y_start = self.fenv_config['dyp_y_start']
            y_end = self.fenv_config['dyp_y_end']       
        elif name == 'edge':
            x_start = self.fenv_config['dyp_x_start']
            x_end = self.fenv_config['dyp_x_end_edge']
            y_start = self.fenv_config['dyp_y_start']
            y_end = self.fenv_config['dyp_y_end']
        else:
            raise ValueError('Invalid name')

        # if terminal_state:
        #     y_start *= self.fenv_config['reward_vertical_scalar']
        #     y_end *= self.fenv_config['reward_vertical_scalar']

        reward_fn = self._get_reward_fn()
        reward = reward_fn(x, x_start, x_end, y_start, y_end)

        if self.fenv_config['dyp_shift_to_negative_y_end']:
            reward -= self.fenv_config['dyp_y_end']

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
    
    
    
