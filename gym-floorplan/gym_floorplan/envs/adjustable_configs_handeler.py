#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 02:05:20 2022

@author: RK
"""
import os
from pathlib import Path

import ast
import json
import numpy as np
import pandas as pd
from pprint import pprint

#%%
class AdjustableConfigsHandeler:
    def __init__(self, fenv_config):
        self.fenv_config = fenv_config        
        
        self.plans_df = None
        
        self.adjustable_configs = self.get_configs()
        
        self.temp_df = pd.DataFrame()
        
    
    def get_configs(self, episode=None):
        # print(f"In _load_random_configs of adjustable_configs_handeler episode:{episode}")
        if self.fenv_config['plan_config_source_name'] == 'fixed_test_config':
            adjustable_configs = self._get_fixed_test_configs(n_rooms=self.fenv_config['n_rooms'])
            
        elif self.fenv_config['plan_config_source_name'] in ['create_fixed_config', 'create_random_config']:
            adjustable_configs = self._create_random_configs()
            
        elif self.fenv_config['plan_config_source_name'] == 'offline_mode':
            adjustable_configs = self._get_from_offline_config(episode=episode)
        
        elif self.fenv_config['plan_config_source_name'] in ['load_fixed_config', 'load_random_config']:
            adjustable_configs = self._load_random_configs(episode=episode)
            
        elif self.fenv_config['plan_config_source_name'] == 'longer_training_config':
            adjustable_configs = self._load_longer_training_configs()
            
        elif self.fenv_config['plan_config_source_name'] == 'inference_mode':
            adjustable_configs = self._load_for_inference_mode(episode=episode)
            
        return adjustable_configs

    
    def _get_fixed_test_configs(self, n_rooms=3):
        if n_rooms == 4:
            n_corners = 4
            mask_numbers = 4
            masked_corners = ['corner_00', 'corner_01', 'corner_10', 'corner_11']
        
            mask_lengths = [8, 4, 2, 6]
            mask_widths = [8, 6, 6, 6]
        
            masked_area = sum([(L+1)*(W+1) for L, W in zip(mask_lengths, mask_widths)])
        
            fixed_desired_areas = [124, 49, 44, 38]
            fixed_desired_areas = np.sort(fixed_desired_areas)[::-1]
            desired_areas = {f"room_{n_corners+1+i}": area for i, area in enumerate(fixed_desired_areas)}
            desired_edge_list = [[1, 2], [2, 3], [2, 4], [3, 4], [1, 4]]
            desired_edge_list = [[min(edge), max(edge)] for edge in desired_edge_list]
            
            room_i_per_size_category, room_area_per_size_category = self._cluster_room_sizes(n_corners, desired_areas)
        
        if n_rooms == 5:
            n_corners = 4
            mask_numbers = 2
            masked_corners = ['corner_01', 'corner_11']
        
            mask_lengths = [2, 2]
            mask_widths = [4, 4]
        
            masked_area = sum([(L+1)*(W+1) for L, W in zip(mask_lengths, mask_widths)])
        
            fixed_desired_areas = [117, 77, 60, 37, 120]
            fixed_desired_areas = np.sort(fixed_desired_areas)[::-1]
            desired_areas = {f"room_{n_corners+1+i}": area for i, area in enumerate(fixed_desired_areas)}
            desired_edge_list = [[4, 5], [1, 5], [2, 4], [3, 4], [2, 3]]
            desired_edge_list = [[min(edge), max(edge)] for edge in desired_edge_list]
            
            room_i_per_size_category, room_area_per_size_category = self._cluster_room_sizes(n_corners, desired_areas)
        
        if n_rooms == 5:
            n_corners = 4
            mask_numbers = 3
            masked_corners = ['corner_00', 'corner_01', 'corner_10']
        
            mask_lengths = [6, 2, 8]
            mask_widths = [4, 8, 8]
        
            masked_area = sum([(L+1)*(W+1) for L, W in zip(mask_lengths, mask_widths)])
        
            fixed_desired_areas = [83, 78, 60, 44, 33]
            fixed_desired_areas = np.sort(fixed_desired_areas)[::-1]
            desired_areas = {f"room_{n_corners+1+i}": area for i, area in enumerate(fixed_desired_areas)}
            desired_edge_list = [[1, 2], [1, 5], [3, 4]]
            desired_edge_list = [[min(edge), max(edge)] for edge in desired_edge_list]
            
            room_i_per_size_category, room_area_per_size_category = self._cluster_room_sizes(n_corners, desired_areas)
        
        if n_rooms == 7:
            n_corners = 4
            mask_numbers = 1
            masked_corners = ['corner_01']
            mask_lengths = [8]
            mask_widths = [4]
            
            masked_area = sum([(L+1)*(W+1) for L, W in zip(mask_lengths, mask_widths)])
            fixed_desired_areas = [37, 53, 53, 52, 34, 33, 70]
            fixed_desired_areas = np.sort(fixed_desired_areas)[::-1]
            desired_areas = {f"room_{n_corners+1+i}": area for i, area in enumerate(fixed_desired_areas)}
            desired_edge_list = []
            
            room_i_per_size_category, room_area_per_size_category = self._cluster_room_sizes(n_corners, desired_areas)
        
        if n_rooms == 7:
            n_corners = 4
            mask_numbers = 3
            masked_corners = ['corner_00', 'corner_01', 'corner_11']
        
            mask_lengths = [4, 4, 8]
            mask_widths = [8, 4, 8]
        
            masked_area = sum([(L+1)*(W+1) for L, W in zip(mask_lengths, mask_widths)])
        
            fixed_desired_areas = [36, 30, 65, 28, 23, 48, 60]
            fixed_desired_areas = np.sort(fixed_desired_areas)[::-1]
            desired_areas = {f"room_{n_corners+1+i}": area for i, area in enumerate(fixed_desired_areas)}
            desired_edge_list = [[2, 3], [1, 5], [6, 7], [3, 4]]
            desired_edge_list = [[min(edge), max(edge)] for edge in desired_edge_list]
            
            room_i_per_size_category, room_area_per_size_category = self._cluster_room_sizes(n_corners, desired_areas)
        
        if n_rooms == 6:
            n_corners = 4
            mask_numbers = 2
            masked_corners = ['corner_00', 'corner_11']
        
            mask_lengths = [8, 2]
            mask_widths = [6, 2]
        
            masked_area = sum([(L+1)*(W+1) for L, W in zip(mask_lengths, mask_widths)])
        
            fixed_desired_areas = [84, 42, 78, 75, 34, 56]
            fixed_desired_areas = np.sort(fixed_desired_areas)[::-1]
            desired_areas = {f"room_{n_corners+1+i}": area for i, area in enumerate(fixed_desired_areas)}
            desired_edge_list = [[1.0, 2.0], [2.0, 4.0], [2.0, 5.0], [2.0, 6.0], [1.0, 5.0], [3.0, 6.0], [4.0, 6.0], [1.0, 3.0], [1.0, 4.0], [3.0, 4.0]]
            desired_edge_list = [[min(edge), max(edge)] for edge in desired_edge_list]
            
            room_i_per_size_category, room_area_per_size_category = self._cluster_room_sizes(n_corners, desired_areas)
        
        if n_rooms == 8:
            n_corners = 4
            mask_numbers = 1
            masked_corners = ['corner_11']
        
            mask_lengths = [4]
            mask_widths = [8]
        
            masked_area = sum([(L+1)*(W+1) for L, W in zip(mask_lengths, mask_widths)])
        
            fixed_desired_areas = [50, 85, 41, 28, 64, 32, 32, 64]
            fixed_desired_areas = np.sort(fixed_desired_areas)[::-1]
            desired_areas = {f"room_{n_corners+1+i}": area for i, area in enumerate(fixed_desired_areas)}
            desired_edge_list = [[2, 8], [4, 6], [1, 7], [2, 7]]#[[5, 6], [4, 6], [6, 7], [1, 4], [7, 8], [2, 8], [4, 5], [4, 8], [1, 3], [2, 7], [1, 8]]
            desired_edge_list = [[min(edge), max(edge)] for edge in desired_edge_list]
            
            room_i_per_size_category, room_area_per_size_category = self._cluster_room_sizes(n_corners, desired_areas)
        
        
        if n_rooms == 9:
            n_corners = 4
            mask_numbers = 4
            masked_corners = ['corner_00', 'corner_01', 'corner_10', 'corner_11']
        
            mask_lengths = [4, 6, 2, 8]	
            mask_widths = [2, 6, 4, 4]
        
            masked_area = sum([(L+1)*(W+1) for L, W in zip(mask_lengths, mask_widths)])
        
            fixed_desired_areas = [45.0, 44.0, 43.0, 42.0, 40.0, 37.0, 29.0, 26.0, 11.0]
            fixed_desired_areas = np.sort(fixed_desired_areas)[::-1]
            desired_areas = {f"room_{n_corners+1+i}": area for i, area in enumerate(fixed_desired_areas)}
            desired_edge_list = [[1.0, 2.0], [1.0, 4.0], [1.0, 8.0], [4.0, 5.0], [5.0, 6.0], [5.0, 7.0], [5.0, 9.0], [4.0, 7.0], [2.0, 8.0], [4.0, 8.0], [3.0, 6.0], [4.0, 6.0]]
            desired_edge_list = [[min(edge), max(edge)] for edge in desired_edge_list]
            
            room_i_per_size_category, room_area_per_size_category = self._cluster_room_sizes(n_corners, desired_areas)
            
        if n_rooms == 9:
            n_corners = 4
            mask_numbers = 3
            masked_corners = ['corner_00', 'corner_01', 'corner_11']
        
            mask_lengths = [6, 8, 8]
            mask_widths = [6, 8, 4]
        
            masked_area = sum([(L+1)*(W+1) for L, W in zip(mask_lengths, mask_widths)])
        
            fixed_desired_areas = [41.0, 38.0, 37.0, 30.0, 30.0, 27.0, 24.0, 23.0, 16.0]
            fixed_desired_areas = np.sort(fixed_desired_areas)[::-1]
            desired_areas = {f"room_{n_corners+1+i}": area for i, area in enumerate(fixed_desired_areas)}
            desired_edge_list = [[1.0, 2.0], [1.0, 4.0], [1.0, 8.0], [4.0, 5.0], [5.0, 6.0], [5.0, 7.0], [5.0, 9.0], [4.0, 7.0], [2.0, 8.0], [4.0, 8.0], [3.0, 6.0], [4.0, 6.0]]
            desired_edge_list = [[min(edge), max(edge)] for edge in desired_edge_list]
            
            room_i_per_size_category, room_area_per_size_category = self._cluster_room_sizes(n_corners, desired_areas)
            
            
            
        adjustable_configs = {
            'plan_config_source_name': 'fixed_test_configs',
            'n_walls': n_rooms-1,
            'n_rooms': n_rooms,
            'n_corners': 4,
            'mask_numbers': mask_numbers,
            'number_of_total_walls': n_rooms-1 + n_corners,
            'number_of_total_rooms': n_rooms + n_corners,
            'masked_corners': masked_corners,
            'mask_lengths': mask_lengths,
            'mask_widths': mask_widths,
            'masked_area': masked_area,
            'desired_areas': desired_areas,
            'room_i_per_size_category': room_i_per_size_category,
            'room_area_per_size_category': room_area_per_size_category,
            'desired_edge_list': desired_edge_list,            
            'adj_vec_desired': self._get_desired_adj_vector(desired_edge_list)
            }
        return adjustable_configs
            

    def _create_random_configs(self):
        if self.fenv_config['n_rooms'] == 'X':
            n_rooms = np.random.choice(self.fenv_config['number_of_rooms_range'], size=1, replace=True).tolist()[0]
        else:
            n_rooms = self.fenv_config['n_rooms']

        n_walls = n_rooms - 1            
        
        blackbox_min_length = 1
        blackbox_max_length = int(self.fenv_config['max_x']/4)
        blackbox_min_width = 1
        blackbox_max_width = int(self.fenv_config['max_y']/4)
        
        while True:
            n_corners = 4
            mask_numbers = np.random.randint(4) + 1
            masked_corners = np.random.choice(list(self.fenv_config['corners'].keys()), size=mask_numbers, replace=False).tolist()
            
            mask_lengths = np.array(np.random.randint(blackbox_min_length, blackbox_max_length, size=mask_numbers)*2, dtype=int).tolist()
            mask_widths = np.array(np.random.randint(blackbox_min_width, blackbox_max_width, size=mask_numbers)*2, dtype=int).tolist()
            
            masked_area = np.sum([(L+1)*(W+1) for L, W in zip(mask_lengths, mask_widths)])
            if masked_area <= self.fenv_config['total_area'] / 2:
                break
        
        desired_areas = self._configure_areas(n_rooms, n_corners, masked_area)
        room_i_per_size_category, room_area_per_size_category = self._cluster_room_sizes(n_corners, desired_areas)
        
        adjustable_configs = {
            'plan_config_source_name': 'create_random_config',
            'n_walls': n_walls,
            'n_rooms': n_rooms,
            'n_corners': n_corners,
            'mask_numbers': mask_numbers,
            'number_of_total_walls': n_walls + n_corners,
            'number_of_total_rooms': n_rooms + n_corners,
            'masked_corners': masked_corners,
            'mask_lengths': mask_lengths,
            'mask_widths': mask_widths,
            'masked_area': masked_area,
            'desired_areas': desired_areas,
            'room_i_per_size_category': room_i_per_size_category,
            'room_area_per_size_category': room_area_per_size_category,
            'desired_edge_list': [],  
            'adj_vec_desired': [],
            }
        
        return adjustable_configs
    
    
    def _create_configs_manually(self):
        n_corners = 4
        import itertools
        a = 0
        b = 0
        X = [2, 4, 6, 8]
        for nr in range(4, 10):
            for mn in range(1, 5):
                mls = mws = [p for p in itertools.product(X, repeat=mn)]
                for ml in mls:
                    for mw in mws:
                        b += 1
                        masked_area = np.sum([(L+1)*(W+1) for L, W in zip(ml, mw)])
                        if masked_area <= self.fenv_config['total_area'] / 2:
                            a+=1
        from sympy.utilities.iterables import multiset_permutations
        X = [2, 4, 6, 8]
        for p in multiset_permutations(X, ):
            print(p)
    
    
    
    def _load_random_configs(self, episode=None):
        self.plans_df = self._load_plans()
        
        if episode is None:
            this_plan_df = self.plans_df.sample()
        else:
            this_plan_df = self.plans_df.iloc[[int(episode % len(self.plans_df))]]
            
        adjustable_configs = self._get_adjustable_configs(this_plan_df)
        
        return adjustable_configs
    
    
    def _get_from_offline_config(self, episode):
        self.plans_df = self._load_plans()
        
        if episode is None:
            this_plan_df = self.plans_df.sample()
        else:
            this_plan_df = self.plans_df.iloc[[int(episode % len(self.plans_df))]]
            # if episode in [  3,  10,  12,  13,  20,  25,  33,  34,  35,  43,  48,  54,  59,
            #                           62,  64,  68,  71,  87,  98, 106, 118, 119, 121, 139, 144, 146,
            #                           149, 150, 151, 155, 156, 160, 166, 168, 169]:
            #     self.temp_df = pd.concat([self.temp_df, this_plan_df], axis=0)
        adjustable_configs = self._get_adjustable_configs(this_plan_df)
        
        return adjustable_configs      
    
    
    def _load_for_inference_mode(self, episode):
        self.plans_df = self._load_plans()
        
        if episode is None:
            this_plan_df = self.plans_df.sample()
        else:
            this_plan_df = self.plans_df.iloc[[int(episode % len(self.plans_df))]]
            
        adjustable_configs = self._get_adjustable_configs(this_plan_df)
        
        return adjustable_configs 
        
            
    def _load_longer_training_configs(self):
        if self.fenv_config['learner_name'] == 'trainer':
            if self.fenv_config['agent_first_name'] == 'dqn':
                base_chkpt_dir = os.path.realpath(Path(self.fenv_config['chkpt_path']).parents[1])
            else:
                base_chkpt_dir = os.path.realpath(Path(self.fenv_config['chkpt_path']).parents[2])
        else:
            if self.fenv_config['agent_first_name'] == 'dqn':
                base_chkpt_dir = os.path.realpath(Path(self.fenv_config['chkpt_path']).parents[1])
            else:
                base_chkpt_dir = os.path.realpath(Path(self.fenv_config['chkpt_path']).parents[2])
            
        configs_for_longer_training_npy_path = os.path.join(base_chkpt_dir, "configs/configs_for_longer_training.json")
        
        with open(configs_for_longer_training_npy_path) as f:
            longer_config = json.load(f)#, allow_pickle=True).tolist()
        
        n_corners = 4
        n_walls = longer_config['n_walls']
        n_rooms = longer_config['n_rooms']
        mask_numbers = longer_config['mask_numbers']
        masked_corners = longer_config['masked_corners']
        mask_lengths = longer_config['mask_lengths']
        mask_widths = longer_config['mask_widths']
        masked_area = longer_config['masked_area']
        desired_areas = longer_config['desired_areas']
        desired_edge_list = longer_config['desired_edge_list']
        potential_good_action_sequence = longer_config['potential_good_action_sequence']
        
        room_i_per_size_category, room_area_per_size_category = self._cluster_room_sizes(n_corners, desired_areas)
        
        adjustable_configs = {
            'plan_config_source_name': 'fixed_test_configs',
            'n_walls': n_walls,
            'n_rooms': n_rooms,
            'mask_numbers': mask_numbers,
            'number_of_total_walls': n_walls + n_corners,
            'number_of_total_rooms': n_rooms + n_corners,
            'masked_corners': masked_corners,
            'mask_lengths': mask_lengths,
            'mask_widths': mask_widths,
            'masked_area': masked_area,
            'desired_areas': desired_areas,
            'room_i_per_size_category': room_i_per_size_category,
            'room_area_per_size_category': room_area_per_size_category,
            'desired_edge_list': desired_edge_list,   
            'potential_good_action_sequence': potential_good_action_sequence,
            'adj_vec_desired': self._get_desired_adj_vector(desired_edge_list),
            }
        
        return adjustable_configs
            
            
    def _load_plans(self):
        if self.plans_df is None:
            if self.fenv_config['plan_config_source_name'] == 'inference_mode':
                plans_df_path = os.path.join(self.fenv_config['the_scenario_dir'], 'plans_df.csv')
                plans_df = pd.read_csv(plans_df_path)
                # plans_df = pd.read_csv(self.fenv_config['plan_path'])
                    
            elif self.fenv_config['plan_config_source_name'] in ['load_fixed_config', 'load_random_config', 'offline_mode']:
                if self.fenv_config['nrows_of_env_data_csv'] is not None:
                    plans_df = pd.read_csv(self.fenv_config['plan_path'], nrows=self.fenv_config['nrows_of_env_data_csv'])
                else:
                    plans_df = pd.read_csv(self.fenv_config['plan_path'])
                    if self.fenv_config['fixed_num_rooms'] is not None:
                        plans_df = plans_df.loc[plans_df['n_rooms'] == self.fenv_config['fixed_num_rooms']]
                        plans_df = plans_df.reset_index()
                
                # try:
                #     plans_df = pd.read_csv(f"{self.fenv_config['the_scenario_dir']}/optim_exp_df_{self.fenv_config['scenario_name']}.csv")
                # except:
                if self.fenv_config['plan_config_source_name'] == 'offline_mode':
                    plans_df = pd.read_csv(f"{self.fenv_config['the_scenario_dir']}/plans_df.csv")
                        
            plans_df = self._ast_literal_eval(plans_df)
            if 'corner' not in str(plans_df.loc[0, 'masked_corners'][0]):
                plans_df['masked_corners'] = plans_df['masked_corners'].apply(lambda x:self._adjust_corner_name(x))
        else:
            plans_df = self.plans_df
            
        return plans_df
            
            
    def _ast_literal_eval(self, df):
        cols_to_ast = ['masked_corners', 'mask_lengths', 'mask_widths', 
                       'desired_areas', 
                       # 'areas', 
                       'desired_edge_list', 
                       # 'edge_list',
                       # 'accepted_action_sequence'
                       ]
        for col in cols_to_ast:
            df[col] = df[col].apply(ast.literal_eval)
        return df
      
    
    def _adjust_corner_name(self, cc):
        if not isinstance(cc, list):
            cc = ast.literal_eval(cc)
        cc_ = []
        for c in cc:
            cc_.append(f"corner_{int(c):02}")
        return cc_

        
    def _get_adjustable_configs(self, this_plan_df):
        n_corners = 4
        adjustable_configs = {}
        adjustable_configs['n_corners'] = n_corners
        adjustable_configs['plan_config_source_name'] = self.fenv_config['plan_config_source_name']
        for col in this_plan_df.columns:
            if col in ['n_rooms', 'mask_numbers', 'mask_lengths', 'mask_widths', 'desired_areas']:
                adjustable_configs[col] = np.asarray(this_plan_df[col].values.tolist(), dtype=int).tolist()[0]
         
        adjustable_configs['n_walls'] = adjustable_configs['n_rooms'] - 1
        
        adjustable_configs['masked_corners'] = this_plan_df['masked_corners'].values.tolist()[0]
        
        adjustable_configs['number_of_total_walls'] = adjustable_configs['n_walls'] + n_corners # adjustable_configs['mask_numbers']
        adjustable_configs['number_of_total_rooms'] = adjustable_configs['n_rooms'] + n_corners # adjustable_configs['mask_numbers']
        
        adjustable_configs['desired_areas'] = np.sort(adjustable_configs['desired_areas'])[::-1]
        adjustable_configs['desired_areas'] = {f"room_{n_corners+i+1}": area for i, area in enumerate(adjustable_configs['desired_areas'])}
        
        desired_edge_list = np.array(this_plan_df['desired_edge_list'].values.tolist()[0], dtype=int).tolist()
        adjustable_configs['desired_edge_list'] = desired_edge_list#[:int(0.75*len(desired_edge_list))]
        
        room_i_per_size_category, room_area_per_size_category = self._cluster_room_sizes(adjustable_configs['n_corners'], adjustable_configs['desired_areas'])
        adjustable_configs.update({
            'room_i_per_size_category': room_i_per_size_category,
            'room_area_per_size_category': room_area_per_size_category,
            })
        
        adjustable_configs['masked_area'] = sum([(L+1)*(W+1) for L, W in zip(adjustable_configs['mask_lengths'], adjustable_configs['mask_widths'])])
        
        adjustable_configs['potential_good_action_sequence'] = list(np.array(this_plan_df['accepted_action_sequence'].apply(lambda x: ast.literal_eval(x))))[0] # this_plan_df['accepted_action_sequence'].apply(lambda x: ast.literal_eval(x)) # np.array(this_plan_df['accepted_action_sequence'].values.tolist()[0], dtype=int).tolist()
        
        adjustable_configs['adj_vec_desired'] = self._get_desired_adj_vector(desired_edge_list)
        
        return adjustable_configs
    
    
    def _get_desired_adj_vector(self, desired_edge_list):
        adj_mat_desired = np.zeros((self.fenv_config['maximum_num_real_rooms'], self.fenv_config['maximum_num_real_rooms']))
        
        desired_edge_list = np.array(desired_edge_list, dtype=int)
        for dedge in desired_edge_list:
            adj_mat_desired[dedge[0]-1][dedge[1]-1] = 1
            adj_mat_desired[dedge[1]-1][dedge[0]-1] = 1
            
        adj_vec_desired = []
        for i, row in enumerate(list(adj_mat_desired)):
            adj_vec_desired.extend(row[i+1:])  
        
        return adj_vec_desired
                
        
    def _configure_areas(self, n_rooms, n_corners, masked_area):
        if self.fenv_config['plan_config_source_name'] in ['create_fixed_config', 'create_random_config']:
            free_area = self.fenv_config['total_area'] - masked_area
            middle_area = np.floor(free_area / n_rooms)
            min_area = np.floor(middle_area/2)
            max_area = min_area + middle_area
            while True:
                areas_config_ = [list(np.random.randint(min_area, max_area, 1)/1.0)[0] for i in range(n_corners, n_rooms-1+n_corners)]
                sum_areas_except_last_room = np.sum(areas_config_)
                last_room_area = free_area - sum_areas_except_last_room
                if (last_room_area >= 10) and (last_room_area >= min_area) and (last_room_area <= max_area):
                    break
            areas_config_.append(last_room_area)
            areas_config_ = np.sort(areas_config_)[::-1]
            areas_config = {f'room_{i+1}': areas_config_[i-n_corners] for i in range(n_corners, n_rooms+n_corners)}
            assert len(areas_config) == n_rooms, "area_configs does not include proper number of rooms"
            assert sum(areas_config_) == free_area, "sum of the all areas must be equal to the free area"
        else:
            raise ValueError("We dont need to call _configure_areas for this case.")
            
        if any(a <= 0 for a in areas_config.values()):
            raise ValueError('Area must be larger than 1.')
            
        return areas_config
    
    
    def _cluster_room_sizes(self, n_corners, desired_areas):
        if isinstance(desired_areas, dict):
            desired_areas = list(desired_areas.values())
        else:
            desired_areas = list(desired_areas)
            raise ValueError('in _cluster_room_sizes of adjustable_config_handeler desiered_areas is a list')
            
        diff = np.abs(np.diff(desired_areas))
        max_ind_2, max_ind_1 = np.argsort(diff, axis=0)[-2:]
        max_ind_1 += 1
        max_ind_2 += 1
        
        left = min(max_ind_1, max_ind_2)
        right = max(max_ind_1, max_ind_2)
        
        room_i_list = np.array(range(len(desired_areas))) + 1 + n_corners
        
        large_ids = room_i_list[:left]
        medium_ids = room_i_list[left:right]
        small_ids = room_i_list[right:]

        large = desired_areas[:left]
        medium = desired_areas[left:right]
        small = desired_areas[right:]
        
        if len(large) == 0 or len(medium) == 0 or len(small) == 0:
            print("wait in _cluster_room_sizes of plan_constructor")
            raise ValueError('Room clusters cannot be empty')
        
        room_i_per_size_category = {'large': list(large_ids),
                                    'medium': list(medium_ids),
                                    'small': list(small_ids)}
        
        room_area_per_size_category = {'large': list(large),
                                       'medium': list(medium),
                                       'small': list(small)}
        
        return room_i_per_size_category, room_area_per_size_category


#%%
if __name__ == '__main__':
    from gym_floorplan.envs.fenv_config import LaserWallConfig
    fenv_config = LaserWallConfig().get_config()
    fenv_config['plan_config_source_name'] = 'fixed_test_config'
    self = AdjustableConfigsHandeler(fenv_config)
    print('- - - - - - - - - - adjustable_configs')
    pprint(self.adjustable_configs)
