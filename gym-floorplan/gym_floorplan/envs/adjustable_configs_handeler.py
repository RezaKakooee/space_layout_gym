#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 02:05:20 2022

@author: Reza Kakooee
"""


#%%
import os

import ast
import copy
import random
import numpy as np
import pandas as pd
from pprint import pprint

from gym_floorplan.envs.fixed_scenarios_lib import get_fixed_scenario




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
            adjustable_configs = self._create_env_configs()
            
        else:
            adjustable_configs = self._load_env_configs(episode=episode)
            
        return adjustable_configs

    

    def _get_fixed_test_configs(self, n_rooms=3):
        fixed_scenario_config = get_fixed_scenario(n_rooms)
        room_i_per_size_category, room_area_per_size_category = self._cluster_room_sizes(fixed_scenario_config['areas_desired'])
        adjustable_configs = {
            'n_walls': n_rooms-1,
            'n_rooms': n_rooms,
            'number_of_total_walls': n_rooms-1 + fixed_scenario_config['n_corners'],
            'room_i_per_size_category': room_i_per_size_category,
            'room_area_per_size_category': room_area_per_size_category,
            }
        adjustable_configs.update(fixed_scenario_config)
        return adjustable_configs
            
    

    def _create_env_configs(self):
        blackbox_min_length = 2
        blackbox_max_length = int(self.fenv_config['max_x']/4)
        blackbox_min_width = 2
        blackbox_max_width = int(self.fenv_config['max_y']/4)
        
        if self.fenv_config['create_random_configs_with_fixed_n_rooms_flag']:
            n_rooms = self.fenv_config['create_random_configs_with_fixed_n_rooms_flag']
        else:
            if self.fenv_config['n_rooms'] == 'X':
                n_rooms = np.random.choice(self.fenv_config['number_of_rooms_range'], size=1, replace=True).tolist()[0]
            else:
                n_rooms = self.fenv_config['n_rooms']
    
        n_walls = n_rooms - 1            
        
        while True:
            n_corners = 4
            
            if self.fenv_config['fixed_outline_flag']:
                mask_numbers = 4
                masked_corners = ['corner_00', 'corner_10', 'corner_01', 'corner_11']
                mask_lengths = [4, 2, 8, 6]
                mask_widths = [6, 2, 8, 2]
                
                mask_numbers = 2
                masked_corners = ['corner_00', 'corner_11']
                mask_lengths = [8, 8]	
                mask_widths = [6, 8]
                
            else:
                mask_numbers = np.random.randint(4) + 1
                masked_corners = np.random.choice(list(self.fenv_config['corners'].keys()), size=mask_numbers, replace=False).tolist()
                
                mask_lengths = np.array(np.random.randint(blackbox_min_length, blackbox_max_length, size=mask_numbers)*2, dtype=int).tolist()
                mask_widths = np.array(np.random.randint(blackbox_min_width, blackbox_max_width, size=mask_numbers)*2, dtype=int).tolist()
                
                # mask_data = {Co: [Le, Wi] for Co, Le, Wi in zip(masked_corners, mask_lengths, mask_widths)}
            
            mask_data = {Co: [Le, Wi] for Co, Le, Wi in zip(masked_corners, mask_lengths, mask_widths)}
            areas_masked_list = [(L+1)*(W+1) for L, W in zip(mask_lengths, mask_widths)]
            area_masked = np.sum(areas_masked_list)
            
            if area_masked <= self.fenv_config['total_area'] / 2:
                break
        
        n_facades_blocked = self.fenv_config['n_facades_blocked'] if self.fenv_config['n_facades_blocked'] else np.random.randint(2) + 1
        facades_blocked = np.random.choice(['n', 's', 'e', 'w'], size=n_facades_blocked, replace=False).tolist()
            
        entrance_coords, entrance_positions, entrance_is_on_facade, extended_entrance_positions, extended_entrance_coords = self._get_entrance_coords(mask_data, facades_blocked)
        
        areas_desired = self._configure_areas(n_rooms, area_masked, self.fenv_config['entrance_area'])
        room_i_per_size_category, room_area_per_size_category = self._cluster_room_sizes(areas_desired)
        
        
        if ( self.fenv_config['plan_config_source_name'] == 'create_random_config' and 
              (self.fenv_config['is_entrance_adjacency_a_constraint'] and
               self.fenv_config['is_entrance_lvroom_connection_a_constraint']) or
             self.fenv_config['zero_constraint_flag'] ):
            lvroom_id = self.fenv_config['min_room_id'] # take the biggest room as the lvroom
        else:
            lvroom_id = None
        
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
            'area_masked': area_masked,
            'areas_masked': {f"room_{i+1}":a for i, a in enumerate(areas_masked_list)},
            'areas_desired': areas_desired,
            'room_i_per_size_category': room_i_per_size_category,
            'room_area_per_size_category': room_area_per_size_category,
            'edge_list_room_desired': [],  
            'edge_list_facade_desired_str': [],
            'edge_list_entrance_desired_str': [],
            'entrance_coords': entrance_coords,
            'entrance_positions': entrance_positions,
            'extended_entrance_coords': extended_entrance_coords,
            'extended_entrance_positions': extended_entrance_positions,
            'entrance_area': self.fenv_config['entrance_area'],   
            'entrance_is_on_facade': entrance_is_on_facade,
            'n_facades_blocked': n_facades_blocked,
            'facades_blocked': facades_blocked,
            'lvroom_id': lvroom_id,
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
                        area_masked = np.sum([(L+1)*(W+1) for L, W in zip(ml, mw)])
                        if area_masked <= self.fenv_config['total_area'] / 2:
                            a+=1
        from sympy.utilities.iterables import multiset_permutations
        X = [2, 4, 6, 8]
        for p in multiset_permutations(X, ):
            print(p)
    
    
    
    def _load_env_configs(self, episode=None):
        self.plans_df = self._load_plans()
        
        if self.fenv_config['plan_id_for_load_fixed_config']:
            this_plan_df = self.plans_df.loc[self.plans_df['plan_id'] == self.fenv_config['plan_id_for_load_fixed_config']]
        else:
            if episode is None:
                this_plan_df = self.plans_df.sample()
            else:
                this_plan_df = self.plans_df.iloc[[int(episode % len(self.plans_df))]]
            
        adjustable_configs = self._get_adjustable_configs(this_plan_df)
        
        return adjustable_configs
    
    
            
    def _load_plans(self):
        # self.fenv_config['plan_path'] = '/home/rdbt/ETHZ/dbt_python/housing_design/storage_nobackup/rnd_agents_storage/Scn__2024_02_02_2222__CRC__XRr__EnZSQR__RND__Stable/plans_test.csv'
        if self.plans_df is None:
            if self.fenv_config['plan_config_source_name'] in ['imitation_mode', 'offline_mode']:
                plans_df = pd.read_csv(self.fenv_config['plan_path'])                
            elif self.fenv_config['plan_config_source_name'] in ['load_fixed_config', 'load_random_config']:
                    plans_df = pd.read_csv(self.fenv_config['plan_path'], nrows=self.fenv_config['nrows_of_env_data_csv'])
            else:
                raise ValueError(f"This is a wrong mode when you want to load saved plans! The current mode is {self.fenv_config['plan_config_source_name']}, and accepted ones are [imitation_mode, offline_mode, load_fixed_config, load_random_config] ")
            
            if self.fenv_config['fixed_num_rooms_for_loading']:
                plans_df = plans_df.loc[plans_df['n_rooms'] == self.fenv_config['fixed_num_rooms_for_loading']]
                plans_df = plans_df.reset_index()
                
            plans_df = self._ast_literal_eval(plans_df)
            
        else:
            plans_df = self.plans_df
            
        # print(f"*-*-*-*-*-*-*-*-*-*-*-*- Plan CSV loaded from: \n {self.fenv_config['plan_path']}")

        return plans_df
            
            
            
    def _ast_literal_eval(self, df):
        cols_to_ast = ['masked_corners', 'mask_lengths', 'mask_widths', 
                       'areas_desired', 
                       'edge_list_room_desired', 
                       'edge_list_facade_desired_str',
                       'edge_list_entrance_desired_str',
                       'entrance_positions',
                       'entrance_coords',
                       'extended_entrance_positions',
                       'extended_entrance_coords',
                       'facades_blocked',
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
        self.this_plan_df = this_plan_df
        n_corners = 4
        adjustable_configs = {}
        adjustable_configs['n_corners'] = n_corners
        adjustable_configs['plan_id'] = self.this_plan_df['plan_id'].values.tolist()[0]
        adjustable_configs['plan_config_source_name'] = self.fenv_config['plan_config_source_name']
        for col in this_plan_df.columns:
            if col in ['n_rooms', 
                       'mask_numbers', 'mask_lengths', 'mask_widths', 'area_masked', 
                       'entrance_positions', 'entrance_coords',
                       'extended_entrance_positions', 'extended_entrance_coords',
                       'n_facades_blocked', 'lvroom_id']:
                adjustable_configs[col] = np.asarray(this_plan_df[col].values.tolist(), dtype=int).tolist()[0]
         
        adjustable_configs['n_walls'] = adjustable_configs['n_rooms'] - 1
        
        adjustable_configs['masked_corners'] = this_plan_df['masked_corners'].values.tolist()[0]
        
        adjustable_configs['number_of_total_walls'] = adjustable_configs['n_walls'] + n_corners # adjustable_configs['mask_numbers']
        adjustable_configs['number_of_total_rooms'] = adjustable_configs['n_rooms'] + n_corners # adjustable_configs['mask_numbers']
        
        areas_masked = this_plan_df['areas_masked'].values.tolist()[0]
        adjustable_configs['areas_masked'] = areas_masked
        
        adjustable_configs['entrance_is_on_facade'] = this_plan_df['entrance_is_on_facade'].values.tolist()[0]
        
        adjustable_configs['facades_blocked'] = this_plan_df['facades_blocked'].values.tolist()[0]
        
        areas_desired = this_plan_df['areas_desired'].values.tolist()[0]
        adjustable_configs['areas_desired'] = areas_desired
        
        edge_list_room_desired = this_plan_df['edge_list_room_desired'].values.tolist()[0]
        edge_list_facade_desired_str = this_plan_df['edge_list_facade_desired_str'].values.tolist()[0]
        edge_list_entrance_desired_str = this_plan_df['edge_list_entrance_desired_str'].values.tolist()[0]
        
        edge_list_room_desired.sort()
        edge_list_facade_desired_str.sort()
        
        adjustable_configs['edge_list_room_desired'] = edge_list_room_desired
        adjustable_configs['edge_list_facade_desired_str'] = edge_list_facade_desired_str
        adjustable_configs['edge_list_entrance_desired_str'] = edge_list_entrance_desired_str
        
        room_i_per_size_category, room_area_per_size_category = self._cluster_room_sizes(adjustable_configs['areas_desired'])
        adjustable_configs.update({
            'room_i_per_size_category': room_i_per_size_category,
            'room_area_per_size_category': room_area_per_size_category,
            })
        
        adjustable_configs['potential_good_action_sequence'] = list(np.array(this_plan_df['accepted_action_sequence'].apply(lambda x: ast.literal_eval(x))))[0] # this_plan_df['accepted_action_sequence'].apply(lambda x: ast.literal_eval(x)) # np.array(this_plan_df['accepted_action_sequence'].values.tolist()[0], dtype=int).tolist()
        
        return adjustable_configs
    
    
    
    def _configure_areas(self, n_rooms, area_masked, entrance_area):
        if self.fenv_config['plan_config_source_name'] in ['create_fixed_config', 'create_random_config']:
            free_area = self.fenv_config['total_area'] - area_masked - entrance_area
            n_rooms_to_create = n_rooms - 1 # because the last room we dont create, as the last_room_areaa is free_area - other_areas
            if self.fenv_config['randomly_create_lvroom_first']:
                lvroom_portion = np.random.uniform(self.fenv_config['lvroom_portion_range'][0], self.fenv_config['lvroom_portion_range'][1])
                lvroom_area = int(lvroom_portion * free_area)
                free_area -= lvroom_area
                n_rooms_to_create -= 1
            
            middle_area = np.floor(free_area / n_rooms)
            min_area = np.floor(middle_area/2) if n_rooms == 9 else self.fenv_config['min_acceptable_area']
            max_area = np.floor(middle_area/2) + middle_area
            
            while True:
                areas_config_ = [list(np.random.randint(min_area, max_area, 1)/1.0)[0] for _ in range(n_rooms_to_create)]
                sum_areas_except_last_room = np.sum(areas_config_)
                last_room_area = free_area - sum_areas_except_last_room
                if (last_room_area >= self.fenv_config['area_inf']) and (last_room_area >= min_area) and (last_room_area <= max_area):
                    break
            areas_config_.append(last_room_area)
            if self.fenv_config['randomly_create_lvroom_first']:
                areas_config_.append(lvroom_area)
            areas_config_ = np.sort(areas_config_)[::-1]
            areas_config = {f"room_{i+self.fenv_config['min_room_id']}": a for i, a in enumerate(areas_config_)}
            assert len(areas_config) == n_rooms, "area_configs does not include proper number of rooms"
            assert (sum(areas_config_) == free_area+lvroom_area) if self.fenv_config['randomly_create_lvroom_first'] else (sum(areas_config_) == free_area), "sum of the all areas must be equal to the free area"
            if self.fenv_config['randomly_create_lvroom_first']: assert max(areas_config_) == lvroom_area, "lvroom should be the largest room"
        else:
            raise ValueError(f"We dont need to call _configure_areas for this mode. The current mode is {self.fenv_config['plan_config_source_name']} while the accepted ones are [create_fixed_config, create_random_config]")
            
        if any(a <= 0 for a in areas_config.values()):
            raise ValueError(f"Area must be larger than 1, while the areas here are: {areas_config}")
            
        return areas_config
    
    
    
    def _cluster_room_sizes(self, areas_desired):
        areas_desired_copy = copy.deepcopy(areas_desired)
        if not self.fenv_config['is_agent_allowed_to_create_lvroom']:
            lvroom_name = f"room_{self.fenv_config['lvroom_id']}"
            del areas_desired_copy[lvroom_name]
        if isinstance(areas_desired_copy, dict):
            areas_desired_list = list(areas_desired_copy.values())
        else:
            areas_desired_list = list(areas_desired_copy)
            # if self.fenv_config['plan_config_source_name'] != 'longer_training_config':
            raise ValueError(f"In _cluster_room_sizes of adjustable_config_handeler desiered_areas should be a list while the current type is {type(areas_desired)}")
            
        areas_desired_list = np.sort(areas_desired_list)[::-1]
        diff = np.abs(np.diff(areas_desired_list))
        max_ind_2, max_ind_1 = np.argsort(diff, axis=0)[-2:]
        max_ind_1 += 1
        max_ind_2 += 1
        
        left = min(max_ind_1, max_ind_2)
        right = max(max_ind_1, max_ind_2)
        
        room_i_arr = np.array(range(len(areas_desired_list))) + self.fenv_config['min_room_id']
        if not self.fenv_config['is_agent_allowed_to_create_lvroom']:
            room_i_arr +=1 #room_i_arr[room_i_arr != self.fenv_config['lvroom_id']]
        
        large_ids = room_i_arr[:left]
        medium_ids = room_i_arr[left:right]
        small_ids = room_i_arr[right:]

        large = areas_desired_list[:left]
        medium = areas_desired_list[left:right]
        small = areas_desired_list[right:]
        
        if len(large) == 0 or len(medium) == 0 or len(small) == 0:
            print("wait in _cluster_room_sizes of plan_constructor")
            raise ValueError(f"Room clusters cannot be empty, while the current clusters are: large: {large}, medium: {medium}, small:{small}")
        
        room_i_per_size_category = {'large': list(large_ids),
                                    'medium': list(medium_ids),
                                    'small': list(small_ids)}
        
        room_area_per_size_category = {'large': list(large),
                                       'medium': list(medium),
                                       'small': list(small)}
        
        return room_i_per_size_category, room_area_per_size_category
    
    
    
    def _get_entrance_coords(self, mask_data, facades_blocked):
        masked_corners = mask_data.keys()
        direction_list = list(set(['n', 's', 'w', 'e']).difference(set(facades_blocked)))
        entrance_is_on_facade = np.random.choice(direction_list, 1).tolist()[0]
        
        offset_minus_one = -1
        offset_plus_one = 1
        offset_minus_two = -2
        offset_plus_two = 2
        if entrance_is_on_facade == 's':
            if ('corner_00' in masked_corners) and ('corner_10' in masked_corners):
                interval = np.random.choice(['1', '2', '3'], 1)[0]
            elif ('corner_00' in masked_corners) and ('corner_10' not in masked_corners):
                interval = np.random.choice(['1', '23'], 1)[0] 
            elif ('corner_00' not in masked_corners) and ('corner_10' in masked_corners):
                interval = np.random.choice(['12', '3'], 1)[0]
            else:
                interval = '123'
                
                
            if interval == '1':
                ColL = np.random.choice( list(range(self.fenv_config['min_x']+offset_plus_one, mask_data['corner_00'][0]+offset_minus_one)) ) 
                ColR = ColL + 1
                RowD = self.fenv_config['max_y'] - mask_data['corner_00'][1] 
                RowU = RowD - 1
            elif interval == '2':
                ColL = np.random.choice(  list(range( mask_data['corner_00'][0]+offset_plus_one, self.fenv_config['max_x']-mask_data['corner_10'][0]+offset_minus_one))  )
                ColR = ColL + 1
                RowD = self.fenv_config['max_y'] 
                RowU = RowD - 1
            elif interval == '3':
                ColL = np.random.choice( list(range( self.fenv_config['max_x']-mask_data['corner_10'][0]+offset_plus_one, self.fenv_config['max_x']+offset_minus_one)) )
                ColR = ColL + 1
                RowD = self.fenv_config['max_y'] - mask_data['corner_10'][1] 
                RowU = RowD - 1
            elif interval == '12':
                ColL = np.random.choice( list(range(self.fenv_config['min_x']+offset_plus_one, self.fenv_config['max_x']-mask_data['corner_10'][0]+offset_minus_one)) ) 
                ColR = ColL + 1
                RowD = self.fenv_config['max_y'] 
                RowU = RowD - 1
            elif interval == '23':
                ColL = np.random.choice( list(range(mask_data['corner_00'][0]+offset_plus_one, self.fenv_config['max_x']+offset_minus_one)) ) 
                ColR = ColL + 1
                RowD = self.fenv_config['max_y'] 
                RowU = RowD - 1
            elif interval == '123':
                ColL = np.random.choice( list(range(self.fenv_config['min_x']+offset_plus_one, self.fenv_config['max_x']+offset_minus_one)) ) 
                ColR = ColL + 1
                RowD = self.fenv_config['max_y'] 
                RowU = RowD - 1
            else:
                raise ValueError('A wrong interval randomly selected! Not possible!')
                
            extended_entrance_positions = [
                [RowD, ColL],
                [RowD, ColR],
                
                [RowU, ColL],
                [RowU, ColR],
                ]
                
        elif entrance_is_on_facade == 'n':
            if ('corner_01' in masked_corners) and ('corner_11' in masked_corners):
                interval = np.random.choice(['1', '2', '3'], 1)[0]
            elif ('corner_01' in masked_corners) and ('corner_11' not in masked_corners):
                interval = np.random.choice(['1', '23'], 1)[0] 
            elif ('corner_01' not in masked_corners) and ('corner_11' in masked_corners):
                interval = np.random.choice(['12', '3'], 1)[0]
            else:
                interval = '123'
                
                
            if interval == '1':
                ColL = np.random.choice( list(range(self.fenv_config['min_x']+offset_plus_one, mask_data['corner_01'][0]+offset_minus_one)) ) 
                ColR = ColL + 1
                RowU = mask_data['corner_01'][1] 
                RowD = RowU + 1
            elif interval == '2':
                ColL = np.random.choice(  list(range( mask_data['corner_01'][0]+offset_plus_one, self.fenv_config['max_x']-mask_data['corner_11'][0]+offset_minus_one))  ) 
                ColR = ColL + 1
                RowU = self.fenv_config['min_y']
                RowD = RowU + 1
            elif interval == '3':
                ColL = np.random.choice( list(range( self.fenv_config['max_x']-mask_data['corner_11'][0]+offset_plus_one, self.fenv_config['max_x']+offset_minus_one)) ) 
                ColR = ColL + 1
                RowU = mask_data['corner_11'][1] 
                RowD = RowU + 1
            elif interval == '12':
                ColL = np.random.choice( list(range(self.fenv_config['min_x']+offset_plus_one, self.fenv_config['max_x']-mask_data['corner_11'][0]+offset_minus_one )) ) 
                ColR = ColL + 1
                RowU = self.fenv_config['min_y']
                RowD = RowU + 1
            elif interval == '23':
                ColL = np.random.choice( list(range(mask_data['corner_01'][0]+offset_plus_one, self.fenv_config['max_x']+offset_minus_one)) ) 
                ColR = ColL + 1
                RowU = self.fenv_config['min_y']
                RowD = RowU + 1
            elif interval == '123':
                ColL = np.random.choice( list(range(self.fenv_config['min_x']+offset_plus_one, self.fenv_config['max_x']+offset_minus_one)) ) 
                ColR = ColL + 1
                RowU = self.fenv_config['min_y']
                RowD = RowU + 1
            else:
                raise ValueError('A wrong interval randomly selected! Not possible!')
                
            extended_entrance_positions = [
                [RowU, ColL],
                [RowU, ColR],
                
                [RowD, ColL],
                [RowD, ColR],
                ]
            
            
        elif entrance_is_on_facade == 'w':
            if ('corner_00' in masked_corners) and ('corner_01' in masked_corners):
                interval = np.random.choice(['1', '2', '3'], 1)[0]
            elif ('corner_00' in masked_corners) and ('corner_01' not in masked_corners):
                interval = np.random.choice(['12', '3'], 1)[0] 
            elif ('corner_00' not in masked_corners) and ('corner_01' in masked_corners):
                interval = np.random.choice(['1', '23'], 1)[0]
            else:
                interval = '123'
                
            if interval == '1':
                RowU = np.random.choice( list(range(self.fenv_config['min_y']+offset_plus_one, mask_data['corner_01'][1]+offset_minus_one)) ) 
                RowD = RowU + 1
                ColL = self.fenv_config['min_x'] + mask_data['corner_01'][0] 
                ColR = ColL + 1
            elif interval == '2':
                RowU = np.random.choice(  list(range( mask_data['corner_01'][1]+offset_plus_one, self.fenv_config['max_y']-mask_data['corner_00'][1]+offset_minus_one))  ) 
                RowD = RowU + 1
                ColL = self.fenv_config['min_x']
                ColR = ColL + 1
            elif interval == '3':
                RowU = np.random.choice( list(range( self.fenv_config['max_y']-mask_data['corner_00'][1]+offset_plus_one, self.fenv_config['max_y']+offset_minus_one)) ) 
                RowD = RowU + 1
                ColL = self.fenv_config['min_x'] + mask_data['corner_00'][0] 
                ColR = ColL + 1
            elif interval == '12':
                RowU = np.random.choice( list(range(self.fenv_config['min_y']+offset_plus_one, self.fenv_config['max_y']-mask_data['corner_00'][1]+offset_minus_one)) ) 
                RowD = RowU + 1
                ColL = self.fenv_config['min_x']
                ColR = ColL + 1
            elif interval == '23':
                RowU = np.random.choice( list(range(mask_data['corner_01'][1]+offset_plus_one, self.fenv_config['max_y']+offset_minus_one)) ) 
                RowD = RowU + 1
                ColL = self.fenv_config['min_x']
                ColR = ColL + 1
            elif interval == '123':
                RowU = np.random.choice( list(range(self.fenv_config['min_y']+offset_plus_one, self.fenv_config['max_y']+offset_minus_one)) ) 
                RowD = RowU + 1
                ColL = self.fenv_config['min_x']
                ColR = ColL + 1
            else:
                raise ValueError('A wrong interval randomly selected! Not possible!')
            
            extended_entrance_positions = [
                [RowD, ColL],
                [RowU, ColL],
                
                [RowD, ColR],
                [RowU, ColR],
                ]
        
        elif entrance_is_on_facade == 'e':
            if ('corner_10' in masked_corners) and ('corner_11' in masked_corners):
                interval = np.random.choice(['1', '2', '3'], 1)[0]
            elif ('corner_10' in masked_corners) and ('corner_11' not in masked_corners):
                interval = np.random.choice(['12', '3'], 1)[0] 
            elif ('corner_10' not in masked_corners) and ('corner_11' in masked_corners):
                interval = np.random.choice(['1', '23'], 1)[0]
            else:
                interval = '123'
                
            if interval == '1':
                RowD = np.random.choice( list(range(self.fenv_config['min_y']+offset_plus_one, mask_data['corner_11'][1]+offset_minus_one)) ) 
                RowU = RowD + 1
                ColR = self.fenv_config['max_x'] - mask_data['corner_11'][0] 
                ColL = ColR - 1
            elif interval == '2':
                RowD = np.random.choice(  list(range( mask_data['corner_11'][1]+offset_plus_one, self.fenv_config['max_y']-mask_data['corner_10'][1]+offset_minus_one))  ) 
                RowU = RowD + 1
                ColR = self.fenv_config['max_x'] 
                ColL = ColR - 1
            elif interval == '3':
                RowD = np.random.choice( list(range( self.fenv_config['max_y']-mask_data['corner_10'][1]+offset_plus_one, self.fenv_config['max_y']+offset_minus_one)) ) 
                RowU = RowD + 1
                ColR = self.fenv_config['max_x'] - mask_data['corner_10'][0] 
                ColL = ColR - 1
            elif interval == '12':
                RowD = np.random.choice( list(range(self.fenv_config['min_y']+offset_plus_one, self.fenv_config['max_y']-mask_data['corner_10'][1]+offset_minus_one)) ) 
                RowU = RowD + 1
                ColR = self.fenv_config['max_x'] 
                ColL = ColR - 1
            elif interval == '23':
                RowD = np.random.choice( list(range(mask_data['corner_11'][1]+offset_plus_one, self.fenv_config['max_y']+offset_minus_one)) ) 
                RowU = RowD + 1
                ColR = self.fenv_config['max_x'] 
                ColL = ColR - 1
            elif interval == '123':
                RowD = np.random.choice( list(range(self.fenv_config['min_y']+offset_plus_one, self.fenv_config['max_y']+offset_minus_one)) ) 
                RowU = RowD + 1
                ColR = self.fenv_config['max_x'] 
                ColL = ColR - 1
            else:
                raise ValueError('A wrong interval randomly selected! Not possible!')
                
            extended_entrance_positions = [
                [RowD, ColR],
                [RowU, ColR],
                
                [RowD, ColL],
                [RowU, ColL],
                ]
        
        else:
            raise ValueError('A wrong direction randomly selected! Not possible!')
                
            
        extended_entrance_coords = []
        for r, c in extended_entrance_positions:
            x, y = self._image_coords2cartesian(r, c, self.fenv_config['n_rows'])
            extended_entrance_coords.append([x, y])
        
        entrance_positions = extended_entrance_positions[:2]
        entrance_coords = extended_entrance_coords[:2]
        
        return entrance_coords, entrance_positions, entrance_is_on_facade, extended_entrance_positions, extended_entrance_coords
    
    

    @staticmethod
    def _cartesian2image_coord(x, y, max_y):
        return int(max_y-y), int(x) 
    
    
    
    def _image_coords2cartesian(self, r, c, n_rows):
        return c, n_rows-1-r #c-1, n_rows-2-r
        


#%%
if __name__ == '__main__':
    from gym_floorplan.envs.fenv_config import LaserWallConfig
    fenv_config = LaserWallConfig().get_config()
    fenv_config['plan_config_source_name'] = 'create_random_config' # only in this mode it works
    fenv_config['create_random_configs_with_fixed_n_rooms_flag'] = 4
    fenv_config['fixed_outline_flag'] = False
    self = AdjustableConfigsHandeler(fenv_config)
    print('- - - - - - - - - - adjustable_configs')
    pprint(self.adjustable_configs)
    obs_mat = np.zeros((fenv_config['max_x']+1, fenv_config['max_y']+1))
    # for r, c in self.adjustable_configs['entrance_positions']:
    #     obs_mat[r][c] = fenv_config['entrance_cell_id'] 
