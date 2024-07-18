#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 17:06:56 2022

@author: Reza Kakooee
"""


#%%
import ast
import copy
import itertools
import numpy as np
import pandas as pd

import torch

# from torch_geometric.data import Data

from gym_floorplan.envs.observation.geometry import Outline, Plan
from gym_floorplan.envs.observation.wall_generator import WallGenerator

from gym_floorplan.envs.observation.sequential_painter import SequentialPainter
from gym_floorplan.envs.observation.room_extractor import RoomExtractor

from gym_floorplan.envs.adjustable_configs_handeler import AdjustableConfigsHandeler




#%%
class PlanConstructor:
    def __init__(self, fenv_config):
        self.fenv_config = fenv_config
        
        self.outline_dict = Outline(fenv_config=self.fenv_config).get_outline_data()
        self.painter = SequentialPainter(fenv_config=self.fenv_config)
        self.rextractor = RoomExtractor(fenv_config=self.fenv_config)
        
        self._reset()
    
    
    
    def _reset(self):
        self.adjustable_configs_handeler = AdjustableConfigsHandeler(fenv_config=self.fenv_config)
        self.initial_plan_data_dict = self._make_very_first_plan()
    
    
    
    def get_plan_data_dict(self, episode=None):
        if self.fenv_config['plan_config_source_name'] in ['fixed_test_config', 'load_fixed_config', 'create_fixed_config']:  
            plan_data_dict = copy.deepcopy(self.initial_plan_data_dict)
        else:
            plan_data_dict = self._make_very_first_plan(episode=episode)
        return plan_data_dict    
    
    
    
    def get_plan_meta_data(self, plan_data_dict):
        n_corners = 4
        masked_corners_name_to_id = {'corner_00': 0, 'corner_01': 1, 'corner_10':2, 'corner_11': 3}
        masked_corners_on_hot = [0] * n_corners
        for masked_corner in plan_data_dict['masked_corners']:
            masked_corners_on_hot[masked_corners_name_to_id[masked_corner]] = 1
            
        c = 0
        mask_len_vec = [0] * n_corners
        mask_wid_vec = [0] * n_corners
        for i, element in enumerate(masked_corners_on_hot):
            if element == 1:
                mask_len_vec[i] = plan_data_dict['mask_lengths'][c]
                mask_wid_vec[i] = plan_data_dict['mask_widths'][c]
                
        facades_blocked_one_hot = [0] * n_corners
        directions_name_to_id = {'n': 0, 's': 1, 'e': 2, 'w': 3}
        for d in plan_data_dict['facades_blocked']:
            facades_blocked_one_hot[directions_name_to_id[d]] = 1
            
        entrance_is_on_facade_one_hot = [0] * n_corners
        entrance_is_on_facade_one_hot[directions_name_to_id[plan_data_dict['entrance_is_on_facade']]] = 1
        
        required_rooms_one_hot = [0] * self.fenv_config['maximum_num_real_rooms']
        required_rooms_one_hot[:plan_data_dict['n_rooms']] = [1] * plan_data_dict['n_rooms']
        
        lvroom_id_one_hot = [0] * self.fenv_config['maximum_num_real_rooms']
        if 'lvroom_id' in plan_data_dict.keys():
            lvroom_id_one_hot[plan_data_dict['lvroom_id'] - self.fenv_config['min_room_id']] = 1
        
        desc = []
        # desc += [plan_data_dict['mask_numbers']]
        desc += masked_corners_on_hot # size = 4*1
        desc += facades_blocked_one_hot # 4*1
        desc += entrance_is_on_facade_one_hot # 4*1
        desc += required_rooms_one_hot # 9*1 
        # desc += lvroom_id_one_hot # 9*1 # 2024-02-05: I excluded this, as lvroom_id is alwasys 11
        desc += mask_len_vec # 4*1
        desc += mask_wid_vec # 4*1
        desc += list(itertools.chain.from_iterable(plan_data_dict['extended_entrance_coords'])) # 8*1
        # desc += list(itertools.chain.from_iterable(plan_data_dict['extended_entrance_positions']))
        return desc
        
        
            
    def _make_very_first_plan(self, episode=None):
        plan_data_dict = self._setup_empty_plan()
        plan_data_dict = self._get_plan_variables(plan_data_dict, episode)
        plan_data_dict = self._make_input_plan(plan_data_dict)
        return plan_data_dict
    

    
    def _setup_empty_plan(self):
        plan_dict = Plan(outline_dict=self.outline_dict).get_plan_data()
        
        obs_mat = np.zeros((self.fenv_config['n_rows'], self.fenv_config['n_cols']), dtype=int)
        
        obs_w_for_prod = np.ones((self.fenv_config['n_rows'], self.fenv_config['n_cols']), dtype=int)
        
        only_boarder = np.zeros((self.fenv_config['n_rows'], self.fenv_config['n_cols']), dtype=int)
        only_boarder[0, :]  = self.fenv_config['north_facade_id']
        only_boarder[-1, :] = self.fenv_config['south_facade_id']
        only_boarder[:, -1] = self.fenv_config['east_facade_id']
        only_boarder[:, 0]  = self.fenv_config['west_facade_id']
        only_boarder[0, 0] = only_boarder[0, -1] = only_boarder[-1, 0] = only_boarder[-1, -1] = self.fenv_config['very_corner_cell_id']
        # only_boarder[self.fenv_config['plan_center_positions'][0], self.fenv_config['plan_center_positions'][1]] = self.fenv_config['very_corner_cell_id']
        
        state_adj_matrix = np.zeros((self.fenv_config['num_plan_components'], self.fenv_config['num_plan_components']), dtype=int)
        
        obs_blocked_cells = np.zeros((self.fenv_config['n_rows'], self.fenv_config['n_cols']), dtype=int)
        obs_moving_ones = np.zeros((self.fenv_config['n_rows'], self.fenv_config['n_cols']), dtype=int)
        obs_mat_mask = np.ones((self.fenv_config['n_rows'], self.fenv_config['n_cols']), dtype=int)
        

        plan_data_dict = {'wall_outline_segments': plan_dict['segments_dict']['outline'],
                          'wall_inline_segments': {},
                          'walls_segments': {},
                          'walls_coords': {},
                          
                          'obs_mat_base': obs_mat,
                          'obs_mat_base_w': obs_mat,
                          'obs_mat_base_for_dot_prod': obs_w_for_prod,
                          
                          'obs_mat': obs_mat,
                          'obs_mat_w': obs_mat,
                          'obs_mat_for_dot_prod': obs_w_for_prod,
                          
                          'obs_moving_labels': obs_mat,
                          
                          'obs_moving_ones': obs_moving_ones,
                          'obs_mat_mask': obs_mat_mask,
                          'obs_blocked_cells': obs_blocked_cells,
                          'obs_blocked_cells_by_shift': obs_blocked_cells,
                          
                          'obs_canvas_mat': obs_mat,
                          'obs_canvas_mat_empty': only_boarder,
                          'only_boarder': only_boarder,
                          
                          'outside_positions': [],
                          'room_wall_occupied_positions': [],
                          
                          'rooms_dict': {},
                          'areas_desired': {},
                          'areas_achieved': {},
                          'areas_delta': {},
                          
                          'number_of_total_walls': self.fenv_config['n_walls'],
                          'number_of_total_rooms': self.fenv_config['n_rooms'], # TODO
                          
                          'edge_list_room_achieved': [],
                          'edge_list_room_desired': [],
                          
                          'edge_list_facade_achieved': [],
                          'edge_list_facade_achieved_str': [],
                          'edge_list_facade_desired': [],
                          'edge_list_facade_desired_str': [],
                          
                          'edge_list_entrance_achieved': [],
                          'edge_list_entrance_achieved_str': [],
                          'edge_list_entrance_desired': [],
                          'edge_list_entrance_desired_str': [],
                          
                          'state_adj_matrix': state_adj_matrix,
                          
                          'mask_numbers': 0,
                          'area_masked': 0,
                          'areas_masked': {},
                          
                          'entrance_area': 0,
                          'entrance_coords': [],
                          'entrance_positions': [],
                          'extented_entrance_coords': [],
                          'extented_entrance_positions': [],
                          
                          'n_facades_blocked': 0,
                          'facades_blocked': [],
                          
                          'actions_accepted': [],
                          
                          'wall_types': {},
                          'wall_order': {},
                          
                          'gnn_edges': [],
                          'gnn_nodes': [],
                          }
            
        return plan_data_dict
    
    
    
    def _get_plan_variables(self, plan_data_dict, episode=None):
        self.adjustable_configs = self.adjustable_configs_handeler.get_configs(episode=episode)
        
        plan_data_dict.update({k: v for k, v in self.adjustable_configs.items()})
        
        return plan_data_dict
        

    
    def _add_entrance_to_plan(self, plan_data_dict):
        for r, c in plan_data_dict['entrance_positions']:
            plan_data_dict['obs_moving_labels'][r][c] = self.fenv_config['entrance_cell_id']
            plan_data_dict['obs_blocked_cells'][r][c] = 1
            plan_data_dict['obs_mat'][r][c] = 10
            plan_data_dict['obs_moving_ones'][r][c] = 1
        return plan_data_dict
            
    

    def _make_input_plan(self, plan_data_dict):
        if self.fenv_config['mask_flag']:
            plan_data_dict = self._mask_plan(plan_data_dict) # only gets the false walls. updating happens afew line later
            
            ## update the plan_data_dict according to false walls
            for fwall_name , fwalls_coord in plan_data_dict['walls_coords_false'].items():
                plan_data_dict = self.update_plan_with_active_wall(plan_data_dict=plan_data_dict, walls_coords=plan_data_dict['walls_coords_false'], active_wall_name=fwall_name)
                plan_data_dict = self.painter.update_obs_mat(plan_data_dict, fwall_name)
                plan_data_dict = self.rextractor.update_room_dict(plan_data_dict, fwall_name)
                
        else:
            self.mask_numbers = 0
        
        walls_coords_outline = {}
        for owall_name , owall_coord in self.fenv_config['outline_walls_coords'].items():
            owall_coords = WallGenerator(self.fenv_config).make_walls(None, owall_coord, wall_name=owall_name)
            walls_coords_outline[owall_name] = owall_coords[owall_name]
        
        for owall_name , owall_coord in walls_coords_outline.items():
            plan_data_dict = self.update_plan_with_active_wall(plan_data_dict=plan_data_dict, walls_coords=self.fenv_config['outline_walls_coords'], active_wall_name=owall_name)
            plan_data_dict = self.painter.update_obs_mat(plan_data_dict, owall_name)
            
        # if plan_data_dict['entrance_coords']:
        #     plan_data_dict = self._add_entrance_to_plan(plan_data_dict)
            
        # plan_data_dict = self._project_facades_on_masked_room_walls(plan_data_dict)
        
        plan_data_dict['room_wall_occupied_positions'].extend(np.argwhere(plan_data_dict['obs_moving_ones']==1).tolist())
        plan_data_dict['obs_mat_mask'] = 1 - plan_data_dict['obs_moving_ones']
        # plan_data_dict = self._adjust_fenv_config(plan_data_dict)
        
        # plan_data_dict = self._add_initial_obs_canvas_mat(plan_data_dict)

        for i in self.fenv_config['fake_room_id_range']:
            masked_room_name = f"room_{i}"
            if masked_room_name not in plan_data_dict['areas_achieved'].keys():
                plan_data_dict['areas_achieved'].update({masked_room_name: 0})
                
        plan_data_dict = self._update_block_cells(plan_data_dict)
        
        return plan_data_dict
    
    
    
    def _update_block_cells(self, plan_data_dict):
        obs_blocked_cells_by_shift_0 = plan_data_dict['obs_moving_labels'].astype(np.int16) - plan_data_dict['obs_mat_w'].astype(np.int16) + plan_data_dict['only_boarder'].astype(np.int16)
        
        obs_blocked_cells_by_shift = copy.deepcopy(obs_blocked_cells_by_shift_0)
        
        obs_blocked_cells_by_shift += self._shift_left(obs_blocked_cells_by_shift_0)
        obs_blocked_cells_by_shift += self._shift_right(obs_blocked_cells_by_shift_0)
        obs_blocked_cells_by_shift += self._shift_down(obs_blocked_cells_by_shift_0)
        obs_blocked_cells_by_shift += self._shift_up(obs_blocked_cells_by_shift_0)
        
        obs_blocked_cells_by_shift += self._shift_left(self._shift_up(obs_blocked_cells_by_shift_0))
        obs_blocked_cells_by_shift += self._shift_right(self._shift_up(obs_blocked_cells_by_shift_0))
        obs_blocked_cells_by_shift += self._shift_left(self._shift_down(obs_blocked_cells_by_shift_0))
        obs_blocked_cells_by_shift += self._shift_right(self._shift_down(obs_blocked_cells_by_shift_0))
        
        obs_blocked_cells_by_shift = np.clip(obs_blocked_cells_by_shift, 0, 1)
        
        plan_data_dict['obs_blocked_cells_by_shift'] = copy.deepcopy(obs_blocked_cells_by_shift)
        
        return plan_data_dict
    
    
    
    @staticmethod
    def _shift_left(arr):
        return  np.append(arr[:, 1:], (arr[:, 0]*0).reshape((-1,1)), axis=1)
    
    
    
    @staticmethod
    def _shift_right(arr):
        return  np.append((arr[:, -1]*0).reshape((-1,1)), arr[:, :-1], axis=1)
    
    
    
    @staticmethod
    def _shift_down(arr):
        return  np.append((arr[-1, :]*0).reshape((1,-1)), arr[:-1, :], axis=0)
    
    
    @staticmethod
    def _shift_up(arr):
        return  np.append(arr[1:, :], (arr[0, :]*0).reshape((1,-1)), axis=0)
        
    
    
    def _project_facades_on_masked_room_walls(self, plan_data_dict):
        obs_moving_labels = copy.deepcopy(plan_data_dict['obs_moving_labels'])
        walls_coords = plan_data_dict['walls_coords']
        
        wall_names = list(walls_coords.keys()) # [f'wall_{i}' for i in range(1,5)]
        for wall_name in wall_names:
            if wall_name in ['wall_1', 'wall_3', 'wall_2', 'wall_4']:
                wall_data = walls_coords[wall_name]
                anchor_coord = wall_data['anchor_coord']
                anchor_pos = self._cartesian2image_coord(anchor_coord[0], anchor_coord[1], self.fenv_config['max_y'])
                wall_positions = wall_data['wall_positions']
                
                if wall_name == 'wall_1':
                    for pos in wall_positions:
                        if pos[0] == anchor_pos[0] and pos[1] != anchor_pos[1]:
                            obs_moving_labels[pos[0]][pos[1]] = self.fenv_config['south_facade_id']
                        if pos[1] == anchor_pos[1] and pos[0] != anchor_pos[0]:
                            obs_moving_labels[pos[0]][pos[1]] = self.fenv_config['west_facade_id']
                            
                    obs_moving_labels[anchor_pos[0]][anchor_pos[1]] = self.fenv_config['very_corner_cell_id']
                    
                    
                elif wall_name == 'wall_2':
                    for pos in wall_positions:
                        if pos[0] == anchor_pos[0] and pos[1] != anchor_pos[1]:
                            obs_moving_labels[pos[0]][pos[1]] = self.fenv_config['north_facade_id']
                        if pos[1] == anchor_pos[1] and pos[0] != anchor_pos[0]:
                            obs_moving_labels[pos[0]][pos[1]] = self.fenv_config['west_facade_id']
                            
                    obs_moving_labels[anchor_pos[0]][anchor_pos[1]] = self.fenv_config['very_corner_cell_id']
                    
                    
                elif wall_name == 'wall_3':
                    for pos in wall_positions:
                        if pos[0] == anchor_pos[0] and pos[1] != anchor_pos[1]:
                            obs_moving_labels[pos[0]][pos[1]] = self.fenv_config['south_facade_id']
                        if pos[1] == anchor_pos[1] and pos[0] != anchor_pos[0]:
                            obs_moving_labels[pos[0]][pos[1]] = self.fenv_config['east_facade_id']
                            
                    obs_moving_labels[anchor_pos[0]][anchor_pos[1]] = self.fenv_config['very_corner_cell_id']
                    
                    
                elif wall_name == 'wall_4':
                    for pos in wall_positions:
                        if pos[0] == anchor_pos[0] and pos[1] != anchor_pos[1]:
                            obs_moving_labels[pos[0]][pos[1]] = self.fenv_config['north_facade_id']
                        if pos[1] == anchor_pos[1] and pos[0] != anchor_pos[0]:
                            obs_moving_labels[pos[0]][pos[1]] = self.fenv_config['east_facade_id']
                            
                    obs_moving_labels[anchor_pos[0]][anchor_pos[1]] = self.fenv_config['very_corner_cell_id']
                    
        plan_data_dict['obs_moving_labels'] = obs_moving_labels
                    
        return plan_data_dict
    
    
    
    def _mask_plan(self, plan_data_dict):
        walls_coords_false = {}
        rectangles_vertex = [] #{}
        for i, mask_cor in enumerate(plan_data_dict['masked_corners']):
            if mask_cor == 'corner_00':
                wall_name = 'wall_2'
                rect_vertex = (0, 0)
                if plan_data_dict['mask_lengths'][i] != 0:
                    fwall_anchor_coord = [plan_data_dict['mask_lengths'][i], plan_data_dict['mask_widths'][i]]
                    fwall_back_coord = [fwall_anchor_coord[0]-1, fwall_anchor_coord[1]]
                    fwall_front_coord = [fwall_anchor_coord[0], fwall_anchor_coord[1]-1]
                else:
                    fwall_anchor_coord = [0, 0]
                    fwall_back_coord = [0, 0]
                    fwall_front_coord = [0, 0]
                    
            elif mask_cor == 'corner_01':
                wall_name = 'wall_3'
                rect_vertex = [0, self.fenv_config['max_y']-plan_data_dict['mask_widths'][i]]
                if plan_data_dict['mask_lengths'][i] != 0:
                    fwall_anchor_coord = [plan_data_dict['mask_lengths'][i], self.fenv_config['max_y']-plan_data_dict['mask_widths'][i]]
                    fwall_back_coord = [fwall_anchor_coord[0], fwall_anchor_coord[1]+1]
                    fwall_front_coord = [fwall_anchor_coord[0]-1, fwall_anchor_coord[1]]
                else:
                    fwall_anchor_coord = [0, self.fenv_config['max_y']]
                    fwall_back_coord = [0, self.fenv_config['max_y']]
                    fwall_front_coord = [0, self.fenv_config['max_y']]
            
            elif mask_cor == 'corner_10':
                wall_name = 'wall_4'
                rect_vertex = [self.fenv_config['max_x']-plan_data_dict['mask_lengths'][i], 0]
                if plan_data_dict['mask_lengths'][i] != 0:
                    fwall_anchor_coord = [self.fenv_config['max_x']-plan_data_dict['mask_lengths'][i], plan_data_dict['mask_widths'][i]]
                    fwall_back_coord = [fwall_anchor_coord[0]+1, fwall_anchor_coord[1]]
                    fwall_front_coord = [fwall_anchor_coord[0], fwall_anchor_coord[1]-1]
                else:
                    fwall_anchor_coord = [self.fenv_config['max_x'], 0]
                    fwall_back_coord = [self.fenv_config['max_x'], 0]
                    fwall_front_coord = [self.fenv_config['max_x'], 0]
                
            elif mask_cor == 'corner_11':
                wall_name = 'wall_5'
                rect_vertex = [self.fenv_config['max_x']-plan_data_dict['mask_lengths'][i], self.fenv_config['max_y']-plan_data_dict['mask_widths'][i]]
                if plan_data_dict['mask_lengths'][i] != 0:
                    fwall_anchor_coord = [self.fenv_config['max_x']-plan_data_dict['mask_lengths'][i], self.fenv_config['max_y']-plan_data_dict['mask_widths'][i]]
                    fwall_back_coord = [fwall_anchor_coord[0], fwall_anchor_coord[1]+1]
                    fwall_front_coord = [fwall_anchor_coord[0]+1, fwall_anchor_coord[1]]
                else:
                    fwall_anchor_coord = [self.fenv_config['max_x'], self.fenv_config['max_y']]
                    fwall_back_coord = [self.fenv_config['max_x'], self.fenv_config['max_y']]
                    fwall_front_coord = [self.fenv_config['max_x'], self.fenv_config['max_y']]
                    
            rectangles_vertex.append(rect_vertex) # rectangles_vertex[mask_cor] =  rect_vertex
            
            walls_coords_false.update({wall_name: {'anchor_coord': fwall_anchor_coord,
                                              'back_open_coord': fwall_back_coord,
                                              'front_open_coord': fwall_front_coord}})
        
        valid_points_for_sampling = self._get_valid_points_for_sampling(plan_data_dict)
        for fwall_name, fwall_coord in walls_coords_false.items():
            fwall_coords = WallGenerator(self.fenv_config).make_walls(valid_points_for_sampling, fwall_coord, wall_name=fwall_name)
            walls_coords_false[fwall_name] = fwall_coords[fwall_name]
        
        plan_data_dict.update({
                               'walls_coords_false': walls_coords_false,
                               'rectangles_vertex': rectangles_vertex,
                               })
        return plan_data_dict
    
    
    
    def update_plan_with_active_wall(self, plan_data_dict, walls_coords, active_wall_name):
        plan_dict = Plan(outline_dict=self.outline_dict, walls_coords=walls_coords).get_plan_data()
        try:
            plan_data_dict['wall_inline_segments'].update({k: v for k, v in plan_dict['segments_dict']['inline'].items() if active_wall_name in k})
            plan_data_dict['walls_segments'].update({k: v for k, v in plan_dict['walls_segments_dict'].items() if active_wall_name in k})
            plan_data_dict['walls_coords'].update({active_wall_name: plan_dict['walls_dict'][active_wall_name]})
        except:
            for k, v in plan_dict['walls_segments_dict'].items():
                if active_wall_name in k:
                    print(k)
            print('wait in _setup_plan of observation')
            raise ValueError("some thing is wrong with updating plan_data_dict")
            
        return plan_data_dict

    

    def _get_valid_points_for_sampling(self, plan_data_dict):
        valid_points_for_sampling = np.argwhere(plan_data_dict['obs_moving_ones']==0)
        valid_points_for_sampling = [[r, c] for r, c in valid_points_for_sampling if (r%2==0 and c%2==0 and r!=0 and c!=0 and r!=self.fenv_config['max_x'] and c!=self.fenv_config['max_y'])]
       
        return np.array(valid_points_for_sampling)
    
    
    
    @staticmethod
    def _cartesian2image_coord(x, y, max_y):
        return max_y-y, x
    
    
    
    
    
#%%
if __name__ == '__main__':
    from gym_floorplan.envs.fenv_config import LaserWallConfig
    fenv_config = LaserWallConfig().get_config()
    
    
    self = PlanConstructor(fenv_config)
    
    for _ in range(1):
        
        plan_data_dict = self.get_plan_data_dict()
        print(plan_data_dict['mask_numbers'])
    
    
    