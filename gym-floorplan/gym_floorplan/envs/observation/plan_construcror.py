#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 17:06:56 2022

@author: RK
"""

#%%
import ast
import copy
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

        self.adjustable_configs_handeler = AdjustableConfigsHandeler(fenv_config=self.fenv_config)
        
        self.initial_plan_data_dict = self._make_very_first_plan()
        
    
    def get_plan_data_dict(self, episode=None):
        if self.fenv_config['plan_config_source_name'] in ['fixed_test_config', 'load_fixed_config', 'create_fixed_config']:  
            plan_data_dict = copy.deepcopy(self.initial_plan_data_dict)
        else:
            plan_data_dict = self._make_very_first_plan(episode=episode)
        return plan_data_dict    
            
            
    def _make_very_first_plan(self, episode=None):
        plan_data_dict = self._setup_empty_plan()
        plan_data_dict = self._get_plan_variables(plan_data_dict, episode)
        plan_data_dict = self._make_input_plan(plan_data_dict)
        return plan_data_dict
        
    
    def _setup_empty_plan(self):
        plan_dict = Plan(outline_dict=self.outline_dict).get_plan_data()
        
        obs_mat = np.zeros((self.fenv_config['n_rows'], self.fenv_config['n_cols']), dtype=int)
        obs_blocked_cells = np.zeros((self.fenv_config['n_rows'], self.fenv_config['n_cols']), dtype=int)
        obs_blocked_cells[0,:] = 1
        obs_blocked_cells[-1,:] = 1
        obs_blocked_cells[:,0] = 1
        obs_blocked_cells[:,-1] = 1
        
        plan_data_dict = {'outline_segments': plan_dict['segments_dict']['outline'],
                          'inline_segments': {},
                          'walls_segments': {},
                          'walls_coords': {},
                          
                          'base_obs_mat': obs_mat,
                          'base_obs_mat_w': obs_mat,
                          'base_obs_mat_for_dot_prod': 1 - obs_mat,
                          
                          'obs_mat': obs_mat,
                          'obs_mat_w': obs_mat,
                          'obs_mat_for_dot_prod': 1 - obs_mat,
                          'obs_blocked_cells': obs_blocked_cells,
                          
                          'moving_labels': obs_mat,
                          'moving_ones': obs_mat,
                          
                          'canvas_mat': obs_mat,
                          'outside_positions': [],
                          'room_wall_occupied_positions': [],
                          
                          'rooms_dict': {},
                          'desired_areas': {},
                          'areas': {},
                          'delta_areas': {},
                          'number_of_total_walls': self.fenv_config['n_walls'],
                          'number_of_total_rooms': self.fenv_config['n_rooms'],
                          
                          'edge_list': [],
                          'desired_edge_list': [],
                          
                          'mask_numbers': 0,
                          'masked_area': 0,
                          
                          'accepted_actions': [],
                          
                          'wall_types': {},
                          }
            
        return plan_data_dict
    
    
    def _get_plan_variables(self, plan_data_dict, episode=None):
        self.adjustable_configs = self.adjustable_configs_handeler.get_configs(episode=episode)
        
        plan_data_dict.update({k: v for k, v in self.adjustable_configs.items()})
        
        plan_data_dict.update({'obs_mat_mask': np.ones((self.fenv_config['n_rows'], self.fenv_config['n_cols']), dtype=int)})
        
        return plan_data_dict
        

    def _make_input_plan(self, plan_data_dict):
        if self.fenv_config['mask_flag']:
            plan_data_dict = self._mask_plan(plan_data_dict) # only gets the false walls. updating happens afew line later
            
            ## update the plan_data_dict according to false walls
            for fwall_name , fwalls_coord in plan_data_dict['fwalls_coords'].items():
                plan_data_dict = self.update_plan_with_active_wall(plan_data_dict=plan_data_dict, walls_coords=plan_data_dict['fwalls_coords'], active_wall_name=fwall_name)
                plan_data_dict = self.painter.updata_obs_mat(plan_data_dict, fwall_name)
                plan_data_dict = self.rextractor.update_room_dict(plan_data_dict, fwall_name)
                
        else:
            self.mask_numbers = 0
            
        plan_data_dict['room_wall_occupied_positions'].extend(np.argwhere(plan_data_dict['moving_ones']==1).tolist())
        plan_data_dict['obs_mat_mask'] = 1 - plan_data_dict['moving_ones']
        # plan_data_dict = self._adjust_fenv_config(plan_data_dict)
        
        
        if self.fenv_config['env_planning'] == 'Dynamic':
            plan_data_dict = self._setup_walls(plan_data_dict)
            
            for iwall_name, iwalls_coord in plan_data_dict['iwalls_coords'].items():
                plan_data_dict = self._setup_plan(plan_data_dict=plan_data_dict, walls_coords=plan_data_dict['iwalls_coords'], active_wall_name=iwall_name)
                plan_data_dict = self.painter.updata_obs_mat(plan_data_dict, iwall_name)
                plan_data_dict = self.rextractor.update_room_dict(plan_data_dict, iwall_name)
        
        plan_data_dict = self._add_initial_convas_cnn(plan_data_dict)

        if self.fenv_config['net_arch'] == 'CnnGcn':
            graph_data_numpy, plan_data_dict = self._get_initil_graph_data_numpy(plan_data_dict)
        
        for i in range(1, self.adjustable_configs['n_corners']+1):
            masked_room_name = f"room_{i}"
            if masked_room_name not in plan_data_dict['areas'].keys():
                plan_data_dict['areas'].update({masked_room_name: 0})
        
        return plan_data_dict
    
    
    def _mask_plan(self, plan_data_dict):
        fwalls_coords = {}
        rectangles_vertex = [] #{}
        for i, mask_cor in enumerate(plan_data_dict['masked_corners']):
            if mask_cor == 'corner_00':
                wall_name = 'wall_1'
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
                wall_name = 'wall_2'
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
                wall_name = 'wall_3'
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
                wall_name = 'wall_4'
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
            
            
            
            fwalls_coords.update({wall_name: {'anchor_coord': fwall_anchor_coord,
                                              'back_open_coord': fwall_back_coord,
                                              'front_open_coord': fwall_front_coord}})
        
        valid_points_for_sampling = self._get_valid_points_for_sampling(plan_data_dict)
        for fwall_name, fwall_coord in fwalls_coords.items():
            fwall_coords = WallGenerator(self.fenv_config).make_walls(valid_points_for_sampling, fwall_coord, wall_name=fwall_name)
            fwalls_coords[fwall_name] = fwall_coords[fwall_name]
        
        plan_data_dict.update({
                               'fwalls_coords': fwalls_coords,
                               'rectangles_vertex': rectangles_vertex,
                               })
        return plan_data_dict
    
    
    def _add_initial_convas_cnn(self, plan_data_dict):
        canvas_cnn = np.zeros((self.fenv_config['n_rows']+2, self.fenv_config['n_cols']+2))
        canvas_cnn[0, :-1] = 1 
        canvas_cnn[:-1, -1] = 2
        canvas_cnn[-1, 1:] = 3
        canvas_cnn[1:, 0] = 4 
        
        obs_cnn = copy.deepcopy(plan_data_dict['moving_labels'])
        obs_cnn += 4
        obs_cnn[obs_cnn==4] = 0
        
        canvas_cnn[1:-1,1:-1] = obs_cnn
        plan_data_dict.update({'canvas_cnn': canvas_cnn})
        
        canvas_cnn_channel = np.expand_dims(canvas_cnn, axis=2)
        canvas_cnn_channel = np.concatenate((canvas_cnn_channel, canvas_cnn_channel), axis=2)
        plan_data_dict.update({'plan_canvas_arr': canvas_cnn_channel})
        return plan_data_dict
    
    
    def _get_initil_graph_data_numpy(self, plan_data_dict):
        desired_areas = plan_data_dict['desired_areas'] # list(plan_data_dict['desired_areas'].values())
        
        partially_desired_graph_features_dict_numpy = {
            room_name: {
                'status': 1, 'desired_area': da, 'current_area': da, 'delta_area': 0
                }  for room_name, da in desired_areas.items()
            }
        
        partially_current_graph_features_dict_numpy = {
            room_name: {
                'status': 0, 'desired_area': da, 'current_area': 0, 'delta_area': da
                }  for room_name, da in desired_areas.items()
            }
        
        fully_current_graph_features_dict_numpy = {
            room_name: {
              'status': 0, 'desired_area': da, 'current_area': 0, 'delta_area': da,
              'start_of_wall': [-1, -1], 
              'before_anchor': [-1, -1], 
              'anchor': [-1, -1], 
              'after_anchor': [-1, -1], 
              'end_of_wall': [-1, -1], 
              # 'anchor_to_corner_00_distance': -1, 
              # 'anchor_to_corner_01_distance': -1, 
              # 'anchor_to_corner_10_distance': -1,
              # 'anchor_to_corner_11_distance': -1,
              'distances': [-1, -1, -1, -1],
              } for room_name, da in desired_areas.items()
            }
        
        graph_features_numpy = {
            'partially_desired_graph_features_dict_numpy': partially_desired_graph_features_dict_numpy,
            'partially_current_graph_features_dict_numpy': partially_current_graph_features_dict_numpy,
            'fully_current_graph_features_dict_numpy': fully_current_graph_features_dict_numpy,
                                }
        
        desired_edge_list = plan_data_dict['desired_edge_list']
        
        partially_desired_graph_edge_list_numpy = copy.deepcopy(desired_edge_list)
        partially_current_graph_edge_list_numpy = []
        fully_current_graph_edge_list_numpy = []
        
        graph_edge_list_numpy = {
            'partially_desired_graph_edge_list_numpy': partially_desired_graph_edge_list_numpy,
            'partially_current_graph_edge_list_numpy': partially_current_graph_edge_list_numpy,
            'fully_current_graph_edge_list_numpy': fully_current_graph_edge_list_numpy,
            }
        
        
        graph_data_numpy = {'graph_features_numpy': graph_features_numpy,
                            'graph_edge_list_numpy': graph_edge_list_numpy}
        
        plan_data_dict.update({'graph_data_numpy_old': graph_data_numpy})
        plan_data_dict.update({'graph_data_numpy': graph_data_numpy})
    
        return graph_data_numpy, plan_data_dict
    
    
    def update_plan_with_active_wall(self, plan_data_dict, walls_coords, active_wall_name):
        plan_dict = Plan(outline_dict=self.outline_dict, walls_coords=walls_coords).get_plan_data()
        try:
            plan_data_dict['inline_segments'].update({k: v for k, v in plan_dict['segments_dict']['inline'].items() if active_wall_name in k})
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
        valid_points_for_sampling = np.argwhere(plan_data_dict['moving_ones']==0)
        valid_points_for_sampling = [[r, c] for r, c in valid_points_for_sampling if (r%2==0 and c%2==0 and r!=0 and c!=0 and r!=self.fenv_config['max_x'] and c!=self.fenv_config['max_y'])]
       
        return np.array(valid_points_for_sampling)
    
    
#%%
if __name__ == '__main__':
    from gym_floorplan.envs.fenv_config import LaserWallConfig
    fenv_config = LaserWallConfig().get_config()
    
    
    self = PlanConstructor(fenv_config)
    
    for _ in range(1):
        
        plan_data_dict = self.get_plan_data_dict()
        print(plan_data_dict['mask_numbers'])
    
    
    