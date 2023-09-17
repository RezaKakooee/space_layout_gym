#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 13:58:10 2023

@author: Reza Kakooee
"""


import copy
import numpy as np
from webcolors import name_to_rgb



#%%
class CnnStateObserver:
    def __init__(self, fenv_config):
        self.fenv_config = fenv_config
        
    
    
    def get_observation(self, plan_data_dict):
        obs_canvas_mat, obs_canvas_arr_1ch = self._get_canvas_cnn(plan_data_dict)
        obs_rooms_cmap, obs_rooms_cmap_1ch = self._get_rooms_color_map(plan_data_dict)
        obs_blocked_cells, obs_blocked_cells_1ch = self._get_obs_blocked_cells(plan_data_dict)
        
        if self.fenv_config['cnn_observation_name'] == 'canvas_1d':
            observation_cnn = obs_canvas_arr_1ch
        elif self.fenv_config['cnn_observation_name'] == 'obs_rooms_cmap':
            observation_cnn = obs_rooms_cmap_1ch
        elif self.fenv_config['cnn_observation_name'] == 'stacked_3d':
             observation_cnn = np.concatenate((obs_canvas_arr_1ch, obs_rooms_cmap_1ch, obs_blocked_cells_1ch))
        else:
            raise ValueError("Invalid cnn_observation_name: {self.fenv_config['cnn_observation_name']}")
            
        plan_data_dict.update({
            'obs_canvas_mat': obs_canvas_mat,
            'obs_canvas_arr_1ch': obs_canvas_arr_1ch,
            
            'obs_rooms_cmap': obs_rooms_cmap,
            'obs_rooms_cmap_1ch': obs_rooms_cmap_1ch,
            
            'obs_blocked_cells': obs_blocked_cells,
            'obs_blocked_cells_1ch': obs_blocked_cells_1ch,
            
            # 'obs_canvas_arr_3ch': obs_canvas_arr_3ch,
            
            'observation_cnn': observation_cnn,
            })
        return plan_data_dict
    
    
    
    def _get_canvas_cnn(self, plan_data_dict):
        obs_canvas_mat = copy.deepcopy(plan_data_dict['obs_moving_labels'])
        obw = copy.deepcopy(plan_data_dict['obs_mat_w'])
        obs_canvas_mat[obw == -self.fenv_config['north_facade_id']] = 0
        obs_canvas_mat[obw == -self.fenv_config['south_facade_id']] = 0
        obs_canvas_mat[obw == -self.fenv_config['east_facade_id']] = 0
        obs_canvas_mat[obw == -self.fenv_config['west_facade_id']] = 0
        
        for r, c in plan_data_dict['extended_entrance_positions']:
            obs_canvas_mat[r][c] = self.fenv_config['entrance_cell_id']
            
        obs_canvas_mat_kroned = np.kron(obs_canvas_mat, 
                                    np.ones((self.fenv_config['cnn_scaling_factor'], self.fenv_config['cnn_scaling_factor']), dtype=obs_canvas_mat.dtype))
        
        obs_canvas_arr_1ch = np.expand_dims(obs_canvas_mat_kroned, axis=0)
        obs_canvas_arr_1ch = obs_canvas_arr_1ch * self.fenv_config['cnn_obs_normalization_factor']
        # obs_canvas_arr_1ch = obs_canvas_arr_1ch.astype(np.uint8)
        
        return obs_canvas_mat, obs_canvas_arr_1ch
    
    
    
    def _get_rooms_color_map(self, plan_data_dict):
        obw = copy.deepcopy(plan_data_dict['obs_mat_w'])
        obw[obw == -self.fenv_config['north_facade_id']] = 0
        obw[obw == -self.fenv_config['south_facade_id']] = 0
        obw[obw == -self.fenv_config['east_facade_id']] = 0
        obw[obw == -self.fenv_config['west_facade_id']] = 0
        
        obs_rooms_cmap = plan_data_dict['obs_moving_labels'] - obw

        # obs_rooms_cmap = copy.deepcopy(plan_data_dict['only_boarder'])
        # obs_rooms_cmap[1:-1, 1:-1] = rw[1:-1, 1:-1]
        
        for r, c in plan_data_dict['extended_entrance_positions'][2:]:
            obs_rooms_cmap[r][c] = self.fenv_config['entrance_cell_id']
            
        obs_rooms_cmap_kroned = np.kron(obs_rooms_cmap, 
                                    np.ones((self.fenv_config['cnn_scaling_factor'], self.fenv_config['cnn_scaling_factor']), dtype=obs_rooms_cmap.dtype))
        
        obs_rooms_cmap_1ch = np.expand_dims(obs_rooms_cmap_kroned, axis=0)
        obs_rooms_cmap_1ch = obs_rooms_cmap_1ch * self.fenv_config['cnn_obs_normalization_factor']
        # obs_rooms_cmap_1ch = obs_rooms_cmap_1ch.astype(np.uint8)
        
        return obs_rooms_cmap, obs_rooms_cmap_1ch
    
    
    
    def _get_obs_blocked_cells(self, plan_data_dict):
        obs_blocked_cells = copy.deepcopy(plan_data_dict['obs_blocked_cells_by_shift'])
        for r, c in plan_data_dict['extended_entrance_positions']:
            obs_blocked_cells[r][c] = 1
            
        obs_blocked_cells_kroned = np.kron(obs_blocked_cells, 
                                       np.ones((self.fenv_config['cnn_scaling_factor'], self.fenv_config['cnn_scaling_factor']), dtype=obs_blocked_cells.dtype))
        
        obs_blocked_cells_1ch = np.expand_dims(obs_blocked_cells, axis=0)
        # obs_blocked_cells_1ch = obs_blocked_cells_1ch.astype(np.uint8)
        
        return obs_blocked_cells_kroned, obs_blocked_cells_1ch
    
    
    
    def _get_canvas_cnn_3d(self, plan_data_dict, obs_canvas_mat_kroned, obs_canvas_arr_1ch):
        obs_canvas_arr_3ch = np.zeros((3, obs_canvas_mat_kroned.shape[0], obs_canvas_mat_kroned.shape[1]), dtype=obs_canvas_mat_kroned.dtype)
        
        for r in range(obs_canvas_arr_1ch.shape[1]): #self.fenv_config['n_rows']*self.fenv_config['cnn_scaling_factor']):
            for c in range(obs_canvas_arr_1ch.shape[2]): #range(self.fenv_config['n_cols']*self.fenv_config['cnn_scaling_factor']):
                for v in list(self.fenv_config['color_map'].keys()):
                    if obs_canvas_arr_1ch[0, r, c] == v:
                        obs_canvas_arr_3ch[:, r, c] = list(  name_to_rgb(self.fenv_config['color_map'][v])  )
                        
        obs_canvas_arr_3ch = obs_canvas_arr_3ch.astype(np.uint8)
        return obs_canvas_arr_3ch
    
    
