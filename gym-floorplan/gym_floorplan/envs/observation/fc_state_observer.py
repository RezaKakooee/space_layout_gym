#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 13:55:50 2023

@author: Reza Kakooee
"""

#%%
import copy
import itertools
import numpy as np



#%%
class FcStateObserver:
    def __init__(self, fenv_config):
        self.fenv_config = fenv_config
        
        
        
    def get_observation(self, plan_data_dict, active_wall_name):
        warooge_data, observation_fc = self._get_fc_obs(plan_data_dict, active_wall_name)
        
        plan_data_dict.update({
                    'warooge_data': warooge_data,
                    'observation_fc': observation_fc,
                    })   
        
        if self.fenv_config['very_short_observation_fc_flag']:
            very_short_observation_fc = self._get_very_short_observation_fc(warooge_data)
            plan_data_dict.update({
                        'observation_fc': very_short_observation_fc,
                        })  
        
        return plan_data_dict
        
        
    
    def _get_fc_obs(self, plan_data_dict, active_wall_name):
        warooge_data, wall_state_vector = self._get_wall_state_vector(plan_data_dict, active_wall_name)
        return warooge_data, wall_state_vector

    

    def _get_very_short_observation_fc(self, warooge_data):
        wall_important_points_dict = warooge_data['wall_important_points_dict']
        very_short_observation_fc_dict = {}
        very_short_observation_fc_vec = []
        for i in range(1, self.fenv_config['cell_id_max']):
            wall_name = f"wall_{i}"
            three_coords = [wall_important_points_dict[wall_name].get(key) for key in ['anchor', 'start_of_wall', 'end_of_wall']]
            very_short_observation_fc_dict.update({
                wall_name: three_coords
                })
            very_short_observation_fc_vec += [item for lst in three_coords for item in lst]
        return very_short_observation_fc_vec
    
    

    def _get_wall_state_vector(self, plan_data_dict, active_wall_name=None):
        warooge_data = copy.deepcopy(plan_data_dict['warooge_data_main'])
        
        wall_important_points_dict = warooge_data['wall_important_points_dict']
        wall_important_coords = [coord for point in wall_important_points_dict.values() for coord in point.values()]
        wall_important_coords_vec = list(itertools.chain.from_iterable(wall_important_coords))
        warooge_data['wall_important_coords_vec'] = wall_important_coords_vec
        
        # walls_to_corners_distance_dict = self.__get_walls_to_corners_distance(plan_data_dict, wall_important_points_dict)
        # walls_to_corners_distance = [ds for ds in walls_to_corners_distance_dict.values()]
        # walls_to_corners_distance_vec = list(itertools.chain.from_iterable(walls_to_corners_distance))
        # warooge_data['walls_to_corners_distance_vec'] = walls_to_corners_distance_vec
        # warooge_data['walls_to_corners_distance_dict'] = walls_to_corners_distance_dict
        walls_to_corners_distance_vec = [] # why empty? bc I already computed the ditance between walls and corners. corner is walls (indeed only cells) with id =1
        
        walls_to_walls_distance_dict = self.__get_walls_to_walls_distance(plan_data_dict, wall_important_points_dict)
        walls_to_walls_distance = [ds for ds in walls_to_walls_distance_dict.values()]
        walls_to_walls_distance_vec = list(itertools.chain.from_iterable(walls_to_walls_distance))
        warooge_data['walls_to_walls_distance_vec'] = walls_to_walls_distance_vec
        warooge_data['walls_to_walls_distance_dict'] = walls_to_walls_distance_dict
        
        wall_state_vec = wall_important_coords_vec + walls_to_corners_distance_vec + walls_to_walls_distance_vec
        wall_state_vec = self.__normalize_walls_state_vector(wall_state_vec)
        return warooge_data, np.array(wall_state_vec)
        
    

    def __get_walls_to_corners_distance(self, plan_data_dict, wall_important_points_dict):
        # 5 wall's important points to 4 corners
        walls_to_corners_distance_dict = {wall_name: list(-1*np.ones((5*4),dtype=float)) for wall_name in wall_important_points_dict.keys()}
        
        for wall_name in walls_to_corners_distance_dict.keys(): # 3
            wall_to_corner_dist_vec = []
            for point_name in self.fenv_config['wall_important_points_names']: #5
                for corner_name, corner_coord in self.fenv_config['corners'].items(): # 4
                    wall_coord = wall_important_points_dict[wall_name][point_name]
                    if -1 in wall_coord:
                        d = -1
                    else:
                        d = self.___distance(wall_coord, corner_coord)
                    wall_to_corner_dist_vec.append(d)
        
            walls_to_corners_distance_dict[wall_name] = copy.deepcopy(wall_to_corner_dist_vec)
        return walls_to_corners_distance_dict
        
            
    
    def __get_walls_to_walls_distance(self, plan_data_dict, wall_important_points_dict):
        walls_to_walls_distance_dict = {}
        for wall_name_s in wall_important_points_dict.keys():
            wall_s_i = int(wall_name_s.split('_')[-1])
            for wall_name_e in wall_important_points_dict.keys():
                wall_e_i = int(wall_name_e.split('_')[-1])
                if wall_e_i > wall_s_i:
                    wall_to_wall_dist_vec = []
                    for point_name_s in self.fenv_config['wall_important_points_names']:
                        for point_name_e in self.fenv_config['wall_important_points_names']:
                            s_coord = wall_important_points_dict[wall_name_s][point_name_s]
                            e_coord = wall_important_points_dict[wall_name_e][point_name_e]
                            if (-1 in s_coord) or (-1 in e_coord):
                                d = -1
                            else:
                                d = self.___distance(s_coord, e_coord)
                            wall_to_wall_dist_vec.append(d)      
                    walls_to_walls_distance_dict[f"{wall_name_s}_to_{wall_name_e}"] = copy.deepcopy(wall_to_wall_dist_vec)
        return walls_to_walls_distance_dict
        

    
    def ___distance(self, coord_s, coord_e):
        return np.linalg.norm(np.array(coord_s)-np.array(coord_e))
    
    
    
    def __normalize_walls_state_vector(self, wall_state_vec):
        normalized_wall_state_vec = np.array([val*self.fenv_config['wall_normalizer_factor'] if val != -1 else -1 for val in wall_state_vec])
        return normalized_wall_state_vec