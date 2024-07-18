#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 17:43:42 2022

@author: Reza Kakooee
"""
#%%
import copy
import numpy as np

import torch

from gym_floorplan.envs.observation.fc_state_observer import FcStateObserver
from gym_floorplan.envs.observation.cnn_state_observer import CnnStateObserver
from gym_floorplan.envs.observation.meta_state_observer import MetaStateObserver
from gym_floorplan.envs.observation.graph_state_observer import GraphStateObserver



#%%
class StateComposer:
    def __init__(self, fenv_config):
        self.fenv_config = fenv_config
        
    #     if (self.fenv_config['net_arch'] == 'Gnn') and (self.fenv_config['gnn_obs_method'] == 'embedded_image_graph'):
    #         self._load_context_model()



    # def _load_context_model(self):
    #         from gym_floorplan.envs.observation.context_model import ContextModel
    #         net_config = {'_in_channels_cnn': 2,
    #                       'dim_pd_features': 4,
    #                       'dim_pc_features': 4,
    #                       'dim_fc_features': 18}
            
    #         self.device = 'cpu' # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #         self.model = ContextModel(net_config)
    #         self.model = torch.load(self.fenv_config['context_mdoel_path'], map_location ='cpu')
    #         self.model = self.model.to(self.device)
        
        
        
    def creat_observation_space_variables(self):
        if self.fenv_config['env_planning'] == 'One_Shot': # we need the last list as it contains 'fc'
            if self.fenv_config['use_areas_info_into_observation_flag']:
                lowest_obs_val_fc = -self.fenv_config['total_area']
                highest_obs_val_fc = self.fenv_config['total_area']
            else:
                lowest_obs_val_fc = -1
                highest_obs_val_fc = 30

        
        low_fc = lowest_obs_val_fc #* np.ones(len_state_vec, dtype=float)
        high_fc = highest_obs_val_fc #* np.ones(len_state_vec, dtype=float)
        shape_fc = (self.fenv_config['len_feature_state_vec'],)
        
        low_cnn = 0
        high_cnn = 255
        shape_cnn = ( (self.fenv_config['n_channels'],
                      (self.fenv_config['n_rows']) * self.fenv_config['cnn_scaling_factor'],
                      (self.fenv_config['n_cols']) * self.fenv_config['cnn_scaling_factor'] ))
        
        low_meta = 0
        high_meta = highest_obs_val_fc
        shape_meta = (self.fenv_config['len_meta_state_vec'],)
        
        low_metafc = 0
        high_metafc = highest_obs_val_fc
        shape_metafc = (self.fenv_config['len_state_vec'],)
            
        low_gnn = -500
        high_gnn = 500
        shape_gnn = (self.fenv_config['num_nodes'], 460) # TODO why 256?
        
        self.state_data_dict = {
            'low_fc': low_fc,
            'high_fc': high_fc,
            'shape_fc': shape_fc,
            
            'low_cnn': low_cnn,
            'high_cnn': high_cnn,
            'shape_cnn': shape_cnn,
            
            'low_meta': low_meta,
            'high_meta': high_meta,
            'shape_meta': shape_meta,
            
            'low_metafc': low_metafc,
            'high_metafc': high_metafc,
            'shape_metafc': shape_metafc,
            
            'low_gnn': low_gnn,
            'high_gnn': high_gnn,
            'shape_gnn': shape_gnn,
            
            'len_feature_state_vec': self.fenv_config['len_feature_state_vec'],
            'len_plan_state_vec': self.fenv_config['len_plan_state_vec'],
            'len_area_state_vec': self.fenv_config['len_area_state_vec'],
            'len_proportion_state_vec': self.fenv_config['len_proportion_state_vec'],
            'len_adjacency_state_vec': self.fenv_config['len_adjacency_state_vec'],
            'len_meta_state_vec': self.fenv_config['len_meta_state_vec'],
            'len_state_vec': self.fenv_config['len_state_vec'],
            }
        
        return self.state_data_dict
    
    
    
    def warooge_data_extractor(self, plan_data_dict):
        main_wall_important_points = self._get_wall_important_points(plan_data_dict)

        
        areas_desired_main = copy.deepcopy(self.fenv_config['room_desired_area_dict_template'])
        areas_desired_main.update(plan_data_dict['areas_achieved'])
        areas_desired_main.update(plan_data_dict['areas_desired'])
        
        areas_acheived_main = copy.deepcopy(self.fenv_config['room_achieved_area_dict_template'])
        areas_acheived_main.update(plan_data_dict['areas_achieved'])
        
        for i in self.fenv_config['facade_id_range']:
            a = np.sum(plan_data_dict['obs_mat_w'] == -i)
            areas_desired_main[f'room_{i}'] = a
            areas_acheived_main[f'room_{i}'] = a
        
        areas_delta_main = copy.deepcopy(self.fenv_config['room_delta_area_dict_template'])
        areas_delta_main.update(plan_data_dict['areas_delta'])

        
        edge_list_room_desired_main = plan_data_dict['edge_list_room_desired']
        edge_list_room_achieved_main = plan_data_dict['edge_list_room_achieved']
        
        edge_list_facade_desired_main = plan_data_dict['edge_list_facade_desired']
        edge_list_facade_achieved_main = plan_data_dict['edge_list_facade_achieved']

        
        warooge_data_main = {
            'wall_important_points_dict': main_wall_important_points,
            'areas_desired': areas_desired_main,
            'areas_acheived': areas_acheived_main,
            'areas_delta': areas_delta_main,
            'edge_list_room_desired': edge_list_room_desired_main,
            'edge_list_room_achieved': edge_list_room_achieved_main,
            'edge_list_facade_desired': edge_list_facade_desired_main,
            'edge_list_facade_achieved': edge_list_facade_achieved_main,
            'obs_moving_labels': copy.deepcopy(plan_data_dict['obs_moving_labels']),
            }
        
        plan_data_dict.update({'warooge_data_main': warooge_data_main})
        
        return plan_data_dict
    
     
        
    def _get_wall_important_points(self, plan_data_dict):
        wall_important_points_dict = {f'wall_{i}': copy.deepcopy(self.fenv_config['wall_coords_template']) for i in range(2, self.fenv_config['cell_id_max'])} 
        
        if self.fenv_config['wall_1_included']:
            wall_important_points_dict.update(self.fenv_config['wall_1'])
        
        for wall_name, wall_data in plan_data_dict['walls_coords'].items():
            wall_i = int(wall_name.split('_')[1])
            if wall_i < self.fenv_config['cell_id_max']:
                try:
                    back_segment = wall_data['back_segment']
                    front_segment = wall_data['front_segment']
                    wall_important_points_dict[wall_name].update({ 'start_of_wall': list(back_segment['reflection_coord']),
                                                                   'before_anchor': list(back_segment['end_coord']),
                                                                   'anchor':        list(back_segment['start_coord']),
                                                                   'after_anchor':  list(front_segment['end_coord']),
                                                                   'end_of_wall':   list(front_segment['reflection_coord']) })
                except:
                    print("wait in _wall_data_extractor_for_single_agent of observation")
                    raise ValueError('Probably number of walls does not match with the number of rooms')
        
        coords = [[0, 0], [self.fenv_config['max_x'], 0], [0, self.fenv_config['max_y']], [self.fenv_config['max_x'], self.fenv_config['max_x']]]
        fake_wall_points = {f"wall_{i}": coord for i, coord in zip(self.fenv_config['fake_room_id_range'], coords)}
        
        for fwall_name, fwall_coord in fake_wall_points.items():
            if fwall_name not in plan_data_dict['walls_coords'].keys():
                for key in wall_important_points_dict[fwall_name].keys():
                    wall_important_points_dict[fwall_name][key] = fwall_coord
        
        x = (plan_data_dict['entrance_coords'][0][0] + plan_data_dict['entrance_coords'][1][0]) / 2.0
        y = (plan_data_dict['entrance_coords'][0][1] + plan_data_dict['entrance_coords'][1][1]) / 2.0        
        wall_important_points_dict[f"wall_{self.fenv_config['entrance_cell_id']}"] = {
            'start_of_wall': plan_data_dict['extended_entrance_coords'][2], 
            'before_anchor': plan_data_dict['extended_entrance_coords'][0], 
            'anchor': [x, y], 
            'after_anchor': plan_data_dict['extended_entrance_coords'][1], 
            'end_of_wall': plan_data_dict['extended_entrance_coords'][3]
                
            }
        return wall_important_points_dict
    
   
    
    def refine_moving_labels(self, plan_data_dict):
        obs_moving_labels_completed = self._complete_moving_labels(plan_data_dict)
        plan_data_dict.update({
            'obs_moving_labels_completed': obs_moving_labels_completed,
            })
        
        obs_moving_labels_completed_projected = obs_moving_labels_refined = self._project_facades_on_masked_room_walls(plan_data_dict)
        plan_data_dict.update({
            'obs_moving_labels_completed_projected': obs_moving_labels_completed_projected,
            'obs_moving_labels_refined': obs_moving_labels_refined,
            })
        return plan_data_dict
        
   
   
    def _complete_moving_labels(self, plan_data_dict):
        obs_moving_labels_completed = copy.deepcopy(np.array(plan_data_dict['obs_moving_labels']))
        obs_moving_labels_completed = self._add_walls_to_moving_labels(obs_moving_labels_completed, plan_data_dict)
        obs_moving_labels_completed = self._add_boarder_to_moving_labels(obs_moving_labels_completed, plan_data_dict)
        obs_moving_labels_completed = self._add_entrance_to_moving_labels(obs_moving_labels_completed, plan_data_dict)
        return obs_moving_labels_completed
        
    
    
    def _add_walls_to_moving_labels(self, obs_moving_labels_, plan_data_dict):
        if plan_data_dict['active_wall_status'] in ['badly_stopped', 'well_finished']:
            obs_moving_labels_[obs_moving_labels_ == 0] = np.max(obs_moving_labels_)+1 # filling with last room id
        obs_moving_labels_ = obs_moving_labels_ * plan_data_dict['obs_mat_for_dot_prod']
        obs_moving_labels_ = obs_moving_labels_ - plan_data_dict['obs_mat_w']
        return obs_moving_labels_
    
    
        
    def _add_boarder_to_moving_labels(self, obs_moving_labels_, plan_data_dict):
        obs_moving_labels_[0, :] = plan_data_dict['only_boarder'][0, :]
        obs_moving_labels_[-1, :] = plan_data_dict['only_boarder'][-1, :]
        obs_moving_labels_[:, 0] = plan_data_dict['only_boarder'][:, 0]
        obs_moving_labels_[:, -1] = plan_data_dict['only_boarder'][:, -1]
        return obs_moving_labels_
    
        
    
    def _add_entrance_to_moving_labels(self, obs_moving_labels_, plan_data_dict):
        for r, c in plan_data_dict['entrance_positions']:
            obs_moving_labels_[r][c] = self.fenv_config['entrance_cell_id']
        return obs_moving_labels_
    
        
    
    def _project_facades_on_masked_room_walls(self, plan_data_dict):
        obs_moving_labels_completed_projected = copy.deepcopy(plan_data_dict['obs_moving_labels_completed'])
        walls_coords = plan_data_dict['walls_coords']
        fake_wall_names = [f"wall_{i}" for i in self.fenv_config['fake_room_id_range']]
        wall_names = list(walls_coords.keys()) # [f'wall_{i}' for i in range(1,5)]
        for wall_name in wall_names:
            if wall_name in fake_wall_names:
                wall_data = walls_coords[wall_name]
                anchor_coord = wall_data['anchor_coord']
                anchor_pos = self._cartesian2image_coord(anchor_coord[0], anchor_coord[1], self.fenv_config['max_y'])
                wall_positions = wall_data['wall_positions']
                
                if wall_name == fake_wall_names[0]:
                    for pos in wall_positions:
                        if pos[0] == anchor_pos[0] and pos[1] != anchor_pos[1]:
                            obs_moving_labels_completed_projected[pos[0]][pos[1]] = self.fenv_config['south_facade_id']
                        if pos[1] == anchor_pos[1] and pos[0] != anchor_pos[0]:
                            obs_moving_labels_completed_projected[pos[0]][pos[1]] = self.fenv_config['west_facade_id']
                            
                    obs_moving_labels_completed_projected[anchor_pos[0]][anchor_pos[1]] = self.fenv_config['very_corner_cell_id']
                    
                    
                elif wall_name == fake_wall_names[1]:
                    for pos in wall_positions:
                        if pos[0] == anchor_pos[0] and pos[1] != anchor_pos[1]:
                            obs_moving_labels_completed_projected[pos[0]][pos[1]] = self.fenv_config['north_facade_id']
                        if pos[1] == anchor_pos[1] and pos[0] != anchor_pos[0]:
                            obs_moving_labels_completed_projected[pos[0]][pos[1]] = self.fenv_config['west_facade_id']
                            
                    obs_moving_labels_completed_projected[anchor_pos[0]][anchor_pos[1]] = self.fenv_config['very_corner_cell_id']
                    
                    
                elif wall_name == fake_wall_names[2]:
                    for pos in wall_positions:
                        if pos[0] == anchor_pos[0] and pos[1] != anchor_pos[1]:
                            obs_moving_labels_completed_projected[pos[0]][pos[1]] = self.fenv_config['south_facade_id']
                        if pos[1] == anchor_pos[1] and pos[0] != anchor_pos[0]:
                            obs_moving_labels_completed_projected[pos[0]][pos[1]] = self.fenv_config['east_facade_id']
                            
                    obs_moving_labels_completed_projected[anchor_pos[0]][anchor_pos[1]] = self.fenv_config['very_corner_cell_id']
                    
                    
                elif wall_name == fake_wall_names[3]:
                    for pos in wall_positions:
                        if pos[0] == anchor_pos[0] and pos[1] != anchor_pos[1]:
                            obs_moving_labels_completed_projected[pos[0]][pos[1]] = self.fenv_config['north_facade_id']
                        if pos[1] == anchor_pos[1] and pos[0] != anchor_pos[0]:
                            obs_moving_labels_completed_projected[pos[0]][pos[1]] = self.fenv_config['east_facade_id']
                            
                    obs_moving_labels_completed_projected[anchor_pos[0]][anchor_pos[1]] = self.fenv_config['very_corner_cell_id']
                    
        
        for r, c in plan_data_dict['entrance_positions']:
            obs_moving_labels_completed_projected[r][c] = self.fenv_config['entrance_cell_id']
                        
        return obs_moving_labels_completed_projected


    
    @staticmethod
    def _cartesian2image_coord(x, y, max_y):
        return max_y-y, x
    
    
    
    def create_x_observation(self, plan_data_dict, active_wall_name):
        if self.fenv_config['phase'] == 'test':
            plan_data_dict = FcStateObserver(self.fenv_config).get_observation(plan_data_dict, active_wall_name)
            plan_data_dict = CnnStateObserver(self.fenv_config).get_observation(plan_data_dict)
            plan_data_dict = GraphStateObserver(self.fenv_config).get_observation(plan_data_dict, active_wall_name)
            
        else:
            if self.fenv_config['net_arch'] in ['Fc', 'MetaFc', 'Gnn']:
                plan_data_dict = FcStateObserver(self.fenv_config).get_observation(plan_data_dict, active_wall_name)
                
                
            if self.fenv_config['net_arch'] in ['Cnn', 'MetaCnn']:
                plan_data_dict = CnnStateObserver(self.fenv_config).get_observation(plan_data_dict)
                
                
            if self.fenv_config['net_arch'] in ['Gnn']:
                plan_data_dict = GraphStateObserver(self.fenv_config).get_observation(plan_data_dict, active_wall_name)
    
            
        if 'Meta' in self.fenv_config['net_arch']:
            plan_data_dict = MetaStateObserver(self.fenv_config).get_observation(plan_data_dict)
            
            
        if self.fenv_config['net_arch'] == 'MetaFc':
            observation_metafc = {
                'observation_fc': plan_data_dict['observation_fc'],
                'observation_meta': plan_data_dict['observation_meta']
                }
            plan_data_dict.update({
                'observation_metafc': observation_metafc
                })
        elif self.fenv_config['net_arch'] == 'MetaCnn':
            observation_metacnn = {
                'observation_cnn': plan_data_dict['observation_cnn'],
                'observation_meta': plan_data_dict['observation_meta']
                }
            plan_data_dict.update({
                'observation_metacnn': observation_metacnn
                })
        
        return plan_data_dict
       
       
