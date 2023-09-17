#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 22:43:27 2023

@author: Reza Kakooee
"""

#%%
import copy
import numpy as np




#%%
class MetaStateObserver:
    def __init__(self, fenv_config):
        self.fenv_config = fenv_config
        
        
        
    def get_observation(self, plan_data_dict):
        plan_data_dict, observatoin_meta = self._get_meta_state(plan_data_dict)
        plan_data_dict.update({
            'observation_meta':observatoin_meta
            })
        return plan_data_dict
        
        
        
    def _get_meta_state(self, plan_data_dict):
        plan_state_vector = self._get_plan_state_vector(plan_data_dict)
        area_state_vector = self._get_area_state_vector(plan_data_dict)
        proportion_state_vector = self._get_proportion_state_vector(plan_data_dict)
        adjacency_state_vector = self._get_adjacency_state_vector(plan_data_dict)
        area_proportion_adjacency_state_vector = np.concatenate([
            plan_state_vector,
            area_state_vector, 
            proportion_state_vector, 
            adjacency_state_vector
            ])
        area_proportion_adjacency_state_dict = {
            'plan_state_vector': plan_state_vector,
            'area_state_vector': area_state_vector,
            'proportion_state_vector': proportion_state_vector,
            'adjacency_state_vector': adjacency_state_vector,
            }
        plan_data_dict.update({
            'area_proportion_adjacency_state_dict': area_proportion_adjacency_state_dict
            })
        return plan_data_dict, area_proportion_adjacency_state_vector
    
    
    
    def _get_plan_state_vector(self, plan_data_dict):
        plan_state_vector = self.__normalize_plan_state_vector(plan_data_dict['plan_description'])
        return plan_state_vector
    
    
    
    def _get_area_state_vector(self, plan_data_dict):
        warooge_data = plan_data_dict['warooge_data_main']
        areas_desired_list = list(warooge_data['areas_desired'].values())
        areas_achieved_list = list(warooge_data['areas_acheived'].values())
        areas_delta_list = list(warooge_data['areas_delta'].values())
        area_state_vec = np.abs(areas_desired_list + areas_achieved_list)# + areas_delta_list)
        area_state_vec = self.__normalize_areas_state_vector(area_state_vec)
        return area_state_vec
        
    
        
    def _get_proportion_state_vector(self, plan_data_dict):
        proportion_desired_list = [self.fenv_config['aspect_ratios_tolerance']/2] * self.fenv_config['maximum_num_real_rooms']
        proportion_desired_list[plan_data_dict['n_rooms']:] = [0] * (len(proportion_desired_list) - plan_data_dict['n_rooms'])
        if not self.fenv_config['is_proportion_considered']:
            proportion_achieved_list = [self.fenv_config['desired_aspect_ratio']] * self.fenv_config['maximum_num_real_rooms']
        else:
            proportion_achieved_list = [0] * self.fenv_config['maximum_num_real_rooms']
            for room_name, room_data in plan_data_dict['rooms_dict'].items():
                room_i = int(room_name.split('_')[1])
                if room_i in self.fenv_config['real_room_id_range']:
                    proportion_achieved_list[room_i - self.fenv_config['min_room_id']] = room_data['delta_aspect_ratio'] + self.fenv_config['desired_aspect_ratio']
        proportions_state_vec = proportion_desired_list + proportion_achieved_list
        if None in proportions_state_vec:
            print("Oh wait")
            raise ValueError("proportions_state_vec has None for delta_aspect_ratio")
        proportions_state_vec = self.__normalize_proportions_state_vector(proportions_state_vec)
        return proportions_state_vec
    
    
        
    def __normalize_plan_state_vector(self, plan_state_vec):
        plan_state_vec[self.fenv_config['len_plan_state_vec_one_hot']:] = list(np.array(plan_state_vec[self.fenv_config['len_plan_state_vec_one_hot']:]) * self.fenv_config['plan_normalizer_factor'])
        return plan_state_vec
    
    
        
    def __normalize_areas_state_vector(self, area_state_vec):
        normalized_area_state_vec = np.array([val*self.fenv_config['room_normalizer_factor'] if val != -1 else -1 for val in area_state_vec])
        return normalized_area_state_vec
    
    
    
    def __normalize_proportions_state_vector(self, proportions_state_vec):
        normalized_proportion_state_vec = np.array(proportions_state_vec) * self.fenv_config['proportion_normalizer_factor']
        return normalized_proportion_state_vec
        
        
        
    def _get_adjacency_state_vector(self, plan_data_dict):
        edge_list_room_desired = copy.deepcopy(plan_data_dict['edge_list_room_desired'])
        edge_list_room_achieved = copy.deepcopy(plan_data_dict['edge_list_room_achieved'])
        room_adj_flat_upper_matrix_desired = self._get_upper_diagonal_elements(self._edge_list_to_adjacency_matrix(edge_list_room_desired))
        room_adj_flat_upper_matrix_achieved = self._get_upper_diagonal_elements(self._edge_list_to_adjacency_matrix(edge_list_room_achieved))
        
        edge_list_facade_achieved = copy.deepcopy(plan_data_dict['edge_list_facade_achieved'])
        room_facade_achieved_one_hot = [0] * self.fenv_config['maximum_num_real_rooms']
        for ed in edge_list_facade_achieved:
            if ed[1] in self.fenv_config['real_room_id_range']:
                room_facade_achieved_one_hot[ed[1] - self.fenv_config['min_room_id']] = 1
        
        edge_state_vec = np.concatenate([room_adj_flat_upper_matrix_desired, room_adj_flat_upper_matrix_achieved, room_facade_achieved_one_hot])
        # state_adj_matrix = plan_data_dict['state_adj_matrix']
        # edge_state_vec = state_adj_matrix.flatten()
        return edge_state_vec
    
    
    
    def _edge_list_to_adjacency_matrix(self, edge_list):
        num_vertices = self.fenv_config['maximum_num_real_rooms']
        adj_matrix = np.zeros((num_vertices, num_vertices), dtype=int)
        for edge in edge_list:
            if min(edge) in self.fenv_config['real_room_id_range']:
                source_vertex = edge[0] - self.fenv_config['min_room_id']
                target_vertex = edge[1] - self.fenv_config['min_room_id']
                adj_matrix[source_vertex][target_vertex] = 1
                adj_matrix[target_vertex][source_vertex] = 1  # If the graph is undirected
        return adj_matrix


    @staticmethod
    def _get_upper_diagonal_elements(matrix):
        return [matrix[i][j] for i, row in enumerate(matrix) for j in range(i + 1, len(row))]
