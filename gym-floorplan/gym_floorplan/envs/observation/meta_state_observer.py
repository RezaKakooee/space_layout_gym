#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 22:43:27 2023

@author: Reza Kakooee
"""

#%%

import os
import copy
import inspect
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
        
    
    
    def convert_oh2ind(self, inp, N=4): # if N >= 2 = pading
        lst = [0] * N
        idx = np.nonzero(inp)[0]
        if len(idx) != 0:  
              lst[:len(idx)] = (idx + 1)
        return lst
    
        
    
    def _get_meta_state(self, plan_data_dict):
        plan_state_vector = self._get_plan_state_vector(plan_data_dict) # 45
        area_state_vector = self._get_area_state_vector(plan_data_dict) # 18
        proportion_state_vector = self._get_proportion_state_vector(plan_data_dict) # 18
        adjacency_state_vector = self._get_adjacency_state_vector(plan_data_dict) # 198
        area_proportion_adjacency_state_vector = np.concatenate([
            plan_state_vector,
            area_state_vector, 
            proportion_state_vector, 
            adjacency_state_vector
            ]) # 279
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
        if self.fenv_config['oh2ind_flag']:
            plan_state_vec = self.convert_oh2ind(plan_data_dict['plan_description'][:4]) # maskd # 4
            plan_state_vec += self.convert_oh2ind(plan_data_dict['plan_description'][4:8]) # blocked # 4
            plan_state_vec += self.convert_oh2ind(plan_data_dict['plan_description'][8:12]) # entrance # 4
            plan_state_vec += self.convert_oh2ind(plan_data_dict['plan_description'][12:21], N=9) # required rooms # 9
            plan_state_vec += plan_data_dict['plan_description'][21:] # 16
        else:
            plan_state_vec = copy.deepcopy(plan_data_dict['plan_description'])
        plan_description_partially_norm = self.__normalize_plan_state_vector(plan_state_vec) # 37
        
        warooge_data = plan_data_dict['warooge_data_main']
        masked_room_and_facades_area = list(warooge_data['areas_desired'].values())[:(self.fenv_config['num_corners']+self.fenv_config['num_facades'])] # 8
        # areas_achieved_list = list(warooge_data['areas_acheived'].values())[:(self.fenv_config['num_corners']+self.fenv_config['num_facades'])]
        masked_room_and_facades_area_norm = self.__normalize_areas_state_vector(masked_room_and_facades_area)
        plan_state_vector = np.concatenate([plan_description_partially_norm, masked_room_and_facades_area_norm], dtype=float) # 45
        return plan_state_vector
    
    
    
    def _get_area_state_vector(self, plan_data_dict):
        warooge_data = plan_data_dict['warooge_data_main']
        areas_desired_list = list(warooge_data['areas_desired'].values())[-self.fenv_config['maximum_num_real_rooms']:]
        areas_achieved_list = list(warooge_data['areas_acheived'].values())[-self.fenv_config['maximum_num_real_rooms']:]
        areas_delta_list = list(warooge_data['areas_delta'].values())[-self.fenv_config['maximum_num_real_rooms']:]
        area_state_vec = np.abs(areas_desired_list + areas_achieved_list)# + areas_delta_list)
        area_state_vec = self.__normalize_areas_state_vector(area_state_vec) # 18
        return area_state_vec # 18
        
    
        
    def _get_proportion_state_vector(self, plan_data_dict):
        proportion_desired_list = [self.fenv_config['aspect_ratios_tolerance']/2] * self.fenv_config['maximum_num_real_rooms']
        proportion_desired_list[plan_data_dict['n_rooms']:] = [0] * (len(proportion_desired_list) - plan_data_dict['n_rooms'])
        if not self.fenv_config['is_proportion_a_constraint']:
            proportion_achieved_list = [self.fenv_config['desired_aspect_ratio']] * self.fenv_config['maximum_num_real_rooms']
        else:
            proportion_achieved_list = [0] * self.fenv_config['maximum_num_real_rooms']
            for room_name, room_data in plan_data_dict['rooms_dict'].items():
                room_i = int(room_name.split('_')[1])
                if room_i in self.fenv_config['real_room_id_range']:
                    proportion_achieved_list[room_i - self.fenv_config['min_room_id']] = room_data['delta_aspect_ratio'] + self.fenv_config['desired_aspect_ratio']
        proportions_state_vec = proportion_desired_list + proportion_achieved_list
        if None in proportions_state_vec:
            np.save(f"plan_data_dict__{os.path.basename(__file__)}_{self.__class__.__name__}_{inspect.currentframe().f_code.co_name}_1.npy", self.plan_data_dict)
            raise ValueError(f"""proportions_state_vec has None for delta_aspect_ratio.
                               proportions_state_vec is: {proportions_state_vec}, 
                               delta_aspect_ratio: {proportions_state_vec['delta_aspect_ratio']}""")
        proportions_state_vec = self.__normalize_proportions_state_vector(proportions_state_vec)
        return proportions_state_vec # 18
    
    
        
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
        edge_list_facade_achieved = copy.deepcopy(plan_data_dict['edge_list_facade_achieved'])
        
        desired_rr_adj_mat = self._edge_list_to_adjacency_matrix(edge_list_room_desired)
        achieved_rr_adj_mat = self._edge_list_to_adjacency_matrix(edge_list_room_achieved)
        achieved_rf_adj_mat = self._edge_list_room_facade_to_adjacency_matrix(edge_list_facade_achieved)
        
        if self.fenv_config['only_upper_diagonal_adj_flag']:
            room_adj_flat_upper_matrix_desired = self._get_upper_diagonal_elements(desired_rr_adj_mat)
            room_adj_flat_upper_matrix_achieved = self._get_upper_diagonal_elements(achieved_rr_adj_mat)
        
        if self.fenv_config['adaptive_window']:
            if self.fenv_config['room_facade_state_as_adj_matrix_flag']:
                pass
            else:
                room_facade_achieved_one_hot = [0] * self.fenv_config['maximum_num_real_rooms']
                for ed in edge_list_facade_achieved:
                    if ed[1] in self.fenv_config['real_room_id_range']:
                        room_facade_achieved_one_hot[ed[1] - self.fenv_config['min_room_id']] = 1
        else:
            raise NotImplementedError("like room-room connections, we need to define room-facade connections for non-adaptive window situation.")
            
        if self.fenv_config['only_upper_diagonal_adj_flag']:
            edge_state_vec = np.concatenate([room_adj_flat_upper_matrix_desired, room_adj_flat_upper_matrix_achieved, room_facade_achieved_one_hot])
            
        else:
            if self.fenv_config['room_facade_state_as_adj_matrix_flag']:
                edge_state_vec = np.concatenate([desired_rr_adj_mat.flatten(), achieved_rr_adj_mat.flatten(), achieved_rf_adj_mat.flatten()])
                
            else:
                if self.fenv_config['oh2ind_flag']:
                    desired_rr_adj_idx_mat = np.zeros_like(desired_rr_adj_mat, dtype=desired_rr_adj_mat.dtype)
                    achieved_rr_adj_idx_mat = np.zeros_like(achieved_rr_adj_mat, dtype=desired_rr_adj_mat.dtype)
                    achieved_rf_adj_idx_mat = np.zeros_like(achieved_rf_adj_mat, dtype=desired_rr_adj_mat.dtype)
                    for i in range(self.fenv_config['maximum_num_real_rooms']):
                        desired_rr_adj_idx_mat[i, :] = self.convert_oh2ind(desired_rr_adj_mat[i, :], N=self.fenv_config['maximum_num_real_rooms'])
                        achieved_rr_adj_idx_mat[i, :] = self.convert_oh2ind(achieved_rr_adj_mat[i, :], N=self.fenv_config['maximum_num_real_rooms'])
                        achieved_rf_adj_idx_mat[i, :] = self.convert_oh2ind(achieved_rf_adj_mat[i, :])
                        
                        
                else:
                    edge_state_vec = np.concatenate([desired_rr_adj_mat.flatten(), achieved_rr_adj_mat.flatten(), room_facade_achieved_one_hot])
                
        
                
        return edge_state_vec # 198
    


    def remove_self_adjacency(self, arr):
        new_arr = np.array([[row[i] for i in range(len(row)) if i != j] for j, row in enumerate(arr)])
        assert new_arr.shape == (self.fenv_config['maximum_num_real_rooms'], self.fenv_config['maximum_num_real_rooms']-1)
        return new_arr

    
    
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
    
    
    
    def _edge_list_room_facade_to_adjacency_matrix(self, edge_list):
        adj_matrix = np.zeros((self.fenv_config['maximum_num_real_rooms'], 4), dtype=int)
        for edge in edge_list:
            try:
                adj_matrix[edge[1] - self.fenv_config['min_room_id'], edge[0] - self.fenv_config['min_facade_id']] = 1
            except:
                print("")
        return adj_matrix


    @staticmethod
    def _get_upper_diagonal_elements(matrix):
        return [matrix[i][j] for i, row in enumerate(matrix) for j in range(i + 1, len(row))]
