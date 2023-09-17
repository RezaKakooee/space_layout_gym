#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 17:48:55 2022

@author: Reza Kakooee
"""

# %% 
import copy
import itertools
import numpy as np
from collections import defaultdict
import networkx as nx
from skimage import graph
from skimage import data, segmentation, filters, color


# %%
class LayoutGraph:
    def __init__(self, plan_data_dict, fenv_config):
        self.plan_data_dict = plan_data_dict
        self.fenv_config = fenv_config
        
        self.obs_moving_labels_completed = self.plan_data_dict['obs_moving_labels_completed'] # self._complete_obs_moving_labels(plan_data_dict)
        self.obs_moving_labels_completed_projected = self.plan_data_dict['obs_moving_labels_completed_projected'] # self._project_facades_on_masked_room_walls(plan_data_dict, copy.deepcopy(self.obs_moving_labels_completed))
        
        try:
            ml = np.kron(self.obs_moving_labels_completed, np.ones((2, 2), dtype=self.obs_moving_labels_completed.dtype))
            pml = np.kron(self.obs_moving_labels_completed_projected, np.ones((2, 2), dtype=self.obs_moving_labels_completed_projected.dtype))
            
            self.rag = self._create_rag(ml)
            self.projected_rag = self._create_rag(pml)
        except:
            print("wait in LaoutGraph")
            raise ValueError('rag does not exist!')

    

    def _create_rag(self, matrix):
        img = np.expand_dims(matrix, axis=-1)
        edge_map = filters.sobel(img)
        edge_map = edge_map[:,:,0]
        rag = graph.rag_boundary(matrix.astype(int), edge_map, connectivity=1)
        return rag


    
    @staticmethod
    def _remove_edge_including_zero(lst):
        return list(filter(lambda sublist: 0 not in sublist, lst))
    
    
    
    def extract_graph_data(self):
        pure_edge_list, pure_facade_edge_list, pure_entrance_edge_list = self._dense_graph_data_extractor(self.rag)
        projected_pure_edge_list, projected_pure_facade_edge_list, projected_pure_entrance_edge_list = self._dense_graph_data_extractor(self.projected_rag)
        
        if self.plan_data_dict['active_wall_status'] not in ['badly_stopped', 'well_finished']:
            pure_edge_list = self._remove_edge_including_zero(pure_edge_list)
            pure_facade_edge_list = self._remove_edge_including_zero(pure_facade_edge_list)
            
            pure_entrance_edge_list = list(filter(lambda x: x != 0, pure_entrance_edge_list))
            
            projected_pure_edge_list = self._remove_edge_including_zero(projected_pure_edge_list)
            projected_pure_facade_edge_list = self._remove_edge_including_zero(projected_pure_facade_edge_list)
            
            projected_pure_entrance_edge_list = list(filter(lambda x: x != 0, projected_pure_entrance_edge_list))
        
        edge_list_room_achieved = np.unique(pure_edge_list + projected_pure_edge_list, axis=0).tolist()
        edge_list_facade_achieved = np.unique(pure_facade_edge_list + projected_pure_facade_edge_list, axis=0).tolist()
        edge_list_entrance_achieved = np.unique(pure_entrance_edge_list + projected_pure_entrance_edge_list, axis=0).tolist()

        edge_list_room_achieved.sort()
        edge_list_facade_achieved.sort()
        
        edge_list_facade_achieved_str = self._convert_facade_edge_list_to_str(edge_list_facade_achieved)
        edge_list_facade_achieved_str.sort()
        
        edge_list_entrance_achieved_str = self._convert_entrance_edge_list_to_str(edge_list_entrance_achieved)
        
        state_adj_matrix = self._get_adj_matrix(self.fenv_config['max_room_id'], edge_list_room_achieved + edge_list_facade_achieved)
        
        # use the 1st two for making adj_vec, and use the 2nd two for making reward
        
        all_edges = (
            edge_list_room_achieved + 
            edge_list_facade_achieved +
            [[self.fenv_config['entrance_cell_id'], e] for e in edge_list_entrance_achieved]
            )
        
        edge_dict = {
            'edge_list_room_achieved': edge_list_room_achieved,
            'edge_list_facade_achieved': edge_list_facade_achieved,
            'edge_list_facade_achieved_str': edge_list_facade_achieved_str,
            'edge_list_entrance_achieved': [[self.fenv_config['entrance_cell_id'], e] for e in edge_list_entrance_achieved],
            'edge_list_entrance_achieved_str': edge_list_entrance_achieved_str,
            'state_adj_matrix': state_adj_matrix,
            'all_eges': all_edges
            }
        
        return edge_dict
    
    
    
    def _dense_graph_data_extractor(self, rg):
        num_nodes = rg.number_of_nodes()
        edge_list = list(rg.edges)
        pure_edge_list, pure_facade_edge_list, pure_entrance_edge_list = self.__separate_facade_edge_list_from_edge_list(edge_list)
        return pure_edge_list, pure_facade_edge_list, pure_entrance_edge_list
        
    
        
    def __separate_facade_edge_list_from_edge_list(self, edge_list):
        pure_facade_edge_list = []
        pure_edge_list = []
        pure_entrance_edge_list = []
        for edge in edge_list:
            edge = list(edge)
            if self.fenv_config['entrance_cell_id'] in edge: # cells connected to the entrance go to the entrance edge list
                edge.remove(self.fenv_config['entrance_cell_id'])
                pure_entrance_edge_list.append(edge[0])
            else:
                if self.fenv_config['very_corner_cell_id'] not in edge: # here I remove the very corner cell
                    if not ( (edge[0] in self.fenv_config['facade_id_range']) and
                             (edge[1] in self.fenv_config['facade_id_range']) ):
                       if ( (edge[0] in self.fenv_config['facade_id_range']) or
                            (edge[1] in self.fenv_config['facade_id_range']) ):
                            if edge not in pure_facade_edge_list:
                                pure_facade_edge_list.append(edge)
                       else:
                            ed = [min(edge), max(edge)]
                            if ed not in pure_edge_list:
                                pure_edge_list.append(ed)
        
        pure_edge_list.sort()
        pure_facade_edge_list.sort()    
        return pure_edge_list, pure_facade_edge_list, pure_entrance_edge_list
    
    
    
    def _convert_facade_edge_list_to_str(self, edge_list_facade_achieved):
        edge_list_facade_achieved_str = []
        for edge in edge_list_facade_achieved:
            if edge[0] in self.fenv_config['facade_id_range']:
                ed = [self.fenv_config['facade_id_to_name_dict'][edge[0]], edge[1]]
            elif edge[1] in self.fenv_config['facade_id_range']:
                ed = [self.fenv_config['facade_id_to_name_dict'][edge[1]], edge[0]]
            else:
                raise ValueError("Either edge[0] or edge[1] must be in self.fenv_config['facade_id_range']")
            edge_list_facade_achieved_str.append(ed)
        return edge_list_facade_achieved_str
                
    
    
    def _convert_entrance_edge_list_to_str(self, edge_list_entrance_achieved):
        edge_list_entrance_achieved_str = [['d', self.plan_data_dict['entrance_is_on_facade']]]
        for node in edge_list_entrance_achieved:
            if node in self.fenv_config['facade_id_range']:
                if ['d', self.fenv_config['facade_id_to_name_dict'][node]] not in edge_list_entrance_achieved_str:
                    edge_list_entrance_achieved_str.append(['d', self.fenv_config['facade_id_to_name_dict'][node]])
            elif node in self.fenv_config['real_room_id_range']:
                if ['d', node] not in edge_list_entrance_achieved_str:
                    edge_list_entrance_achieved_str.append(['d', node])
        return edge_list_entrance_achieved_str
    
    
    
    def _get_adj_matrix(self, num_nodes, edge_list):
        try:
            state_adj_matrix = np.zeros((num_nodes, num_nodes))
            for edge in edge_list:
                state_adj_matrix[edge[0]-1, edge[1]-1] = 1
                state_adj_matrix[edge[1]-1, edge[0]-1] = 1
        except:
            print('in layout graph: some index might be wrong')
            raise IndexError('some index might be wrong')
        return state_adj_matrix
    
    
    
    # def _get_sparsed_reduced_edges(self, edge_list_room_achieved_, edge_list_facade_achieved_):
    #     sparsed_reduced_edge_list_room_achieved_ = []
    #     for edge in edge_list_room_achieved_:
    #         if min(edge) > self.fenv_config['maximum_num_masked_rooms']:
    #             reduced_edge = [edge[0]-self.fenv_config['maximum_num_masked_rooms'], edge[1]-self.fenv_config['maximum_num_masked_rooms']]
    #             sparsed_reduced_edge_list_room_achieved_.append(reduced_edge)
        
    #     sparsed_reduced_edge_list_facade_achieved_ = []
    #     for edge in edge_list_facade_achieved_:
    #         if min(edge) > self.fenv_config['maximum_num_masked_rooms']:
    #             max_idx = edge.index(max(edge))
    #             min_idx = edge.index(min(edge))
    #             reduced_mapped_edge = [self.fenv_config['facade_id_to_name_dict'][max(edge)], edge[min_idx] - self.fenv_config['maximum_num_masked_rooms']]
    #             sparsed_reduced_edge_list_facade_achieved_.append(reduced_mapped_edge)
            
    #     return sparsed_reduced_edge_list_room_achieved_, sparsed_reduced_edge_list_facade_achieved_
    
            

    
#%%
if __name__ == "__main__":
     m = np.array(
       [[19, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 19],
        [18,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  7,  7,  7,  7, 17],
        [18,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  7,  7,  7,  7, 17],
        [18,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  7,  7,  7,  7, 17],
        [18,  9,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  7,  7,  7,  7, 17],
        [18,  9,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  7,  7,  7,  7, 17],
        [18,  9,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  7,  7,  7,  7, 17],
        [18,  9,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  7,  7,  7,  7, 17],
        [18,  9,  8,  8,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  7,  7,  7,  7, 17],
        [18,  9,  8,  8,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  7,  7,  7,  7, 17],
        [18,  9,  8,  8,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  7,  7,  7,  7, 17],
        [18,  9,  8,  8,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  7,  7,  7,  7, 17],
        [18,  9,  8,  8,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  7,  7,  7,  7, 17],
        [18,  9,  8,  8,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  7,  7,  7,  7, 17],
        [16, 16, 16, 16, 16, 16, 19,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  7,  7,  7,  7, 17],
        [ 1,  1,  1,  1,  1,  1, 18,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  7,  7,  7,  7, 17],
        [ 1,  1,  1,  1,  1,  1, 18,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  7,  7,  7,  7, 14],
        [ 1,  1,  1,  1,  1,  1, 18,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  7,  7,  7,  7, 14],
        [ 1,  1,  1,  1,  1,  1, 18,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  7,  7,  7,  7, 17],
        [ 1,  1,  1,  1,  1,  1, 18,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  7,  7,  7,  7, 17],
        [ 1,  1,  1,  1,  1,  1, 18,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  7,  7,  7,  7, 17],
        [ 1,  1,  1,  1,  1,  1, 18,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  7,  7,  7,  7, 17],
        [ 1,  1,  1,  1,  1,  1, 18, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 19]])
 
 
     m = np.kron(m, np.ones((2, 2), dtype=m.dtype)) 
     # since the connection betwween 9 and 16 is very narrow, rag does not recognized their connection.
     # so I kron the me to thicker the 9. then it works
     
     img = np.expand_dims(m, axis=-1)
     edge_map = filters.sobel(img)
     edge_map = edge_map[:,:,0]
     rag = graph.rag_boundary(m.astype(int), edge_map, connectivity=1)
     num_nodes = rag.number_of_nodes()
     edge_list = list(rag.edges)