#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 17:48:55 2022

@author: RK
"""

# %% 
import copy
import numpy as np
import networkx as nx
from skimage.future import graph
from skimage import data, segmentation, filters, color


# %%
class LayoutGraph:
    def __init__(self, plan_data_dict):
        self.plan_data_dict = plan_data_dict
        try:
            self.rag = self._obs_to_rag(plan_data_dict)
        except:
            print("wain in LaoutGraph")
            raise ValueError('rag does not exist!')

        
    def _obs_to_rag(self, plan_data_dict):
        moving_labels = np.array(plan_data_dict['moving_labels'])
        moving_labels[moving_labels == 0] = np.max(moving_labels)+1
        moving_labels = moving_labels * plan_data_dict['obs_mat_for_dot_prod']
        moving_labels = moving_labels - plan_data_dict['obs_mat_w']
        self.moving_labels = moving_labels - self.plan_data_dict['mask_numbers']
        
        img = np.expand_dims(moving_labels, axis=-1)
        edge_map = filters.sobel(img)
        edge_map = edge_map[:,:,0]
        rag = graph.rag_boundary(moving_labels, edge_map, connectivity=1)
        return rag
    
    def extract_graph_data(self):
        num_nodes = self.rag.number_of_nodes()
        mask_numbers = self.plan_data_dict['mask_numbers']
        n_corners = 4
        num_nodes = num_nodes - mask_numbers
        
        edge_list = list(self.rag.edges)
        
        edge_list = [ edge for edge in edge_list if ( (edge[0] > n_corners) and ((edge[1] > n_corners)) ) ]
        
        real_drawn_room_i_list = [int(room_name.split('_')[1]) for room_name in list(self.plan_data_dict['delta_areas'].keys())]
        edge_list = [ edge for edge in edge_list if ( (edge[0] in real_drawn_room_i_list) and ((edge[1] in real_drawn_room_i_list)) ) ]
        
        edge_list = [(edge[0]-n_corners, edge[1]-n_corners) for edge in edge_list]
        
        return num_nodes, edge_list
    
    
    def get_adj_matrix(self, num_nodes, edge_list):
        adj_matrix = np.zeros((num_nodes, num_nodes))
        for edge in edge_list:
            adj_matrix[edge[0], edge[1]] = 1
            adj_matrix[edge[1], edge[0]] = 1
        return adj_matrix
    
    
    def show_graph(self, adj_matrix):
        N = nx.from_numpy_matrix(adj_matrix)
        labels_dict = {i:f"Room_{i+1:02}" for i in range(len(adj_matrix))}
        nx.draw(N, with_labels=True, labels=labels_dict)

        
# %%        
if __name__ == '__main__':
    moving_labels = np.array([
                        [3, 3, 3, 1, 1, 1],
                        [3, 3, 3, 2, 2, 2],
                        [3, 3, 3, 2, 2, 2],
                        [4, 4, 4, 6, 6, 6],
                        [4, 4, 4, 7, 7, 7],
                        [5, 5, 5, 7, 7, 7]
                        ])
    obs_mat_for_dot_prod = np.array([
                        [1, 1, 0, 1, 1, 1],
                        [1, 1, 0, 1, 1, 1],
                        [1, 1, 0, 1, 1, 1],
                        [1, 1, 0, 1, 1, 1],
                        [1, 1, 0, 0, 0, 0],
                        [1, 1, 1, 1, 1, 1]
                        ])
    plan_data_dict_ = {'moving_labels': moving_labels, 
                       'obs_mat_for_dot_prod': obs_mat_for_dot_prod,
                       'mask_numbers':2}
    
    self = LayoutGraph(plan_data_dict_)
    num_nodes, edge_list = self.extract_graph_data()
    adj_matrix = self.get_adj_matrix(num_nodes, edge_list)
    self.show_graph(adj_matrix)
    print(adj_matrix)
    
    