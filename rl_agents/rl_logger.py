#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 10:57:56 2023

@author: rdbt
"""

import numpy as np

class RlLogger:
    def __init__(self, fevn_config, agent_config):
        self.fevn_config = fevn_config
        self.agent_config = agent_config
        
        self.percentage_satisfied_adjacency_list = []
        self.episode_lens = []

        
    
    def start_episode_logger(self):
        pass
    
    
    
    def in_episode_logger(self):
        pass
              
    
    
    def end_episode_logger(self, plan_data_dict, ep_last_reward, episode_len):
        self.plan_data_dict = plan_data_dict
        
        areas_delta_abs = np.abs(list(self.plan_data_dict['areas_delta'].values()))
        areas_diff_mean = np.mean(areas_delta_abs)
            
        n_desired_adjacencies = len(self.plan_data_dict['edge_list_room_desired']) + len(self.plan_data_dict['edge_list_facade_desired_str'])
        if not n_desired_adjacencies:
            n_desired_adjacencies = 1
        
        sum_graph_diff = n_desired_adjacencies
        if 'edge_color_data_dict_room' in self.plan_data_dict.keys():
            room_edge_diff = len(self.plan_data_dict['edge_color_data_dict_room']['red'])
            facade_edge_diff = len(self.plan_data_dict['edge_color_data_dict_facade']['red'])
            sum_graph_diff = room_edge_diff + facade_edge_diff
        
        percentage_satisfied_adjacency = round((n_desired_adjacencies - sum_graph_diff) / n_desired_adjacencies * 100, 2)
        # self.edge_diff_list.append(edge_diff+facade_edge_diff)
        
        
        self.percentage_satisfied_adjacency_list.append(percentage_satisfied_adjacency)
        self.episode_lens.append(episode_len)
        
        
        # room_names = self.plan_data_dict['rooms_dict'].keys()
        # delta_aspect_ratio = list(np.around([self.plan_data_dict['rooms_dict'][room_name]['delta_aspect_ratio'] 
        #                       for room_name in list(room_names)[self.plan_data_dict['mask_numbers']:]], decimals=2))
        
        
        if self.agent_config['test_end_episode_verbose'] >= 2: 
            print('===============================================================')
            print(f"----- areas_desired: { self.plan_data_dict['areas_desired'] }")
            print(f"----- areas_delta: { areas_delta_abs }")
            print(f"----- areas_achieved: { self.plan_data_dict['areas_achieved']}")
            print(f"----- areas_diff_mean: { areas_diff_mean}\n")
            
            print('===============================================================')
            print(f"----- edge_list_room_desired: {self.plan_data_dict['edge_list_room_desired']}")
            print(f"----- edge_list_room_achieved: {self.plan_data_dict['edge_list_room_achieved']}")
            print(f"----- room_edge_diff: {room_edge_diff}")
            print(f"----- facade_edge_diff: {facade_edge_diff}")
            print(f"----- n_desired_adjacencies: {n_desired_adjacencies}")
            print(f"----- sum_graph_diff: {sum_graph_diff}")
            print(f"----- percentage_satisfied_adjacency: {percentage_satisfied_adjacency}\n")
            
            print(f"----- edge_list_room_desired_facade_achieved_str: {self.plan_data_dict['edge_list_facade_desired_str']}")
            print(f"----- edge_list_facade_achieved_str: {self.plan_data_dict['edge_list_facade_achieved_str']}\n")
            
            print('===============================================================')        
            print(f"----- ep_last_reward: {ep_last_reward}\n")
            
        if self.agent_config['test_end_episode_verbose'] >= 0 and len(self.episode_lens) % 10 == 0: 
            print(f"Min performance: {np.min(self.percentage_satisfied_adjacency_list)}") 
            print(f"Avg performance: {np.mean(self.percentage_satisfied_adjacency_list)}")
            print(f"Std performance: {np.std(self.percentage_satisfied_adjacency_list)}")
            print(f"Max performance: {np.max(self.percentage_satisfied_adjacency_list)}\n")
            
            print(f"Avg ep len: {np.mean(self.episode_lens)}")
            print(f"Std ep len: {np.std(self.episode_lens)}\n")
            
            
            # print(f"----- delta_aspect_ratio: {delta_aspect_ratio}")
            # print(f"----- desired_aspect_ratio: {self.env.fenv_config['desired_aspect_ratio']}")
            # print(f"----- sum_delta_aspect_ratio: {np.sum(delta_aspect_ratio):.2f}")
            # print(f"----- mean_delta_aspect_ratio: {np.mean(delta_aspect_ratio):.2f}")
        