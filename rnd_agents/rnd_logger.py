#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 12:30:35 2023

@author: Reza Kakooee
"""

#%%
import os
import logging
import numpy as np
import pandas as pd
from pprint import pprint



#%%
class RandomAgentLogger:
    def __init__(self, fenv_config, agent_config):
        self.fenv_config = fenv_config
        self.agent_config = agent_config
    
    
    
    def start_interaction_logger(self):
        if self.agent_config['print_verbose'] >= 1:  print(f" ########## Interact for {self.agent_config['n_episodes']} episodes ... ########## ")
                    
    
    
    def start_episode_logger(self, episode_i, observation):
        if self.agent_config['print_verbose'] >= 2:  print(" - - - - -  - - - - - - - - - - - - - - - - - - -- - - -")
        if self.agent_config['print_verbose'] >= 1: print(f"==================================== Episode: {episode_i}")
        if self.agent_config['print_verbose'] >= 3 and self.fenv_config['action_masking_flag']: print(f"n_non_masked_actions: {np.sum(observation['action_mask'])}")
    
    
    
    def in_episode_logger(self, action, observation , reward, done, info):
        if self.agent_config['print_verbose'] >= 2:    
            if self.fenv_config['net_arch'] in ['fccnn', 'cnnfc']:
                experience = f"({action:06}, {observation[0].shape}, {observation[1].shape}, {reward:05}, {done}, {info})" 
            else:
                if self.fenv_config['action_masking_flag']:
                    experience = f"({action:06}, {observation['real_obs'].shape,}, {reward:05}, {done}, {info})"
                else:
                    experience = f"({action:06}, {observation.shape}, {reward:05}, {done}, {info})"
            print(f"- - - - - experience: {experience}") 
                
            if self.fenv_config['action_masking_flag']:
                    if self.agent_config['print_verbose'] >= 3: print(f"\nn_non_masked_actions: {np.sum(observation['action_mask'])}")
                    if self.agent_config['print_verbose'] >= 3: print(f"selected action: {action}")
            
              
    
    def end_episode(self, plan_data_dict, episode_good_action_sequence, end_episode_reward, ep_time_step, episode_finishing_status):
        if self.agent_config['print_verbose'] >= 1: 
            print("========== done!")
            print(f"episode_finishing_status: {episode_finishing_status}")
            print(f"----- good_action_sequence: {episode_good_action_sequence}")
            print(f"----- End episode reward: {end_episode_reward:.2f}")
            print(f"----- Episode len: {ep_time_step:03}\n")
            
        
            areas_delta = plan_data_dict['areas_delta']
            areas_delta_mean = np.mean([abs(da) for da in list(areas_delta.values())]) #[:-1]])
            
            
            print(f"----- n_rooms: {plan_data_dict['n_rooms']}")
            print("----- areas_desired:")
            pprint(plan_data_dict['areas_desired'])
            print("----- areas_achieved: ")
            pprint(plan_data_dict['areas_achieved'])
            print("----- areas_delta:")
            pprint(areas_delta)
            print(f"----- areas_delta_mean: { areas_delta_mean}\n")
            
            
            
            if self.agent_config['agent_name'] != 'RND':
                n_desired_adjacencies = len(plan_data_dict['edge_list_room_desired']) + len(plan_data_dict['edge_list_facade_desired_str'])
                
                room_edge_diff = len(plan_data_dict['edge_color_data_dict_room']['red'])
                facade_edge_diff = len(plan_data_dict['edge_color_data_dict_facade']['red'])
                sum_graph_diff = room_edge_diff + facade_edge_diff
                
                room_names = plan_data_dict['rooms_dict'].keys()                        
                delta_aspect_ratio = list(
                    np.around(
                        [plan_data_dict['rooms_dict'][room_name]['delta_aspect_ratio'] for room_name in plan_data_dict['areas_desired']], 
                        decimals=2)
                    )
                
                
                print(f"----- edge_list_room_desired: {plan_data_dict['edge_list_room_desired']}")
                print(f"----- edge_list_room_achieved: {plan_data_dict['edge_list_room_achieved']}")
                print(f"----- room_edge_diff: {room_edge_diff}\n")
                
                print(f"----- edge_list_facade_desired_str: {plan_data_dict['edge_list_facade_desired_str']}")
                print(f"----- edge_list_facade_achieved_str: {plan_data_dict['edge_list_facade_achieved_str']}")
                print(f"----- facade_edge_diff: {facade_edge_diff}\n")
                
                print(f"----- n_desired_adjacencies: {n_desired_adjacencies}")
                print(f"----- sum_graph_diff: {sum_graph_diff}\n")
                
                print(f"----- edge_performance: {(n_desired_adjacencies - sum_graph_diff)/n_desired_adjacencies * 100}%\n")
            
    
    def end_of_interaction(self, last_episode_rewards, episode_lens, num_failures):
        print("\n==================================== Summary:")
        if self.agent_config['print_verbose'] >= 1: 
            print(f"- - - - - Mean of episode len: {np.mean(episode_lens)}")
            print(f"- - - - - Mean of end episode reward: {np.mean(last_episode_rewards)}")
            print(f"- - - - - Num of episodes: {self.agent_config['n_episodes']}")
            print(f"- - - - - Num of failures: {num_failures}")


    def _save_plan_values(self, env_data_dict):
        if not os.path.exists(self.fenv_config['plan_path_cc']):
            plan_values_df_new = pd.DataFrame.from_dict(env_data_dict, orient='index')
            plan_values_df_new.to_csv(self.fenv_config['plan_path_cc'], index=False)
        else:
            plan_values_df_old = pd.read_csv(self.fenv_config['plan_path_cc'])
            plan_values_df_new = pd.DataFrame.from_dict(env_data_dict, orient='index')
            plan_values_df_old = pd.concat([plan_values_df_old, plan_values_df_new], axis=0)
            plan_values_df_old.to_csv(self.fenv_config['plan_path_cc'], index=False)
