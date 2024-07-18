#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 16:54:47 2023

@author: Reza Kakooee
"""

import os
import inspect
import numpy as np
import pandas as pd
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns



#%%
class Metrics:
    def __init__(self, fenv_config):
        self.fenv_config = fenv_config

        self.final_info_dict = defaultdict()
        if len(fenv_config) >= 1:
            self.final_info_dict.update({
                'project_name': self.fenv_config['project_name'],
                'scenario_config': {
                    'scenario_name': self.fenv_config['scenario_name'],
                    'action_masking_flag': self.fenv_config['action_masking_flag'],
                    'agent_name': self.fenv_config['agent_name'],
                    'cnn_observation_name': self.fenv_config['cnn_observation_name'],
                    'cnn_scaling_factor': self.fenv_config['cnn_scaling_factor'],
                    'model_last_name': self.fenv_config['model_last_name'],
                    'model_source': self.fenv_config['model_source'],
                    'plan_config_source_name': self.fenv_config['plan_config_source_name'],
                    'reward_vertical_scalar': self.fenv_config['reward_vertical_scalar'],
                    'rewarding_method_name': self.fenv_config['rewarding_method_name'],
                    'shift_zc_reward_bottom_to_zero': self.fenv_config['shift_zc_reward_bottom_to_zero'],
                    'zc_ignore_immidiate_reward': self.fenv_config['zc_ignore_immidiate_reward'],
                    'zc_reward_weighening_mode': self.fenv_config['zc_reward_weighening_mode'],
                    'zero_constraint_flag': self.fenv_config['zero_constraint_flag'],
                },                
            })

        self.final_info_dict_list = []



    def get_end_episode_metrics(self, plan_data_dict):
        self.update_info_by_geometry_metrics(plan_data_dict)
        self.update_info_by_topological_metrics(plan_data_dict)
        self.append_metrics_to_list()
        return self.final_info_dict



    def update_info_by_geometry_metrics(self, plan_data_dict):
        room_delta_area_list = []
        room_delta_aspect_ratio_list = []

        for room_name in plan_data_dict['areas_desired'].keys():
            if room_name != f"room_{self.fenv_config['lvroom_id']}":
                delta_area = plan_data_dict['areas_delta'][room_name]
                delta_aspect_ratio = plan_data_dict['rooms_dict'][room_name]['delta_aspect_ratio']
    
                room_delta_area_list.append(delta_area)
                room_delta_aspect_ratio_list.append(delta_aspect_ratio)

        self.final_info_dict.update({
            'geometry_metrics': {
                'n_rooms': plan_data_dict['n_rooms'],
                'room_delta_area_mean': sum(room_delta_area_list)/len(room_delta_area_list),
                'room_delta_area_max': max(room_delta_area_list),
                'room_delta_area_min': min(room_delta_area_list),
                'room_delta_area_std': np.std(room_delta_area_list),
                'room_delta_aspect_ratio_mean': sum(room_delta_aspect_ratio_list)/len(room_delta_aspect_ratio_list),
                'room_delta_aspect_ratio_max': max(room_delta_aspect_ratio_list),
                'room_delta_aspect_ratio_min': min(room_delta_aspect_ratio_list),
                'room_delta_aspect_ratio_std': np.std(room_delta_aspect_ratio_list),
            }
        })
            

    
    def update_info_by_topological_metrics(self, plan_data_dict):
        edge_list_room_desired = plan_data_dict['edge_list_room_desired']
        if 1: # self.fenv_config['zero_constraint_flag']:
            n_desired_room_room_connections = len(edge_list_room_desired)
            
            if self.fenv_config['adaptive_window']:
                edge_color_data_dict_facade_adaptive_rooms = plan_data_dict['edge_color_data_dict_facade']['adaptive_rooms']
                # edge_color_data_dict_facade_adaptive_rooms_no_living_room = list(set(edge_color_data_dict_facade_adaptive_rooms).difference(set([plan_data_dict['lvroom_id']])))
                n_desired_room_facade_connections = len(edge_color_data_dict_facade_adaptive_rooms) 
            else:
                n_desired_room_facade_connections = len(plan_data_dict['edge_list_facade_desired_str']) 
            
            n_desired_connections = n_desired_room_room_connections + n_desired_room_facade_connections
            ## so far we count the whole number of desired connections.  
        
            edge_color_data_dict_room_red = plan_data_dict['edge_color_data_dict_room']['red']
            n_missed_room_room_connections = len(edge_color_data_dict_room_red)
            
            if self.fenv_config['adaptive_window']:
                edge_color_data_dict_facade_blind_rooms = plan_data_dict['edge_color_data_dict_facade']['blind_rooms']
                n_missed_room_facade_connections = len(edge_color_data_dict_facade_blind_rooms)
                # if ( not self.fenv_config['does_living_room_need_a_facade'] and 
                #         plan_data_dict['lvroom_id'] in edge_color_data_dict_facade_blind_rooms ):
                #     n_missed_room_facade_connections -= 1                  
            else:
                n_missed_room_facade_connections = len(plan_data_dict['edge_color_data_dict_facade']['red'])
                
            n_missed_connections = n_missed_room_room_connections + n_missed_room_facade_connections
            ## so far we count the whole number of missed connections. 
            
            edge_color_data_dict_room_green = plan_data_dict['edge_color_data_dict_room']['green']
            n_nicely_achieved_room_room_connections = len(edge_color_data_dict_room_green)
    
            if self.fenv_config['adaptive_window']:
                sighted_rooms = plan_data_dict['edge_color_data_dict_facade']['sighted_rooms']
                sighted_real_rooms = {r for r in sighted_rooms if r in self.fenv_config['real_room_id_range']}
                # if ( not self.fenv_config['does_living_room_need_a_facade'] and 
                #         plan_data_dict['lvroom_id'] in sighted_real_rooms ):
                #     sighted_real_rooms = set(sighted_real_rooms).difference(set([plan_data_dict['lvroom_id']])) 
                n_nicely_achieved_room_facade_connections = len(sighted_real_rooms)
            else:
                n_nicely_achieved_room_facade_connections = len(plan_data_dict['edge_color_data_dict_facade']['green'])
                            
                
            n_nicely_achieved_connections = n_nicely_achieved_room_room_connections + n_nicely_achieved_room_facade_connections
            
            try:
                if plan_data_dict['active_wall_status'] == 'well_finished' and self.fenv_config['plan_config_source_name'] != 'create_random_config':
                    assert n_nicely_achieved_connections == n_desired_connections - n_missed_connections, "Sth is wrong in calculating the adj performance"
            except Exception as e:
                np.save(f"plan_data_dict__{os.path.basename(__file__)}_{self.__class__.__name__}_{inspect.currentframe().f_code.co_name}.npy", plan_data_dict)
                raise ValueError(f"""Sth is wrong in calculating the adj performance for badly_finished situation
                                 n_nicely_achieved_connections: {n_nicely_achieved_connections}, 
                                 n_desired_connections: {n_desired_connections}, 
                                 n_missed_connections: {n_missed_connections}
                                 plan_id: {plan_data_dict['plan_id']}, 
                                 """) from e
    
            self.final_info_dict.update({
                'topological_metrics': {
                    'n_desired_connections': n_desired_connections,
                    'n_missed_connections': n_missed_connections,
                    'n_nicely_achieved_connections': n_nicely_achieved_connections,
                }
            })
            


    def append_metrics_to_list(self):
        new_row = {
            **self.final_info_dict['geometry_metrics'],
            **self.final_info_dict['topological_metrics']
        }
        self.final_info_dict_list.append(new_row)



    def get_metrics_df(self):
        # columns = ['room_delta_area_max', 'room_delta_area_mean', 'room_delta_area_min', 'room_delta_area_std', 
        #            'room_delta_aspect_ratio_max', 'room_delta_aspect_ratio_mean', 'room_delta_aspect_ratio_min', 'room_delta_aspect_ratio_std', 
        #            'n_desired_connections', 'n_missed_connections', 'n_nicely_achieved_connections']        
        self.metrics_df = pd.DataFrame(self.final_info_dict_list)
        return self.metrics_df
    


    def save_metrics_df(self, result_dir):
        df_path = os.path.join(result_dir, 'metrics.csv')
        self.metrics_df.to_csv(df_path, index=False)



    def visualize_metrics(self, result_dir):
        self.metrics_df['adj_performance'] = self.metrics_df['n_nicely_achieved_connections'] / self.metrics_df['n_desired_connections']

        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set(style="whitegrid")

        # Create a figure and a set of subplots
        fig, ax = plt.subplots(3, 1, figsize=(10, 15))

        rotation = 0

        # Boxplot for room area related metrics
        sns.boxplot(data=self.metrics_df[['room_delta_area_max', 'room_delta_area_mean', 'room_delta_area_min', 'room_delta_area_std']], ax=ax[0])
        # ax[0].set_title('Room Area Metrics')

        # Set xtick labels
        ax[0].set_xticklabels(['Max Delta Area', 'Mean Delta Area', 'Min Delta Area', 'Std Delta Area'], rotation=rotation)

        # Boxplot for room aspect ratio related metrics
        sns.boxplot(data=self.metrics_df[['room_delta_aspect_ratio_max', 'room_delta_aspect_ratio_mean', 'room_delta_aspect_ratio_min', 'room_delta_aspect_ratio_std']], ax=ax[1])
        # ax[1].set_title('Room Aspect Ratio Metrics')

        # Set xtick labels
        ax[1].set_xticklabels(['Max Delta Aspect Ratio', 'Mean Delta Aspect Ratio', 'Min Delta Aspect Ratio', 'Std Delta Aspect Ratio'], rotation=rotation)

        # Violin plot for adj_performance
        arr = self.metrics_df['adj_performance'].values
        mean_value = np.mean(arr)  # Calculate the mean value
        std_value = np.std(arr)

        violin_parts = ax[2].violinplot(arr, vert=False, showmedians=True)
        jitter = np.random.normal(0, 0.02, size=len(arr))  # Adjust the jitter value for the sake of better visualization
        ax[2].scatter(arr, np.ones(len(arr)) + jitter, color='k', s=20, alpha=0.7)
        
        # Plot the mean value on the violin plot
        mean_line = ax[2].axvline(mean_value, color='r', linestyle='dashed', linewidth=2, label='Mean')
        
        # Plot lines for mean +/- standard deviation
        std_line1 = ax[2].axvline(mean_value + std_value, color='g', linestyle='dotted', linewidth=2, label='+1 Std.')
        std_line2 = ax[2].axvline(mean_value - std_value, color='g', linestyle='dotted', linewidth=2, label='-1 Std.')
        
        # Add the legend to the plot
        ax[2].legend()
        
        # ax[2].set_title("Agent Performance on Adjacency")
        ax[2].set_xlabel("Adjacency Accuracy")
        ax[2].set_yticks([])  # Hide y ticks as they are not meaningful in this context

        plt.tight_layout()  # Adjust the layout
        fig_path = os.path.join(result_dir, 'metrics.png')
        plt.savefig(fig_path, dpi=300) 
        plt.show()  # Display the plots
        
        
# #%%
if __name__ == '__main__':
    path = "/scicore/home/graber0001/kakooe0000/housing_design/storage_nobackup/sb_agents_storage/Prj__2024_04_20_1010__sb__bc2ppo/Scn__2024_04_20_082331__PTM____ZSLR__BC/result/metrics.csv"
    result_dir = os.path.dirname(path)
    self = Metrics({})
    self.metrics_df = pd.read_csv(path)
    self.visualize_metrics(result_dir)

    
