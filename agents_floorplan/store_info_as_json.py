#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 12:21:59 2022

@author: RK
"""
import os
import json
import numpy as np


def store(fenv_config, shape_fc, shape_cnn, agent_config, plan_data_dict, chkpt_path, store_dir):
    info_json = {
                'env_name': fenv_config['env_name'],
                'env_planning': fenv_config['env_planning'],
                'env_type': fenv_config['env_type'],
                'env_space': fenv_config['env_space'],
                'plan_config_source_name': fenv_config['plan_config_source_name'],
                
                'min_x': fenv_config['min_x'],
                'max_x': fenv_config['max_x'],
                'min_y': fenv_config['min_y'],
                'max_y': fenv_config['max_y'],
                'n_channels': fenv_config['n_channels'],
                'n_actions': fenv_config['n_actions'],
                'fc_obs_shape': np.array(shape_fc).tolist(),
                'cnn_obs_shape': shape_cnn,
                
                'scenario_name': fenv_config['scenario_name'],
                
                'net_arch': fenv_config['net_arch'],
                'model_source': fenv_config['model_source'],
                'model_name': fenv_config['model_name'],
                
                'action_masking_flag': fenv_config['action_masking_flag'],
                
                'mask_flag': fenv_config['mask_flag'],
                'mask_numbers': plan_data_dict['mask_numbers'],
                'fixed_fc_observation_space': fenv_config['fixed_fc_observation_space'],
                
                'is_area_considered': fenv_config['is_area_considered'],
                'is_adjacency_considered': fenv_config['is_adjacency_considered'],
                'is_proportion_considered': fenv_config['is_proportion_considered'],
                
                'stop_time_step': fenv_config['stop_time_step'], 
                
                'include_wall_in_area_flag': fenv_config['include_wall_in_area_flag'],
                'areas_config': np.array(list(plan_data_dict['desired_areas'].values()), dtype=int).tolist(),
                'desired_edge_list': plan_data_dict['desired_edge_list'],
                'area_tolerance': fenv_config['area_tolerance'],
                'n_walls': plan_data_dict['n_walls'],
                'use_areas_info_into_observation_flag': fenv_config['use_areas_info_into_observation_flag'],
                
                'rewarding_method_name': fenv_config['rewarding_method_name'],
                'positive_done_reward': fenv_config['positive_done_reward'],
                'negative_action_reward': fenv_config['negative_action_reward'],
                'negative_wrong_area_reward': fenv_config['negative_wrong_area_reward'],
                'negative_rejected_by_room_reward': fenv_config['negative_rejected_by_room_reward'], 
                'negative_rejected_by_canvas_reward': fenv_config['negative_rejected_by_canvas_reward'], 
                'negative_wrong_blocked_cells_reward': fenv_config['negative_wrong_blocked_cells_reward'],
                
                'reward_decremental_flag': fenv_config['reward_decremental_flag'],
                'reward_increment_within_interval_flag': fenv_config['reward_increment_within_interval_flag'],
                
                'learner_name': agent_config['learner_name'],
                'agent_first_name': agent_config['agent_first_name'],
                'agent_last_name': agent_config['agent_last_name'],
                'some_agents': agent_config['some_agents'],
                'num_policies': agent_config['num_policies'],
                'hyper_tune_flag': agent_config['hyper_tune_flag'],
                }
        
    info_json_path = os.path.join(store_dir, "info_json.json")
    with open(info_json_path, 'w') as f:
        json.dump(info_json, f, indent=4)
        
        
    if isinstance(plan_data_dict['desired_areas'], dict):
        desired_areas = np.array(list(plan_data_dict['desired_areas'].values())).astype(float).tolist()
        
    configs_for_longer_training = {
        'n_walls': plan_data_dict['n_walls'],
        'n_rooms': plan_data_dict['n_rooms'],    
        'mask_numbers': plan_data_dict['mask_numbers'],
        'masked_corners': plan_data_dict['masked_corners'],
        'mask_lengths': plan_data_dict['mask_lengths'],
        'mask_widths': plan_data_dict['mask_widths'],
        'masked_area': plan_data_dict['masked_area'],
        'desired_areas': desired_areas,
        'desired_edge_list': plan_data_dict['desired_edge_list'],
        
        'is_area_considered': fenv_config['is_area_considered'],
        'is_adjacency_considered': fenv_config['is_adjacency_considered'],
        'is_proportion_considered': fenv_config['is_proportion_considered'],
        
        'area_tolerance': fenv_config['area_tolerance'],
        'aspect_ratios_tolerance': fenv_config['aspect_ratios_tolerance'],
    
        'rewarding_method_name': fenv_config['rewarding_method_name'],
        'only_final_reward_flag': fenv_config['only_final_reward_flag'],
        'area_diff_in_reward_flag': fenv_config['area_diff_in_reward_flag'],
        'proportion_diff_in_reward_flag': fenv_config['proportion_diff_in_reward_flag'],
        }

    configs_for_longer_training_json_path = os.path.join(store_dir, "configs/configs_for_longer_training.json")
    with open(configs_for_longer_training_json_path, 'w') as f:
        json.dump(configs_for_longer_training, f, indent=4)