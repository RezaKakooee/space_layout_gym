#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 09:51:39 2023

@author: Reza Kakooee
"""

import os
import json
import numpy as np

class ConfigSaver:
    def __init__(self, fenv_config, agent_config, configs_dir):
        self.fenv_config = fenv_config
        self.agent_config = agent_config
        self.configs_dir = configs_dir
        
        

    def store(self):
        config_json = {
            'scenario_name': self.fenv_config['scenario_name'],
            
            # "fenv_config": {
                    'env_name': self.fenv_config['env_name'],
                    'env_type': self.fenv_config['env_type'],
                    'env_planning': self.fenv_config['env_planning'],
                    'env_space': self.fenv_config['env_space'],
                    'mask_flag': self.fenv_config['mask_flag'],
                    'resolution': self.fenv_config['resolution'],
                    'plan_config_source_name': self.fenv_config['plan_config_source_name'],
                    'min_x': self.fenv_config['min_x'],
                    'max_x': self.fenv_config['max_x'],
                    'min_y': self.fenv_config['min_y'],
                    'max_y': self.fenv_config['max_y'],
                    'n_channels': self.fenv_config['n_channels'],
                    'n_actions': self.fenv_config['n_actions'],
                    # },
            
            # "constraints_objective_config": {
                    'is_area_considered': self.fenv_config['is_area_considered'],
                    'is_adjacency_considered': self.fenv_config['is_adjacency_considered'],
                    'is_proportion_considered': self.fenv_config['is_proportion_considered'],
                    'area_tolerance': self.fenv_config['area_tolerance'],
                    'rewarding_method_name': self.fenv_config['rewarding_method_name'],
                    'positive_done_reward': self.fenv_config['positive_done_reward'],
                    'positive_final_reward': self.fenv_config['positive_final_reward'],
                    'edge_diff_min': 0,
                    'edge_diff_max': 20,
                    'linear_reward_coeff': 50,
                    'quad_reward_coeff': 2.49,
                    'exp_reward_coeff': 50,
                    'exp_reward_temperature': 144.7,
                    'stop_ep_time_step': self.fenv_config['stop_ep_time_step'], 
                    'action_masking_flag': self.fenv_config['action_masking_flag'],
                    # },
            
            # "model_config": {
                    'net_arch': self.fenv_config['net_arch'],
                    'model_source': self.fenv_config['model_source'],
                    'model_name': self.fenv_config['model_name'],
                    'agent_name': self.agent_config['agent_name'],
                    # },
            
            # "agent_config": self.agent_config,
            }
            
        info_json_path = os.path.join(self.configs_dir, "config_json.json")
        with open(info_json_path, 'w') as f:
            json.dump(config_json, f, indent=4)
            
            
        configs_for_longer_training = {
            'is_area_considered': self.fenv_config['is_area_considered'],
            'is_adjacency_considered': self.fenv_config['is_adjacency_considered'],
            'is_proportion_considered': self.fenv_config['is_proportion_considered'],
            
            'area_tolerance': self.fenv_config['area_tolerance'],
            'aspect_ratios_tolerance': self.fenv_config['aspect_ratios_tolerance'],
        
            'rewarding_method_name': self.fenv_config['rewarding_method_name'],
            'positive_done_reward': self.fenv_config['positive_done_reward'],
            'area_diff_in_reward_flag': self.fenv_config['area_diff_in_reward_flag'],
            'proportion_diff_in_reward_flag': self.fenv_config['proportion_diff_in_reward_flag'],
            }
    
        configs_for_longer_training_json_path = os.path.join(self.configs_dir, "configs_for_longer_training.json")
        with open(configs_for_longer_training_json_path, 'w') as f:
            json.dump(configs_for_longer_training, f, indent=4)
            
            
        fenv_config_path = os.path.join(self.configs_dir, "fenv_config.npy")
        with open(fenv_config_path, "wb") as f:
            np.save(f, self.fenv_config)
            
        agent_config_path = os.path.join(self.configs_dir, "agent_config.npy")
        with open(agent_config_path, "wb") as f:
            np.save(f, self.agent_config)