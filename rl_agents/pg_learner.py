#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 18:27:43 2023

@author: Reza Kakooee
"""

import os
from pathlib import Path

root_dir = os.path.realpath(Path(os.getcwd()).parents[0])
print(f"root_dir: {root_dir}")



#%%
import time
import json

# import gymnasium as gym
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter

from env_maker import EnvMaker

from ppo import PPO
from pg_trainer import PgTrainer
from config_saver import ConfigSaver



#%%
class PpoLearner:
    def __init__(self, fenv_config, agent_config, trained_scenario_name=None):
        self.start_time = time.time()
        
        self.fenv_config = fenv_config
        self.agent_config = agent_config
        self.trained_scenario_name = trained_scenario_name
        
        self._update_configs()
        self._make_writer()
        
            
        self.env = EnvMaker(self.fenv_config).make()
        self._set_seed()
        
        self._get_env_vars()
            
        config_saver = ConfigSaver(self.fenv_config, self.agent_config, self.configs_dir)
        config_saver.store()
        

    
    def _set_seed(self, seed=41):
        np.random.seed(seed)
        
        torch.manual_seed(seed)
        if torch.backends.cudnn.enabled:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            
        # self.env.seed(seed)
        
    
    
    def _update_configs(self):
        self.storage_dir = os.path.join(root_dir, 'storage')
        self.chkpt_txt_fixed_path = os.path.join(self.storage_dir, 'chkpt_dir/chkpt_path.txt')
        self.rl_agents_storage_dir = os.path.join(self.storage_dir, 'rl_agents_storage')
        
        self.scenario_dir = os.path.join(self.rl_agents_storage_dir, self.fenv_config['scenario_name'])
        if not os.path.exists(self.scenario_dir):
            os.makedirs(self.scenario_dir)
            
        self.tb_logs_dir = os.path.join(self.scenario_dir, 'tb_logs')
        if not os.path.exists(self.tb_logs_dir):
            os.makedirs(self.tb_logs_dir)
        
        self.model_dir = os.path.join(self.scenario_dir, 'models')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            
        self.configs_dir = os.path.join(self.scenario_dir, 'configs')
        if not os.path.exists(self.configs_dir):
            os.makedirs(self.configs_dir)
            
        self.results_dir = os.path.join(self.scenario_dir, 'results')
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        self.plan_df_path = os.path.join(self.results_dir, 'plan_df.csv')
        self.res_df_path = os.path.join(self.results_dir, 'res_df.csv')
        self.res_summary_df_path = os.path.join(self.results_dir, 'res_summary_df.csv')
        
        
        self.agent_config.update({
            'scenario_dir': self.scenario_dir,
            'chkpt_txt_fixed_path': self.chkpt_txt_fixed_path,
            'tb_logs_dir': self.tb_logs_dir,
            'model_dir': self.model_dir,
            'configs_dir': self.configs_dir,
            'results_dir': self.results_dir,
            'plan_df_path': self.plan_df_path,
            'res_df_path': self.res_df_path,
            'res_summary_df_path': self.res_summary_df_path,
            'n_shuffles_for_data_augmentation': self.fenv_config['n_shuffles_for_data_augmentation'],
            # 'device': 'cpu',
            })
        
        
        if self.agent_config['load_model_flag']:
            self.chkpt_path, self.latest_chkpt_number = self._get_latest_chkpt()
            self.agent_config.update({
                'chkpt_path': self.chkpt_path,
                'latest_chkpt_number': self.latest_chkpt_number,
                })
        
        self.fenv_config['library'] = '__RLlib'
        self.fenv_config['phase'] = 'train'
        self.fenv_config['results_dir'] = self.results_dir



    def _get_env_vars(self):
        self.agent_config.update({
            'action_space_type': self.fenv_config['action_space_type'],
            'action_masking': self.fenv_config.get('action_masking_flag', False),
            'model_name': self.fenv_config['model_name'],
            'net_arch': self.fenv_config['net_arch'],
            })
        
        self.agent_config.update({
            'act_dim': self.fenv_config['act_dim'],
            'action_space_type': self.fenv_config['action_space_type'],
            'obs_dim': self.fenv_config['obs_dim'],
            })
        
        if self.fenv_config['net_arch'] == 'MetaFc':
            self.agent_config.update({
                'obs_fc_dim': self.fenv_config['obs_fc_dim'],
                'obs_meta_dim': self.fenv_config['obs_meta_dim'],
            })
            
        if self.fenv_config['net_arch'] == 'MetaCnn':
            self.agent_config.update({
                'obs_cnn_dim': self.fenv_config['obs_cnn_dim'],
                'obs_meta_dim': self.fenv_config['obs_meta_dim'],
            })
        
        
        
    def _get_latest_chkpt(self):
        chkpt_txt_fixed_path = os.path.join(self.storage_dir, 'chkpt_dir/chkpt_path.txt') 
        with open(chkpt_txt_fixed_path) as f:
            chkpt_path = f.readline()
        latest_chkpt_number = int(chkpt_path.split("/")[-1].split('_')[-1].split('.')[0])
        return chkpt_path, latest_chkpt_number
            
        
    
    def _make_writer(self):
        self.writer = SummaryWriter(self.tb_logs_dir)
        self.writer.add_text("env_config", "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in self.fenv_config.items()])))
        self.writer.add_text("agent_config", "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in self.agent_config.items()])))

        

    def run_trainer(self):
        print("Training started for:")
        print(f"Scenario_Name: {self.fenv_config['scenario_name']}")
        self.agent = PPO(self.env, self.fenv_config, self.agent_config)
        self.trainer = PgTrainer(self.env, self.agent, self.writer)
        self.trainer.train()
        
        self._terminator()
    
    
    
    def _terminator(self):
        self.writer.close()
        print("The tb-writer closed.")
        
        end_time = time.time()
        elapsed_time = end_time - self.start_time
        
        run_info_json = {
            'elapsed_time (minutes)': elapsed_time/60,
            'device': str(self.agent_config['device']),
            'num_workers': self.agent_config['n_workers'],
            'n_updates': self.agent_config['n_updates'],
        }
        
        run_info_json_path = os.path.join(self.scenario_dir, 'run_info_json.json')
        with open(run_info_json_path, 'w') as f:
            json.dump(run_info_json, f, indent=4)
            
        print(f"- - - - - - Scenario name: {self.fenv_config['scenario_name']}")
        print(f"- - - - Total time is: {elapsed_time} (s)")   
        print(" = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =")
        

    
    
#%%
if __name__ == '__main__':
    env_name = 'DOLW-v0' 
    agent_name = "PPO"
    phase = 'train'
    action_masking_flag = False
    
    from gym_floorplan.envs.fenv_scenarios import FEnvScenarios
    from gym_floorplan.envs.fenv_config import LaserWallConfig
    from pg_config import PgConfig

    scenarios_dict = FEnvScenarios(agent_name, action_masking_flag).get_scenarios()
    
    fenv_config = LaserWallConfig(agent_name, phase, scenarios_dict).get_config()
    agent_config = PgConfig(env_name, agent_name, phase).get_config()
    
    fenv_config.update({'n_envs': agent_config['n_envs']})
    fenv_config.update({'phase': phase})
    
    self = PpoLearner(fenv_config, agent_config)
    self.run_trainer()