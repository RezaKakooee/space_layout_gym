#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 13:32:36 2023

@author: Reza Kakooee
"""


import os
from pathlib import Path

root_dir = os.path.realpath(Path(os.getcwd()).parents[0])
print(f"root_dir: {root_dir}")

import torch
import numpy as np

from gym_floorplan.envs.fenv_config import LaserWallConfig
from env_maker import EnvMaker

from pg_rollout import PgRollout
from visualizer import Visualizer

from pg_config import PgConfig
from ppo import PPO




#%%
class PgTester:
    def __init__(self, env_config, agent_config, trained_scenario_name):
        self.fenv_config = env_config
        self.agent_config = agent_config
        self.trained_scenario_name = trained_scenario_name
        
        for k, v in self.agent_config.items():
            setattr(self, k, v)
            
        self._load_configs()
        
        self.visualizer = Visualizer(self.agent_config)


    
    def _load_configs(self):
        self.storage_dir = os.path.join(root_dir, 'storage')
        self.rl_agents_storage_dir = os.path.join(self.storage_dir, 'rl_agents_storage')
        
        self.trained_scenario_dir = os.path.join(root_dir, f'storage/rl_agents_storage/{self.trained_scenario_name}')
        self.configs_dir = os.path.join(self.trained_scenario_dir, 'configs')
        self.results_dir = os.path.join(self.trained_scenario_dir, 'results')
        
        try:
            self.fenv_config_path = os.path.join(self.configs_dir, 'fenv_config.npy')
            self.agent_config_path = os.path.join(self.configs_dir, 'agent_config.npy')
            self.fenv_config = np.load(self.fenv_config_path, allow_pickle=True).tolist()
            self.agent_config = np.load(self.agent_config_path, allow_pickle=True).tolist()
        except:
            self.agent_config.update({
                'act_dim': self.fenv_config['act_dim'],
                'obs_dim': self.fenv_config['obs_dim'],
                'action_space_type': self.fenv_config['action_space_type'],
                'action_masking': self.fenv_config.get('action_masking_flag', False),
                'model_name': self.fenv_config['model_name'],
                })
            
        self.fenv_config.update({
            'phase': 'test',
            'show_render_flag': True,
            'so_thick_flag': True,
            'show_graph_on_plan_flag': True,
            'save_render_flag': True,
            'results_dir': self.results_dir,
            'scenario_name': self.trained_scenario_name,
            'scenario_dir': self.trained_scenario_dir,
            # 'num_of_facades': 4,
            # 'num_plan_components': self.fenv_config['num_of_fixed_walls_for_masked'] + 4,
            })
        
        
        self.agent_config.update({
            'phase': 'test',
            'load_model_flag': True,
            'save_model_flag': False,
            'device': 'cpu',
            'n_workers': 8,
            'n_envs': 8,
            'n_test_episodes': 1,
            'test_render_verbose': 1,
            'test_print_verbose': 1,
            'test_end_episode_verbose': 2,
            'results_dir': self.results_dir,
            })
        
        
        if test_mode == 'offline':
            self.fenv_config.update({
                'plan_config_source_name': 'load_random_config',
                'plan_id_for_load_fixed_config': None,
                'nrows_of_env_data_csv': None,
                'fixed_num_rooms_for_loading': None,
                })
        
        self.agent_config.update({
            'test_mode': test_mode,
            })
        
        self.trained_model_dir = os.path.join(self.trained_scenario_dir, 'models')
        self.trained_model_path = os.path.join(self.trained_model_dir, 'model_0000.pth')
        self.agent_config.update({'model_path': self.trained_model_path})
        
        
     
    def _set_seed(self, seed=41):
        np.random.seed(seed)
        
        torch.manual_seed(seed)
        if torch.backends.cudnn.enabled:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            
        # self.env.seed(seed)
        
    
    
    def load_model(self, model_path):
        self.agent.net.load_state_dict(torch.load(model_path))
        self.agent.net.eval()
        
        
    
    def test(self, n_episodes):
        for i in range(n_episodes):
            try:
                episode_data = self.pg_rollout.one_episode_rollout()
                print(f"-------------------- Episode: {i:02} --------------------")
                print(f"MeanEpisodeReward: {episode_data['mean_episode_reward']}")
                print(f"EpisodeLen: {episode_data['episode_len']}\n")
                    
            except KeyboardInterrupt:
                break        
                
        self.visualizer.plot_tb()
        
        
        
    def run_tester(self):
        self.env = EnvMaker(self.fenv_config).make()
        self._set_seed()
        
        self.agent = PPO(self.env, self.fenv_config, self.agent_config)
        if self.load_model_flag:
            self.load_model(self.trained_model_path)
            print("The model loaded for testing.")
        
        self.pg_rollout = PgRollout(self.env, self.agent)
        self.test(self.agent_config['n_test_episodes'])
        
        
        
        

#%%
if __name__ == '__main__':
    env_name = 'DOLW-v0' #"CartPole-v0" # CartPole-v0 DOLW Pendulum-v1
    agent_name = "PPO"
    phase = 'test'
    test_mode = '_offline'

    trained_scenario_name = 'Scn__2023_04_11_1701__PPO__LinearNet__FTC__08_Rooms__Area__Adj__Ent__AW__SER'
    
    if env_name == 'DOLW-v0':
        fenv_config = LaserWallConfig(agent_name).get_config()
    else:
        fenv_config = {'env_name': env_name}
    
    
    agent_config = PgConfig(env_name, agent_name, phase).get_config()
    
    fenv_config.update({'n_envs': agent_config['n_envs']})
    fenv_config.update({'phase': phase})
    
    agent_config.update({'test_mode': test_mode})
    
    self = PgTester(fenv_config, agent_config, trained_scenario_name)
    self.run_tester()
