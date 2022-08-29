# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 09:41:35 2021

@author: RK
"""

# %%
import os
from pathlib import Path
import shutil

import json
import numpy as np
import pandas as pd


from ray.rllib.agents import dqn
from ray.rllib.agents import ppo
from ray.rllib.agents import sac
from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.rllib.agents.dqn.dqn import DQNTrainer
from ray.rllib.agents.sac.sac import SACTrainer

from ray.tune.suggest.bayesopt import BayesOptSearch
# from hebo.optimizers.hebo import HEBO
# from ray.tune.suggest.hebo import HEBOSearch

import my_learner_config

# %%
class MyTrainer:
    def __init__(self, env, agent_config):
        self.env = env
        self.env_name = self.env.env_name
        
        self.agent_config = agent_config
        
        if self.agent_config['agent_first_name'] == 'ppo':
            self.default_config = ppo.DEFAULT_CONFIG.copy()
        elif self.agent_config['agent_first_name'] == 'dqn':
            self.default_config = dqn.DEFAULT_CONFIG.copy()
        elif self.agent_config['agent_first_name'] == 'sac':
            self.default_config = sac.DEFAULT_CONFIG.copy()
            
        self.config = self._get_trainer_config(self.default_config)

        if self.agent_config['agent_first_name'] == 'ppo':
            self.agent = PPOTrainer(env=self.env_name, config=self.config)
        elif self.agent_config['agent_first_name'] == 'dqn':
            self.agent = DQNTrainer(env=self.env_name, config=self.config)
        elif self.agent_config['agent_first_name'] == 'sac':
            self.agent = SACTrainer(env=self.env_name, config=self.config)

        self.search_alg = None
        if self.agent_config['hyper_tune_flag']:
            self.search_alg = BayesOptSearch(metric="mean_loss", mode="min")
            # self.search_alg = HEBOSearch(metric="mean_loss", mode="min")

        if self.agent_config['load_agent_flag']:
            self.agent = self._load_model()
            
        
    def _the_trainer(self, save_outputs=True):
        self.trainer_results = []
        
        self.episode_df = self._define_episode_df()
        
        for n in range(self.agent_config['num_episodes']):
            print(f"---------- episode: {n}")
            self.result = self.agent.train()
            self.trainer_results.append(self.result)

            iteration_df = pd.DataFrame({'n': n,
                                'episode_reward_min': self.result['episode_reward_min'],
                                'episode_reward_mean': self.result['episode_reward_mean'],
                                'episode_reward_max': self.result['episode_reward_max'],
                                'episode_len_mean': self.result['episode_len_mean']},
                                index=[n])
                
            self.episode_df = pd.concat([self.episode_df, iteration_df])
            
            print(f'{n:3d}: Min  reward: {self.result["episode_reward_min"]:8.4f}')
            print(f'{n:3d}: Mean reward: {self.result["episode_reward_mean"]:8.4f}')
            print(f'{n:3d}: Max  reward: {self.result["episode_reward_max"]:8.4f}')

            if self.agent_config['save_agent_flag']:
                if n % self.agent_config['checkpoint_freq'] == 0:
                    self.chkpt_path = self._save_model()
                elif n == (self.agent_config['n_episodes'] - 1):
                    self.chkpt_path = self._save_model()

        if self.agent_config['save_agent_flag']:
            if save_outputs:
                self._save_trainer_outputs()
    
    
    def _define_episode_df(self):
        if self.agent_config['load_agent_flag']:
            chkpt_path = Path(self.agent_config['chkpt_path'])
            if "tunner" in str(chkpt_path):
                trial_df_csv_path = os.path.join(chkpt_path.parents[2], "outputs", "trial_df.csv")
            elif "trainer" in str(chkpt_path):
                trial_df_csv_path = os.path.join(chkpt_path.parents[1], "outputs", "trial_df.csv")

            episode_df = pd.read_csv(trial_df_csv_path)
        else:
            episode_df = pd.DataFrame({'n': [], 
                                       'episode_reward_min': [],
                                       'episode_reward_mean': [],
                                       'episode_reward_max': [],
                                       'episode_len_mean': []})
        return episode_df
        
    def _get_trainer_config(self, config):
        config = my_learner_config.get_learner_config(self.env, 
                                                     self.env_name, 
                                                     self.agent_config, 
                                                     config)
        return config
    
    
    def _load_model(self):
        # agent.get_policy().model.base_model = load_model('saved_models/model.h5')
        self.agent.restore(self.agent_config['chkpt_path'])
        return self.agent
    
    
    def _save_model(self):
        # agent.get_policy().model.base_model.save('saved_models/model.h5')
        chkpt_path = self.agent.save(checkpoint_dir=self.agent_config['trainer_chkpt_dir'])
        print(f"chkpt_path: {chkpt_path}")
        return Path(chkpt_path)
    
    
    def _save_trainer_outputs(self):
        with open(self.agent_config['chkpt_txt_fixed_path'], "w") as f:
            f.write(str(self.chkpt_path))
            
        with open(self.agent_config['chkpt_config_npy_fixed_path'], 'wb') as f:
            np.save(f, self.config)

        outputs_dir = os.path.join(self.agent_config['trainer_chkpt_dir'], "outputs")
        if not os.path.exists(outputs_dir):
            os.makedirs(outputs_dir)
            
        trial_df_csv_path = os.path.join(outputs_dir, "trial_df.csv")
        self.episode_df.to_csv(trial_df_csv_path, index=False)
        
        
        configs_dir = os.path.join(self.agent_config['trainer_chkpt_dir'], "configs")
        if not os.path.exists(configs_dir):
            os.makedirs(configs_dir)
        
        learner_config_path = os.path.join(configs_dir, "learner_config.npy")
        with open(learner_config_path, "wb") as f:
            np.save(f, self.config)
        
        if self.env.env_name == 'master_env-v0':
            fenv_config_path = os.path.join(configs_dir, "fenv_config.npy")
            with open(fenv_config_path, "wb") as f:
                np.save(f, self.env.fenv_config)
            
        agent_config_path = os.path.join(configs_dir, "agent_config.npy")
        with open(agent_config_path, "wb") as f:
            np.save(f, self.agent_config)

        if self.env.fenv_config['load_fixed_walls_sets_flag']:
            src_wall_data_path = os.path.join(Path(self.agent_config['storage']), 'walls_data', 'fixed_walls.p')
            des_wall_data_dir = os.path.join(self.agent_config['trainer_chkpt_dir'], "walls_data")
            if not os.path.exists(des_wall_data_dir):
                os.makedirs(des_wall_data_dir)
            des_wall_data_path = os.path.join(des_wall_data_dir, 'fixed_walls.p')
            shutil.copyfile(src_wall_data_path, des_wall_data_path)
            
        ## save info as a json file
        info_json = {
                'env_name': self.env.fenv_config['env_name'],
                'env_planning': self.env.fenv_config['env_planning'],
                'env_type': self.env.fenv_config['env_type'],
                'env_space': self.env.fenv_config['env_space'],
                'plan_config_source_name': self.env.fenv_config['plan_config_source_name'],
                
                'min_x': self.env.fenv_config['min_x'],
                'max_x': self.env.fenv_config['max_x'],
                'min_y': self.env.fenv_config['min_y'],
                'max_y': self.env.fenv_config['max_y'],
                'n_channels': self.env.fenv_config['n_channels'],
                'n_actions': self.env.fenv_config['n_actions'],
                'fc_obs_shape': np.array(self.env.obs.shape_fc).tolist(),
                'cnn_obs_shape': self.env.obs.shape_cnn,
                
                'scenario_name': self.env.fenv_config['scenario_name'],
                
                'net_arch': self.env.fenv_config['net_arch'],
                'custom_model_flag': self.env.fenv_config['custom_model_flag'],
                
                'mask_flag': self.env.fenv_config['mask_flag'],
                'mask_numbers': self.env.fenv_config['mask_numbers'],
                'fixed_fc_observation_space': self.env.fenv_config['fixed_fc_observation_space'],
                
                'is_area_considered': self.env.fenv_config['is_area_considered'],
                'is_proportion_considered': self.env.fenv_config['is_proportion_considered'],
                
                
                'stop_time_step': self.env.fenv_config['stop_time_step'], 
                
                'include_wall_in_area_flag': self.env.fenv_config['include_wall_in_area_flag'],
                # 'areas_config': self.env.obs.plan_data_dict['desired_areas'],
                'area_tolerance': self.env.fenv_config['area_tolerance'],
                'n_walls': self.env.fenv_config['n_walls'],
                'use_areas_info_into_observation_flag': self.env.fenv_config['use_areas_info_into_observation_flag'],
                
                'rewarding_method_name': self.env.fenv_config['rewarding_method_name'],
                'positive_done_reward': self.env.fenv_config['positive_done_reward'],
                'negative_action_reward': self.env.fenv_config['negative_action_reward'],
                'negative_wrong_area_reward': self.env.fenv_config['negative_wrong_area_reward'],
                'negative_rejected_by_room_reward': self.env.fenv_config['negative_rejected_by_room_reward'], 
                'negative_rejected_by_canvas_reward': self.env.fenv_config['negative_rejected_by_canvas_reward'], 
                
                'reward_shaping_flag': self.env.fenv_config['reward_shaping_flag'],
                'reward_decremental_flag': self.env.fenv_config['reward_decremental_flag'],
                'reward_increment_within_interval_flag': self.env.fenv_config['reward_increment_within_interval_flag'],
                
                'learner_name': self.agent_config['learner_name'],
                'agent_first_name': self.agent_config['agent_first_name'],
                'agent_last_name': self.agent_config['agent_last_name'],
                'some_agents': self.agent_config['some_agents'],
                'num_policies': self.agent_config['num_policies'],
                'hyper_tune_flag': self.agent_config['hyper_tune_flag'],
                }
            
        info_json_path = os.path.join(self.agent_config['trainer_chkpt_dir'], "info_json.json")
        with open(info_json_path, 'w') as f:
            json.dump(info_json, f, indent=4)
