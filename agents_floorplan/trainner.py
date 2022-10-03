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

import learner_config

import store_info_as_json

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
        config = learner_config.get_learner_config(self.env, 
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

        ## save info as a json file
        store_info_as_json.store(fenv_config=self.env.fenv_config,
                                 shape_fc=self.env.obs.state_data_dict['shape_fc'],
                                 shape_cnn=self.env.obs.state_data_dict['shape_cnn'],
                                 agent_config=self.agent_config,
                                 plan_data_dict=self.env.obs.plan_data_dict,
                                 chkpt_path=self.chkpt_path,
                                 store_dir=self.agent_config['trainer_chkpt_dir'])
            
