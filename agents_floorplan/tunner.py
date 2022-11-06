# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 09:35:41 2021

@author: RK
"""

# %%
import os
from pathlib import Path
import shutil

import json
import numpy as np
import pandas as pd

from ray import tune
from ray.rllib.agents import dqn
from ray.rllib.agents import ppo
from ray.rllib.agents import sac
from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.rllib.agents.dqn.dqn import DQNTrainer
# from ray.rllib.agents.sac.sac import SACTrainer

from ray.tune.suggest.bayesopt import BayesOptSearch
# from hebo.optimizers.hebo import HEBO
# from ray.tune.suggest.hebo import HEBOSearch

import learner_config

import store_info_as_json


# %%
class MyTunner:
    def __init__(self, env, agent_config):
        self.env = env
        self.env_name = self.env.env_name

        self.agent_config = agent_config

        if self.agent_config['agent_first_name'] == 'ppo':
            self.default_config = ppo.DEFAULT_CONFIG.copy()
            self.agent = PPOTrainer
        elif self.agent_config['agent_first_name'] == 'dqn':
            self.default_config = dqn.DEFAULT_CONFIG.copy()
            self.agent = DQNTrainer
        # elif self.agent_config['agent_first_name'] == 'sac':
        #     self.default_config = sac.DEFAULT_CONFIG.copy()
        #     self.agent = SACTrainer

        self.config = self._get_tunner_config(self.default_config)

        if self.env_name == 'master_env-v0':
            self.stop = {"training_iteration": self.agent_config['stop_tunner_iteration']}
        else:
            self.stop = {"training_iteration": self.agent_config['stop_tunner_iteration'], "episode_reward_mean": 200}

        self.search_alg = None
        if self.agent_config['hyper_tune_flag']:
            self.search_alg = BayesOptSearch(metric="mean_loss", mode="min")
            # self.search_alg = HEBOSearch(metric="mean_loss", mode="min")

    def _the_tunner(self, save_outputs=True):
        if self.agent_config['load_agent_flag']:
            self.tunner_analysis = tune.run(
                self.agent,
                local_dir=self.agent_config['local_dir'],
                config=self.config,
                stop=self.stop,
                restore=self.agent_config['chkpt_path'],
                checkpoint_freq=self.agent_config['checkpoint_freq'],
                checkpoint_at_end=True,
                # search_alg=self.search_alg,
                # verbose=0,
            )

        else:
            self.tunner_analysis = tune.run(
                self.agent,
                local_dir=self.agent_config['local_dir'],
                config=self.config,
                stop=self.stop,
                checkpoint_freq=self.agent_config['checkpoint_freq'],
                checkpoint_at_end=True,
                # search_alg=self.search_alg,
            )

        if save_outputs:
            self._save_tunner_outputs()

    def _get_tunner_config(self, config):
        config = learner_config.get_learner_config(self.env,
                                                      self.env_name,
                                                      self.agent_config,
                                                      config)
        return config

    def _save_tunner_outputs(self):
        self.tunner_analysis.default_metric = self.agent_config['default_metric']
        self.tunner_analysis.default_mode = self.agent_config['default_mode']
        checkpoint_path = self.tunner_analysis.get_best_checkpoint(trial=self.tunner_analysis.get_best_trial())

        checkpoints = self.tunner_analysis.get_trial_checkpoints_paths(
            trial=self.tunner_analysis.get_best_trial(metric=self.agent_config['default_metric'],
                                                      mode=self.agent_config['default_mode']),
            metric=self.agent_config['default_metric'])

        self.chkpt_path = checkpoints[-1][0]  # checkpoint_path
        with open(self.agent_config['chkpt_txt_fixed_path'], "w") as f:
            f.write(self.chkpt_path)

        ## Best config is for when you have hyper-parameters tuning
        best_config = self.tunner_analysis.get_best_config(metric=self.agent_config['default_metric'],
                                                           mode=self.agent_config['default_mode'])

        with open(self.agent_config['chkpt_config_npy_fixed_path'], 'wb') as f:
            np.save(f, best_config)

        self.checkpoint_dir = Path(checkpoint_path).parents[2]

        outputs_dir = os.path.join(self.checkpoint_dir, "outputs")
        if not os.path.exists(outputs_dir):
            os.makedirs(outputs_dir)

        trial_df_csv_path = os.path.join(outputs_dir, "trial_df.csv")
        for key, val in self.tunner_analysis.trial_dataframes.items():
            trial_df = pd.DataFrame(val)
        trial_df.to_csv(trial_df_csv_path, index=False)

        configs_dir = os.path.join(self.checkpoint_dir, "configs")
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

        if self.agent_config['save_env_data_flag']:
            src_env_data_path = Path(self.agent_config['env_data_dir'])
            des_wall_data_dir = os.path.join(self.checkpoint_dir, 'env_data')

            shutil.move(src_env_data_path, des_wall_data_dir)
            
        
        ## save info as a json file
        store_info_as_json.store(fenv_config=self.env.fenv_config,
                                 shape_fc=self.env.obs.state_data_dict['shape_fc'],
                                 shape_cnn=self.env.obs.state_data_dict['shape_cnn'],
                                 agent_config=self.agent_config,
                                 plan_data_dict=self.env.obs.plan_data_dict,
                                 chkpt_path=self.chkpt_path,
                                 store_dir=self.checkpoint_dir)
        