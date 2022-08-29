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
from ray.rllib.agents.sac.sac import SACTrainer

from ray.tune.suggest.bayesopt import BayesOptSearch
# from hebo.optimizers.hebo import HEBO
# from ray.tune.suggest.hebo import HEBOSearch

import my_learner_config


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
        elif self.agent_config['agent_first_name'] == 'sac':
            self.default_config = sac.DEFAULT_CONFIG.copy()
            self.agent = SACTrainer

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
        config = my_learner_config.get_learner_config(self.env,
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

        chkpt_path = checkpoints[-1][0]  # checkpoint_path
        with open(self.agent_config['chkpt_txt_fixed_path'], "w") as f:
            f.write(chkpt_path)

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

        if self.env.fenv_config['load_fixed_walls_sets_flag']:
            src_wall_data_path = os.path.join(Path(self.agent_config['storage']), 'walls_data', 'fixed_walls.p')
            des_wall_data_dir = os.path.join(self.checkpoint_dir, "walls_data")
            if not os.path.exists(des_wall_data_dir):
                os.makedirs(des_wall_data_dir)
            des_wall_data_path = os.path.join(des_wall_data_dir, 'fixed_walls.p')
            shutil.copyfile(src_wall_data_path, des_wall_data_path)

        if self.agent_config['save_env_data_flag']:
            src_env_data_path = Path(self.agent_config['env_data_dir'])
            des_wall_data_dir = os.path.join(self.checkpoint_dir, 'env_data')

            shutil.move(src_env_data_path, des_wall_data_dir)
            
        
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
                'areas_config': self.env.obs.plan_data_dict['desired_areas'],
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
        
        info_json_path = os.path.join(self.checkpoint_dir, "info_json.json")
        with open(info_json_path, 'w') as f:
            json.dump(info_json, f, indent=4)