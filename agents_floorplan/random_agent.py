# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 13:40:40 2021

@author: Reza Kakooee
"""


# %%
import os
from pathlib import Path

housing_design_dir = os.path.realpath(Path(os.getcwd()).parents[0])
gym_floorplan_dir = f"{housing_design_dir}/gym-floorplan"
print(f"{os.path.basename(__file__)} -> housing_design_dir: {housing_design_dir}")
print(f"{os.path.basename(__file__)} -> gym_floorplan_dir:  {gym_floorplan_dir}")

# %%
# import sys

# print(f"{os.path.basename(__file__)} -> Appending: ........ {gym_floorplan_dir}")
# sys.path.append(f"{gym_floorplan_dir}")


# %%
import gym
import time
import pickle
import logging
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import gym_floorplan
from gym_floorplan.envs.fenv_config import LaserWallConfig

from gif_maker import make_gif


# %% 
class FEnv():
    def __init__(self, fenv_config):
        self.fenv_config = fenv_config
        self.env_name = fenv_config['env_name']
        
        
    def get_env(self):
        if self.env_name == 'gym':
            env = gym.make(self.env_name)
            env.env_name = self.env_name
        else:
            from gym_floorplan.envs.master_env import MasterEnv
            env = MasterEnv(self.fenv_config)
        return env
    
    
# %%
class RandomAgent:
    def __init__(self, fenv_config, print_verbose, render_verbose, trial,
                 save_valid_plans_flag=False):
        self.fenv_config = fenv_config
        self.print_verbose = print_verbose
        self.render_verbose = render_verbose
        self.trial = trial
        self.save_valid_plans_flag = save_valid_plans_flag
        
        self._adjust_configs()
        
        self.env = FEnv(fenv_config).get_env()
        
        self.logger = self._get_logger()
        
    
    def _adjust_configs(self):
        fenv_config.update({'trial': self.trial})
        self.base_generated_plans_dir = self.fenv_config['generated_plans_dir']
        generated_plans_dir = f"{self.fenv_config['generated_plans_dir']}/{self.fenv_config['scenario_name']}"
        if not os.path.exists(generated_plans_dir):
            os.mkdir(generated_plans_dir)
        fenv_config.update({'generated_plans_dir': generated_plans_dir})
        fenv_config.update({'show_render_flag': True})
        fenv_config.update({'save_render_flag': False})
        
        if self.fenv_config['is_area_considered'] and self.fenv_config['is_proportion_considered']:
            self.plan_values_path = f"{self.base_generated_plans_dir}/plan_values__{self.fenv_config['n_walls']:02}_walls__area_proportion.csv"
        elif self.fenv_config['is_area_considered']:
            self.plan_values_path = f"{self.base_generated_plans_dir}/plan_values__{self.fenv_config['n_walls']:02}_walls__area.csv"
        else:
            self.plan_values_path = f"{self.base_generated_plans_dir}/plan_values__{self.fenv_config['n_walls']:02}_walls__proportion.csv"
            
        
        
    def _get_logger(self):
        LOG_FORMAT = "%(message)s"
        filename = f"{self.fenv_config['generated_plans_dir']}/log_random_agent_trial_{self.fenv_config['trial']:02}.log"
        logging.basicConfig(filename=filename, 
                            level=logging.INFO, 
                            format=LOG_FORMAT, filemode="w")
        logging.getLogger('PIL').setLevel(logging.WARNING)
        logger = logging.getLogger()
        # logger.addHandler(logging.StreamHandler())
        logger.propagate = False
        return logger  
    
    
    def run_random_agent(self, n_episodes):
        if self.env.env_name == 'master_env-v0':
            self._random_agent_for_master_env(n_episodes)
        else: 
            self._random_agent_for_gym_env(n_episodes)
            
            
    def _get_random_action(self):
        action = np.random.randint(0, self.fenv_config['n_actions'], 1)
        return action[0]


    def _random_agent_for_master_env(self, n_episodes):
        keys = ['env_name', 'env_type', 'env_planning', 'env_space', 'plan_config_source_name',
                'net_arch', 'mask_flag', 'mask_numbers', 'fixed_fc_observation_space',
                'include_wall_in_area_flag', 'area_tolerance', 'reward_shaping_flag']                
        reward_per_episode = []
        episode_len = []
        obs_min_list = []
        obs_max_list = []
        num_failures = 0
        info_dict = defaultdict(list)
        env_data_dict = defaultdict(list)
        self.accepted_action_sequence_dict = defaultdict()
        TQDM = 1
        iterator = tqdm(range(n_episodes)) if TQDM else range(n_episodes)
        for i in iterator:
            if i == 0: 
                if self.print_verbose >= 1: print("################################################## Start")
                if self.print_verbose >= 0: print(f"- - - - - scenario_name: {self.fenv_config['scenario_name']}")
                if self.print_verbose >= 0: self.logger.info(f"- - - - - scenario_name: {self.fenv_config['scenario_name']}")
                for key in keys:    
                    if self.print_verbose >= 1: print(f"- - - - - {key}: {self.fenv_config[key]}")
                    if self.print_verbose >= 1: self.logger.info(f"- - - - - {key}: {self.fenv_config[key]}")
                        
            if self.print_verbose >= 1: print(f"==================================== Episode: {i}")
            this_episode_reward = 0
            observation = self.env.reset()
            if self.fenv_config['net_arch'] in ['fccnn', 'cnnfc']:
                if self.print_verbose >= 2: print(f"fc observation_space: {self.env.obs.observation_space[0].shape}")
                if self.print_verbose >= 2: print(f"Observation shape: {observation[0].shape}")
            else:
                if self.print_verbose >= 2: print(f"fc observation_space: {self.env.obs.observation_space.shape}")
                if self.print_verbose >= 2: print(f"Observation shape: {observation.shape}")
                
            time_step = self.env.time_step
            if self.render_verbose >= 2: self.env.render()
            if self.render_verbose == 3: self.env._illustrate()
            if self.render_verbose >= 4: self.env._display()
            done = False
            good_action_sequence = []
            while not done:
                if self.print_verbose >= 2:  print(" - - - - -  - - - - - - - - - - - - - - - - - - -- - - -")
                action = self._get_random_action()
                if self.fenv_config['env_type'] == 'multi':
                    action = {"wall_1": action}
                # action_sequence = [1136, 959, 766, 961, 261, 838, 570] #
                # action = action_sequence[time_step]
                observation, reward, done, info = self.env.step(action)
                time_step = self.env.time_step
                if self.fenv_config['net_arch'] in ['fccnn', 'cnnfc']:
                    experience = (action, observation[0].shape, observation[1].shape, reward, done, info)
                else:
                    experience = (action, observation.shape, reward, done, info)
                if self.env.obs.active_wall_status in ['accepted', 'finished']:
                    good_action_sequence.append(action)
                obs_min_list.append(np.min(observation[1]))
                obs_max_list.append(np.max(observation[1]))
                this_episode_reward += reward
                
                if self.print_verbose >= 2: print(f"- - - - - {time_step:02}-> experience: {experience}")
                
                # if done:
                    # print(f"Sum of the created areas: {sum(list(info['env_data'].values()))}")
                    
                if done:
                    if self.env.obs.active_wall_status == 'finished':
                        # info_dict[i] = info
                        env_data_dict[i] = info['env_data']
                        self.accepted_action_sequence_dict[i] = good_action_sequence
                    if self.print_verbose >= 4: print(f"========== done: {done}!")
                    reward_per_episode.append(this_episode_reward)
                    if self.render_verbose >= 1: self.env.render()
                    if self.render_verbose >= 4: self.env._illustrate()
                    if self.render_verbose >= 2: self.env._display()
                    if time_step == self.fenv_config['stop_time_step']:
                        num_failures += 1
                        if self.print_verbose >= 2: print('Failed')
                    if self.print_verbose >= 1: print(f"good_action_sequence: {good_action_sequence}")
                    if self.fenv_config['mask_flag']:
                        if self.print_verbose >= 1: print(f"----- mask_numbers: {self.env.obs.mask_numbers}")
                        if self.print_verbose >= 1: print(f"----- masked_corners: {self.env.obs.masked_corners}")
                        if self.print_verbose >= 1: print(f"----- mask_lengths: {self.env.obs.mask_lengths}")
                        if self.print_verbose >= 1: print(f"----- mask_widths: {self.env.obs.mask_widths}")
                        if self.print_verbose >= 1: print(f"----- areas: {self.env.obs.plan_data_dict['areas']}")
                    break
                    
                else:
                    if self.render_verbose >= 2: self.env.render()
                    if self.render_verbose >= 3: self.env._illustrate()
                    if self.render_verbose >= 4: self.env._display()

                if time_step % 100 == 0:
                    if self.print_verbose >= 2: print(f"TimeStep: {time_step}")
            episode_len.append(time_step)
            
            if self.save_valid_plans_flag:
                if i % 10 == 0:
                    self._save_plan_values(env_data_dict)
                
            if self.print_verbose >= 2: print("==========================================================")
        
        if self.save_valid_plans_flag:
            self._save_plan_values(env_data_dict)
        
        if self.print_verbose >= 3: print(f"min(obs): {np.min(obs_min_list)} , max(obs): {np.max(obs_max_list)}")
        if self.print_verbose >= 0: print(f"- - - - - episode_len: {episode_len}")
        if self.print_verbose >= 0: print(f"- - - - - Mean of episode_len: {np.mean(episode_len)}")
        if self.print_verbose >= 0: print(f"- - - - - reward_per_episode: {reward_per_episode}")
        if self.print_verbose >= 0: print(f"- - - - - num_failures: {num_failures}")
        
        if self.print_verbose >= 0: 
            self.logger.info(f"- - - - - episode_len: {episode_len}")
            self.logger.info(f"- - - - - Mean of episode_len: {np.mean(episode_len)}")
            self.logger.info(f"- - - - - reward_per_episode: {reward_per_episode}")
            self.logger.info(f"- - - - - num_failures: {num_failures}")
            self.logger.info("################################################## End")
            
        
    def _save_plan_values(self, env_data_dict):
        plan_values_df_old = pd.read_csv(self.plan_values_path)
        plan_values_df_new = pd.DataFrame.from_dict(env_data_dict, orient='index')
        plan_values_df_new['accepted_action_sequence'] = list(self.accepted_action_sequence_dict.values())
        plan_values_df = pd.concat([plan_values_df_old, plan_values_df_new], axis=0)
        plan_values_df.to_csv(self.plan_values_path, index=False)
    
# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some parameters for random agent.')
    parser.add_argument('--n_episodes', type=int, default=5, help='n_episodes')
    args = parser.parse_args()
    
    start_time = time.time()
    fenv_config = LaserWallConfig().get_config()
    
    print_verbose, render_verbose, trial = 1, 1, 1
    self = RandomAgent(fenv_config, print_verbose, render_verbose, trial, save_valid_plans_flag=True)
    self.run_random_agent(n_episodes=args.n_episodes)
    
    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time}")
    
    # make_gif(input_dir=fenv_config['generated_plans_dir'], output_dir=fenv_config['generated_gif_dir'])
