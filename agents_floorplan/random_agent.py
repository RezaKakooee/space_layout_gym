# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 13:40:40 2021

@author: RK
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
from gym_floorplan.envs.layout_graph import LayoutGraph

import matplotlib.pyplot as plt
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
    def __init__(self, fenv_config, params):
        
        self.fenv_config = fenv_config
        self.params = params
                
        generated_plans_dir = self.fenv_config['generated_plans_dir']
        self.scenario_name = self.fenv_config['scenario_name']
        self.scenario_dir = f"{generated_plans_dir}/{self.scenario_name}"
        
        if self.params['save_log_flag'] or self.params['save_render_flag']:
            if not os.path.exists(self.scenario_dir):
                os.mkdir(self.scenario_dir)
            
        
        self._adjust_configs()
        
        if self.params['save_log_flag']:
            self.logger = self._get_logger()
        
        self.env = FEnv(fenv_config).get_env()
        
        
    def _adjust_configs(self):
        self.fenv_config.update({k: v for k, v in self.params.items()})
        # self.fenv_config.update({'trial': self.params['trial']})
        # self.fenv_config.update({'scenario_dir': self.scenario_dir})
        # self.fenv_config.update({'show_render_flag': self.params['show_render_flag']})
        # self.fenv_config.update({'save_render_flag': self.params['save_render_flag']})
        # self.fenv_config.update({'so_thick_flag': self.params['so_thick_flag']})
    
    def _get_logger(self):
        LOG_FORMAT = "%(message)s"
        filename = f"{self.scenario_dir}/log_random_agent_trial_{self.params['trial']:02}.log"
        print(f"filename: {filename}")
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
            self.random_agent_for_master_env(n_episodes)
        else: 
            self._random_agent_for_gym_env(n_episodes)
            
            
    def _get_random_action(self, obs):
        if self.fenv_config['action_masking_flag']:
            action_mask = obs['action_mask']
            try:
                action = np.random.choice(np.arange(self.fenv_config['n_actions']), 
                                      size=1, replace=False, 
                                      p=np.array(action_mask)/(sum(action_mask)))
            except:
                print("wait for checking action_mask")
                raise("seems no action left to be selected!")
        else:
            action = np.random.randint(0, self.fenv_config['n_actions'], 1)
        return action[0]


    def random_agent_for_master_env(self, n_episodes):
        fkeys = ['env_name', 'env_type', 'env_planning', 'env_space', 'net_arch', 
                'fixed_fc_observation_space', 'include_wall_in_area_flag', 
                'area_tolerance', 'reward_shaping_flag', 'mask_flag']
        obs_min_list = []
        obs_max_list = []
        num_failures = 0
        info_dict = defaultdict(list)
        self.env_data_dict = defaultdict(list)
        self.accepted_action_sequence_dict = defaultdict()
        TQDM = 1
        iterator = tqdm(range(n_episodes)) if TQDM else range(n_episodes)
        self.reward_list_in = []
        self.reward_per_episode = []
        self.episode_len = []
        self.status = [0] * n_episodes
        self.last_reward = []
        for i in iterator:
            if i == 0: 
                if self.params['print_verbose'] >= 1: print("\n################################################## Start")
                if self.params['print_verbose'] >= 0: print(f"\n- - - - - scenario_name: {self.fenv_config['scenario_name']}")
                for fkey in fkeys:    
                    if self.params['print_verbose'] >= 1: print(f"- - - - - {fkey}: {self.fenv_config[fkey]}")
                        
            this_episode_reward = 0
            if self.fenv_config['action_masking_flag']:
                if self.params['print_verbose'] >= 3: print(f"n_non_masked_actions: {self.fenv_config['n_actions']}")
                
                
            observation = self.env.reset()
            if self.params['print_verbose'] >= 1: print(f"==================================== Episode: {self.env.episode}")
            
            
            if self.fenv_config['action_masking_flag']:
                if self.params['print_verbose'] >= 3: print(f"n_non_masked_actions: {np.sum(observation['action_mask'])}")
                if self.fenv_config['net_arch'] in ['fccnn', 'cnnfc']:
                    if self.params['print_verbose'] >= 2: print(f"fc observation_space: {self.env.obs.observation_space[0].shape}")
                    if self.params['print_verbose'] >= 2: print(f"Observation shape: {observation[0].shape}")
                else:
                    if self.params['print_verbose'] >= 2: print(f"fc observation_space: {self.env.obs.observation_space.shape}")
                    # if self.params['print_verbose'] >= 2: print(f"Observation shape: {observation.shape}")
                
            time_step = self.env.time_step
            if self.params['render_verbose'] >= 2: self.env.render()
            if self.params['render_verbose'] == 3: self.env._illustrate()
            if self.params['render_verbose'] >= 4: self.env._display()
            done = False
            self.good_action_sequence = []
            while not done:
                if self.params['print_verbose'] >= 2:  print(" - - - - -  - - - - - - - - - - - - - - - - - - -- - - -")
                action = self._get_random_action(observation)
                if self.fenv_config['env_type'] == 'multi':
                    action = {"wall_1": action}
                
                if self.params['fixed_action_seq_flag']:
                    action = self.params['action_sequence'][time_step]
                    
                    
                observation, reward, done, info = self.env.step(action)
                
                self.reward_list_in.append((i, reward))
                
                
                time_step = self.env.time_step
                if self.fenv_config['net_arch'] in ['fccnn', 'cnnfc']:
                    experience = (action, observation[0].shape, observation[1].shape, reward, done, info)
                else:
                    if self.fenv_config['action_masking_flag']:
                        experience = (action, observation['real_obs'].shape, reward, done, info)
                    else:
                        experience = f"({action:06}, {observation.shape}, {reward}, {done}, {info})"
                        
                if self.env.obs.active_wall_status in ['accepted', 'well_finished']:
                    self.good_action_sequence.append(action)
                    if self.fenv_config['action_masking_flag']:
                        if self.params['print_verbose'] >= 3: print(f"\nn_non_masked_actions: {np.sum(observation['action_mask'])}")
                        if self.params['print_verbose'] >= 3: print(f"selected action: {action}")
                    
                if not self.fenv_config['action_masking_flag']:
                    obs_min_list.append(np.min(observation[1]))
                    obs_max_list.append(np.max(observation[1]))
                    
                this_episode_reward += reward
                
                if self.params['print_verbose'] >= 2: print(f"- - - - - {time_step:05}-> experience: {experience}")
                
                # if done:
                    # print(f"Sum of the created areas: {sum(list(info['env_data'].values()))}")
                    
                if done:
                    self.last_reward.append(reward)
                    if self.env.obs.active_wall_status == 'well_finished':
                        self.env.obs.plan_data_dict.update({'rooms_gravity_coord_dict': self._get_rooms_gravity_coord(self.env.obs.plan_data_dict)})
                        info_dict[i] = info
                        self.env_data_dict[i] = info['env_data']
                        self.accepted_action_sequence_dict[i] = self.good_action_sequence
                        self.status[i] = 1
                        
                    if self.params['print_verbose'] >= 4: print(f"========== done: {done}!")
                    self.reward_per_episode.append(this_episode_reward)
                    if self.params['render_verbose'] >= 1: self.env.render()
                    if self.params['render_verbose'] >= 4: self.env.illustrate()
                    if self.params['render_verbose'] >= 2: self.env.display()
                    if time_step >= self.fenv_config['stop_time_step']-1:
                        num_failures += 1
                        if self.params['print_verbose'] >= 2: print('Failed')
                    if self.params['print_verbose'] >= 1: print(f"\ngood_action_sequence: {self.good_action_sequence}")
                    if self.fenv_config['mask_flag']:
                        if self.params['print_verbose'] >= 1: print(f"----- mask_numbers: {self.env.obs.plan_data_dict['mask_numbers']}")
                        if self.params['print_verbose'] >= 1: print(f"----- masked_corners: {self.env.obs.plan_data_dict['masked_corners']}")
                        if self.params['print_verbose'] >= 1: print(f"----- mask_lengths: {self.env.obs.plan_data_dict['mask_lengths']}")
                        if self.params['print_verbose'] >= 1: print(f"----- mask_widths: {self.env.obs.plan_data_dict['mask_widths']}")
                    
                    delta_areas = self.env.obs.plan_data_dict['delta_areas']
                    mean_delta_areas = np.mean([abs(da) for da in list(delta_areas.values())]) #[:-1]])
                    
                    edge_diff = np.sum([1 for edge in self.env.obs.plan_data_dict['desired_edge_list'] if edge not in self.env.obs.plan_data_dict['edge_list']])
                    
                    room_names = self.env.obs.plan_data_dict['rooms_dict'].keys()
                    try:
                        delta_aspect_ratio = list(
                            np.around(
                                [self.env.obs.plan_data_dict['rooms_dict'][room_name]['delta_aspect_ratio'] 
                                          for room_name in list(room_names)[self.env.obs.plan_data_dict['mask_numbers']:]], 
                                decimals=2)
                            )
                    except:
                        print('wait in random_agent')
                        raise ValueError('sth is wrong!')
                    
                    if self.params['print_verbose'] >= 1: print(f"----- delta_areas: { delta_areas }")
                    if self.params['print_verbose'] >= 1: print(f"----- areas: { self.env.obs.plan_data_dict['areas']}")
                    if self.params['print_verbose'] >= 1: print(f"----- mean_delta_areas: { mean_delta_areas}")
                    
                    if self.params['print_verbose'] >= 1: print(f"----- desired_edge_list: {self.env.obs.plan_data_dict['desired_edge_list']}")
                    if self.params['print_verbose'] >= 1: print(f"----- edge_list: {self.env.obs.plan_data_dict['edge_list']}")
                    if self.params['print_verbose'] >= 1: print(f"----- edge_diff: {edge_diff}")
                    
                    if self.params['print_verbose'] >= 1: print(f"----- last reward: {reward:.2f}")
                    # if self.params['print_verbose'] >= 1: print(f"----- desired_aspect_ratio: {self.env.fenv_config['desired_aspect_ratio']}")
                    # if self.params['print_verbose'] >= 1: print(f"----- delta_aspect_ratio: {delta_aspect_ratio}")
                    # if self.params['print_verbose'] >= 1: print(f"----- sum_delta_aspect_ratio: {np.sum(delta_aspect_ratio):.2f}")
                    # if self.params['print_verbose'] >= 1: print(f"----- mean_delta_aspect_ratio: {np.mean(delta_aspect_ratio):.2f}")
                    
                    if self.params['print_verbose'] >= 1: print(f"- - - - - Timestep: {time_step:03}")
                    
                    break
                    
                else:
                    if self.params['render_verbose'] >= 2: self.env.render()
                    if self.params['render_verbose'] >= 3: self.env._illustrate()
                    if self.params['render_verbose'] >= 4: self.env._display()

                if time_step % 5000 == 0:
                    if self.params['print_verbose'] >= 2: print(f"TimeStep: {time_step}")
            self.episode_len.append(time_step+1)
            
            if self.params['save_env_data_dict']:
                if i % 10 == 0:
                    self._save_plan_values(self.env_data_dict)
                
            if self.params['print_verbose'] >= 2: print("==========================================================")
        
        if self.params['save_env_data_dict']:
            self._save_plan_values(self.env_data_dict)
        
        # if self.params['print_verbose'] >= 1: print(f"min(obs): {np.min(obs_min_list)} , max(obs): {np.max(obs_max_list)}")
        if self.params['print_verbose'] >= 1: print(f"- - - - - episode_len: {self.episode_len}")
        if self.params['print_verbose'] >= 1: print(f"- - - - - Mean of episode_len: {np.mean(self.episode_len)}")
        if self.params['print_verbose'] >= 1: print(f"- - - - - reward_per_episode: {self.reward_per_episode}")
        if self.params['print_verbose'] >= 1: print(f"- - - - - n_episodes: {n_episodes}")
        if self.params['print_verbose'] >= 1: print(f"- - - - - num_failures: {num_failures}")
        
        if self.params['save_log_flag']:
            if self.params['print_verbose'] >= 1: 
                self.logger.info(f"- - - - - scenario_name: {self.fenv_config['scenario_name']}")
                for fkey in fkeys:
                    self.logger.info(f"- - - - - {fkey}: {self.fenv_config[fkey]}")
                self.logger.info(f"- - - - - episode_len: {self.episode_len}")
                self.logger.info(f"- - - - - Mean of episode_len: {np.mean(self.episode_len)}")
                self.logger.info(f"- - - - - reward_per_episode: {self.reward_per_episode}")
                self.logger.info(f"- - - - - n_episodes: {n_episodes}")
                self.logger.info(f"- - - - - num_failures: {num_failures}")
                self.logger.info("################################################## End")
            
        
    def _save_plan_values(self, env_data_dict):
        self.generated_plans_dir = self.fenv_config['generated_plans_dir']
        if self.fenv_config['is_area_considered'] and self.fenv_config['is_adjacency_considered']:
            if self.params['high_resolution']:
                self.plan_values_path = f"{self.generated_plans_dir}/plan_values__{self.fenv_config['room_count_str']}__area_adjacency_hr.csv"
            else:
                self.plan_values_path = f"{self.generated_plans_dir}/plan_values__{self.fenv_config['room_count_str']}__area_adjacency.csv"
        elif self.fenv_config['is_area_considered'] and self.fenv_config['is_proportion_considered']:
            if self.params['high_resolution']:
                self.plan_values_path = f"{self.generated_plans_dir}/plan_values__{self.fenv_config['room_count_str']}__area_proportion_hr.csv"
            else:
                self.plan_values_path = f"{self.generated_plans_dir}/plan_values__{self.fenv_config['room_count_str']}__area_proportion.csv"
        elif self.fenv_config['is_area_considered']:
            if self.params['high_resolution']:
                self.plan_values_path = f"{self.generated_plans_dir}/plan_values__{self.fenv_config['room_count_str']}__area_hr.csv"
            else:
                self.plan_values_path = f"{self.generated_plans_dir}/plan_values__{self.fenv_config['room_count_str']}__area_3.csv"
        else:
            if self.params['high_resolution']:
                self.plan_values_path = f"{self.generated_plans_dir}/plan_values__{self.fenv_config['room_count_str']}__proportion_hr.csv"
            else:
                self.plan_values_path = f"{self.generated_plans_dir}/plan_values__{self.fenv_config['room_count_str']}__proportion.csv"
                
        # if os.path.exists(self.plan_values_path):
        #     plan_values_df_old = pd.read_csv(self.plan_values_path)
        #     plan_values_df_new = pd.DataFrame.from_dict(env_data_dict, orient='index')
        #     plan_values_df_new['accepted_action_sequence'] = list(self.accepted_action_sequence_dict.values())
        #     plan_values_df = pd.concat([plan_values_df_old, plan_values_df_new], axis=0)
        #     plan_values_df.to_csv(self.plan_values_path, index=False)
        # else:
        plan_values_df_new = pd.DataFrame.from_dict(env_data_dict, orient='index')
        plan_values_df_new['accepted_action_sequence'] = list(self.accepted_action_sequence_dict.values())
        plan_values_df_new.to_csv(self.plan_values_path, index=False)
            
    
    def make_graph(self, show_adj_graph_flag=False):
        layout_graph = LayoutGraph(self.env.obs.plan_data_dict)
        num_nodes, edge_list = layout_graph.extract_graph_data()
        # print(edge_list)
        if show_adj_graph_flag:
            adj_matrix = layout_graph.get_adj_matrix(num_nodes, np.array(edge_list)-1)
            fig = plt.figure()
            layout_graph.show_graph(adj_matrix)
            # print(adj_matrix)
        return edge_list
        

    def _get_rooms_gravity_coord(self, plan_data_dict):
        rooms_dict = plan_data_dict['rooms_dict']
        mask_numbers = plan_data_dict['mask_numbers']
        
        rooms_gravity_coord_dict = {}
        for i, (room_name, this_room) in enumerate(rooms_dict.items()):
            if i+1>mask_numbers:
                room_shape = this_room['room_shape']
                room_positions = this_room['room_positions']
                room_coords = [self.__image_coords2cartesian(p[0], p[1], self.fenv_config['max_y']) for p in room_positions]
                
                if room_shape == 'rectangular':
                    gravity_coord = self.__get_gravity(room_coords)
                    rooms_gravity_coord_dict[room_name] = gravity_coord
                    
                else:
                    sub_rects = this_room['sub_rects']
                    max_area_ind = np.argmax(sub_rects['areas'])+1
                    max_sub_rects_positions = sub_rects['all_rects_positions'][max_area_ind]
                    max_sub_rects_coords = [self.__image_coords2cartesian(p[0], p[1], self.fenv_config['max_y']) for p in max_sub_rects_positions]
                    gravity_coord = self.__get_gravity(max_sub_rects_coords)
                    rooms_gravity_coord_dict[room_name] = gravity_coord
                    
        return rooms_gravity_coord_dict
    
    
    def __get_gravity(self, room_coords):
        room_coords = np.array(room_coords)
        median = np.median(room_coords,axis=0).tolist() #  np.array(np.median(room_coords,axis=0), dtype=int).tolist()
        dists = [np.linalg.norm(median-rc) for rc in room_coords]
        gravity_coord = list(room_coords[np.argmin(dists)])
        return gravity_coord
    
    
    @staticmethod
    def __image_coords2cartesian(r, c, n_rows):
        return c, n_rows-r
        
# %%
if __name__ == '__main__':
    start_time = time.time()
    
    df_10 = pd.DataFrame()
    
    for tr in range(1):
        
        
        parser = argparse.ArgumentParser(description='Process some parameters for random agent.')
        parser.add_argument('--n_episodes', type=int, default=1, help='n_episodes') #15660
        args = parser.parse_args()
        
        fenv_config = LaserWallConfig().get_config()
        
            
        fenv_config.update({'random_agent_flag': True})
        
        
        investigation_mode = True
        
    
        params = {
        'print_verbose': 1 if investigation_mode else 0,
        'render_verbose' : 1 if investigation_mode else 0,
        'trial': tr,
        'n_episodes': args.n_episodes,
        
        'fixed_action_seq_flag': True,
        'action_sequence': [997, 178, 1085, 1355, 867, 1406, 1423],#[597, 1170, 308, 689, 129, 705, 147],
        
        'save_log_flag': False,
        'show_render_flag': True if investigation_mode else False,
        'save_render_flag': False,
        'so_thick_flag': True,
        
        'save_env_data_dict': True if investigation_mode else True,
        'show_adj_graph_flag': False,
        
        'show_graph_on_plan_flag': True,
        
        'high_resolution': True if fenv_config['max_x'] == 40 else False,
        
        }
        
        params.update({
            'plan_config_source_name': 'fixed_test_config' , # 'create_random_config', fixed_test_config, load_random_config
            'is_adjacency_considered': True,
            'net_arch': 'Fc',
            'rewarding_method_name': 'OnlyFinalRewardSimple',
        })
        
        
        if fenv_config['plan_config_source_name'] == 'oflline_mode':
            raise ValueError("For random_agent, the value of plan_config_source_name cannot be oflline_mode")
            
        params.update({'stop_time_step': 2000 if params['plan_config_source_name'] == 'create_random_config' else fenv_config['stop_time_step']})
    
        self = RandomAgent(fenv_config, params)
        self.run_random_agent(n_episodes=args.n_episodes)
        # self.make_graph(show_adj_graph_flag=True)
        
        temp_df_10 = pd.DataFrame({
            'trial': [tr] * args.n_episodes,
            'reward_per_episode': self.reward_per_episode, 
            'last_reward': self.last_reward,
            'episode_len': self.episode_len,
            'status': self.status
            })
        
        df_10 = pd.concat([df_10, temp_df_10], axis=0)
        
    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time}")
    
    # make_gif(input_dir=fenv_config['generated_plans_dir'], 
    #           output_dir=fenv_config['generated_gif_dir'])
        
        
        