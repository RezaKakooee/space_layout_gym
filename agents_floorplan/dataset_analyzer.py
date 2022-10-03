#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 10:59:06 2022

@author: RK
"""

import os
import ast
import json
import copy
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

import matplotlib.pyplot as plt

from gym_floorplan.envs.fenv_config import LaserWallConfig
from gym_floorplan.envs.master_env import MasterEnv




# %%
class DatasetAnalyzer:
    def __init__(self, params):
        self.params = params
        
        self.fenv_config = LaserWallConfig().get_config()
        
        self.fenv_config.update({k: v for k, v in self.params.items()})
        
        # self.fenv_config.update({
        #     'plan_config_source_name': self.params['plan_config_source_name'],
        #     'show_render_flag': self.params['show_render_flag'],
        #     'so_thick_flag': self.params['so_thick_flag'],
        #     'show_graph_on_plan_flag': self.params['show_graph_on_plan_flag'],
        #     'show_room_dots_flag': self.params['show_room_dots_flag'],
        #     'save_render_flag': self.params['save_render_flag'],
        #     })
        
        
    def design(self, n_episodes=None):
        self.env = MasterEnv(self.fenv_config)
        # print(self.env.episode)
        
        if not n_episodes:
            self.plans_df = self.env.obs.plan_construcror.adjustable_configs_handeler.plans_df
            n_episodes = len(self.plans_df)
        
        
        actions_list = []
        self.reward_list_in = []
        self.reward_list = []
        self.episode_edgelist = []
        # self.emb_dict = {}
        # self.temp_emb_dict = {}
        mean_delta_areas_list = []
        edge_diff_list = []
        for i in tqdm(range(n_episodes)):
                observation = self.env.reset()
            # if self.env.episode in [  3,  10,  12,  13,  20,  25,  33,  34,  35,  43,  48,  54,  59,
            #                           62,  64,  68,  71,  87,  98, 106, 118, 119, 121, 139, 144, 146,
            #                           149, 150, 151, 155, 156, 160, 166, 168, 169]:
                actions = self.env.obs.plan_data_dict['potential_good_action_sequence']
                for j, action in enumerate(actions):
                    observation, reward, done, info = self.env.step(action)
                    
                    experience = (action, reward, done)
                    self.reward_list_in.append(reward)
                    
                    # print("-----------------------------------------")
                    # print(f"Episode: {i}, TimeStep: {self.env.time_step}, Action: {action}, Reward: {reward}, Done: {done}")
                    
                        
                    # if self.fenv_config['net_arch'] == 'CnnGcn':
                    #     if self.params['context_data_save_flag']:# and done:
                    #         self._get_and_save_context_data_dict(self.env.obs.plan_data_dict, reward, i, j, done=False)
                     
                    
                    # if self.params['obs_grayscale_save_flag']:
                    #         obs_grayscale = self._get_and_save_obs_grayscale(self.env.obs.plan_data_dict, i, j, save=True)
                    
                    if done:
                        if self.params['context_data_save_flag']:# and done:
                            self._get_and_save_context_data_dict(self.env.obs.plan_data_dict, reward, i, j, done=True)
                        
                        self.reward_list.append(reward)
                        self.episode_edgelist.append((i, self.env.obs.edge_list))
                        
                        mean_delta_areas = np.mean([abs(da) for da in list(self.env.obs.plan_data_dict['delta_areas'].values())])
                        edge_diff = np.sum([1 for edge in self.env.obs.plan_data_dict['desired_edge_list'] if edge not in self.env.obs.plan_data_dict['edge_list']])
                        
                        mean_delta_areas_list.append(mean_delta_areas)
                        edge_diff_list.append(edge_diff)
                      
                        
                    if done and self.params['show_render_flag'] and edge_diff == 0:
                        # print(f"\nEpisode: {i:05}, TimeStep: {self.env.time_step+1:02}, Action: {action:04}, Reward: {reward:.2f}, Done: {done}")
                        self.env.obs.plan_data_dict.update({'rooms_gravity_coord_dict': self._get_rooms_gravity_coord(self.env.obs.plan_data_dict)})
                        # self.env.demonestrate()
                        self.env.render()
                        adjustable_configs = self.env.obs.plan_construcror.adjustable_configs
                        print(f"episode: {i}, adjustable_configs: {adjustable_configs}")
                        
                        
                                    
                if self.params['split_obs_grayscale_data_flag']:
                    self._split_obs_grayscale_data()
                
                print(f"\nNum of unique plans: {len(actions_list)}")
                print(f"Avg of mean_delta_areas: {np.mean(mean_delta_areas_list)}")
                print(f"Avg of edge_diff: {np.mean(edge_diff_list)}")
                    
                    
    def _get_and_save_context_data_dict(self, plan_data_dict, reward, i, j, done=False, save=True):
        if i == 0:
            self.scenario_dir = self.fenv_config['scenario_dir']
            self.context_data_dict_dir = os.path.join(self.scenario_dir, 'inference_files/context_data_dict/')
            if not os.path.exists(self.context_data_dict_dir):
                os.mkdir(self.context_data_dict_dir)
                
            self.context_data_dict_done_dir = os.path.join(self.scenario_dir, 'inference_files/context_data_dict_done/')
            if not os.path.exists(self.context_data_dict_done_dir):
                os.mkdir(self.context_data_dict_done_dir)
        
        normalized_graph_data_numpy_old = self.__graph_normalization(plan_data_dict['graph_data_numpy_old'])
        normalized_graph_data_numpy = self.__graph_normalization(plan_data_dict['graph_data_numpy'])
        normalized_plan_canvas_arr_old = self.__image_normalization(plan_data_dict['plan_canvas_arr_old'])
        normalized_plan_canvas_arr = self.__image_normalization(plan_data_dict['plan_canvas_arr'])
        
        context_data_dict = {'graph_data_numpy_old': normalized_graph_data_numpy_old,
                             'graph_data_numpy': normalized_graph_data_numpy,
                             'plan_canvas_arr_old': normalized_plan_canvas_arr_old,
                             'plan_canvas_arr': normalized_plan_canvas_arr,
                             'reward': reward,
            }
        
        if save and done:
            context_data_done_file_path = os.path.join(self.context_data_dict_done_dir, f'context_data__done__episode_{i+1:05}__timestep_{j+1:02}.npy')
            np.save(context_data_done_file_path, context_data_dict)
        else:
            context_data_file_path = os.path.join(self.context_data_dict_dir, f'context_data__episode_{i+1:05}__timestep_{j+1:02}.npy')
            np.save(context_data_file_path, context_data_dict)
            
    
    def __graph_normalization(self, graph_):
        graph = copy.deepcopy(graph_)
        for feat_cat_name, feat_cat_val in graph['graph_features_numpy'].items():
            for room_name, room_feat in feat_cat_val.items():
                for akey in ['desired_area', 'current_area', 'delta_area']:
                    graph['graph_features_numpy'][feat_cat_name][room_name][akey] /= 400
                    
                if feat_cat_name == 'fully_current_graph_features_dict_numpy':
                    for ckey in ['start_of_wall', 'before_anchor', 'anchor', 'after_anchor', 'end_of_wall']:
                        coord = graph['graph_features_numpy'][feat_cat_name][room_name][ckey]
                        if -1 not in coord:
                            graph['graph_features_numpy'][feat_cat_name][room_name][ckey] = list(np.array(coord)/20)
                        else:
                            break
                    
                    distances = graph['graph_features_numpy'][feat_cat_name][room_name]['distances']
                    if -1 not in distances:
                        graph['graph_features_numpy'][feat_cat_name][room_name]['distances'] = list(np.array(distances)/30)
                    
        return graph           
                    
    
    def __image_normalization(self, image_):
        image = copy.deepcopy(image_)
        image[:,:,0] /= 25
        return image
    
    
    def _split_context_data(self, num_valids):
        self.scenario_dir = f"{self.fenv_config['storage_dir']}/offline_datasets"
        self.context_data_dict_dir = os.path.join(self.scenario_dir, 'inference_files/context_data_dict/')
        context_names = os.listdir(self.context_data_dict_dir)
        
        train_context_dir = os.path.join(self.context_data_dict_dir, 'train')
        valid_context_dir = os.path.join(self.context_data_dict_dir, 'valid')
        
        if not os.path.exists(train_context_dir):
            os.mkdir(train_context_dir)
                
        if not os.path.exists(valid_context_dir):
            os.mkdir(valid_context_dir)
        
        num_contexts = len(context_names)
        num_trains = num_contexts - num_valids        
            
        valid_image_indices = np.random.randint(0, num_contexts, num_valids)
        
        for i, context_name in enumerate(context_names):
            source = os.path.join(self.context_data_dict_dir, context_name)
            if i in valid_image_indices:
                destination = os.path.join(valid_context_dir, context_name)
            else:
                destination = os.path.join(train_context_dir, context_name)
            dest = shutil.move(source, destination) 
            

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
    
            
    @staticmethod
    def __shift_to_origin(coords, anchor_coord):
        if isinstance(coords, list):
            coords = np.array(coords)
        if isinstance(anchor_coord, list):
            anchor_coord = np.array(anchor_coord) 
        
        new_coords = coords - anchor_coord
        return new_coords
    
    
    @staticmethod
    def __return_to_anchor(coords, anchor_coord):
        if isinstance(coords, list):
            coords = np.array(coords)
        if isinstance(anchor_coord, list):
            anchor_coord = np.array(anchor_coord) 
        
        new_coords = coords + anchor_coord
        return new_coords
    
    
    @staticmethod
    def __image_coords2cartesian(r, c, n_rows):
        return c, n_rows-r


    @staticmethod
    def __cartesian2image_coord(x, y, max_y):
        return max_y-y, x
    
    
    def __get_gravity(self, room_coords):
        room_coords = np.array(room_coords)
        median = np.median(room_coords,axis=0).tolist() #  np.array(np.median(room_coords,axis=0), dtype=int).tolist()
        dists = [np.linalg.norm(median-rc) for rc in room_coords]
        gravity_coord = list(room_coords[np.argmin(dists)])
        return gravity_coord


    def _plot_emb_as_vector(self):#, emb, time_step, done):
        # if time_step == 0:
        #     self.fig = plt.figure()
        #     self.time_steps = np.arange(len(emb))
        # plt.plot(self.time_steps, emb)
        
        # if done:
        #     plt.show()
        
        fig = plt.figure()
        all_obs = np.array(list(self.emb_dict.values()))
        df = pd.DataFrame(all_obs).T
        df.plot(legend=False)
        plt.show()
        

    def _get_and_save_obs_grayscale(self, plan_data_dict, i, j, save=True):
        if i == 0:
            self.scenario_dir = self.fenv_config['the_scenario_dir']
            self.obs_grayscale_dir = os.path.join(self.scenario_dir, 'inference_files/obs_grayscale/')
            if not os.path.exists(self.obs_grayscale_dir):
                os.mkdir(self.obs_grayscale_dir)
            
        obs_m = plan_data_dict['obs_mat']/10
        n_rows, n_cols = obs_m.shape
        obs_mat_w = plan_data_dict['obs_mat_w']
        moving_labels = plan_data_dict['moving_labels']
        obs_grayscale = copy.deepcopy(obs_m)
        obs_grayscale[0, :] = 1
        obs_grayscale[-1, :] = 1
        obs_grayscale[:, 0] = 1
        obs_grayscale[:, -1] = 1
        for r in range(n_rows):
            for c in range(n_cols):
                if (moving_labels[r][c] > 0) and  (moving_labels[r][c] <= 2):
                    obs_grayscale[r][c] = 0
        
        if save:
            obs_grayscale_file_path = os.path.join(self.obs_grayscale_dir, f'obs_grayscale__episode_{i+1:05}__timestep_{j+1:02}.npy')
            np.save(obs_grayscale_file_path, obs_grayscale)
                        
        return obs_grayscale
# %%        
if __name__ == '__main__':
    scenario_name = 'Scenario__DOLW__Masked_LFC__XX_Rooms__LRres__Area__Adj__Prop__FC__2022_10_02_1647' # 'Scenario__DOLW__Masked_LRC__XX_Rooms__LRres__Area__Adj__CNNGCN__2022_08_24_2349' # 'random' # random 'Scenario__DOLW__Masked_TC__06_Walls__Area__FC__2022_07_17_0947' # random 
    
    storage_dir = '/home/rdbt/ETHZ/dbt_python/housing_design_making-general-env/agents_floorplan/storage'
    if scenario_name == 'random':
        the_scenario_dir = f"{storage_dir}/random"
    else:
        the_tunner_dir = f"{storage_dir}/tunner/local_dir"
        the_scenario_dir = f"{the_tunner_dir}/{scenario_name}"
    
        
    investigation_mode = True
    
    params = {
    'scenario_name': scenario_name,
    # 'the_tunner_dir': the_tunner_dir,
    'the_scenario_dir': the_scenario_dir,
    'high_resolution': False,
    'n_episodes': None,#1 if investigation_mode else None,
    'context_data_save_flag': False,
    'split_obs_grayscale_data_flag': False,
    'obs_grayscale_save_flag': True,
    'random_order_flag': False,
    
    'plan_config_source_name': 'offline_mode',
    'show_render_flag': True if investigation_mode else False,
    'so_thick_flag': True,
    'show_graph_on_plan_flag': True if investigation_mode else False,
    'show_room_dots_flag': False if investigation_mode else False,
    'save_render_flag': True,
    
    'show_emb_vector_flag': False,
    
    'trial': 1,
    }
          
    self = DatasetAnalyzer(params)
    self.design(n_episodes=params['n_episodes'])
    
    # self._split_context_data(6957)
    
    # episode_edgelist_df = pd.DataFrame(self.episode_edgelist)
    # episode_edgelist_df.columns = ['episode', 'edge_list']
    # episode_edgelist_df.to_csv('episode_edgelist_df.csv', index=False)
    

    if params['show_emb_vector_flag']:
        self._plot_emb_as_vector()