# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 09:45:21 2021

@author: RK
"""
import os
from pathlib import Path

housing_design_dir = os.path.realpath(Path(os.getcwd()).parents[0])
gym_floorplan_dir = f"{housing_design_dir}/gym-floorplan"
print(f"{os.path.basename(__file__)} -> housing_design_dir: {housing_design_dir}")
print(f"{os.path.basename(__file__)} -> gym_floorplan_dir:  {gym_floorplan_dir}")

# %%
import sys

# print(f"{os.path.basename(__file__)} -> Appending: ........ {gym_floorplan_dir}")
# sys.path.append(f"{gym_floorplan_dir}")

# %%
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# %%
# import re
import ast
import json
import logging 
import numpy as np
import pandas as pd
from pprint import pformat
from collections import defaultdict

from torchsummary import summary

#import seaborn as sns
# import matplotlib.pylab as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import ray
from ray.rllib.agents import dqn
from ray.rllib.agents import ppo
from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.rllib.agents.dqn.dqn import DQNTrainer
# from ray.rllib.models import ModelCatalog
# from ray.rllib.agents.callbacks import DefaultCallbacks


import torch
from torch_intermediate_layer_getter import IntermediateLayerGetter as MidGetter

# from agent_config import AgentConfig
from env_to_agent import EnvToAgent
# from my_callbacks import MyCallbacks
# from gym_floorplan.envs.fenv_config import LaserWallConfig

from gym_floorplan.envs.layout_graph import LayoutGraph



# %%
class MyInferencer(EnvToAgent):
    def __init__(self, fenv_config, agent_config, learner_chkpt_config, params):
        
        super().__init__(fenv_config=fenv_config, agent_config=agent_config)

        self.params = params
        self.fenv_config = fenv_config
        self.agent_config = agent_config
        self.learner_chkpt_config = learner_chkpt_config
        
        if self.params['compute_non_greedy_actions']:
            self.learner_chkpt_config.update({'explore': False})
            
        self.optim_exp_df_path = f"{self.params['the_scenario_dir']}/plans_df.csv"
        
            
        info = ray.init(
                        ignore_reinit_error=True,
                        log_to_driver=False,
                        local_mode=True,
                        object_store_memory=10**8,
                        )


        self.agent = self._restore_agent()

   
    def _restore_agent(self):
        if self.agent_config['agent_first_name'] == 'ppo':
            self.config = ppo.DEFAULT_CONFIG.copy()
        elif self.agent_config['agent_first_name'] == 'dqn':
            self.config = dqn.DEFAULT_CONFIG.copy()

        for key in self.learner_chkpt_config:
            self.config[key] = self.learner_chkpt_config[key]
        
        self.config['framework'] = 'torch'
        self.config['num_workers'] = 1
        if self.agent_config['agent_first_name'] == 'ppo':
            agent = PPOTrainer(env=self.env_name, config=self.config)
        elif self.agent_config['agent_first_name'] == 'dqn':
            agent = DQNTrainer(env=self.env_name, config=self.config)

        agent.restore(str(self.params['chkpt_path']))
        
        return agent
        
    
    def _get_plan_row(self, i):
        if self.params['random_order_flag']:
            if i == 0:
                self.randomized_plan_oder = np.random.randint(0, len(self.plan_df), len(self.plan_df))
            j = self.randomized_plan_oder[i]
            row = self.plan_df.iloc[j,:]
        else:
            row = self.plan_df.iloc[i%len(self.plan_df),:]
        return row
    
    
    def _evaluate(self):
        fkeys = ['env_name', 'env_type', 'env_planning', 'env_space', 'net_arch', 
                'fixed_fc_observation_space', 'include_wall_in_area_flag', 
                'area_tolerance', 'reward_shaping_flag', 'mask_flag']
        
        if self.env_name == 'master_env-v0':
            reward_per_episode = []
            episode_len = []
            num_failures = 0
            areas_per_episode = defaultdict()
            delta_areas_per_episode = defaultdict()
            
            self.edge_diff_list = []
            self.mean_delta_areas_list = []
            
            if self.params['load_action_from_buffer_flag']:
                optim_exp_df = pd.read_csv(self.optim_exp_df_path)
                optim_exp_df['accepted_action_sequence'] = optim_exp_df['accepted_action_sequence'].apply(ast.literal_eval)  
                optim_action_sequences_list = optim_exp_df['accepted_action_sequence'].values
              
            if self.params['n_episodes'] is None:
                self.plans_df = self.env.obs.plan_construcror.adjustable_configs_handeler.plans_df
                n_episodes = len(self.plans_df)
            else: 
                n_episodes = self.params['n_episodes']
            
            for i in range(n_episodes):
                if i == 0: 
                    if self.params['print_verbose'] >= 1: print("################################################## Start")
                    for fkey in fkeys:    
                        if self.params['print_verbose'] >= 1: print(f"- - - - - {fkey}: {self.fenv_config[fkey]}")
                        if self.params['print_verbose'] >= 1: self.params['logger'].info(f"- - - - - {fkey}: {self.fenv_config[fkey]}")
                 
                if self.params['print_verbose'] >= 2: print(f"================================= Episode: {i:03}")
                
                if self.fenv_config['use_lstm']:
                    state = self.agent.get_policy().model.get_initial_state()     
                    
                observation = self.env.reset()
                time_step = self.env.time_step
                if self.params['render_verbose'] >= 3: self.env.render()
                if self.params['render_verbose'] >= 4: self.env.illustrate()
                done = False
                this_episode_reward = 0
                
                if self.params['load_action_from_buffer_flag']:
                    action_sequence = optim_action_sequences_list[i]
                while not done:
                    if self.params['print_verbose'] >= 2:  print(" - - - - -  - - - - - - - - - - - - - - - - - - -- - - -")
                    if self.params['load_action_from_buffer_flag']:
                        action = action_sequence[time_step]
                    else:
                        if self.params['compute_non_greedy_actions']:
                            if self.fenv_config['use_lstm']:
                                action, state, logit = self.agent.compute_single_action(observation=observation, 
                                                                                        prev_action=1.0, 
                                                                                        prev_reward=0.0, 
                                                                                        state=state)
                            else:   
                                action = self.agent.compute_single_action(observation, explore=False)
                        else:
                           if self.fenv_config['use_lstm']:
                                action, state, logit = self.agent.compute_single_action(observation=observation, 
                                                                                        prev_action=1.0, 
                                                                                        prev_reward=0.0, 
                                                                                        state=state)
                           else:   
                                action = self.agent.compute_single_action(observation)
                                
                                
                    observation, reward, done, info = self.env.step(action)
                    time_step = self.env.time_step
                    this_episode_reward += reward
                    areas_per_episode[i] = self.env.obs.plan_data_dict['areas']
                    delta_areas_per_episode[i] = self.env.obs.plan_data_dict['delta_areas']
                    if self.fenv_config['net_arch'] in ['fccnn', 'cnnfc']:
                        experience = (action, (observation[0].shape, observation[1].shape), reward, done, info)
                    else:
                        if self.fenv_config['action_masking_flag']:
                            experience = (action, observation['real_obs'].shape, reward, done, info)
                        else:
                            experience = (action, observation[1].shape, reward, done, info)
                    
                    if self.params['print_verbose'] >= 3: print(f"- - - - -{time_step:02} -> experience: {experience}")

                    if done and self.env.obs.active_wall_status == 'well_finished':
                        # emb = self._get_embeding(observation)
                        
                        if time_step == self.fenv_config['stop_time_step']:
                            num_failures += 1
                            
                            
                        self.env.obs.plan_data_dict.update({'rooms_gravity_coord_dict': self._get_rooms_gravity_coord(self.env.obs.plan_data_dict)})
                        
                        if self.params['render_verbose'] >= 1: self.env.render(i)
                        if self.params['render_verbose'] >= 2: self.env.illustrate(i)
                        if self.params['render_verbose'] >= 3: self.env.display(i)
                        # if self.params['render_verbose'] >= 1: self.env.demonestrate(i)
                        
                        delta_areas = self.env.obs.plan_data_dict['delta_areas']
                        mean_delta_areas = np.mean([abs(da) for da in list(delta_areas.values())[:-1]])
                        self.mean_delta_areas_list.append(mean_delta_areas)
                        
                        edge_diff = np.sum([1 for edge in self.env.obs.plan_data_dict['desired_edge_list'] if edge not in self.env.obs.plan_data_dict['edge_list']])
                        self.edge_diff_list.append(edge_diff)
                        
                        room_names = self.env.obs.plan_data_dict['rooms_dict'].keys()
                        delta_aspect_ratio = list(np.around([self.env.obs.plan_data_dict['rooms_dict'][room_name]['delta_aspect_ratio'] 
                                              for room_name in list(room_names)[self.env.obs.plan_data_dict['mask_numbers']:]], decimals=2))
                        
                        if self.params['print_verbose'] >= 1: print(f"----- desired_areas: { self.env.obs.plan_data_dict['desired_areas'] }")
                        if self.params['print_verbose'] >= 1: print(f"----- delta_areas: { delta_areas }")
                        if self.params['print_verbose'] >= 1: print(f"----- areas: { self.env.obs.plan_data_dict['areas']}")
                        if self.params['print_verbose'] >= 1: print(f"----- mean_delta_areas: { mean_delta_areas}")
                        
                        if self.params['print_verbose'] >= 1: print(f"----- desired_edge_list: {self.env.obs.plan_data_dict['desired_edge_list']}")
                        if self.params['print_verbose'] >= 1: print(f"----- edge_list: {self.env.obs.plan_data_dict['edge_list']}")
                        if self.params['print_verbose'] >= 1: print(f"----- edge_diff: {edge_diff}")
                        
                        if self.params['print_verbose'] >= 1: print(f"----- delta_aspect_ratio: {delta_aspect_ratio}")
                        if self.params['print_verbose'] >= 1: print(f"----- desired_aspect_ratio: {self.env.fenv_config['desired_aspect_ratio']}")
                        if self.params['print_verbose'] >= 1: print(f"----- sum_delta_aspect_ratio: {np.sum(delta_aspect_ratio):.2f}")
                        if self.params['print_verbose'] >= 1: print(f"----- mean_delta_aspect_ratio: {np.mean(delta_aspect_ratio):.2f}")
                        
                        if self.params['print_verbose'] >= 2: print(f"----- {time_step:002} -> experience: {experience}")
                        if self.params['print_verbose'] >= 2: print(f"========== done: {done}!")
                        if self.params['print_verbose'] >= 2: print(f"- - - - - Status: {done}")
                        
                        if self.params['print_verbose'] >= 1: print(f"- - - - - Timestep: {time_step:03}")
                        
                        break
                    
                    else:
                        if self.params['render_verbose'] >= 3: self.env.render()
                        if self.params['render_verbose'] >= 4: self.env.illustrate()

                episode_len.append(time_step+1)
                reward_per_episode.append(this_episode_reward)
                if self.params['print_verbose'] >= 2: print("==========================================================")
            
            if self.params['print_verbose'] >= 2: 
                print(f"- - - - - episode_len: {episode_len}")
                print(f"- - - - - Mean of episode_len: {np.mean(episode_len)}")
                print(f"- - - - - reward_per_episode: {reward_per_episode}")
                print(f"- - - - - num_failures: {num_failures}")
                self.params['logger'].info("################################################## End")
            if self.params['print_verbose'] >= 1: 
                self.params['logger'].info(f"- - - - - areas_per_episode: {pformat(areas_per_episode.values())}")
                self.params['logger'].info(f"- - - - - delta_areas_per_episode: {pformat(delta_areas_per_episode.values())}")
                self.params['logger'].info(f"- - - - - episode_len: {episode_len}")
                self.params['logger'].info(f"- - - - - Mean of episode_len: {np.mean(episode_len)}")
                self.params['logger'].info(f"- - - - - reward_per_episode: {reward_per_episode}")
                self.params['logger'].info(f"- - - - - num_failures: {num_failures}")
                self.params['logger'].info("################################################## End")
        
    
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
    
    
    def _get_embeding(self, inputs, return_layers=None):
        inputs = {"obs": inputs}
        self.model = self.agent.get_policy().model
        
        print(f"area_layer_out dist to origin: {torch.linalg.norm(self.model.area_layer_out, dim=1).mean()}")
        print(f"adj_layer_out dist to origin: {torch.linalg.norm(self.model.adj_layer_out, dim=1).mean()}")
        
        
        # if return_layers is None:
        #     return_layers = {
        #             'meta_linear_layer.0': 'emb',
        #         }
            
        # self.model.eval()
        # with torch.no_grad():
        #     mid_getter = MidGetter(self.model, return_layers=return_layers, keep_output=True)
        #     mid_outputs, model_output = mid_getter(inputs)
        # emb = mid_outputs['emb'].detach().cpu().numpy().flatten()
        
        emb = self.model.meta_out
        
        print(f"emb dist to origin: {torch.linalg.norm(emb, dim=1).mean()}")
        return emb

        
    def _check_tunner_performance(self, trial_df=None):
        if trial_df is None:
            trial_df = self._load_trial_df_from_local_dir()
        cols = ['episode_len_mean', 'episode_reward_mean']
        df = trial_df[cols]
        df = df.assign(n=list(range(len(df))))
        self._plot_TB(df)
        # df.plot(x='n', y=cols, secondary_y=True)
      
        
    def _plot_TB_subplot(self, df):
        fig = make_subplots(rows=2, cols=1, shared_yaxes=True)
        x = df['timesteps_total']
        fig.add_trace( go.Scatter(x=x, y=df['episode_reward_mean'], mode='lines', name='Episode Reward Mean'), row=1, col=1 )
        fig.add_trace( go.Scatter(x=x, y=df['episode_len_mean'], mode='lines', name='Episode Length Mean'), row=2, col=1 )
        fig.update_traces(line=dict(color="Black", width=5))
        fig.update_layout(height=1000, width=800, hovermode='x')
        fig.update_layout({
                        # 'plot_bgcolor': 'rgba(0, 255, 255, 0.2)', 
                        'paper_bgcolor': 'rgba(0, 0, 0, 0)'
                            })
        fsize = 35
        fig.update_xaxes(title_font=dict(size=fsize, family='Courier', color='black'))
        fig.update_yaxes(title_font=dict(size=fsize, family='Courier', color='black'))
        fig.update_xaxes(tickfont=dict(family='Rockwell', color='black', size=fsize))
        fig.update_yaxes(tickfont=dict(family='Rockwell', color='black', size=fsize))
        fig.update_xaxes(range=[0, 200_000])
        # fig.update_yaxes(range=[ymax, ymin])
        # fig.update_layout(template="presentation")
        fig.show() 
        agent_performance_metrics_dir = os.path.join(self.params['the_scenario_dir'], 'inference_files/agent_performance_metrics')
        if not os.path.exists(agent_performance_metrics_dir):
            os.makedirs(agent_performance_metrics_dir)
           
        fig.write_image(f"{agent_performance_metrics_dir}//tb_plots.png", scale=3)
        
    
    def _plot_TB_multiple(self, df):
        fig = go.Figure()
        x = df['timesteps_total']
        fig.add_trace( go.Scatter(x=x, y=df['episode_reward_mean'], mode='lines', name='Episode Reward Mean') )
        fig.add_trace( go.Scatter(x=x, y=df['episode_len_mean'], mode='lines', name='Episode Length Mean') )
        fig.update_traces(line=dict(color="Black", width=5))
        fig.update_layout(height=1000, width=800, hovermode='x')
        fig.update_layout({
                        # 'plot_bgcolor': 'rgba(0, 255, 255, 0.2)', 
                        'paper_bgcolor': 'rgba(0, 0, 0, 0)'
                            })
        fsize = 35
        fig.update_xaxes(title_font=dict(size=fsize, family='Courier', color='black'))
        fig.update_yaxes(title_font=dict(size=fsize, family='Courier', color='black'))
        fig.update_xaxes(tickfont=dict(family='Rockwell', color='black', size=fsize))
        fig.update_yaxes(tickfont=dict(family='Rockwell', color='black', size=fsize))
        # fig.update_xaxes(range=[0, 200_000])
        # fig.update_yaxes(range=[ymax, ymin])
        # fig.update_layout(template="presentation")
        fig.show() 
        agent_performance_metrics_dir = os.path.join(self.params['the_scenario_dir'], 'inference_files/agent_performance_metrics')
        if not os.path.exists(agent_performance_metrics_dir):
            os.makedirs(agent_performance_metrics_dir)
           
        fig.write_image(f"{agent_performance_metrics_dir}//tb_plots.png", scale=3)
        
        
    def _plot_TB(self, df, y_name='episode_reward_mean'):
        if isinstance(y_name, list):
            fig = px.line(df, x="timesteps_total", y=y_name, #["episode_reward_mean", "episode_len_mean"],
                           labels={"timesteps_total": "Time Step", y_name: y_name})
            fig.update_yaxes(title="Episode Mean Rewartd & Length", title_font=dict(size=20))
        else:
            fig = px.line(df, x="timesteps_total", y=y_name, #"episode_reward_mean",
                          labels={"timesteps_total": "Time Step", y_name: y_name})
        
        
        fig.update_traces(line=dict(width=3))
        fig.update_layout(height=600, width=800, hovermode='x')
        fig.update_layout({
                        # 'plot_bgcolor': 'rgba(0, 255, 255, 0.2)', 
                        'paper_bgcolor': 'rgba(0, 0, 0, 0)'})
        fsize = 30
        fig.update_xaxes(title_font=dict(size=fsize, family='Courier', color='black'), tickangle = 270)
        fig.update_yaxes(title_font=dict(size=25, family='Courier', color='black'))
        fig.update_xaxes(tickfont=dict(family='Rockwell', color='black', size=fsize), tickangle = 270)
        fig.update_yaxes(tickfont=dict(family='Rockwell', color='black', size=fsize))
        # fig.update_xaxes(range=[1_600_000, 2_200_000])
            
        
        # fig.update_yaxes(range=[ymax, ymin])
        fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99))
        fig.update_layout(legend={'title_text':''})
        fig.update_layout(legend_font_size=30)
        # fig.update_layout(template="presentation")

        fig.show()
        
        agent_performance_metrics_dir = os.path.join(self.params['the_scenario_dir'], 'inference_files/agent_performance_metrics')
        if not os.path.exists(agent_performance_metrics_dir):
            os.makedirs(agent_performance_metrics_dir)
        fig.write_image(f"{agent_performance_metrics_dir}/tb_plots_{y_name}.png", scale=3)

        
    def _check_trainer_performance(self, trial_df=None):
        if trial_df is None:
            trial_df = self._load_trial_df_from_local_dir()
        df = pd.DataFrame(data=trial_df)
        df.plot(x="n", y=['episode_len_mean', 'episode_reward_mean'], secondary_y=True)

        
    def _load_trial_df_from_local_dir(self):
        checkpoint_dir = Path(self.params['chkpt_path']).parents[2]
        outputs_dir = os.path.join(checkpoint_dir, "outputs")
        trial_df_csv_path = os.path.join(outputs_dir, "trial_df.csv")
        trial_df = pd.read_csv(trial_df_csv_path)
        return trial_df


    def _get_model(self, chkpt_config=None, chkpt_path=None):
        # policy_id  = 'policy_0'
        model = self.agent.get_policy().model
        print(model)
        obs_shape = self.env.obs.observation.shape
        summary(model, input_size=(obs_shape[2], obs_shape[0], obs_shape[1]))
        return model
    
    
# %%
class RLDesigner:
    def __init__(self, params, fenv_config=None):
        
        self.params = params
        self.fenv_config = fenv_config
        
        self._restore_trained_config()
        self._adjust_configs()
        self.params['logger'] = self._get_logger()
        
        
    def _restore_trained_config(self):
        chkpt_path_str = str(self.params['chkpt_path'])
        if "Scenario" not in chkpt_path_str: 
            chkpt_path_ = chkpt_path_str.split('local_dir')
            chkpt_path_ = chkpt_path_[0] + f"local_dir/{self.params['scenario_name']}" + chkpt_path_[1]
            self.chkpt_path = Path(chkpt_path_)
    
        if "trainer" in str(self.params['chkpt_path']):
            configs_dir = os.path.join(Path(self.params['chkpt_path']).parents[1], 'configs')
        else:
            configs_dir = os.path.join(Path(self.params['chkpt_path']).parents[2], 'configs')
        
        if self.fenv_config is None:
            fenv_config_npy_path = os.path.join(configs_dir, 'fenv_config.npy')
            self.fenv_config = np.load(fenv_config_npy_path, allow_pickle=True).tolist()
            
            if self.fenv_config['scenario_name'] != self.params['scenario_name']:
                raise ValueError("Scenario names do not match!")
            
        agent_config_npy_path = os.path.join(configs_dir, 'agent_config.npy')
        learner_config_npy_path = os.path.join(configs_dir, 'learner_config.npy')
    
        self.agent_config = np.load(agent_config_npy_path, allow_pickle=True).tolist()
        self.learner_chkpt_config = np.load(learner_config_npy_path, allow_pickle=True).tolist()
        
    
    def _adjust_configs(self):
        self.agent_config.update({'learner_name': "tunner"})
        self.agent_config.update({'RLLIB_NUM_GPUS': 0})
        
        ## update and rewrite
        # self.fenv_config.update({'desired_edge_list':[]})
        # with open(fenv_config_npy_path, "wb") as f:
        #     np.save(f, self.fenv_config)
        
        self.fenv_config.update({k: v for k, v in self.params.items()})
        
        # self.fenv_config.update({'trial': self.params['trial']})
        # self.fenv_config.update({'show_render_flag': self.params['show_render_flag']})
        # self.fenv_config.update({'save_render_flag': self.params['save_render_flag']})
        # self.fenv_config.update({'scenario_dir': self.params['scenario_dir']})
        # self.fenv_config.update({'so_thick_flag': self.params['so_thick_flag']})
        # self.fenv_config.update({'use_lstm': False})
        
        # if self.params['plan_config_source_name']:
        #     self.fenv_config.update({'plan_config_source_name': self.params['plan_config_source_name'],
        #                              'learner_name': 'tunner',
        #                              'agent_first_name': 'ppo',
        #                              'chkpt_path': self.params['chkpt_path']})
        
        
        # self.generated_plans_dir = f"{self.fenv_config['generated_plans_dir']}/{self.parmas['scenario_name']}"
        # if not os.path.exists(self.generated_plans_dir):
        #     os.mkdir(self.generated_plans_dir)
        
    
    def _get_logger(self):
            LOG_FORMAT = "%(message)s"
            inference_files_dir = os.path.join(self.params['the_scenario_dir'], 'inference_files')
            if not os.path.exists(inference_files_dir):
                os.makedirs(inference_files_dir)
                                    
            filename = f"{inference_files_dir}/log_trial_{self.fenv_config['trial']:02}.log"
            logging.basicConfig(filename=filename, 
                                level=logging.INFO, 
                                format=LOG_FORMAT, filemode="w")
            logging.getLogger('PIL').setLevel(logging.WARNING)
            logger = logging.getLogger()
            # logger.addHandler(logging.StreamHandler())
            logger.propagate = False
            return logger
        
        
    def design(self):
        self.inferencer = MyInferencer(
                            self.fenv_config, 
                            self.agent_config, 
                            self.learner_chkpt_config,
                            self.params,
                            )

        if self.agent_config['learner_name'] == "tunner":
            self.inferencer._evaluate()
            if self.params['n_episodes'] == 0:
                self.inferencer._check_tunner_performance()
        else:
            self.inferencer._evaluate()
            if self.params['n_episodes'] == 0:
                self.inferencer._get_model()
        
        self._close()


    def _close(self):
        print('- - - - - - __exit__')
        handlers = self.params['logger'].handlers[:]
        for handler in handlers:
            handler.close()
            self.params['logger'].removeHandler(handler)
            
        ray.shutdown()
        
        
        
        
# %%
def make_graph(self, plan_data_dict, show_graph_flag=False):
    layout_graph = LayoutGraph(plan_data_dict)
    num_nodes, edge_list = layout_graph.extract_graph_data()
    # print(edge_list)
    if show_graph_flag:
        adj_matrix = layout_graph.get_adj_matrix(num_nodes, np.array(edge_list)-1)
        layout_graph.show_graph(adj_matrix)
        # print(adj_matrix)
    return edge_list, layout_graph.moving_labels
        



# %%
if __name__ == '__main__':
    scenario_name = "Scenario__DOLW__Masked_LFC__XX_Rooms__LRres__Area__Adj__METAFC__2022_09_28_1857"
    
    agents_floorlan_dir = os.getcwd()
    scenario_dir = os.path.join(agents_floorlan_dir, f"storage/tunner/local_dir/{scenario_name}")
    
    chkpt_json_path = os.path.join(scenario_dir, 'chkpt_json.json')
    json_file = open(chkpt_json_path)
    chkpt_json = json.load(json_file)
    chkpt_path = chkpt_json['chkpt_path']
    
    
    params = {
    'scenario_name': scenario_name,
    'the_scenario_dir': scenario_dir, ## TODO! Scenario dir needs to be set into the fenv_configs
    'chkpt_path': chkpt_path,
    
    'compute_non_greedy_actions' : False,
    'load_action_from_buffer_flag' : False,
    
    'print_verbose': 1,
    'render_verbose' : 1,
    'trial': 5,
    'n_episodes': 1,
    
    'random_order_flag': False,
    
    'save_log_flag': True,
    'show_render_flag': True,
    'save_render_flag': True,
    'so_thick_flag': True,
    
    'show_graph_on_plan_flag': True,
    'show_room_dots_flag': False,
    
    'show_graph_flag': True,
    
    'plan_config_source_name': 'inference_mode',
    
    'random_agent_flag': True,
    
    'maximum_num_real_rooms': 9,
    'use_edge_info_into_observation_flag': True,
    }
    
    
    from gym_floorplan.envs.fenv_config import LaserWallConfig
    fenv_config = LaserWallConfig().get_config()
    
    
    rl_designer = RLDesigner(params, fenv_config=None)
    rl_designer.design()
    
    df = rl_designer.inferencer._load_trial_df_from_local_dir()
    rl_designer.inferencer._plot_TB(df, y_name='episode_reward_mean')#['episode_reward_mean', 'episode_len_mean'])
    rl_designer.inferencer._plot_TB(df, y_name='episode_len_mean')
    # rl_designer.inferencer._plot_TB_multiple(df)
    # rl_designer.inferencer._plot_TB_subplot(df)
    # model = rl_designer.inferencer._get_model()
    
    logging.shutdown()
    
    plan_data_dict = rl_designer.inferencer.env.obs.plan_data_dict
    # edge_list, moving_labels = make_graph(plan_data_dict, show_graph_flag=params['show_graph_flag'])