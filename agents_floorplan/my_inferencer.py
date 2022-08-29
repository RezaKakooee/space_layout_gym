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
# import sys

# print(f"{os.path.basename(__file__)} -> Appending: ........ {gym_floorplan_dir}")
# sys.path.append(f"{gym_floorplan_dir}")

# %%
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# %%
# import re
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

# from agent_config import AgentConfig
from env_to_agent import EnvToAgent
# from my_callbacks import MyCallbacks
# from gym_floorplan.envs.fenv_config import LaserWallConfig


# %%
class MyInferencer(EnvToAgent):
    def __init__(self, chkpt_path=None, 
                 fenv_config=None, agent_config=None, learner_chkpt_config=None,
                 print_verbose=0, render_verbose=0,
                 logger=None):
        
        super().__init__(fenv_config=fenv_config, agent_config=agent_config)

        self.chkpt_path = chkpt_path
        self.fenv_config = fenv_config
        self.agent_config = agent_config
        self.learner_chkpt_config = learner_chkpt_config
        
        self.print_verbose = print_verbose
        self.render_verbose = render_verbose
        self.logger = logger
    
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

        agent.restore(str(self.chkpt_path))
        
        return agent
        
    
    def _evaluate(self, n_episodes):
        keys = ['env_name', 'env_type', 'env_planning', 'env_space', 'plan_config_source_name',
                'net_arch', 'mask_flag', 'mask_numbers', 'fixed_fc_observation_space',
                'include_wall_in_area_flag', 'area_tolerance', 'reward_shaping_flag']
        
        if self.env_name == 'master_env-v0':
            reward_per_episode = []
            episode_len = []
            num_failures = 0
            areas_per_episode = defaultdict()
            delta_areas_per_episode = defaultdict()
            for i in range(n_episodes):
                if i == 0: 
                    if self.print_verbose >= 1: print("################################################## Start")
                    for key in keys:    
                        if self.print_verbose >= 1: print(f"- - - - - {key}: {self.fenv_config[key]}")
                        if self.print_verbose >= 1: self.logger.info(f"- - - - - {key}: {self.fenv_config[key]}")
                 
                if self.print_verbose >= 2: print(f"================================= Episode: {i:03}")
                observation = self.env.reset()
                time_step = self.env.time_step
                if self.render_verbose >= 3: self.env.render()
                if self.render_verbose >= 4: self.env._illustrate()
                done = False
                this_episode_reward = 0
                while not done:
                    if self.print_verbose >= 2:  print(" - - - - -  - - - - - - - - - - - - - - - - - - -- - - -")
                    action = self.agent.compute_single_action(observation)
                    observation, reward, done, info = self.env.step(action)
                    time_step = self.env.time_step
                    this_episode_reward += reward
                    areas_per_episode[i] = self.env.obs.plan_data_dict['areas']
                    delta_areas_per_episode[i] = self.env.obs.plan_data_dict['delta_areas']
                    if self.fenv_config['net_arch'] in ['fccnn', 'cnnfc']:
                        experience = (action, (observation[0].shape, observation[1].shape), reward, done, info)
                    else:
                        experience = (action, observation.shape, reward, done, info)
                    
                    if self.print_verbose >= 3: print(f"- - - - -{time_step:02} -> experience: {experience}")

                    if done:
                        if time_step == self.fenv_config['stop_time_step']:
                            num_failures += 1
                             
                        if self.render_verbose >= 1: self.env.render(i)
                        if self.render_verbose >= 2: self.env._illustrate(i)
                        
                        if self.print_verbose >= 1: print(f"- - - - - Timestep: {time_step:03}")
                        if self.print_verbose >= 1: print(f"delta_areas: { self.env.obs.plan_data_dict['delta_areas'] }")
                        if self.print_verbose >= 1: print(f"areas: { self.env.obs.plan_data_dict['areas']}")
                        if self.print_verbose >= 2: print(f"{time_step:002} -> experience: {experience}")
                        if self.print_verbose >= 2: print(f"========== done: {done}!")
                        if self.print_verbose >= 2: print(f"- - - - - Status: {done}")
                        
                        break
                    else:
                        if self.render_verbose >= 3: self.env.render()
                        if self.render_verbose >= 4: self.env._illustrate()

                episode_len.append(time_step+1)
                reward_per_episode.append(this_episode_reward)
                if self.print_verbose >= 2: print("==========================================================")
            
            if self.print_verbose >= 2: 
                print(f"- - - - - episode_len: {episode_len}")
                print(f"- - - - - Mean of episode_len: {np.mean(episode_len)}")
                print(f"- - - - - reward_per_episode: {reward_per_episode}")
                print(f"- - - - - num_failures: {num_failures}")
                self.logger.info("################################################## End")
            if self.print_verbose >= 1: 
                self.logger.info(f"- - - - - areas_per_episode: {pformat(areas_per_episode.values())}")
                self.logger.info(f"- - - - - delta_areas_per_episode: {pformat(delta_areas_per_episode.values())}")
                self.logger.info(f"- - - - - episode_len: {episode_len}")
                self.logger.info(f"- - - - - Mean of episode_len: {np.mean(episode_len)}")
                self.logger.info(f"- - - - - reward_per_episode: {reward_per_episode}")
                self.logger.info(f"- - - - - num_failures: {num_failures}")
                self.logger.info("################################################## End")
        
            
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
        fig.add_trace( go.Scatter(x=x, y=df['episode_reward_mean'], mode='lines', name='episode_reward_mean'), row=1, col=1 )
        fig.add_trace( go.Scatter(x=x, y=df['episode_len_mean'], mode='lines', name='episode_len_mean'), row=2, col=1 )
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
        fig.show() 
        fig.write_image(f"{self.fenv_config['generated_plans_dir']}/tb_plots.png", scale=3)
        
    def _plot_TB(self, df):
        # fig = px.line(df, x="timesteps_total", y=["episode_reward_mean", "episode_len_mean"],
        #               labels={"timesteps_total": "Time Step", "episode_reward_mean": "Episode Mean Reward"})
        
        fig = px.line(df, x="timesteps_total", y="episode_reward_mean",
                      labels={"timesteps_total": "Time Step", "episode_reward_mean": "Episode Mean Reward"})
        
        
        fig.update_traces(line=dict(width=3))
        fig.update_layout(height=600, width=800, hovermode='x')
        fig.update_layout({
                        # 'plot_bgcolor': 'rgba(0, 255, 255, 0.2)', 
                        'paper_bgcolor': 'rgba(0, 0, 0, 0)'})
        fsize = 35
        fig.update_xaxes(title_font=dict(size=fsize, family='Courier', color='black'))
        fig.update_yaxes(title_font=dict(size=fsize, family='Courier', color='black'))
        fig.update_xaxes(tickfont=dict(family='Rockwell', color='black', size=fsize))
        fig.update_yaxes(tickfont=dict(family='Rockwell', color='black', size=fsize))
        # fig.update_xaxes(range=[200_000, 400_000])
        # fig.update_yaxes(range=[ymax, ymin])
        fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99))
        fig.update_layout(legend={'title_text':''})
        fig.update_layout(legend_font_size=30)

        fig.show() 
        fig.write_image(f"{self.fenv_config['generated_plans_dir']}/tb_plots.png", scale=3)

        
    def _check_trainer_performance(self, trial_df=None):
        if trial_df is None:
            trial_df = self._load_trial_df_from_local_dir()
        df = pd.DataFrame(data=trial_df)
        df.plot(x="n", y=['episode_len_mean', 'episode_reward_mean'], secondary_y=True)

        
    def _load_trial_df_from_local_dir(self):
        checkpoint_dir = Path(self.chkpt_path).parents[2]
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
    def __init__(self, scenario_name, chkpt_txt_fixed_path,
                 print_verbose, render_verbose, trial):
        self.scenario_name = scenario_name
        self.chkpt_txt_fixed_path = chkpt_txt_fixed_path
        self.print_verbose = print_verbose
        self.render_verbose = render_verbose
        self.trial = trial
    
        self._restore_trained_config()
        self._adjust_configs()
        self.logger = self._get_logger()
        
    def _restore_trained_config(self):
        with open(self.chkpt_txt_fixed_path) as f:
            self.chkpt_path = Path(f.readline())
        
        chkpt_path_str = str(self.chkpt_path)
        if "Scenario" not in chkpt_path_str: 
            chkpt_path_ = chkpt_path_str.split('local_dir')
            chkpt_path_ = chkpt_path_[0] + f"local_dir/{self.scenario_name}" + chkpt_path_[1]
            self.chkpt_path = Path(chkpt_path_)
    
        if "trainer" in str(self.chkpt_path):
            configs_dir = os.path.join(self.chkpt_path.parents[1], 'configs')
        else:
            configs_dir = os.path.join(self.chkpt_path.parents[2], 'configs')
        
        #configs_dir = configs_dir.replace("/app/main/", "/home/iakakooe/dbt/housing_design/")
        fenv_config_npy_path = os.path.join(configs_dir, 'fenv_config.npy')
        agent_config_npy_path = os.path.join(configs_dir, 'agent_config.npy')
        learner_config_npy_path = os.path.join(configs_dir, 'learner_config.npy')
    
        self.fenv_config = np.load(fenv_config_npy_path, allow_pickle=True).tolist()
        self.agent_config = np.load(agent_config_npy_path, allow_pickle=True).tolist()
        self.learner_chkpt_config = np.load(learner_config_npy_path, allow_pickle=True).tolist()
        
        ## update and rewrite
        # self.fenv_config.update({'mask_numbers': 0})
        # with open(fenv_config_npy_path, "wb") as f:
        #     np.save(f, self.fenv_config)
        
        
        if self.fenv_config['scenario_name'] != self.scenario_name:
            raise ValueError("Scenario names do not match!")
    
    
    def _adjust_configs(self):
        self.agent_config.update({'learner_name': "tunner"})
        self.agent_config.update({'RLLIB_NUM_GPUS': 0})
        
        self.fenv_config.update({'trial': self.trial})
        self.fenv_config.update({'show_render_flag': True})
        self.fenv_config.update({'save_render_flag': True})
        
        # self.fenv_config.update({'is_proportion_considered': True})
        # self.fenv_config.update({'min_desired_proportion': 0.5})
        # self.fenv_config.update({'max_desired_proportion': 2})
        
        self.generated_plans_dir = f"{self.fenv_config['generated_plans_dir']}/{self.scenario_name}"
        #self.generated_plans_dir = self.generated_plans_dir.replace('/app/main/', '/home/iakakooe/dbt/housing_design/')
        if not os.path.exists(self.generated_plans_dir):
            os.mkdir(self.generated_plans_dir)
        self.fenv_config.update({'generated_plans_dir': self.generated_plans_dir})
        
    
    def _get_logger(self):
            LOG_FORMAT = "%(message)s"
            filename = f"{self.fenv_config['generated_plans_dir']}/log_trial_{self.fenv_config['trial']:02}.log"
            logging.basicConfig(filename=filename, 
                                level=logging.INFO, 
                                format=LOG_FORMAT, filemode="w")
            logging.getLogger('PIL').setLevel(logging.WARNING)
            logger = logging.getLogger()
            # logger.addHandler(logging.StreamHandler())
            logger.propagate = False
            return logger
        
        
    def design(self, n_episodes):
        self.inferencer = MyInferencer(self.chkpt_path, 
                            self.fenv_config, 
                            self.agent_config, 
                            self.learner_chkpt_config,
                            self.print_verbose, 
                            self.render_verbose, 
                            self.logger)
    
        if self.agent_config['learner_name'] == "tunner":
            self.inferencer._evaluate(n_episodes)
            if n_episodes == 0:
                self.inferencer._check_tunner_performance()
        else:
            self.inferencer._evaluate(n_episodes)
            if n_episodes == 0:
                self.inferencer._get_model()
        
        self._close()


    def _close(self):
        print('- - - - - - __exit__')
        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)
            
        ray.shutdown()
        
        
# %%
if __name__ == '__main__':
    scenario_name = "Scenario__DOLW__Masked_TC__07_Walls__Area__FC__2022_06_05_2018"
    chkpt_txt_fixed_path = os.path.join(os.getcwd(), 'storage/chkpt_dir/chkpt_path.txt')
    
    print_verbose, render_verbose, trial = 1, 1, 4
    rl_designer = RLDesigner(scenario_name, chkpt_txt_fixed_path,
                             print_verbose, render_verbose, trial)
    
    rl_designer.design(n_episodes=1)

    df = rl_designer.inferencer._load_trial_df_from_local_dir()
    rl_designer.inferencer._plot_TB(df)
    
    # model = rl_designer.inferencer._get_model()
    
    logging.shutdown()
