# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 13:40:40 2021

@author: Reza Kakooee
"""

from dotenv import load_dotenv
load_dotenv('../.env')

# %%
import os

def get_housing_design_root_dir():
    housing_design_root_dir = os.getenv('HOUSING_DESIGN_ROOT_DIR')
    housing_design_root_dir = os.path.expandvars(housing_design_root_dir)
    if housing_design_root_dir is None:
        raise EnvironmentError("The 'HOUSING_DESIGN_ROOT_DIR' environment variable is not set.")
    return housing_design_root_dir
root_dir = get_housing_design_root_dir()
print(f"Root directory: {root_dir}")


import time
import ast
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
from pprint import pprint
import gymnasium as gym
from collections import defaultdict, OrderedDict

from rnd_logger import RandomAgentLogger

import gym_floorplan
from gym_floorplan.envs.fenv_config import LaserWallConfig
from gym_floorplan.envs.reward.metrics import Metrics



# %% 
class EnvMaker():
    def __init__(self, fenv_config):
        self.fenv_config = fenv_config
        self.env_name = fenv_config['env_name']
        
        
        
    def make(self):
        if self.env_name == 'gym':
            env = gym.make(self.env_name)
            env.env_name = self.env_name
        else:
            from gym_floorplan.envs.master_env import SpaceLayoutGym
            env = SpaceLayoutGym(self.fenv_config)
            # from ray.rllib.utils import check_env
            # check_env(env)
        return env
    
    
    
    
# %%
class RandomAgent:
    def __init__(self, fenv_config, agent_config):
        self.fenv_config = fenv_config
        self.agent_config = agent_config
                
        self.env = EnvMaker(self.fenv_config).make()
        
        self.logger = RandomAgentLogger(fenv_config, agent_config)

        self.time_dict_random_agent = defaultdict(dict)
        
        

    def _get_random_action(self, obs):
        if self.fenv_config['action_masking_flag']:
            action_mask = obs['action_mask']
            # print(f"Num of left actions: {np.sum(action_mask)}")
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



    def _reset_env_data_dict(self):
        self.env_data_dict = defaultdict(list)
        
        

    def run_random_agent(self):
        self.logger.start_interaction_logger()
        self._reset_env_data_dict()
        num_failures = 0
        info_dict = defaultdict(list)
        self.end_episode_reward = defaultdict(int)
        self.episode_len = defaultdict(int)
        
        if self.fenv_config['plan_config_source_name'] not in ['fixed_test_config'] and self.agent_config['n_episodes'] is None:
            self.plans_df = self.env.obs.plan_constructor.adjustable_configs_handeler.plans_df
            self.episode_finishing_status = [0] * len(self.plans_df)
            self.agent_config['n_episodes'] = len(self.plans_df)
        else:    
            self.episode_finishing_status = [0] * self.agent_config['n_episodes']
            
        
        self.low_high_quality_data = defaultdict(dict)
        
        metrics = Metrics(self.fenv_config)
        
        TQDM = 0 if self.agent_config['investigation_mode'] else 1 
        iterator = tqdm(range(self.agent_config['n_episodes'])) if TQDM else range(self.agent_config['n_episodes'])
        try:
            for i in iterator:
                observation, _ = self.env.reset()

                self.logger.start_episode_logger(i, observation)
                
                if self.fenv_config['plan_config_source_name'] == 'imitation_mode':
                    adjustable_configs = self.env.obs.plan_constructor.adjustable_configs
                    accepted_action_sequence = adjustable_configs['potential_good_action_sequence']
                elif self.fenv_config['plan_config_source_name'] == 'fixed_test_config' and self.agent_config['use_action_sequence']:
                    adjustable_configs = self.env.obs.plan_constructor.adjustable_configs
                    accepted_action_sequence = adjustable_configs['sample_action_sequence']
                else:
                    accepted_action_sequence = self.agent_config['accepted_action_sequence']
                
                self.sum_episode_rewards = 0
                episode_rewards = []
                self.episode_good_action_sequence = []
                done = False
                ep_info_dict = defaultdict(list)
                ep_steper = 0
                
                ep_obs_list = []
                ep_action_list = []
                ep_reward_list = []
                ep_done_list = []
                ep_next_obs_list = []

                while not done:# and ep_steper<=7:
                    if ( (self.agent_config['fixed_action_seq_flag'] or self.fenv_config['plan_config_source_name'] == 'imitation_mode') and 
                        self.env.ep_time_step < len(accepted_action_sequence)):
                        action = accepted_action_sequence[self.env.ep_time_step]
                    elif (self.fenv_config['plan_config_source_name'] == 'fixed_test_config' and self.agent_config['use_action_sequence'] and 
                          self.env.ep_time_step < len(accepted_action_sequence) ):
                        action = accepted_action_sequence[self.env.ep_time_step]
                    else:
                        action = self._get_random_action(observation)
                        
                    # action = action if ep_steper <= 5 else 0
                    # ep_obs_list.append(observation)
                    observation, reward, done, truncated, info = self.env.step(action=action)

                    # ep_next_obs_list.append(observation)
                    ep_action_list.append(action)
                    # ep_reward_list.append(reward)
                    # ep_done_list.append(done)
                    
                    # print(np.max(observation['observation_cnn']))
                    self.sum_episode_rewards += reward
                    episode_rewards.append(reward)
                    
                    if self.env.obs.active_wall_status in ['accepted', 'well_finished']:
                        self.episode_good_action_sequence.append(action)
                        
                    self.logger.in_episode_logger(action, observation, reward, done, info)
                    
                    if self.fenv_config['only_save_high_quality_env_data']:
                        if done:
                            # self.logger.end_episode(self.env.obs.plan_data_dict, self.episode_good_action_sequence, reward, self.env.ep_time_step+1, self.env.obs.active_wall_status)
                            self.end_episode_reward[i] = reward
                            self.episode_len[i] = self.env.ep_time_step+1
                            
                            if self.env.obs.active_wall_status == 'well_finished':
                                if self.agent_config['save_env_data_dict']:
                                    info_dict[i] = info
                                    self.env_data_dict[i] = info['env_data']
                                    self.episode_finishing_status[i] = 1
                                    final_info_dict = metrics.get_end_episode_metrics(self.env.obs.plan_data_dict)
                                    pprint(final_info_dict)
                                    
                            if self.env.ep_time_step >= self.fenv_config['stop_ep_time_step']-1:
                                num_failures += 1
                                if self.agent_config['print_verbose'] >= 1: print('Failed')
                            if self.agent_config['render_verbose'] >= 1: self.env.render()
                            if self.agent_config['render_verbose'] >= 3: self.env.display()
                            if self.agent_config['render_verbose'] >= 3: self.env.illustrate()
                            if self.agent_config['render_verbose'] >= 3: self.env.demonestrate()
                            if self.agent_config['render_verbose'] >= 3: self.env.exibit()
                            if self.agent_config['render_verbose'] >= 3: self.env.view()
                            if self.agent_config['render_verbose'] >= 3: self.env.portray()
                            
                            break
                            
                        else:
                            if self.agent_config['render_verbose'] >= 2 and self.env.obs.active_wall_status == 'accepted': self.env.render()
                    
                    else:
                        ep_info_dict[self.env.ep_time_step-1] = info
                        if done:
                            if self.env.obs.active_wall_status == 'well_finished':
                                lvroom_id = info['lvroom_id']
                            else:
                                if info['env_data']['lvroom_id'] == -1:
                                    lvroom_id = int(list(ast.literal_eval(info['env_data']['areas_achieved']).keys())[-1].split('_')[1])
                                else:
                                    lvroom_id = info['env_data']['lvroom_id']
                                    
                                start_i = len(self.env_data_dict)
                                for j in range(len(ep_info_dict)):
                                    info_ = ep_info_dict[j]
                                    info_['env_data']['lvroom_id'] = lvroom_id
                                    self.env_data_dict[start_i+j] = info_
                                    self.logger._save_plan_values(self.env_data_dict)
                                    
                                
                            if self.agent_config['render_verbose'] >= 1: self.env.render()
                            
                            break

                    ep_steper +=1 
                    if ep_steper >= self.fenv_config['stop_ep_time_step'] + 10:
                        break
        
                if self.fenv_config['plan_config_source_name'] == 'load_random_config' and self.agent_config['save_low_high_quality_data_flag']:
                    self.low_high_quality_data[self.env.obs.plan_data_dict['plan_id']] = {'action': ep_action_list, 'status': self.env.obs.active_wall_status, 'last_reward': reward}
                    if (i+1) == self.agent_config['lh_save_freq']:
                        self.save_lh_df()
                    
            self.save_lh_df()
            
        except KeyboardInterrupt:
            if (self.agent_config['save_env_data_dict'] and 
                i % self.agent_config['save_env_data_dict_freq'] == 0 and
                len(self.env_data_dict)):
                    self.logger._save_plan_values(self.env_data_dict)
                    self._reset_env_data_dict()
            if self.agent_config['save_low_high_quality_data_flag']:
                self.save_lh_df()
                
        self.logger.end_of_interaction(list(self.end_episode_reward.values()), list(self.episode_len.values()), num_failures)
        
        if self.agent_config['save_env_data_dict'] and len(self.env_data_dict):
            self.logger._save_plan_values(self.env_data_dict)
            self._reset_env_data_dict()
        
        if self.agent_config['save_log_flag']:
            self.logger.save_logs()
            
            
    def save_lh_df(self):
        if self.agent_config['save_low_high_quality_data_flag']:
            ## convert to dataframes and save
            self.df_lh = pd.DataFrame.from_dict(self.low_high_quality_data).T
            if not os.path.exists(self.agent_config['scenario_dir']):
                os.makedirs(self.agent_config['scenario_dir'])
            self.df_lh.to_csv(os.path.join(self.agent_config['scenario_dir'], 'plan_lh.csv'))

    
    
# %%
if __name__ == '__main__':
    start_time = time.time()
    default_hyper_params_yaml_path = os.path.join(root_dir, 'rnd_agents/default_hps_env.yaml')
    with open(default_hyper_params_yaml_path, 'r') as f:
        hyper_params = yaml.load(f, Loader=yaml.FullLoader)
    hyper_params.update({
        'agent_name': 'RND',
        'phase': 'train',
        'n_rooms': 7,
        'plan_config_source_name': 'imitation_mode',
        'model_last_name': 'MetaCnnEncoder',
        'scenario_name': 'Scn__2024_04_14_1037__FTC__7Rr__ZSLR__RND',
        })
    
    fenv_config = LaserWallConfig(phase=hyper_params['phase'], 
                                  hyper_params=hyper_params).get_config()
    
    investigation_mode = True
    
    fenv_config.update({'random_agent_flag': True,
                        'show_graph_on_plan_flag': True if fenv_config['plan_config_source_name'] in ['create_random_config', 'load_fixed_config'] else True,
                        'graph_line_style': 'bezier', # stright
                        'save_env_info_on_callback': False if investigation_mode else False,
                        'only_save_high_quality_env_data': True,
                        'only_draw_room_gravity_points_flag': False,
                        'save_render_flag': True,
    })
        
    if fenv_config['plan_config_source_name'] == 'offline_mode':
        raise ValueError("In random agent, plan_config_source_name cannot be offline_mode.")
    elif fenv_config['plan_config_source_name'] in ['create_random_config', 'create_fixed_config']:
        new_plan_name = 'plans_cc.csv' 
        plan_path_cc = os.path.join(fenv_config['rnd_agents_storage_dir'], fenv_config['scenario_name'], new_plan_name)
        fenv_config['plan_path_cc'] = plan_path_cc
        print(f"plan_path_cc: {plan_path_cc}")
        
    random_agent_config = {
        'agent_name': hyper_params['agent_name'],
        'investigation_mode': investigation_mode,
        
        'trial': 1,
        'n_episodes': 1,
        
        'fixed_action_seq_flag': True,
        'accepted_action_sequence': [1314, 145, 1965, 1085, 2491, 3503],
        
        'print_verbose': 1 if investigation_mode else 0,
        'render_verbose' : 1 if investigation_mode else 0,
        'show_render_flag': True if investigation_mode else True,
        
        'save_env_data_dict': False if investigation_mode else False,
        'save_env_data_dict_freq': 1,
        
        'save_log_flag': False,
        
        'use_action_sequence': False,
        
        'save_low_high_quality_data_flag': True,
        'lh_save_freq': 1000,
        
        'scenario_dir': os.path.join(fenv_config['rnd_agents_storage_dir'], fenv_config['scenario_name']),
    }
    if random_agent_config['save_log_flag'] or fenv_config['save_render_flag']:
        if not os.path.exists(random_agent_config['scenario_dir']):
            os.mkdir(random_agent_config['scenario_dir'])
    fenv_config['results_dir'] = random_agent_config['scenario_dir']
    
    if random_agent_config['print_verbose'] >= 1: print("\n################################################## Start")
    print(f"- - - - - - - - scenario_name: {fenv_config['scenario_name']}")


    self = RandomAgent(fenv_config, random_agent_config)
    self.run_random_agent()
    
        
    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time}")
    

