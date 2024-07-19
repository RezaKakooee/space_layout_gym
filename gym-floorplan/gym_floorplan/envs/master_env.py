# -*- coding: utf-8 -*-
"""
Created on Wed May 11 18:01:36 2022

@author: Reza Kakooee
"""


# %%
import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# %%
import gymnasium as gym
import numpy as np
from datetime import datetime
from collections import defaultdict

import gym_floorplan
from gym_floorplan.envs.action.action import Action
from gym_floorplan.envs.observation.observation import Observation
from gym_floorplan.envs.reward.reward import Reward
from gym_floorplan.envs.render.render import Render



# %%
class SpaceLayoutGym(gym.Env):#(MultiAgentEnv):
    def __init__(self, env_config={'fenv_name': 'SpaceLayoutGym-v0'}):
        self.fenv_config = self.env_config = env_config
        self.env_name = 'SpaceLayoutGym-v0'
        
        self.obs = Observation(fenv_config=self.fenv_config)
        self.act = Action(fenv_config=self.fenv_config)
        self.rew = Reward(fenv_config=self.fenv_config)
        self.vis = Render(fenv_config=self.fenv_config)

        self.action_space = self.act.action_space
        self.observation_space = self.obs.observation_space

        self.episode = 0
        # observation = self.reset()
        
        # self.seed(42)
        
        self.max_episode_steps = self.fenv_config['stop_ep_time_step']
        
        self.load_plan_randomly = False



    def reset(self, *, seed=None, options=None):
        self.episode += 1
        episode_counter = None if self.load_plan_randomly else self.episode
        observation = self.obs.obs_reset(episode_counter)
        self.observation_space = self.obs.observation_space
        self.accepted_action_sequence = []
        self.ep_time_step = 0
        self.ep_sum_reward = 0
        self.ep_max_reward = float('-inf')
        self.ep_mean_reward = 0
        if self.fenv_config['load_good_action_sequence_for_on_policy_pre_training'] and np.random.rand() <= self.fenv_config['load_good_action_sequence_prob']:
            self.load_good_action_sequence_for_on_policy_pre_training = True
        else:
            self.load_good_action_sequence_for_on_policy_pre_training = False
        self.action_counter = 0
        self.prev_action = 0
        self.prev_reward = 0
        return observation, {}
        
    
    
    def seed(self, seed: int = None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
    
    
    
    def set_seed(self, seed: int = None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]



    def step(self, action):
        if self.load_good_action_sequence_for_on_policy_pre_training:
            action = self.obs.plan_data_dict['potential_good_action_sequence'][self.action_counter] # [321, 1338, 976, 933, 915, 848][self.action_counter]#
        
        observation = self.obs.update(self.episode, action, self.ep_time_step)
        
        if isinstance(observation, dict) and 'cnn' in self.fenv_config['model_last_name']:
            assert observation['observation_cnn'].shape == (23, 23, 1) if self.fenv_config['cnn_scaling_factor'] == 1 else (46, 46, 1)
            assert observation['observation_meta'].shape == (172,)
        
        done = truncated = self.obs.done
        
        if self.obs.active_wall_status == 'badly_stopped_':
            reward = -2000
        else:
            reward = self.rew.reward(self.obs.plan_data_dict,
                                     self.obs.decoded_action_dict['active_wall_name'], 
                                     self.obs.active_wall_status,
                                     self.ep_time_step, 
                                     done)

        self.ep_sum_reward += reward
        self.ep_max_reward = max(self.ep_max_reward, reward)
        self.ep_mean_reward = 1/(self.ep_time_step+1) * (self.ep_time_step*self.ep_mean_reward + reward)
        
        
        if self.fenv_config['only_save_high_quality_env_data']:
            if self.obs.active_wall_status in ['accepted', 'well_finished']:
                self.accepted_action_sequence.append(action)
            info = self._get_info(done, reward) if (done and self.fenv_config['save_env_info_on_callback']) else {}
        
        else:
            self.accepted_action_sequence.append(action)
            info = self._get_info(done, reward) if self.fenv_config['save_env_info_on_callback'] else {}
        
        
        if not done:
            self.ep_time_step += 1
        
        self.done = done
        self.info = info
        
        self.action_counter += 1

        return observation, reward, done, truncated, info
    
    
    
    def _get_info(self, done, reward):
        if self.fenv_config['only_save_high_quality_env_data']:
            if self.obs.active_wall_status == 'well_finished':
                info = {'env_data': self._update_env_stats(reward)}
            else:
                info = {} if self.fenv_config['library'] == 'RLlib' else {'episode_end_data': self._get_episode_end_data(reward)}
        else:
            info = {'env_data': self._update_env_stats(reward)}
        return info
    
    

    def _get_episode_end_data(self, reward):
        self.episode_end_data = {
            'episode': self.episode,
            'ep_len': self.ep_time_step,
            'ep_mean_reward': self.ep_mean_reward,
            'ep_sum_reward': self.ep_sum_reward,
            'ep_max_reward': self.ep_max_reward,
            'ep_last_reward': reward/1.0,
            }
        return self.episode_end_data
        
        
    
    def _update_env_stats(self, reward):
        self.env_stats = {
            'plan_id': datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f'),
            'episode': self.episode,
            'n_walls': self.obs.plan_data_dict['n_walls'],
            'n_rooms': self.obs.plan_data_dict['n_rooms'],
            'mask_numbers': self.obs.plan_data_dict['mask_numbers'],
            'masked_corners': str(self.obs.plan_data_dict['masked_corners']),
            'mask_lengths': str(self.obs.plan_data_dict['mask_lengths']),
            'mask_widths': str(self.obs.plan_data_dict['mask_widths']), 
            'area_masked': self.obs.plan_data_dict['area_masked'],
            'areas_masked': str(self.obs.plan_data_dict['areas_masked']),
            'entrance_positions': str(self.obs.plan_data_dict['entrance_positions']),
            'entrance_coords': str(self.obs.plan_data_dict['entrance_coords']),
            'extended_entrance_positions': str(self.obs.plan_data_dict['extended_entrance_positions']),
            'extended_entrance_coords': str(self.obs.plan_data_dict['extended_entrance_coords']),
            'entrance_is_on_facade': self.obs.plan_data_dict['entrance_is_on_facade'],
            'lvroom_id': self.obs.plan_data_dict['lvroom_id'],
            'n_facades_blocked': self.obs.plan_data_dict['n_facades_blocked'],
            'facades_blocked': str(self.obs.plan_data_dict['facades_blocked']),
            'areas_desired': str(self.obs.plan_data_dict['areas_desired']),
            'areas_achieved': str(self.obs.plan_data_dict['areas_achieved']),
            "accepted_action_sequence": str(self.accepted_action_sequence), 
            'ep_len': self.ep_time_step,
            'ep_mean_reward': self.ep_mean_reward,
            'ep_sum_reward': self.ep_sum_reward,
            'ep_max_reward': self.ep_max_reward,
            'ep_last_reward': reward/1.0,
            }
        
        
        
        def _get_obs_stats(obs):
            if isinstance(obs, dict):
                pass
            
            
            
        edge_list_facade_achieved_str = self.obs.plan_data_dict['edge_list_facade_achieved_str'].copy()
        edge_list_entrance_achieved_str = self.obs.plan_data_dict['edge_list_entrance_achieved_str'].copy()
        if self.fenv_config['plan_config_source_name'] in ['create_fixed_config', 'create_random_config']:
            lvroom_id = [room for edge in edge_list_entrance_achieved_str for room in edge if isinstance(room, int)]
            if len(lvroom_id):
                lvroom_id = lvroom_id[0]
                
                rooms_connected_to_lvroom = []
            else:
                lvroom_id = -1
                
            self.env_stats.update({'edge_list_room_desired': str(self.obs.plan_data_dict['edge_list_room_achieved']),
                                   'edge_list_room_achieved': str(self.obs.plan_data_dict['edge_list_room_achieved']),
                                   'edge_list_facade_desired_str': str(edge_list_facade_achieved_str),
                                   'edge_list_facade_achieved_str': str(edge_list_facade_achieved_str),
                                   'edge_list_entrance_desired_str': str(edge_list_entrance_achieved_str),
                                   'edge_list_entrance_achieved_str': str(edge_list_entrance_achieved_str),
                                   'lvroom_id': lvroom_id,
                                   })
        else:
            if self.fenv_config['only_save_high_quality_env_data']:
                self.env_stats.update({'edge_list_room_desired': str(self.obs.plan_data_dict['edge_list_room_desired']),
                                       'edge_list_room_achieved': str(self.obs.plan_data_dict['edge_list_room_achieved']),
                                       'edge_list_facade_desired_str': str(self.obs.plan_data_dict['edge_list_facade_desired_str']),
                                       'edge_list_facade_achieved_str': str(edge_list_facade_achieved_str),
                                       'edge_list_entrance_desired_str': str(self.obs.plan_data_dict['edge_list_entrance_desired_str']),
                                       'edge_list_entrance_achieved_str': str(edge_list_entrance_achieved_str)
                                       })
            else:
                self.env_stats.update({'edge_list_room_desired': str(self.obs.plan_data_dict['edge_list_room_desired']),
                                       'edge_list_room_achieved': str(self.obs.plan_data_dict['edge_list_room_achieved']),
                                       'edge_list_facade_desired_str': str(self.obs.plan_data_dict['edge_list_facade_desired_str']),
                                       'edge_list_facade_achieved_str': str(edge_list_facade_achieved_str),
                                       'edge_list_entrance_desired_str': str(self.obs.plan_data_dict['edge_list_entrance_desired_str']),
                                       'edge_list_entrance_achieved_str': str(edge_list_entrance_achieved_str),
                                       })
        
        return self.env_stats



    def render(self, episode=0, mode='human'):
        image_path = self.vis.render(self.obs.plan_data_dict, self.episode, self.ep_time_step)
        return image_path


    
    def portray(self, episode=0):
        self.vis.portray(self.obs.plan_data_dict, self.episode, self.ep_time_step)
        
        

    def display(self, episode=0):
        self.vis.display(self.obs.plan_data_dict, self.episode, self.ep_time_step)



    def illustrate(self, episode=0):
        self.vis.illustrate(self.obs.plan_data_dict, self.episode, self.ep_time_step)
        
    
    
    def demonestrate(self, episode=0):
        self.vis.demonestrate(self.obs.plan_data_dict, self.episode, self.ep_time_step)
        
        
        
    def exibit(self, episode=0):
        self.vis.exibit(self.obs.plan_data_dict, self.episode, self.ep_time_step)
         
         

    def view(self, episode=0):
        self.vis.view(self.obs.plan_data_dict, self.episode, self.ep_time_step)
            
    



#%% This is only for testing and debugging
if __name__ == "__main__":
    from gym_floorplan.envs.fenv_scenarios import FEnvScenarios
    from gym_floorplan.envs.fenv_config import LaserWallConfig
    
    default_config = {
        'agent_name': 'RND',
        'phase': 'debug',
        
        'action_masking_flag': False,
        'rewarding_method_name': 'Smooth_Quad_Reward', # Smooth_Linear_Reward, Smooth_Exp_Reward, Smooth_Quad_Reward
        'cnn_observation_name': 'rooms_cmap',
        'model_last_name': 'MetaFcNet', # TinyFcNet, TinyCnnNet, MetaCnnNet, MetaFcNet
        }
    
    fenv_config = LaserWallConfig(agent_name=default_config['agent_name'], 
                                  phase=default_config['phase'], 
                                  scenarios_dict=scenarios_dict).get_config()
    
    
    self = SpaceLayoutGym(fenv_config)
    s0, _ = self.reset()
    a = self.action_space.sample()
    s, _, _, _, _= self.step(a)
    
    from ray.rllib.utils import check_env
    check_env(self)
