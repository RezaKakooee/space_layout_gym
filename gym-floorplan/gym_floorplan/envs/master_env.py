# -*- coding: utf-8 -*-
"""
Created on Wed May 11 18:01:36 2022

@author: RK
"""

# %%
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# %%
import gym
import numpy as np

# from ray.rllib.env.multi_agent_env import MultiAgentEnv

import gym_floorplan
from gym_floorplan.envs.action.action import Action
from gym_floorplan.envs.observation.observation import Observation
from gym_floorplan.envs.reward.reward import Reward
from gym_floorplan.envs.render.render import Render


# %%
class MasterEnv(gym.core.Env):#(MultiAgentEnv):
    def __init__(self, env_config={'fenv_name': 'master_env'}):
        self.fenv_config = env_config
        self.env_name = 'master_env-v0'
        
        self.act = Action(fenv_config=self.fenv_config)
        self.obs = Observation(fenv_config=self.fenv_config)
        self.rew = Reward(fenv_config=self.fenv_config)
        self.vis = Render(fenv_config=self.fenv_config)

        self.action_space = self.act.action_space
        self.observation_space = self.obs.observation_space

        self.episode = -1
        observation = self.reset()
        
        self.seed(42)
        

    def reset(self):
        self.episode += 1
        observation = self.obs.obs_reset(self.episode)
        self.observation_space = self.obs.observation_space
        self.time_step = 0
        self.accepted_action_sequence = []
        return observation
        
    
    def seed(self, seed: int = None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]


    def step(self, action):
        observation = self.obs.update(self.episode, action, self.time_step)
        
        done = self.obs.done

        reward = self.rew.reward(self.obs.plan_data_dict,
                                 self.obs.decoded_action_dict['active_wall_name'], 
                                 self.obs.active_wall_status,
                                 self.time_step, 
                                 done)
        
        if self.obs.active_wall_status in ['accepted', 'well_finished']:
            self.accepted_action_sequence.append(action)
            
        info = self._get_info(done, reward)
        
        if not done:
            self.time_step += 1
        
        return observation, reward, done, info
    
    
    def _get_info(self, done, reward):
        if self.obs.active_wall_status == 'well_finished':
            info = {'time_step': self.time_step, 'env_data': self._update_env_stats(reward)}
        else:
            info = {}
        return info
        
        
    def render(self, episode=0, mode='human'):
        image_arr = self.vis.render(self.obs.plan_data_dict, self.episode, self.time_step)
        return np.array(image_arr)


    def display(self, episode=0):
        self.vis.display(self.obs.plan_data_dict, episode, self.time_step)


    def illustrate(self, episode=0):
        self.vis.illustrate(self.obs.plan_data_dict['obs_arr_conv'], episode, self.time_step)
        
        
    def demonestrate(self, episode=0):
        self.vis.demonestrate(self.obs.plan_data_dict, episode, self.time_step)

    
    def _update_env_stats(self, reward):
        masked_corners = [int(corner.split('_')[1]) for corner in self.obs.plan_data_dict['masked_corners']]
        only_real_areas = [v for k, v in self.obs.plan_data_dict['areas'].items() if k not in ['room_1', 'room_2', 'room_3', 'room_4'] ]
        self.env_stats = {
            'episode': self.episode,
            'n_walls': self.obs.plan_data_dict['n_walls'],
            'n_rooms': self.obs.plan_data_dict['n_rooms'],
            'mask_numbers': self.obs.plan_data_dict['mask_numbers']/1.0,
            'masked_corners': list(np.array(masked_corners).astype(float)),
            'mask_lengths': list(np.array(self.obs.plan_data_dict['mask_lengths']).astype(float)),
            'mask_widths': list(np.array(self.obs.plan_data_dict['mask_widths']).astype(float)),
            'desired_areas': list(np.array([*self.obs.plan_data_dict['desired_areas'].values()])/1.0),
            'areas': list(np.array(only_real_areas)/1.0),
            "accepted_action_sequence": list(np.array(self.accepted_action_sequence).astype(float)),
            'reward': reward/1.0,
            'time_step': self.time_step}
        
        if self.fenv_config['plan_config_source_name'] in ['create_fixed_config', 'create_random_config']:
            self.env_stats.update({
                                    'edge_list': self.obs.plan_data_dict['edge_list'],
                                    'desired_edge_list': self.obs.plan_data_dict['edge_list'],
                                    })
        else:
            self.env_stats.update({
                                    'edge_list': self.obs.plan_data_dict['edge_list'],
                                    'desired_edge_list': (np.array(self.obs.plan_data_dict['desired_edge_list'])/1.0).tolist(),
                                  })
        
        return self.env_stats


# %%
if __name__ == "__main__":
    from gym_floorplan.envs.fenv_config import LaserWallConfig
    fenv_config = LaserWallConfig().get_config()
    self = MasterEnv(fenv_config)
    
    keys = ['env_name', 'net_arch', 'mask_flag', 
                'fixed_fc_observation_space',
                'include_wall_in_area_flag', 'area_tolerance', 'reward_shaping_flag']                
    for key in keys:    
        print(f"- - - - - {key}: {self.fenv_config[key]}")   
        
    # from stable_baselines3.common.env_checker import check_env
    # check_env(self)
        
    self.reset()
    done = False
    c = 0
    # good_action_sequence = [1878, 1427, 826]
    good_action_sequence = []
    while not done:
        # for a in good_action_sequence:
            a = self.action_space.sample()
            if fenv_config['env_type'] == 'multi':
                a = {"wall_1": a}
            # print(f"action: {a}")
            
            observation, reward, done, info = self.step(a)
            
            # print(f"active_wall_status: {info['active_wall_status']}")
            # if info['active_wall_status'] in ["accepted", "well_finished"]:
            if not done:
                if self.obs.active_wall_status in ['accepted', 'well_finished']:
                    good_action_sequence.append(a)  
            else:
                self.render()
            
            c += 1
            # if c > 20:
            #     break
    
    print(f"good_action_sequence: {good_action_sequence}")