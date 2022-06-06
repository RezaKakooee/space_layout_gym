# -*- coding: utf-8 -*-
"""
Created on Wed May 11 18:01:36 2022

@author: Reza Kakooee
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

        observation = self.reset()
        

    def reset(self):
        observation = self.obs.obs_reset()
        self.observation_space = self.obs.observation_space
        self.time_step = 0
        self.accepted_action_sequence = []
        return observation
        
    
    def seed(self, seed: int = None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]


    def step(self, action):
        active_wall_name = self.obs.wall_names_deque[0] # list(action.keys())[0] for multi 
        
        observation = self.obs.update(active_wall_name=active_wall_name, action=action)
        reward = self.rew.reward(self.obs.active_wall_name, self.obs.active_wall_status,
                                 self.obs.plan_data_dict, self.obs.done)
        done = self.obs.done

        if self.time_step >= self.fenv_config['stop_time_step']:
            if self.fenv_config['env_type'] == 'multi':
                self.wall_names_deque.append(active_wall_name)
                done['__all__'] = True
            else:
                done = True
         
        if self.obs.active_wall_status:
            self.accepted_action_sequence.append(action)
            
        info = self._get_info(done)
        
        if not done:
            self.time_step += 1
            
        return observation, reward, done, info
    
    
    def _get_info(self, done):
        if done and self.obs.active_wall_status == 'finished':
            info = {'time_step': self.time_step, 'env_data': self._update_env_stats()}
        else:
            info = {}
        return info
        
        
    def render(self, episode=0, mode='human'):
        image_arr = self.vis.render(self.obs.plan_data_dict, episode, self.time_step)
        return np.array(image_arr)


    def _display(self, episode=0):
        self.vis.show_segmentation_map(self.obs.plan_data_dict, episode, self.time_step)


    def _illustrate(self, episode=0):
        self.vis.show_obs_arr_conv(self.obs.obs_arr_conv, episode, self.time_step)

    
    def _update_env_stats(self):
        # self.env_stats = {
        #     'mask_numbers': self.obs.mask_numbers/1.0,
        #     'masked_corners': list(self.obs.masked_corners),
        #     'mask_lengths': list(np.array(self.obs.mask_lengths).astype(float)),
        #     'mask_widths': list(np.array(self.obs.mask_widths).astype(float)),
        #     'desired_areas': list(np.array([*self.obs.plan_data_dict['desired_areas'].values()])/1.0),
        #     'time_step': self.time_step,
        #     }
        
        self.env_stats = {
            "accepted_action_sequence": self.accepted_action_sequence,
            }
        
        return self.env_stats


# %%
if __name__ == "__main__":
    from gym_floorplan.envs.fenv_config import LaserWallConfig
    fenv_config = LaserWallConfig().get_config()
    self = MasterEnv(fenv_config)
    
    keys = ['env_name', 'net_arch', 'mask_flag', 'fixed_maksing_flag', 
                'mask_numbers', 'fixed_fc_observation_space',
                'include_wall_in_area_flag', 'area_tolerance', 'reward_shaping_flag']                
    for key in keys:    
        print(f"- - - - - {key}: {self.fenv_config[key]}")   
        
    # from stable_baselines3.common.env_checker import check_env
    # check_env(self)
        
    self.reset()
    done = False
    c = 0
    good_action_sequence = [1878, 1427, 826]
    accepted_action_list = []
    while not done:
        # for a in good_action_sequence:
            a = self.action_space.sample()
            if fenv_config['env_type'] == 'multi':
                a = {"wall_1": a}
            # print(f"action: {a}")
            
            observation, reward, done, info = self.step(a)
            
            # print(f"active_wall_status: {info['active_wall_status']}")
            # if info['active_wall_status'] in ["accepted", "finished"]:
            if done:
                accepted_action_list.append(a)
                self.render()
            
            c += 1
            # if c > 20:
            #     break
    
    print(f"accepted_action_list: {accepted_action_list}")