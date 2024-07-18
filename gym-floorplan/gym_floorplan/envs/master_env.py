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

import gym_floorplan
from gym_floorplan.envs.action.action import Action
from gym_floorplan.envs.observation.observation import Observation
from gym_floorplan.envs.reward.reward import Reward
from gym_floorplan.envs.render.render import Render




# %%
class MasterEnv(gym.Env):#(MultiAgentEnv):
    def __init__(self, env_config={'fenv_name': 'DOLW-v0'}):
        self.fenv_config = self.env_config = env_config
        self.env_name = 'DOLW-v0'
        
        self.obs = Observation(fenv_config=self.fenv_config)
        self.act = Action(fenv_config=self.fenv_config)
        self.rew = Reward(fenv_config=self.fenv_config)
        self.vis = Render(fenv_config=self.fenv_config)

        self.action_space = self.act.action_space
        self.observation_space = self.obs.observation_space

        self.episode = 0
        # observation = self.reset()
        
        # self.seed(42)
        
        self.max_episode_steps = 1000
        


    def reset(self, *, seed=None, options=None):
        self.episode += 1
        observation = self.obs.obs_reset(self.episode)
        self.observation_space = self.obs.observation_space
        self.accepted_action_sequence = []
        self.ep_time_step = 0
        self.ep_sum_reward = 0
        self.ep_max_reward = float('-inf')
        self.ep_mean_reward = 0
        return observation, {}
        
    
    
    def seed(self, seed: int = None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]



    def step(self, action):
        observation = self.obs.update(self.episode, action, self.ep_time_step)
        
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
        
        if self.obs.active_wall_status in ['accepted', 'well_finished']:
            self.accepted_action_sequence.append(action)
            
        info = self._get_info(done, reward) if (done and self.fenv_config['save_env_info_on_callback']) else {}
        
        if not done:
            self.ep_time_step += 1
        
        self.done = done
        self.info = info
        return observation, reward, done, truncated, info
    
    
    
    def _get_info(self, done, reward):
        if self.obs.active_wall_status == 'well_finished':
            info = {'env_data': self._update_env_stats(reward)}
        else:
            info = {} if self.fenv_config['library'] == 'RLlib' else {'episode_end_data': self._get_episode_end_data(reward)}
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
            'corridor_id': self.obs.plan_data_dict['corridor_id'],
            'living_room_id': self.obs.plan_data_dict['living_room_id'],
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
        
        
        def _get_living_room_id(edges, corridor_id):
            rooms_connected_to_corridor = []
            for ed in edges:
                if corridor_id in ed:
                    rooms_connected_to_corridor.extend(ed)
            rooms_connected_to_corridor = list(set(rooms_connected_to_corridor).difference(set([corridor_id])))
            return np.random.choice(rooms_connected_to_corridor, 1)[0]
        
        
        def _get_obs_stats(obs):
            if isinstance(obs, dict):
                pass
            
            
            
        edge_list_facade_achieved_str = self.obs.plan_data_dict['edge_list_facade_achieved_str'].copy()
        edge_list_entrance_achieved_str = self.obs.plan_data_dict['edge_list_entrance_achieved_str'].copy()
        if self.fenv_config['plan_config_source_name'] in ['create_fixed_config', 'create_random_config']:
            corridor_id = [room for edge in edge_list_entrance_achieved_str for room in edge if isinstance(room, int)][0]
            rooms_connected_to_corridor = []
            living_room_id = ( self.obs.plan_data_dict['living_room_id'] if self.obs.plan_data_dict['living_room_id'] 
                               else _get_living_room_id(self.obs.plan_data_dict['edge_list_room_achieved'], corridor_id) )
            self.env_stats.update({'edge_list_room_desired': str(self.obs.plan_data_dict['edge_list_room_achieved']),
                                   'edge_list_room_achieved': str(self.obs.plan_data_dict['edge_list_room_achieved']),
                                   'edge_list_facade_desired_str': str(edge_list_facade_achieved_str),
                                   'edge_list_facade_achieved_str': str(edge_list_facade_achieved_str),
                                   'edge_list_entrance_desired_str': str(edge_list_entrance_achieved_str),
                                   'edge_list_entrance_achieved_str': str(edge_list_entrance_achieved_str),
                                   'corridor_id': corridor_id,
                                   'living_room_id': living_room_id,
                                   })
        else:
            self.env_stats.update({'edge_list_room_desired': str(self.obs.plan_data_dict['edge_list_room_desired']),
                                   'edge_list_room_achieved': str(self.obs.plan_data_dict['edge_list_room_achieved']),
                                   'edge_list_facade_desired_str': str(self.obs.plan_data_dict['edge_list_facade_desired_str']),
                                   'edge_list_facade_achieved_str': str(edge_list_facade_achieved_str),
                                   'edge_list_entrance_desired_str': str(self.obs.plan_data_dict['edge_list_entrance_desired_str']),
                                   'edge_list_entrance_achieved_str': str(edge_list_entrance_achieved_str)
                                   })
        
        return self.env_stats



    def render(self, episode=0, mode='human'):
        image_arr = self.vis.render(self.obs.plan_data_dict, self.episode, self.ep_time_step)
        return np.array(image_arr)


    
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
            
    



# %%
if __name__ == "__main__":
    from gym_floorplan.envs.fenv_scenarios import FEnvScenarios
    from gym_floorplan.envs.fenv_config import LaserWallConfig
    
    
    default_config = {
        'agent_name': 'RND',
        'phase': 'debug',
        
        'action_masking_flag': False,
        'rewarding_method_name': 'Simple_Quad_Reward', # ['Simple_Linear_Reward', 'Simple_Exp_Reward', 'Simple_Quad_Reward'],
        'cnn_observation_name': 'canvas_1d',
        'model_name': 'MetaFcNet', #['TinyLinearNet', 'TinyCnnNet'],
        }
    
    
    scenarios_dict = FEnvScenarios(agent_name=default_config['agent_name'], 
                                   action_masking_flag=default_config['action_masking_flag'], 
                                   cnn_observation_name=default_config['cnn_observation_name'],
                                   rewarding_method_name=default_config['rewarding_method_name'],
                                   model_name=default_config['model_name']).get_scenarios()
    
    fenv_config = LaserWallConfig(agent_name=default_config['agent_name'], 
                                  phase=default_config['phase'], 
                                  scenarios_dict=scenarios_dict).get_config()
    
    
    self = MasterEnv(fenv_config)
    s0, _ = self.reset()
    a = self.action_space.sample()
    s, _, _, _, _= self.step(a)
    
    from ray.rllib.utils import check_env
    check_env(self)
