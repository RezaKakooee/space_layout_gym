# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 01:22:19 2021

@author: RK
"""

# %%
from gym_floorplan.base_env.render.base_render import BaseRender
import gym_floorplan.envs.render.laser_wall_vis_utils as visutils
from gym_floorplan.envs.observation.room_extractor import RoomExtractor


# %%
class Render(BaseRender):
    def __init__(self, fenv_config: dict = {}):
        super().__init__()
        self.fenv_config = fenv_config


    def render(self, plan_data_dict, episode, time_step):
        visutils.render_plan(plan_data_dict, episode, time_step,
                           fenv_config=self.fenv_config)


    def display(self, plan_data_dict, episode, time_step):
        rextractor = RoomExtractor(fenv_config=self.fenv_config)
        obs_mat, labels = rextractor._get_segmentation_map(plan_data_dict['obs_mat'])
        
        visutils.display_env(obs_mat, labels, episode, time_step, 
                          fenv_config=self.fenv_config)


    def illustrate(self, obs_conv_arr, episode, time_step):
        visutils.illustrate_obs_arr_conv(obs_conv_arr, episode, time_step, 
                                   fenv_config=self.fenv_config)
    
    
    def demonestrate(self, plan_data_dict, episode, time_step):
        visutils.demonestrate_rooms_with_walls(plan_data_dict, episode, time_step,
                                       fenv_config=self.fenv_config)
