# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 01:22:19 2021

@author: Reza Kakooee
"""


# %%
from gym_floorplan.base_env.render.base_render import BaseRender
from gym_floorplan.envs.render.render_plan import RenderPlan, DisplayPlan




# %%
class Render(BaseRender):
    def __init__(self, fenv_config: dict = {}):
        super().__init__()
        self.fenv_config = fenv_config
        self.render_plan = RenderPlan(fenv_config)
        self.display_plan = DisplayPlan(fenv_config)
        


    def render(self, plan_data_dict, episode, ep_time_step):
        self.render_plan.render(plan_data_dict, episode, ep_time_step)

    

    def portray(self, plan_data_dict, episode, ep_time_step):
        self.render_plan.portray(plan_data_dict, episode, ep_time_step)


    
    def display(self, plan_data_dict, episode, ep_time_step):
        self.display_plan.show_room_map(plan_data_dict, episode, ep_time_step)
        
        

    def illustrate(self, plan_data_dict, episode, ep_time_step):
        self.display_plan.show_1d_obs(plan_data_dict, episode, ep_time_step)
    
    
    
    def demonestrate(self, plan_data_dict, episode, ep_time_step):
        self.display_plan.show_3d_obs(plan_data_dict, episode, ep_time_step)
        
        

    def exibit(self, plan_data_dict, episode, ep_time_step):
        self.display_plan.show_segmentation_map(plan_data_dict, episode, ep_time_step)


    
    def view(self, plan_data_dict, episode, ep_time_step):
        self.display_plan.view_obs_mat(plan_data_dict, episode, ep_time_step)



    