 # -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 01:19:28 2021

@author: Reza Kakooee
"""

# %%


from gym_floorplan.envs.reward.reward_base_simple import RewardBaseSimple
from gym_floorplan.envs.reward.reward_base_smooth import RewardBaseSmooth
from gym_floorplan.envs.reward.reward_zc_smooth import RewardZcSmooth
from gym_floorplan.envs.reward.reward_sensor import RewardSensor




# %%

class Reward:
    def __init__(self, fenv_config:dict={}):
        super().__init__()
        self.fenv_config = fenv_config
        


    def reward(self, plan_data_dict, 
                     active_wall_name, active_wall_status, 
                     ep_time_step, done):
        
        if active_wall_status in ['accepted', 'well_finished']:
            self.rew_sensor = RewardSensor(self.fenv_config, plan_data_dict, active_wall_name, active_wall_status, done)
            inspection_output_dict = self.rew_sensor.inspect()
        else:
            inspection_output_dict = {}
        
        
        if self.fenv_config['rewarding_method_name'] in ['Constrain_Satisfaction', 'Binary_Reward', 'Simple_Reward']:
            reward_cls = RewardBaseSimple(self.fenv_config, active_wall_status, done, inspection_output_dict)
            
        elif self.fenv_config['rewarding_method_name'] in ['Smooth_Linear_Reward', 'Smooth_Quad_Reward', 'Smooth_Log_Reward', 'Smooth_Exp_Reward']:
            reward_cls = RewardBaseSmooth(self.fenv_config, plan_data_dict, active_wall_name, active_wall_status, done, inspection_output_dict)
            
        elif self.fenv_config['rewarding_method_name'] in ['ZC_Smooth_Linear_Reward', 'ZC_Smooth_Quad_Reward', 'ZC_Smooth_Log_Reward', 'ZC_Smooth_DLin_Reward', 'ZC_Smooth_Perc_Reward', 'ZC_Smooth_FNorm_Reward']:
            reward_cls = RewardZcSmooth(self.fenv_config, plan_data_dict, active_wall_name, active_wall_status, done, inspection_output_dict)
            
        else:
            raise ValueError(f"Invalid rewarding method! The current one is {self.fenv_config['rewarding_method_name']}")
        
        reward = reward_cls.get_reward()
        
        return reward