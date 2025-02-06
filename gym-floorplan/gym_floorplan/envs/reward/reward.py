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
from gym_floorplan.envs.reward.dynamic_planning_reward_sensor import DynamicPlanningRewardSensor
from gym_floorplan.envs.reward.dynamic_planning_reward_zc_smooth import DynamicPlanningRewardZcSmooth



# %%

class Reward:
    def __init__(self, fenv_config:dict={}):
        super().__init__()
        self.fenv_config = fenv_config

        if self.fenv_config['env_planning'] == 'One_Shot':
            self.rew_sensor = RewardSensor(self.fenv_config)
            if self.fenv_config['rewarding_method_name'] in ['Constrain_Satisfaction', 'Binary_Reward', 'Simple_Reward']:
                self.reward_base_simple_cls = RewardBaseSimple(self.fenv_config)
                
            elif self.fenv_config['rewarding_method_name'] in ['Smooth_Linear_Reward', 'Smooth_Quad_Reward', 'Smooth_Log_Reward', 'Smooth_Exp_Reward']:
                self.reward_base_smooth_cls = RewardBaseSmooth(self.fenv_config)
                
            elif self.fenv_config['rewarding_method_name'] in ['ZC_Smooth_Linear_Reward', 'ZC_Smooth_Quad_Reward', 'ZC_Smooth_Log_Reward', 'ZC_Smooth_DLin_Reward', 'ZC_Smooth_Perc_Reward', 'ZC_Smooth_FNorm_Reward']:
                self.reward_zc_smooth_cls = RewardZcSmooth(self.fenv_config)
                
            else:
                raise ValueError(f"Invalid rewarding method! The current one is {self.fenv_config['rewarding_method_name']}")
            
        elif self.fenv_config['env_planning'] == 'Dynamic':
            self.dyp_rew_sensor = DynamicPlanningRewardSensor(self.fenv_config)
            self.dyp_reward_cls = DynamicPlanningRewardZcSmooth(self.fenv_config)
        


    def reward(self, plan_data_dict, 
                     active_wall_name, active_wall_status, 
                     ep_time_step, done):

        if self.fenv_config['env_planning'] == 'One_Shot':        
            if active_wall_status in ['accepted', 'well_finished']:
                inspection_output_dict = self.rew_sensor.inspect(plan_data_dict, active_wall_name, active_wall_status, done)
            else:
                inspection_output_dict = {}
            
            if self.fenv_config['rewarding_method_name'] in ['Constrain_Satisfaction', 'Binary_Reward', 'Simple_Reward']:
                reward = self.reward_base_simple_cls.get_reward(active_wall_status, done, inspection_output_dict)
                
            elif self.fenv_config['rewarding_method_name'] in ['Smooth_Linear_Reward', 'Smooth_Quad_Reward', 'Smooth_Log_Reward', 'Smooth_Exp_Reward']:
                reward = self.reward_base_smooth_cls.get_reward(plan_data_dict, active_wall_name, active_wall_status, done, inspection_output_dict)
                
            elif self.fenv_config['rewarding_method_name'] in ['ZC_Smooth_Linear_Reward', 'ZC_Smooth_Quad_Reward', 'ZC_Smooth_Log_Reward', 'ZC_Smooth_DLin_Reward', 'ZC_Smooth_Perc_Reward', 'ZC_Smooth_FNorm_Reward']:
                reward = self.reward_zc_smooth_cls.get_reward(plan_data_dict, active_wall_name, active_wall_status, done, inspection_output_dict)
                
            else:
                raise ValueError(f"Invalid rewarding method! The current one is {self.fenv_config['rewarding_method_name']}")
            
        
        elif self.fenv_config['env_planning'] == 'Dynamic':
            self.well_finished_condition = False
            if active_wall_status in ['accepted']: # in Dynamic planning each time_step is like 'well_finished'
                inspection_output_dict = self.dyp_rew_sensor.inspect(plan_data_dict, active_wall_status)
                
                reward, self.well_finished_condition = self.dyp_reward_cls.get_reward(plan_data_dict, active_wall_status, inspection_output_dict) # -1 to (2 + bonus)
                
            elif 'reject' in active_wall_status:
                reward = self.fenv_config['dyp_rejected_reward'] # -2
            
            elif active_wall_status == 'badly_stopped':
                reward = self.fenv_config['dyp_negative_badly_stop_reward'] # -200
                
            else:
                raise ValueError(f"Invalid active_wall_status: {active_wall_status}")
            
        else:
            raise ValueError(f"Invalid env_planning: {self.fenv_config['env_planning']}")

        return reward