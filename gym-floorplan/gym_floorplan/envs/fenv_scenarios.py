# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 01:34:16 2021

@author: Reza Kakooee
"""

import datetime

# %%
### scenario_name = Scenario_
class FEnvScenarios:
    # env_name = {'DOLW', 'DMLS'} # discrete one shot laser wall, discrete dymamic laser wall env
    # env_type = {'Single', 'Multi'}
    # env_planning = {'One_Shot', 'Dynamic'}
    # env_space = {'Discrete', 'Continous'}
    # plan_config_source_name = {'test_config', 'load_fixed_config', 'load_random_config', 'create_fixed_config', 'create_random_config'}
    def __init__(self):
        ## Define Scenarios
        self.env_name = 'DOLW' 
        self.env_type = 'Single'
        self.env_planning = 'One_Shot'
        self.env_space = 'Discrete'
        
        self.plan_config_source_name = 'test_config'
        
        self.mask_flag = True
        
        # self.fixed_maksing_flag = True
        # self.load_valid_plans_flag = True
        
        self.is_area_considered = True
        self.is_proportion_considered = False
        
        self.n_walls = 10
        
        self.net_arch = 'fc'
        
        self.scenario_name = f"Scenario__{self.env_name}__"
        if self.mask_flag: self.scenario_name += "Masked_"
        self.scenario_name += ''.join(s[0].upper() for s in self.plan_config_source_name.split('_'))
        # self.scenario_name += '_Plan__'
        self.scenario_name += f"__{self.n_walls:02}_Walls__"
        if self.is_area_considered: self.scenario_name += "Area__"
        if self.is_proportion_considered: self.scenario_name += "Proportion__"
        self.scenario_name += f"{self.net_arch.upper()}__{datetime.datetime.now().strftime('%Y_%m_%d_%H%M')}"
        
        
        self.fixed_fc_observation_space = True
        self.include_wall_in_area_flag = True
        self.use_areas_info_into_observation_flag = False
        self.rewarding_method_name = 'MinMax_Threshold'
        self.reward_shaping_flag = True
        self.reward_increment_within_interval_flag = False
        self.reward_decremental_flag = False

    def get_scenarios(self):
        return self.__dict__


# %%
if __name__ == '__main__':
    scenarios_dict = FEnvScenarios().get_scenarios()
    from pprint import pprint
    pprint(scenarios_dict)
