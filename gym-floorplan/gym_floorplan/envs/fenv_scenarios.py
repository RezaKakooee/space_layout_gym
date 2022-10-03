# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 01:34:16 2021

@author: RK
"""

import datetime

# %%
### scenario_name = Scenario_
class FEnvScenarios:
    """
    env_name = {'DOLW', 'DMLS'} # discrete one shot laser wall, discrete dymamic laser wall env
    env_type = {'Multi', 'Multi'}
    env_planning = {'Dynamic', 'Dynamic'}
    env_space = {'Discrete', 'Continous'}
    plan_config_source_name = {
            'fixed_test_config', 'offline_mode', 'longer_training_config'
            'load_fixed_config', 'load_random_config', 
            'create_fixed_config', 'create_random_config'
            }
    custom_model_name = {
        'SimpleFc', 'SimpleCnn',
        'SimpleCnnFc', 'SimpleFcCnn', 'SimpleActionMaskFc',
         
        'MySimpleFc', 'MySimpleCnn', 'MySimpleConv', 'MySimpleActionMaskFc',
        'MySimpleFcCnn', 'MySimpleGnnCnn'
        }
    net_arch = {
        'Fc', 'Cnn',
        'FcCnn',
        'CnnGcn' # only for generating dataset for sup-learning. for RL training we can use Fc or RNN
        
        'ActionMaskFc', 'ActionMaskCnn'
        }
    model_source = {
        'RllibBase', 'RllibCustomConfig', 
        'RllibCustomModel',
        'MyCustomModel',
        }
    """

    def __init__(self):
        ## Define Scenarios
        
        ### Env info
        self.env_name = 'DOLW' 
        self.env_type = 'Single'
        self.env_planning = 'One_Shot'
        self.env_space = 'Discrete'
        
        self.mask_flag = True
        
        ### Plan generation method and room/wall count
        self.plan_config_source_name = 'offline_mode'  # fixed_test_config  create_random_config   offline_mode   inference_mode load_random_config
        self.room_set_cardinality = 'Fixed' if self.plan_config_source_name == 'fixed_test_config' else 'X'
        
        self.nrows_of_env_data_csv = None #650 None
        self.fixed_num_rooms = 4
        
        if self.room_set_cardinality == 'Fixed':
            self.n_rooms = 4
            self.n_walls = self.n_rooms - 1
            self.wall_count_str = f"{self.n_walls:02}_walls"
            self.room_count_str = f"{self.n_rooms:02}_Rooms"
        elif self.room_set_cardinality == 'X':
            self.n_walls = self.room_set_cardinality
            self.n_rooms = self.room_set_cardinality
            self.wall_count_str = f"{self.n_walls*2}_walls"
            self.room_count_str = f"{self.n_rooms*2}_Rooms"
            
        
        ### Constraints info
        self.is_area_considered = True
        self.is_adjacency_considered = True
        self.is_proportion_considered = False
        
        
        ### Model info
        self.resolution = 'Low' # Low High
        self.action_masking_flag = False
        self.net_arch = 'Fc' # 'gnncnn'
        
        self.gcn_obs_method = 'embedded_image_graph' # image   embedded_image_graph dummy_vector
        
        self.use_lstm = False
        self.model_source = 'RllibModel' # 'MyCustomModel' # 'RllibCustom' RllibCustomConfig RllibModel
        
        if self.model_source == 'MyCustomModel':
            if self.action_masking_flag:
                self.model_name = f"MySimpleActionMask{self.net_arch}"
            else:
                self.model_name = f"MySimple{self.net_arch}"
        elif self.model_source == 'RllibCustomModel':
            if self.action_masking_flag:
                self.model_name = f"SimpleActionMask{self.net_arch}"
            else:
                self.model_name = f"Simple{self.net_arch}"
        else:
            self.model_name = 'RllibModel'
        
        
        ### Scenario name        
        self.scenario_name = f"Scenario__{self.env_name}__"
        if self.mask_flag: self.scenario_name += "Masked_"
        self.scenario_name += ''.join(s[0].upper() for s in self.plan_config_source_name.split('_'))
        # self.scenario_name += '_Plan__'
        # self.scenario_name += f"__{self.wall_count_str}__"
        self.scenario_name += f"__{self.room_count_str}__"
        self.scenario_name += "HRres__" if self.resolution == 'High' else "LRres__"
        if self.action_masking_flag: self.scenario_name += "ActionMasking__"
        if self.is_area_considered: self.scenario_name += "Area__"
        if self.is_adjacency_considered: self.scenario_name += "Adj__"
        if self.is_proportion_considered: self.scenario_name += "Prop__"
        self.scenario_name += f"{self.net_arch.upper()}__{datetime.datetime.now().strftime('%Y_%m_%d_%H%M')}"
        
        
        ### Some obs config
        self.fixed_fc_observation_space = True
        self.include_wall_in_area_flag = True
        self.use_areas_info_into_observation_flag = True
        self.use_edge_info_into_observation_flag = True
        
        self.maximum_num_masked_rooms = 4
        self.maximum_num_real_rooms = 9 if self.fixed_fc_observation_space else self.n_rooms
        self.num_of_fixed_walls_for_masked = self.maximum_num_masked_rooms + self.maximum_num_real_rooms 
        
        
        ### Some reward config
        self.rewarding_method_name = 'OnlyFinalRewardSimple' # 'OnlyFinalRewardSimple' if self.is_adjacency_considered else 'BasicReward' # BasicReward MinMax_Threshold
        self.binary_reward_flag = True if self.rewarding_method_name == 'BinaryReward' else False
        self.basic_reward_flag = True if self.rewarding_method_name == 'BasicReward' else False
        self.only_final_reward_flag = True if self.rewarding_method_name == 'OnlyFinalReward' else False
        self.only_final_reward_simple_flag = True if self.rewarding_method_name == 'OnlyFinalRewardSimple' else False
        self.reward_shaping_flag = True if self.rewarding_method_name == 'MinMaxThresholdReward' else False
        
        self.area_diff_in_reward_flag = True if self.is_area_considered else False
        self.proportion_diff_in_reward_flag = False #True if self.is_proportion_considered else False  
        
        self.reward_increment_within_interval_flag = False
        self.reward_decremental_flag = False
        
        
    def get_scenarios(self):
        return self.__dict__



# %%
if __name__ == '__main__':
    scenarios_dict = FEnvScenarios().get_scenarios()
    from pprint import pprint
    pprint(scenarios_dict)
