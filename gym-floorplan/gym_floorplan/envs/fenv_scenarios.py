# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 01:34:16 2021

@author: Reza Kakooee
"""

import datetime

# %%
### scenario_name = Scenario_
class FEnvScenarios:
    """
    plan_config_source_name = {
            'fixed_test_config', 
            'offline_mode', 
            'imitation_mode'
            'load_fixed_config',
            'load_random_config', 
            'create_fixed_config', 
            'create_random_config',
            'inference_mode',
            }
    custom_model_name = {
        'TinyFcNet', 
        'FcNet',
        'MetaFcNet',
        
        'TinyCnnNet',
         
        'CnnNet', 
        
        'MetaCnnNet', 
        'MetaCnnResNet', 
        'MetaCnnResidual', 
        
        'TineyGnnNet'
        
        }
    net_arch = {
        'Fc', 'Cnn',
        'MetaFc', 'MetaCnn'
        'ActionMaskFc', 'ActionMaskCnn'
        'Gnn'
        }
    model_source = {
        'RllibBase', 
        'RllibCustomConfig', 
        'RllibCustomModel',
        'MyCustomModel',
        }
    rewarding_method_name = {
        'Constrain_Satisfaction',
        'Very_Binary_Reward',
        'Binary_Reward',
        'Simple_Reward',
        'Simple_Exp_Reward',
        'Simple_Quad_Reward',
        'Simple_Lin_Reward',
        'Detailed_Reward',
        }
    resolution = {
        'Low',
        'High'
        }
    cnn_observation_name = {
        'canvas_1d',
        'room_cmap',
        'stacked_3d',
        'canvas_3d',
        }
    """
    
    
    net_archs = {
        # model_name -> model_arch_name
        'TinyFcNet':'Fc', 
        'FcNet':'Fc', 
        'MetaFcNet': 'MetaFc',
        
        
        'TinyCnnNet':'Cnn', 
        'CnnNet':'Cnn', 
        'MetaCnnNet':'MetaCnn',
        'MetaCnnResNet':'MetaCnn',
        'MetaCnnResidualNet':'MetaCnn',
        
        
        'TinyFcNetActionMasking': 'Fc', 
        'FcNetActionMasking': 'Fc',
        'MetaFcNetActionMasking': 'MetaFc',
        
        'TinyCnnNetActionMasking': 'Cnn',
        'CnnNetActionMasking': 'Cnn',
        'MetaCnnNetActionMasking': 'MetaCnn',
        'MetaCnnResNetActionMasking': 'MetaCnn',
        'MetaCnnResidualNetActionMasking':'MetaCnn',
        
        
        'TinyGnnNet': 'Gnn',
        'GnnNet': 'Gnn',
        
        'SimpleFc': 'Fc', 
        'SimpleCnn': 'Cnn',

        }
   
    
        
    def __init__(self, agent_name=None, 
                 action_masking_flag=None, 
                 cnn_observation_name=None, 
                 rewarding_method_name=None,
                 model_name=None,
                 n_rooms=None,
                 batch_norm=None,
                 project_name=None):
        
        self.agent_name = 'PPO' if agent_name is None else agent_name
        self.action_masking_flag = False if action_masking_flag is None else action_masking_flag
        self.cnn_observation_name = 'canvas_1d' if cnn_observation_name is None else cnn_observation_name
        self.rewarding_method_name = 'Simple_Quad_Reward' if rewarding_method_name is None else rewarding_method_name
        self.model_name = 'TinyFcNet' if model_name is None else model_name
        self.n_rooms = 4 if n_rooms is None else n_rooms
        self.batch_norm = False if batch_norm is None else batch_norm
        
        self.project_name = f"Prj__{datetime.datetime.now().strftime('%Y_%m_%d_%H%M')}" if project_name is None else project_name  
        
        ######################################################################
        ## Define Scenarios
        
        
        ### Constraints info
        self.stop_ep_time_step = 1000
        self.is_area_considered = True
        self.is_proportion_considered = True
        self.is_entrance_considered = True
        self.is_entrance_adjacency_a_constraint = True
        self.is_cooridor_entrance_a_constraint = True
        self.adaptive_window = True
        self.is_adjacency_considered = True
        
        
        ## Hyper-parametres:
        self.resnet_pretrained_flag = True
        self.resolution = 'Low'
        self.cnn_scaling_factor = 1 if self.resolution == 'Low' else 1
        self.n_channels = 1 if self.cnn_observation_name in ['canvas_1d', 'room_cmap'] else 3
        
        self.use_redidual = False
        
        self.gnn_obs_method = 'embedded_image_graph' # image   embedded_image_graph dummy_vector
        self.use_lstm = False
            
        
        ### Plan generation method and room/wall count
        
        self.plan_config_source_name = 'load_random_config'
        
        # when create_random_config: if you set is_entrance_adjacency_a_constraint and is_cooridor_entrance_a_constraint to True, then you have to set the corridor id in advance. otherwise it will be set in the end in master_env
        
        if self.plan_config_source_name in ['create_random_config', 'create_random_config']:
            self.create_random_configs_with_fixed_n_rooms_flag = False # self.n_rooms # only for create_random_config -> set it to False, or the number of rooms you want 
            self.fixed_outline_flag = False # only for create_random_config
        
        if self.plan_config_source_name in ['load_random_config', 'load_fixed_config', 'offline_mode', 'imitation_mode']:
            self.plan_id_for_load_fixed_config = None #'2023-08-08_16-47-42.456011' #'2023-08-09_17-49-38.634210' #None # None or plan_id
            self.nrows_of_env_data_csv = None # None or an integer
            self.fixed_num_rooms_for_loading = None # None or an integer
        
        self.room_set_cardinality = 'Fixed' if self.plan_config_source_name == 'fixed_test_config' else 'X'
        
        if self.room_set_cardinality == 'Fixed':
            self.n_rooms = self.n_rooms
            self.n_walls = self.n_rooms - 1
            self.wall_count_str = f"{self.n_walls:02}_walls"
            self.room_count_str = f"{self.n_rooms:02}_Rooms"
        elif self.room_set_cardinality == 'X':
            self.n_walls = self.room_set_cardinality
            self.n_rooms = self.room_set_cardinality
            self.wall_count_str = f"{self.n_walls*2}_walls"
            self.room_count_str = f"{self.n_rooms*2}_Rooms"
            

        ## Library and model
        if self.action_masking_flag: self.model_name += 'ActionMasking'
        if self.use_redidual: self.model_name += 'Residual'
        self.net_arch = self.net_archs[self.model_name]
        self.library = 'RLlib' 
        self.model_source = 'MyCustomModel' if "Simple" not in self.model_name else 'RllibCustomModel'

        
        ## Scenario name        
        self.scenario_name = "Scn__"
        self.scenario_name += f"{datetime.datetime.now().strftime('%Y_%m_%d_%H%M')}__"
        self.scenario_name += f"{self.agent_name}__"
        # self.scenario_name += f"{self.model_name}__"
        self.scenario_name += ''.join(s[0].upper() for s in self.plan_config_source_name.split('_'))
        self.scenario_name += f"__{self.n_rooms}Rr__"
        if self.resolution == 'High': self.scenario_name += "Hr__"
        if self.is_area_considered: self.scenario_name += "Ar"
        if self.is_proportion_considered: self.scenario_name += "Pr"
        if self.is_entrance_considered: self.scenario_name += "En"
        if self.adaptive_window: self.scenario_name += "Aw"
        if self.is_adjacency_considered: self.scenario_name += "Ad__"
        self.scenario_name += ''.join(s[0].upper() for s in self.rewarding_method_name.split('_'))
        # if self.action_masking_flag: self.scenario_name += "__ActMask__"
        
        
        
    def get_scenarios(self):
        return self.__dict__




# %%
if __name__ == '__main__':
    self = FEnvScenarios()
    scenarios_dict = self.get_scenarios()
    from pprint import pprint
    pprint(scenarios_dict)
