# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 01:34:16 2021

@author: Reza Kakooee
"""

import datetime

# %%
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
        'pre_train_mode'
        }
custom_model_last_name = {
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
    'Simple_Lin_Reward',
    'Simple_Exp_Reward',
    'Simple_Quad_Reward',
    'Smooth_Linear_Reward',
    'Smooth_Log_Reward',
    'Smooth_Quad_Reward',
    'Detailed_Reward',
    'OO_Simple_Linear_Reward', # Objected Oriented reward, only for zero constraint
    'OO_Simple_Log_Reward',
    'OO_Simple_Quad_Reward',
    }
resolution = {
    'Low',
    'High'
    }
cnn_observation_name = {
    'canvas_1d',
    'rooms_cmap',
    'stacked_3d',
    'canvas_3d',
    }
"""

net_archs = {
        # model_last_name -> model_arch_name
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
        'MetaCnnResidualNetActionMaskingQ':'MetaCnn',
        'MetaCnnResidualNetActionMaskingP':'MetaCnn',
        
        'TinyGnnNet': 'Gnn',
        'GnnNet': 'Gnn',
        
        'SimpleFc': 'Fc', 
        'SimpleCnn': 'Cnn',
        
        'DQNTinyFcNet': 'Fc',
        'DQNTinyFcNetActionMasking': 'Fc',

        'DQNCnnNetActionMasking': 'Cnn',

        'BaseCnnNet': 'Cnn',
        'CnnMiniResNet': 'Cnn',
        
        'MetaCnnNetFineTune': 'MetaCnn',
        'MetaCnnNetFromPreTrain': 'MetaCnn',

        }

class FEnvScenarios:
    net_archs = net_archs
    def __init__(self, hyper_params):
        self.agent_name = hyper_params.get('agent_name', 'PPO')
        self.action_masking_flag = hyper_params.get('action_masking_flag', False)
        self.cnn_observation_name = hyper_params.get('cnn_observation_name', 'canvas_1d')
        self.rewarding_method_name = hyper_params.get('rewarding_method_name', 'Smooth_Quad_Reward')
        self.zc_reward_weighening_mode = hyper_params.get('zc_reward_weighening_mode', '15_15_15_15_40')
        self.reward_vertical_scalar = hyper_params.get('reward_vertical_scalar', 1)
        self.zc_ignore_immidiate_reward = hyper_params.get('zc_ignore_immidiate_reward', 0)
        self.shift_zc_reward_bottom_to_zero = hyper_params.get('shift_zc_reward_bottom_to_zero', 1)
        self.zc_has_geom_tolerance = hyper_params.get('zc_has_geom_tolerance', True)
        self.learn_room_order_directly = hyper_params.get('learn_room_order_directly', False)
        self.lv_mode = hyper_params.get('lv_mode', False)
        self.model_last_name = hyper_params.get('model_last_name', 'TinyFcNet')
        self.n_rooms = hyper_params.get('n_rooms', 4)
        self.batch_norm = hyper_params.get('batch_norm', False)
        self.project_name = hyper_params.get('project_name', None)


        self.project_name = f"Prj__{datetime.datetime.now().strftime('%Y_%m_%d_%H%M')}" if project_name is None else project_name  
        
        
        ######################################################################
        ## Define Scenarios
        self.pre_train_mode = True if self.agent_name in ['BC'] else False
        
        self.plan_config_source_name = 'fixed_test_config'
        
        self.encode_img_obs_by_vqvae_flag = False
        
        self.exactly_mimic_an_existing_plan = False
        self.removing_lv_room_from_action_set = True
        self.is_agent_allowed_to_create_lvroom = False
        self.randomly_create_lvroom_first = True # randomly_create_lvroom_first and lvroom_portion_range are only for the create_random_config. it allows to assign an area first to the lv_room and then distribute the rest of region to the other rooms
        
        self.action_mode = False

        ### Constraints info
        self.stop_ep_time_step = 1000
        self.is_area_a_constraint = True
        self.is_proportion_a_constraint = True
        self.is_entrance_a_constraint = True # Ensures no walls hits the entrance cells. This always has to be True even for zero constraint mode
        self.is_entrance_adjacency_a_constraint = True # Ensures no non-lvroom rooms occupy the cells adjacent to the entrance
        self.is_entrance_lvroom_connection_a_constraint = True # Ensures the lvroom is positioned (if positioned at all) such that it becomes an adjacent cell to the entrance.
        self.adaptive_window = True
        self.is_adjacency_considered = True
        
        self.adaptive_window = False if self.exactly_mimic_an_existing_plan else True
        
        self.zero_constraint_flag = True if 'ZC' in self.rewarding_method_name else False
        
        if self.zero_constraint_flag:
            self.is_area_a_constraint = False
            self.is_proportion_a_constraint = False
            # self.is_entrance_adjacency_a_constraint = False
            # self.is_entrance_lvroom_connection_a_constraint = False
            
        if self.plan_config_source_name == 'create_random_config':
            self.adaptive_window = False
            self.is_adjacency_considered = False
            self.load_resnet_from_pretrained_weights = False
            
        # TODO: On 2024-01-16, I added the following two lines
        # if self.plan_config_source_name == 'load_random_config':
        #     self.adaptive_window = False

        # self.does_living_room_need_a_facade = False
        
        if self.action_mode:
            self.is_area_a_constraint = False
            self.is_proportion_a_constraint = False
            self.is_entrance_adjacency_a_constraint = True
            self.is_entrance_cooridor_connection_a_constraint = True
            self.adaptive_window = False
            self.does_living_room_need_a_facade = True
        
        ## Hyper-parametres:
        self.load_resnet_from_pretrained_weights = True
        self.resolution = 'Low'
        self.cnn_scaling_factor = 1 if self.resolution == 'Low' else 1
        self.n_channels = 1 if self.cnn_observation_name in ['canvas_1d', 'rooms_cmap'] else 3
        self.very_short_observation_fc_flag = False
        
        self.use_redidual = False
        
        self.gnn_obs_method = 'embedded_image_graph' # image   embedded_image_graph dummy_vector
        
        # self.use_lstm = False
        # self.lstm_cell_size = 4465
            
        
        ### Plan generation method and room/wall count
        # when create_random_config: if you set is_entrance_adjacency_a_constraint and is_entrance_lvroom_connection_a_constraint to True, then you have to set the corridor id in advance. otherwise it will be set in the end in master_env
        
        if self.plan_config_source_name in ['create_random_config', 'create_fixed_config']:
            self.create_random_configs_with_fixed_n_rooms_flag = None # self.n_rooms # only for create_random_config -> set it to False, or the number of rooms you want 
            self.fixed_outline_flag = False # only for create_random_config
        
        if self.plan_config_source_name in ['load_random_config', 'load_fixed_config', 'offline_mode', 'imitation_mode']:
            # 4: 2024-02-01_17-36-53.949841 , 5: 2024-02-03_00-09-44.602456 , 6: 2024-02-04_18-09-42.133013 , 7: 2024-02-02_22-18-27.417868, 8: 2024-02-05_01-34-15.036061 9: 2024-02-03_00-09-44.602456

            self.plan_id_for_load_fixed_config = '2024-02-03_10-31-37.911766' # '2024-02-04_18-09-42.133013' #'2024-02-02_17-46-29.464807' # '2024-02-02_21-01-51.028423' # '2024-02-02_13-16-33.865762' # '2024-02-02_16-59-04.762614' # '2024-02-04_01-13-40.249609' # '2024-02-04_15-52-42.304696' #  None or plan_id
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
            
        if self.agent_name in ['BC_', 'CQL_']:
            self.n_walls, self.n_rooms = 'X', 'X'
            

        ## Library and model
        if self.action_masking_flag: self.model_last_name += 'ActionMasking'
        if self.use_redidual: self.model_last_name += 'Residual'
        if self.agent_name == 'CQL': 
            self.policy_model_last_name = self.model_last_name + 'P'
            self.q_model_last_name = self.model_last_name + 'Q'
        self.net_arch = self.net_archs[self.model_last_name]
        self.library = 'RLlib' 
        self.model_source = 'MyCustomModel' # 'RllibCustomConfig' # 'MyCustomModel' if "Simple" not in self.model_last_name else 'RllibCustomModel' RllibCustomConfig

        self.dropout_prob = 0.0
        
        ## Scenario name        
        self.scenario_name = "Scn__"
        self.scenario_name += f"{datetime.datetime.now().strftime('%Y_%m_%d_%H%M')}__"
        # self.scenario_name += f"{self.model_last_name}__"
        self.scenario_name += 'PTM__' if self.pre_train_mode else ''.join(s[0].upper() for s in self.plan_config_source_name.split('_'))
        self.scenario_name += f"__{self.n_rooms}Rr__" if (self.plan_config_source_name == 'fixed_test_config' and not self.pre_train_mode) else ''
        # if self.is_area_a_constraint: self.scenario_name += "Ar"
        # if self.is_proportion_a_constraint: self.scenario_name += "Pr"
        # if self.is_entrance_a_constraint: self.scenario_name += "En"
        # if self.adaptive_window: self.scenario_name += "Aw"
        # if self.is_adjacency_considered: self.scenario_name += "Ad__"
        self.scenario_name += ''.join(s[0].upper() for s in self.rewarding_method_name.split('_')) + '__'
        if self.resolution == 'High': self.scenario_name += "Hr__"
        self.scenario_name += f"{self.agent_name}"
        
        # if self.action_masking_flag: self.scenario_name += "__ActMask__"
        
        
        
    def get_scenarios(self):
        return self.__dict__




# %%
if __name__ == '__main__':
    self = FEnvScenarios()
    scenarios_dict = self.get_scenarios()
    from pprint import pprint
    pprint(scenarios_dict)