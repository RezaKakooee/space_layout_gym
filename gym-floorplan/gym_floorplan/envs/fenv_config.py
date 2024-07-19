# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 00:58:16 2021

@author: Reza Kakooee
"""


# %%
import os
def get_housing_design_root_dir():
    housing_design_root_dir = os.getenv('HOUSING_DESIGN_ROOT_DIR')
    if housing_design_root_dir is None:
        raise EnvironmentError("The 'HOUSING_DESIGN_ROOT_DIR' environment variable is not set.")
    return housing_design_root_dir
root_dir = get_housing_design_root_dir()
    

import yaml
import datetime
import numpy as np
from gym_floorplan.envs.name_spaces import net_archs, color_map



#%%
class LaserWallConfig:
    def __init__(self, phase='train', hyper_params={}):
        self.phase = phase
        
        #######################################################################
        ### Hyper params ######################################################
        #######################################################################
        default_hyper_params_yaml_path = os.path.join(root_dir, 'default_hps_env.yaml')
        with open(default_hyper_params_yaml_path, 'r') as f:
            default_hyper_params = yaml.load(f, Loader=yaml.FullLoader)
        default_hyper_params.update(hyper_params)
        for dhpattribute in default_hyper_params.keys():
            setattr(self, dhpattribute, default_hyper_params[dhpattribute])

        self.resolution = 'Low'
        self.n_channels = 1 if self.cnn_observation_name in ['canvas_1d', 'rooms_cmap'] else 3
        self.use_redidual = False
        
        #######################################################################
        ### Env info ##########################################################
        #######################################################################
        self.env_name = 'SpaceLayoutGym-v0' 
        self.env_type = 'Single'
        self.env_planning = 'One_Shot'
        self.env_space = self.action_space_type = 'Discrete'
        self.mask_flag = True
        self.net_archs = net_archs
        
        #######################################################################
        ### Constraints info ##################################################
        #######################################################################
        self.stop_ep_time_step = 1000
        self.is_area_a_constraint = True
        self.is_proportion_a_constraint = True
        self.is_entrance_a_constraint = True # Ensures no walls hits the entrance cells. This always has to be True even for zero constraint mode
        self.is_entrance_adjacency_a_constraint = True # Ensures no non-lvroom rooms occupy the cells adjacent to the entrance
        self.is_entrance_lvroom_connection_a_constraint = True # Ensures the lvroom is positioned (if positioned at all) such that it becomes an adjacent cell to the entrance.
        self.adaptive_window = True
        self.is_adjacency_considered = True
        
        #######################################################################
        ### Define Scenarios ##################################################
        #######################################################################
        self.pre_train_mode = True if self.agent_name in ['BC'] else False
        self.encode_img_obs_by_vqvae_flag = False
        self.exactly_mimic_an_existing_plan = False
        self.removing_lv_room_from_action_set = True
        self.is_agent_allowed_to_create_lvroom = False
        self.randomly_create_lvroom_first = True # randomly_create_lvroom_first and lvroom_portion_range are only for the create_random_config. it allows to assign an area first to the lv_room and then distribute the rest of region to the other rooms
        self.action_mode = False
        self.adaptive_window = False if self.exactly_mimic_an_existing_plan else True
        self.zero_constraint_flag = True if 'ZC' in self.rewarding_method_name else False
        self.very_short_observation_fc_flag = False
        if self.zero_constraint_flag:
            self.is_area_a_constraint = False
            self.is_proportion_a_constraint = False
        if self.plan_config_source_name == 'create_random_config':
            self.adaptive_window = False
            self.is_adjacency_considered = False
            self.load_resnet_from_pretrained_weights = False
        if self.action_mode:
            self.is_area_a_constraint = False
            self.is_proportion_a_constraint = False
            self.is_entrance_adjacency_a_constraint = True
            self.is_entrance_cooridor_connection_a_constraint = True
            self.adaptive_window = False
            self.does_living_room_need_a_facade = True
        
        self.stop_ep_time_step = self.stop_ep_time_step if self.resolution == 'Low' else self.stop_ep_time_step * 2
        self.stop_ep_time_step = int(self.stop_ep_time_step/2) if self.zero_constraint_flag else self.stop_ep_time_step
        self.gnn_obs_method = 'embedded_image_graph' # image   embedded_image_graph dummy_vector
        
        ### Define plan info ##################################################
        if self.plan_config_source_name in ['create_random_config', 'create_fixed_config']:
            self.create_random_configs_with_fixed_n_rooms_flag = None # self.n_rooms # only for create_random_config -> set it to False, or the number of rooms you want 
            self.fixed_outline_flag = False # only for create_random_config
        
        if self.plan_config_source_name in ['load_random_config', 'load_fixed_config', 'offline_mode', 'imitation_mode']:
            plans_ = {
                4: '2024-02-03_07-20-08.979178', # 2024-02-03_07-20-08.979178
                5: '2024-02-03_12-32-29.675151',
                6: '2024-02-02_00-03-43.008442',
                7: '2024-02-03_08-21-52.677010',
                8: '2024-02-04_09-14-55.906332',
                9: '2024-02-02_21-03-07.288144',
            }
            self.plan_id_for_load_fixed_config = None #  None or plan_id plans_[4]
            self.nrows_of_env_data_csv = None # None or an integer
            self.fixed_num_rooms_for_loading = 4 # None or an integer
        
        ### Room-wall cardinality info ########################################
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
            
        ### Library and model info ############################################
        if self.action_masking_flag: self.model_last_name += 'ActionMasking'
        if self.use_redidual: self.model_last_name += 'Residual'
        if self.agent_name == 'CQL': 
            self.policy_model_last_name = self.model_last_name + 'P'
            self.q_model_last_name = self.model_last_name + 'Q'
        self.net_arch = self.net_archs[self.model_last_name]
        # self.library = 'RLlib' 
        self.model_source = 'MyCustomModel' # 'RllibCustomConfig' # 'MyCustomModel' if "Simple" not in self.model_last_name else 'RllibCustomModel' RllibCustomConfig

        ### Scenario name #####################################################
        if 'scenario_name' not in self.__dict__:
            self.scenario_name = "Scn__"
            self.scenario_name += f"{datetime.datetime.now().strftime('%Y_%m_%d_%H%M%S')}__"
            # self.scenario_name += f"{self.model_last_name}__"
            self.scenario_name += 'PTM__' if self.pre_train_mode else ''.join(s[0].upper() for s in self.plan_config_source_name.split('_'))
            self.scenario_name += f"__{self.n_rooms}Rr" if (self.plan_config_source_name == 'fixed_test_config' and not self.pre_train_mode) else ''
            # if self.is_area_a_constraint: self.scenario_name += "__Ar"
            # if self.is_proportion_a_constraint: self.scenario_name += "Pr"
            # if self.is_entrance_a_constraint: self.scenario_name += "En"
            # if self.adaptive_window: self.scenario_name += "Aw"
            # if self.is_adjacency_considered: self.scenario_name += "Ad__"
            self.scenario_name += '__' + ''.join(s[0].upper() for s in self.rewarding_method_name.split('_')) + '__'
            if self.resolution == 'High': self.scenario_name += "Hr__"
            self.scenario_name += f"{self.agent_name}"
    
        #######################################################################
        ### Hyper params ######################################################
        #######################################################################
        # ## New params
        self.oh2ind_flag = True
        self.room_facade_state_as_adj_matrix_flag = True
        self.only_upper_diagonal_adj_flag = False
        self.distance_field_flag = False
        ## for lv_mode
        self.weight_for_missing_entrance_lvroom_connection = 1
        self.lvroom_portion_range = [0.3, 0.6]
        self.lvroom_id = 11
        self.load_good_action_sequence_for_on_policy_pre_training = False
        self.load_good_action_sequence_prob = 0.0
        self.exclude_fake_rooms_from_adjacency_from_beginning = False
        
        #######################################################################
        ### Creating datase ###################################################
        #######################################################################
        self.meta_observation_type = 'dict' # 'dict' 'tuple' 'list'
        self.meta_observation_type_for_saving = 'list' 
        assert self.meta_observation_type_for_saving == 'list', "meta_observation_type_for_saving should be list"
        if 'Meta' not in self.model_last_name:
            self.meta_observation_type = 'list'
            self.meta_observation_type_for_saving = 'list'
        ## for storing the env info
        self.save_env_info_on_callback = False
        self.only_save_high_quality_env_data = False

        #######################################################################
        ### Reqward info ######################################################
        #######################################################################
        self.positive_final_reward = 1000 # self.stop_ep_time_step
        self.positive_done_reward = 100 # 0.1*self.stop_ep_time_step
        self.negative_badly_stop_reward = -200 # -0.2 * self.stop_ep_time_step
        
        self.last_good_reward_scalar = 1 # I hope this could help to prioritize adjacency over episode_len
        self.last_good_reward_threshold = 500
        self.last_good_reward_high_coeff = 3
        self.last_good_reward_low_val = 100

        self.non_accepted_negative_reward = -0.05 ## (seems 0.5 works better)
        self.zc_reward_badly_stop = -5
        self.zc_reward_bottom = 0 if self.shift_zc_reward_bottom_to_zero == 1 else -1
        self.zc_reward_up = 1
        self.zc_reward_terminal_state_factor = 1 # self.positive_final_reward / (self.zc_reward_up - self.zc_reward_bottom)
        
        self.zc_intra_episode_wa = 0.7
        self.zc_intra_episode_wp = 0.3

        self.zc_end_episode_wa = 0.3
        self.zc_end_episode_wp = 0.2
        self.zc_end_episode_we = 0.5

        self.zc_terminal_state_wa_sum = 0.3
        self.zc_terminal_state_wp_sum = 0.3
        self.zc_terminal_state_we = 0.5
               
        if self.zc_reward_weighening_mode == '15_15_15_15_40': # Balanced Topo_Oriented Geom_Oriented 15_15_15_15_40
            self.zc_terminal_state_wa_mean = 0.15
            self.zc_terminal_state_wa_std = 0.15
            self.zc_terminal_state_wp_mean = 0.15
            self.zc_terminal_state_wp_std = 0.15
            self.zc_terminal_state_we_mean = 0.4
        if self.zc_reward_weighening_mode == '05_05_05_05_80': # Balanced Topo_Oriented Geom_Oriented
            self.zc_terminal_state_wa_mean = 0.05
            self.zc_terminal_state_wa_std = 0.05
            self.zc_terminal_state_wp_mean = 0.05
            self.zc_terminal_state_wp_std = 0.05
            self.zc_terminal_state_we_mean = 0.8
        if self.zc_reward_weighening_mode == '25_25_25_25_20': # Balanced Topo_Oriented Geom_Oriented
            self.zc_terminal_state_wa_mean = 0.25
            self.zc_terminal_state_wa_std = 0.25
            self.zc_terminal_state_wp_mean = 0.15
            self.zc_terminal_state_wp_std = 0.15
            self.zc_terminal_state_we_mean = 0.2
        if self.zc_reward_weighening_mode == '02_01_02_01_94': # Balanced Topo_Oriented Geom_Oriented
            self.zc_terminal_state_wa_mean = 0.02
            self.zc_terminal_state_wa_std = 0.01
            self.zc_terminal_state_wp_mean = 0.02
            self.zc_terminal_state_wp_std = 0.01
            self.zc_terminal_state_we_mean = 0.94
            
        ## Constraints and Objective
        # Geometrical
        self.area_tolerance = 10 if self.resolution == 'Low' else 20
        self.desired_aspect_ratio = 1
        self.aspect_ratios_tolerance = 6 if self.is_proportion_a_constraint else 20

        self.area_tolerance_for_zero_constraint = 20
        self.aspect_ratios_tolerance_for_zero_constraint = 6 if self.is_proportion_a_constraint else 6
        
        if self.exactly_mimic_an_existing_plan:
            self.area_tolerance = 1
            self.aspect_ratios_tolerance = 1
            self.area_tolerance_for_zero_constraint = 1
            self.aspect_ratios_tolerance_for_zero_constraint = 1
        
        if 'create' in self.plan_config_source_name:
            self.area_tolerance = 10
            self.aspect_ratios_tolerance = 6
            self.area_tolerance_for_zero_constraint = 15
            self.aspect_ratios_tolerance_for_zero_constraint = 6

        ### fnorm Reqward #####################################################
        self.reward_concavity_factor = 8
        self.fnorm_area_factor = 10
        self.fnorm_aspect_ratio_factor = 10
        self.fnorm_edge_factor = 1

        self.fnorm_delta_inf = 0
        self.fnorm_delta_sup = 1
        self.fnorm_reward_bottom = -0.4
        self.fnorm_reward_up = 1
        self.fnorm_delta_factor = 1
        self.fnorm_reward_concavity_factor = 8

        self.fnorm_non_accepted_negative_reward = -0.01 ## (-0.01 works better) #### ### No geom tolerance and vertical scalar 1000 PPO_SpaceLayoutGym_3a8a1_00000: -0.01 #### No geom tolerance PPO_SpaceLayoutGym_d76ad_00000: -0.1, PPO_SpaceLayoutGym_6018d_00000: -0.001 # ### FNorm with geom tolerance PPO_SpaceLayoutGym_21c39_00000: -1 # PPO_SpaceLayoutGym_0024e_00000: -0.5 # PPO_SpaceLayoutGym_e3619_00000: -0.4 # PPO_SpaceLayoutGym_bf16d_00000: -0.1 # PPO_SpaceLayoutGym_4eb60_00000:-0.01 
        self.fnorm_negative_badly_stop_reward = -self.stop_ep_time_step

        self.fnorm_intra_episode_wa = 0.7
        self.fnorm_intra_episode_wp = 0.3

        self.fnorm_terminal_state_wa_mean = 0.25
        self.fnorm_terminal_state_wp_mean = 0.15
        self.fnorm_terminal_state_wa_std = 0.1
        self.fnorm_terminal_state_wp_std = 0.1
        self.fnorm_terminal_state_we_mean = 0.399
        self.fnorm_terminal_state_wl = 0.001

        ## fnorm and
        self.area_inf = 10
        self.area_sup = 220 if self.resolution == 'Low' else 440
        self.area_delta_inf = 0
        self.area_delta_sup = self.area_sup - self.area_inf
        self.aspect_ratio_inf = 1
        self.aspect_ratio_sup = 21 if self.resolution == 'Low' else 43
        self.aspect_ratio_delta_inf = 0
        self.aspect_ratio_delta_sup = self.aspect_ratio_sup - self.aspect_ratio_inf
        self.num_edges_inf = 1
        self.num_edge_sup = 20 
        self.edge_delta_inf = 0
        self.edge_delta_sup = self.num_edge_sup - self.num_edges_inf

        self.area_delta_shift = 20
        self.area_delta_scalar = 0.4
        self.area_delta_power = 0.7
        self.aspect_ratio_delta_shift = 3
        self.aspect_ratio_delta_scalar = 0.4
        self.aspect_ratio_delta_power = 4
        self.edge_delta_shift = 2
        self.edge_delta_scalar = 0.7
        self.edge_delta_power = 3
        
        self.delta_area_tolerance_for_fnorm = 5
        self.delta_aspect_ratios_tolerance_for_fnorm = 5

        ### General Reqward info ##############################################
        self.min_acceptable_area = 10 if (self.plan_config_source_name == 'fixed_test_config' and self.n_rooms == 7) else 12
        self.max_acceptable_area = 150 if self.resolution == 'Low' else 320 # TODO: 420 has to be adjusted probably
        self.min_acceptable_aspect_ratio = 1
        self.max_acceptable_aspect_ratio = 10 if self.resolution == 'Low' else 20

        self.area_diff_left = 0
        self.area_diff_right = 30
        self.aspect_ratio_diff_left = 0
        self.aspect_ratio_diff_right = 6

        self.area_diff_min = 0
        self.area_diff_max = 30
        self.aspect_ratio_diff_min = 0
        self.aspect_ratio_diff_max = 10
        
        # adjacencies
        self.edge_diff_min = 0
        self.edge_diff_max = 30 if 'Smooth' in self.rewarding_method_name else 20
        
        self.edge_diff_left = 0
        self.edge_diff_right = 10

        # common
        self.diff_min = 0
        self.diff_max = 20
        self.rew_bottom = -1 * 10
        self.rew_up = 1
        
        # smooth reward
        self.linear_reward_coeff = 50
        self.quad_reward_coeff = 2.49
        self.exp_reward_coeff = 50
        self.exp_reward_temperature = 144.7
        
        # others
        self.n_facades_blocked = 1
        self.area_diff_in_reward_flag = False #True if self.is_area_a_constraint else False
        self.proportion_diff_in_reward_flag = False #True if self.is_proportion_a_constraint else False  
        
        ## TODO:
        # if self.resolution == 'High':
        #     raise ValueError("edge_diff_max needs to be adjusted for high resolution with high number of rooms")
        
        #######################################################################
        ### Plan components ###################################################
        #######################################################################
        self.maximum_num_masked_rooms = 4
        self.num_of_facades = 4
        self.num_entrance = 1
        self.maximum_num_real_rooms = 9 if self.resolution == 'Low' else 20
        self.maximum_num_real_plus_masked_rooms = self.maximum_num_masked_rooms + self.maximum_num_real_rooms
        self.num_plan_components = self.maximum_num_real_plus_masked_rooms + self.num_of_facades + self.num_entrance
        
        # self.facade_id_min_threshold = self.maximum_num_real_plus_masked_rooms + 1 # = 14
        
        self.very_corner_cell_id = 1
        self.min_fake_room_id = 2
        self.max_fake_room_id = 5
        self.fake_room_id_range = list(range(self.min_fake_room_id, self.max_fake_room_id+1))
        
        self.north_facade_id = self.min_facade_id = 6
        self.south_facade_id = 7
        self.east_facade_id = 8
        self.west_facade_id = self.max_facade_id = 9
        self.facade_id_range = list(range(self.min_facade_id, self.max_facade_id+1))
        
        self.entrance_cell_id = 10
        
        self.min_room_id = 11
        self.max_room_id = self.min_room_id + self.maximum_num_real_rooms - 1
        self.real_room_id_range = list(range(self.min_room_id, self.max_room_id+1))
        
        self.cell_id_max = self.max_room_id + 1
        
        self.num_nodes = self.max_room_id - 1
        
        self.facade_id_to_name_dict = {self.north_facade_id: 'n',
                                       self.south_facade_id: 's',
                                       self.east_facade_id: 'e',
                                       self.west_facade_id: 'w'}
        self.facade_name_to_id_dict = {'n': self.north_facade_id,
                                       's': self.south_facade_id,
                                       'e': self.east_facade_id,
                                       'w': self.west_facade_id}
        
        self.number_of_walls_range = list(range(3, self.maximum_num_real_rooms))
        self.number_of_rooms_range = list(range(4, self.maximum_num_real_rooms+1))
        
        self.room_desired_area_dict_template = {f'room_{i}':0 for i in range(self.min_fake_room_id, self.cell_id_max)}
        self.room_achieved_area_dict_template = {f'room_{i}':0 for i in range(self.min_fake_room_id, self.cell_id_max)}
        self.room_delta_area_dict_template = {f'room_{i}':0 for i in range(self.min_fake_room_id, self.cell_id_max)}
        
        self.wall_important_points_names = ['start_of_wall', 'before_anchor', 'anchor', 'after_anchor', 'end_of_wall']
        self.wall_coords_template = {k: [-1, -1] for k in self.wall_important_points_names}
        
        self.treat_facade_as_wall = True
        
        #######################################################################
        ### Observation #######################################################
        #######################################################################
        self.num_entrance_coords_elements = 8
        self.num_masked_region_and_facades = 8
        ## TODO how the agent knows that what is the corriror and what is the living room id? we need to embed it in the observation for generalization - Done!
        self.len_feature_state_vec = 114 if self.very_short_observation_fc_flag else (4465 if self.resolution == 'Low' else 11175) # TODO : needs more cuation, particulary its dependencies
        self.len_plan_state_vec_one_hot = self.maximum_num_masked_rooms + self.num_of_facades + self.num_of_facades + self.maximum_num_real_rooms
        ## self.num_of_facades * 2 -> *2 because we need to count which facased are blocked, and on which facade the entrance located
        self.len_plan_state_vec_continous = self.maximum_num_masked_rooms * 2 + self.num_entrance_coords_elements + self.num_masked_region_and_facades # the last 8 refers to the num of elements in entrance_coords - continous here means continous and discrete, excluding one-hot
        ## self.maximum_num_masked_rooms * 2 -> *2 becasue it contains with and lengh for each of 4 masked areas
        self.len_plan_state_vec = self.len_plan_state_vec_one_hot + self.len_plan_state_vec_continous # 45
        self.len_area_state_vec = self.maximum_num_real_rooms * 2 # 18
        self.len_proportion_state_vec = self.maximum_num_real_rooms * 2 # 18
        if self.only_upper_diagonal_adj_flag:
            self.len_adjacency_state_vec = int(sum(list(range(1, self.maximum_num_real_rooms))) * 2 + self.maximum_num_real_rooms)
        else:
            if self.room_facade_state_as_adj_matrix_flag:
                self.len_adjacency_state_vec = 2*self.maximum_num_real_rooms*(self.maximum_num_real_rooms-0) + self.maximum_num_real_rooms * self.num_of_facades
            else:
                self.len_adjacency_state_vec = 2*self.maximum_num_real_rooms*(self.maximum_num_real_rooms-0) + self.maximum_num_real_rooms # 171
        
        self.len_meta_state_vec = self.len_plan_state_vec + self.len_area_state_vec + self.len_proportion_state_vec + self.len_adjacency_state_vec # 218
        
        self.len_feature_plan_state_vec = self.len_feature_state_vec + self.len_plan_state_vec
        self.len_feature_plan_area_state_vec = self.len_feature_plan_state_vec + self.len_area_state_vec
        self.len_feature_plan_area_prop_state_vec = self.len_feature_plan_area_state_vec + self.len_proportion_state_vec
        
        self.len_plan_area_state_vec = self.len_plan_state_vec + self.len_area_state_vec
        self.len_plan_area_prop_state_vec = self.len_plan_area_state_vec + self.len_proportion_state_vec
        self.len_plan_area_prop_adjacency_state_vec = self.len_plan_area_state_vec + self.len_proportion_state_vec + self.len_adjacency_state_vec
        
        self.len_state_vec = self.len_feature_plan_area_prop_state_vec + self.len_adjacency_state_vec # = 4717 for 
        
        if self.encode_img_obs_by_vqvae_flag:
            self.len_feature_state_vec = 128
            self.len_state_vec = 128 + self.len_plan_area_prop_adjacency_state_vec

        ## normalization factor
        self.plan_normalizer_factor = 1 / self.cell_id_max # it should be indeed max_x. even better to have two separate factors for length/width and coordinates
        self.wall_normalizer_factor = 1 / self.cell_id_max
        self.room_normalizer_factor = 1 / self.cell_id_max
        self.proportion_normalizer_factor = 1 / (self.aspect_ratios_tolerance + 1)
        self.cnn_obs_normalization_factor = 1 / self.cell_id_max
        
        ### Some obs config
        self.include_wall_in_area_flag = True
        self.use_areas_info_into_observation_flag = True
        self.use_edge_info_into_observation_flag = True
        self.n_shuffles_for_data_augmentation = 0 # TODO: set it to zero to avoid data augmentation.
        
        #######################################################################
        ### network params ####################################################
        #######################################################################
        self.max_rr_connections_per_room = 9
        self.max_rf_connections_per_room = 4
        self.num_corners_lw = 8
        self.num_entrance_coords = 8
        
        self.plan_desc_input_dim = 45
        self.plan_desc_embedding_dim = 16
        self.plan_desc_hidden_dim = 128 # 7 * self.plan_desc_embedding_dim = 56 
        self.plan_desc_output_dim = 128 # 32 64
        
        self.area_prop_input_dim = 36
        self.area_input_dim = 18
        self.prop_input_dim = 18
        self.area_embedding_dim = 64
        self.prop_embedding_dim = 64
        self.area_prop_hidden_dim = 128
        self.area_prop_output_dim = 128 # 32 64
        
        self.edge_input_dim = 81 + 81 + 36  
        self.edge_embedding_dim = 16
        self.edge_hidden_dim = 128 # 72
        self.edge_output_dim = 128 # 64 128
        
        self.meta_input_dim = self.plan_desc_output_dim + self.area_prop_output_dim + self.edge_output_dim
        self.meta_hidden_dim = 256 # 128 256
        self.meta_output_dim = 256 # 64 128
        
        self.image_input_dim = (1, 23*self.cnn_scaling_factor, 23*self.cnn_scaling_factor)
        self.image_output_dim = 256 # 64 128
        self.resnetx_output_dim = 256
        
        self.latent_input_dim = self.meta_output_dim + self.image_output_dim
        self.latent_hidden_dim = 2*256 # 128 256
        self.latent_output_dim = 256 # 128 256
        
        self.plan_desc_drop = 0.01
        self.area_prop_drop = 0.01
        self.edge_drop = 0.01
        self.meta_drop = 0.01
        self.mini_res_drop = 0.01
        self.img_meta_drop = 0.01
        
        self.feature_hidden_dim = 256 # 128 256
        self.feature_output_dim = 256 # 128 256

        self.actor_hidden_dim = 256 # 128 256
        self.critic_hidden_dim = 256 # 128 256
        
        self.plan_desc_one_hot_input_dim = self.len_plan_state_vec_one_hot
        self.plan_desc_continous_input_dim = self.len_plan_state_vec_continous
        
        self.feature_input_dim = self.n_channels
        
        cmc_attribute_names = ['n_channels', 'distance_field_flag', 'resolution', 'cnn_scaling_factor', 
                               'len_meta_state_vec', 'len_feature_state_vec', 'len_plan_state_vec', 'len_plan_area_state_vec', 
                               'len_plan_area_prop_state_vec', 'len_plan_state_vec_one_hot', 'len_plan_state_vec_continous', 
                               'len_area_state_vec', 'len_proportion_state_vec', 'len_adjacency_state_vec', 
                               'maximum_num_real_rooms', 'maximum_num_masked_rooms', 'num_of_facades',
                               'max_rr_connections_per_room', 'max_rf_connections_per_room', 'num_corners_lw', 'num_entrance_coords',
                               'plan_desc_input_dim', 'plan_desc_embedding_dim', 'plan_desc_hidden_dim', 'plan_desc_output_dim',
                               'area_prop_input_dim', 'area_input_dim', 'prop_input_dim', 'area_embedding_dim', 
                               'prop_embedding_dim', 'area_prop_hidden_dim', 'area_prop_output_dim', 
                               'edge_input_dim', 'edge_embedding_dim', 'edge_hidden_dim', 'edge_output_dim', 
                               'meta_input_dim', 'meta_hidden_dim', 'meta_output_dim', 
                               'image_input_dim', 'image_output_dim', 'resnetx_output_dim',
                               'latent_input_dim', 'latent_hidden_dim', 'latent_output_dim', 
                               'plan_desc_drop', 'area_prop_drop', 'edge_drop', 'meta_drop', 'mini_res_drop', 'img_meta_drop',
                               'actor_hidden_dim', 'critic_hidden_dim', 'encoding_type', 'image_encoder_type', 'load_resnet_from_pretrained_weights', 'feature_hidden_dim',
                               'activation_fn_name', 'model_first_name', 'model_last_name', 'library',
        ]
        self.custom_model_config = {}
        for att_name in cmc_attribute_names:
            self.custom_model_config[att_name] = getattr(self, att_name)
        # self.custom_model_config['use_cuda'] = self.RLLIB_NUM_GPUS > 0
        
        #######################################################################
        ### data model ########################################################
        #######################################################################
        ## Plan data
        self.min_x_background = 0
        self.max_x_background = 22
        self.min_y_background = 0
        self.max_y_background = 22
        self.min_x = 0
        self.max_x = 22
        self.min_y = 0
        self.max_y = 22
        if self.resolution == 'High':
            self.max_x = 44
            self.max_y = 44

        self.non_squared_plan_flag = False if (self.max_x == self.max_x_background and self.max_y == self.max_y_background) else True
            
        self.plan_center_coords = [self.max_x//2, self.max_y//2]
        self.plan_center_positions = [self.max_x//2, self.max_y//2]
        self.wall_1_included = True
        self.wall_1 = {'wall_1': {
            'start_of_wall': [self.min_x, self.min_y],
            'before_anchor': [self.min_x, self.max_y],
            'anchor':  [self.max_x//2, self.max_y//2],
            'after_anchor': [self.max_x, self.min_y],
            'end_of_wall':  [self.max_x, self.max_y],
            }}
        
        self.scaling_factor = 1
        self.seg_length = 2
        
        ## coords info
        self.num_corners = 4
        self.num_facades = 4
        self.corners = {'corner_00': [self.min_x, self.min_y],
                        'corner_01': [self.min_x, self.max_y],
                        'corner_10': [self.max_x, self.max_y],
                        'corner_11': [self.max_x, self.max_y]}

        self.n_rows = self.max_y + 1
        self.n_cols = self.max_x + 1
        
        if self.plan_config_source_name == 'fixed_test_config':
            from gym_floorplan.envs.fixed_scenarios_lib import get_fixed_scenario
            fixed_scenario_config = get_fixed_scenario(self.n_rooms)
            plan_id = fixed_scenario_config['plan_id']
        else:
            plan_id = None
        self.north_anchor_coord = [self.max_x//2, self.max_y]
        self.south_anchor_coord = [self.max_x//2, self.min_y] 
        if plan_id is not None:
            if plan_id == 'rplan_24_r7':
                self.south_anchor_coord = [self.max_x//2+4, self.min_y]
        self.east_anchor_coord = [self.max_x, self.max_y//2]
        self.west_anchor_coord = [self.min_x, self.max_y//2]
        self.outline_walls_coords = {f"wall_{self.north_facade_id}": {'anchor_coord': self.north_anchor_coord,
                                                                      'back_open_coord': [self.north_anchor_coord[0]-1, self.north_anchor_coord[1]],
                                                                      'front_open_coord': [self.north_anchor_coord[0]+1, self.north_anchor_coord[1]]},
                                     f"wall_{self.south_facade_id}": {'anchor_coord': self.south_anchor_coord,
                                                                      'back_open_coord': [self.south_anchor_coord[0]-1, self.south_anchor_coord[1]],
                                                                      'front_open_coord': [self.south_anchor_coord[0]+1, self.south_anchor_coord[1]]},
                                     f"wall_{self.east_facade_id}": {'anchor_coord': self.east_anchor_coord,
                                                                      'back_open_coord': [self.east_anchor_coord[0], self.east_anchor_coord[1]-1],
                                                                      'front_open_coord': [self.east_anchor_coord[0], self.east_anchor_coord[1]+1]},
                                     f"wall_{self.west_facade_id}": {'anchor_coord': self.west_anchor_coord,
                                                                      'back_open_coord': [self.west_anchor_coord[0], self.west_anchor_coord[1]-1],
                                                                      'front_open_coord': [self.west_anchor_coord[0], self.west_anchor_coord[1]+1]},
                                     }
        
        offset = 4
        self.facade_coords = {'room_n': [self.max_x/2, self.max_y+offset], 'room_s': [self.max_x/2, self.min_y-offset], 
                              'room_e': [self.max_x+offset, self.max_y/2], 'room_w': [self.min_x-offset, self.max_y/2]}
        
        self.total_area = (self.n_rows - 2) * (self.n_cols - 2)
        self.entrance_area = 0
        
        self.wall_pixel_value = 10
        self.mask_pixel_value = -100
        
        #######################################################################
        ### Actions ########################################################
        #######################################################################
        self.action_dict = {
            0: 'move_up',
            1: 'move_right',
            2: 'move_down',
            3: 'move_left',
            4: 'move_up_right',
            5: 'move_down_right',
            6: 'move_down_left',
            7: 'move_up_left',
            8: 'rotate_cw',
            9: 'rotate_ccw',
            10: 'rotate_cw_front_seg',
            11: 'rotate_ccw_front_seg',
            12: 'rotate_cw_back_seg',
            13: 'rotate_ccw_back_seg',
            14: 'no_action',
            15: 'flip_x',
            16: 'flip_y',
            17: 'grow_front_seg',
            18: 'cut_front_seg',
            19: 'grow_back_seg',
            20: 'cut_back_seg',
        }

        if self.env_planning == 'One_Shot':
            self.wall_lib = [
                [[ 1, 0], [ 0, 0], [ 0, 1]], # 0 # |__ 
                [[ 0, 1], [ 0, 0], [-1, 0]], # 1 # __|
                [[-1, 0], [ 0, 0], [ 0,-1]], # 2 # --|
                [[ 0,-1], [ 0, 0], [ 1, 0]], # 3 # |--
                
                [[-1, 0], [ 0, 0], [ 1, 0]], # 4 # __
                [[ 0,-1], [ 0, 0], [ 0, 1]], # 5 # |
                ]
            self.wall_lib_length = len(self.wall_lib)
            self.wall_type = {0: 'angeled', 1: 'angeled', 2: 'angeled', 3: 'angeled',
                              4: 'horizental', 5: 'vertical'}
            
            self.room_size_category = {0: 'large', 
                                       1: 'medium',
                                       2: 'small'}
            
            self.learn_room_size_category_order_flag = False if self.learn_room_order_directly else True
            assert (self.learn_room_size_category_order_flag != self.learn_room_order_directly), "learn_room_size_category_order_flag and learn_room_order_directly cannot be True simultanously!"
            if self.learn_room_size_category_order_flag:
                self.zero_index_action_size = len(self.room_size_category)
            elif self.learn_room_order_directly:
                if self.plan_config_source_name == 'fixed_test_config':
                    self.zero_index_action_size = self.n_rooms # n_rooms maximum_num_real_rooms
                else:
                    self.zero_index_action_size = self.maximum_num_real_rooms 
                
                if self.lv_mode and self.removing_lv_room_from_action_set:
                    self.zero_index_action_size -= 1 # -1 bc the agent never selects the lvroom to draw
                        
            
            # self.action_to_acts_tuple_dic = {}
            # # self.acts_tuple_to_action_dic = {}
            # a = 0
            # for c in range(self.zero_index_action_size):
            #     for i in range(len(self.wall_lib)):
            #         for j in range(2, self.max_x-1, 2):
            #             for k in range(2, self.max_y-1, 2):
            #                 self.action_to_acts_tuple_dic.update({a: (c, i, j, k)})
            #                 # self.acts_tuple_to_action_dic.update({(c, i, j, k): a})
            #                 a += 1
            
            self.n_actions = self.zero_index_action_size * len(self.wall_lib) * ((self.max_x-1)//2) * ((self.max_y-1)//2) # len(self.action_to_acts_tuple_dic) 
            
        ### Transformations
        self.translation_mat_dict = {'move_up': [0, 1],
                                     'move_right': [1, 0],
                                     'move_down': [0, -1],
                                     'move_left': [-1, 0],
                                     'move_up_right': [1, 1],
                                     'move_down_right': [1, -1],
                                     'move_down_left': [-1, -1],
                                     'move_up_left': [-1, 1]}

        self.flip_mat_dict = {'flip_x': [-1, 1],
                              'flip_y': [1, -1]}

        self.rotation_mat_dict = {'rotate_cw': -np.pi / 2,
                                  'rotate_ccw': np.pi / 2,
                                  'rotate_cw_front_seg': -np.pi / 2,
                                  'rotate_ccw_front_seg': np.pi / 2,
                                  'rotate_cw_back_seg': -np.pi / 2,
                                  'rotate_ccw_back_seg': np.pi / 2}
        
        
        #######################################################################
        ### Gym env info ######################################################
        #######################################################################
        ## set act_dim and obs_dim
        self.act_dim = self.n_actions
        self.action_space_type = 'discrete'
        
        if self.net_arch == 'Fc':
            self.obs_dim = self.len_feature_state_vec
            
        elif self.net_arch == 'Cnn':
            # self.obs_dim = ( self.n_channels, (self.n_rows+2) * self.cnn_scaling_factor, (self.n_cols+2) * self.cnn_scaling_factor )
            self.obs_dim = ( self.n_channels, (self.n_rows+0) * self.cnn_scaling_factor, (self.n_cols+0) * self.cnn_scaling_factor )
            
        elif self.net_arch == 'MetaFc':
            self.obs_fc_dim = self.len_feature_state_vec
            self.obs_meta_dim = self.len_plan_area_prop_state_vec
            self.obs_dim = self.len_state_vec
            
        elif self.net_arch == 'MetaCnn':
            self.obs_cnn_dim = (self.n_channels, (self.n_rows+2) * self.cnn_scaling_factor, (self.n_cols+2) * self.cnn_scaling_factor)
            self.obs_meta_dim = self.len_plan_area_prop_state_vec
            self.obs_dim = tuple((self.obs_cnn_dim, self.obs_meta_dim))
        
        elif self.net_arch == 'Gnn':
            pass
        else:
            raise NotImplementedError
            
        #######################################################################
        ### Rendering #########################################################
        #######################################################################
        self.show_render_flag = False
        self.so_thick_flag = False
        self.show_graph_on_plan_flag = False
        self.graph_line_style = 'bezier' # bezier hanging stright
        self.show_room_dots_flag = False
        self.save_render_flag = False
        self.show_edges_for_fake_rooms_flag = False
        self.random_agent_flag = False
        self.color_map = color_map
    
        #######################################################################
        ### Directories and paths #############################################
        #######################################################################
        self.root_dir = root_dir
        print('fenv_config.py -> housing_design_dir:', root_dir)
        self.storage_dir = os.path.join(self.root_dir, 'storage_nobackup')
        self.rnd_agents_storage_dir = os.path.join(self.storage_dir, "rnd_agents_storage")
        self.ana_agents_storage_dir = os.path.join(self.storage_dir, "ana_agents_storage")
        self.rlb_agents_storage_dir = os.path.join(self.storage_dir, "rlb_agents_storage")
        
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)
        if not os.path.exists(self.rnd_agents_storage_dir):
            os.makedirs(self.rnd_agents_storage_dir)
        if not os.path.exists(self.ana_agents_storage_dir):
            os.makedirs(self.ana_agents_storage_dir)
            
        ## dataset path
        self.rnd_scenario_name = 'Scn__2024_02_02_2222__CRC__XRr__EnZSQR__RND__Stable'
        self.ana_scenario_name = 'Scn__2024_02_20_2020__MetaCnnNet__ZC_Smooth_Log_Reward__all_plans__ANA__Stable' ##'Scn__2024_01_19_1444__MetaCnnNet__ZC_Smooth_Log_Reward__all_plans__ANA__Stable' 
        
        ## from random agent
        if 'create' in self.plan_config_source_name:
            self.plan_path_cc = os.path.join(self.rnd_agents_storage_dir, self.scenario_name, 'plan_path_cc.csv')
        else:
            if 'bc' in self.phase:
                phase = self.phase.split('_')[1]
                
            plan_name = "plans_test.csv" if self.agent_name == 'PPO' else f"plans_valid.csv" 
            self.plan_path = os.path.join(self.rnd_agents_storage_dir, self.rnd_scenario_name, f"plans_test.csv")
            self.plan_path_cc = os.path.join(self.rnd_agents_storage_dir, self.rnd_scenario_name, 'plans_cc.csv')

            # self.plan_path = os.path.join(self.rnd_agents_storage_dir, 'plans_4-9.csv')
            # self.plan_path = os.path.join(self.rnd_agents_storage_dir, 'Scn__2024_01_26_1716__CRC__XRr__ArPrEnSLR__RND/plans_cc.csv')
            # self.plan_path = "/home/rdbt/ETHZ/dbt_python/housing_design/storage_nobackup/rnd_agents_storage/Scn__2024_01_31_1447__CRC__XRr__EnZSQR__RND/plans_cc.csv"
        ## from ana agent
        self.ana_scenario_dir = os.path.join(self.ana_agents_storage_dir, self.ana_scenario_name)


        #######################################################################
        ### rlb pretrained agents #############################################
        #######################################################################
        if self.load_agent_from_pre_train_flag: # type: ignore
            self.old_project_name = 'Prj__2024_04_10_1800__rlb__bc2ppo__2nd_paper'
            self.old_scenario_name = 'Scn__2024_04_10_1805__PTM__ZSLR__BC'
            self.old_agent_first_name = 'BC_2024-04-10_18-05-17'
            self.old_agent_run_name = 'BC_SpaceLayoutGym_1fbb6_00000_0_2024-04-10_18-05-18'
            self.old_checkpoint_number = 'checkpoint_000019'
            self.old_agent_name = 'BC'

            self.old_scenario_dir = os.path.join(self.rlb_agents_storage_dir, 'tunner', self.old_project_name, self.old_scenario_name)
            self.old_agent_dir = os.path.join(self.old_scenario_dir, self.old_agent_first_name)
            self.trained_checkpoint_dir = os.path.join(self.old_agent_dir, self.old_agent_run_name, self.old_checkpoint_number)
            
            self.chkpt_path_ = self.trained_checkpoint_dir
            
            trained_agent_dir = os.path.join(self.rlb_agents_storage_dir, 'tunner', self.old_project_name, self.old_scenario_name, self.old_agent_first_name)
            self.old_model_path = trained_agent_dir
            # all_checkpoints = [ch for ch in os.listdir(trained_agent_dir) if 'checkpoint_' in ch]
            # latest_checkpoint = np.sort(all_checkpoints)[-1]
            # chkpt_path_ = os.path.join(trained_agent_dir, latest_checkpoint)
            # self.old_model_path = chkpt_path_

        if self.load_agent_from_pre_train_flag:
            self.custom_model_config.update({
                'old_scenario_dir': self.old_scenario_dir,
                'pretrained_modelsd_path': os.path.join(self.old_scenario_dir, 'model', 'modelsd.pt'),
                'pretrained_model_path': os.path.join(self.old_scenario_dir, 'model', 'model.pt'),
            })
            
        
    def get_config(self):
        return self.__dict__




#%% This is only for testing and debugging
if __name__ == '__main__':
    self = LaserWallConfig()
    fenv_config = self.get_config()
