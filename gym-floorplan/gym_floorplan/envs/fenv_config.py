# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 00:58:16 2021

@author: Reza Kakooee
"""


# %%
import os
from pathlib import Path

# agent_dir = os.path.join(os.path.realpath(Path(os.getcwd()).parents[2]), 'agents_floorplan') \
#     if __name__ == "__main__" else os.path.realpath(Path(os.getcwd()))

root_dir = os.path.realpath(Path(os.getcwd()).parents[2]) if __name__ == "__main__" else os.path.realpath(Path(os.getcwd()).parents[0])
# root_dir = os.path.normpath("/home/rdbt/ETHZ/dbt_python/housing_design_making-general-env/")


# %%
# import ast
import numpy as np
# import pandas as pd
from gym_floorplan.envs.fenv_scenarios import FEnvScenarios




# %%
class LaserWallConfig:
    def __init__(self, agent_name='PPO', phase='train', scenarios_dict=None):
        
        ### Env info
        self.env_name = 'DOLW-v0' 
        self.env_type = 'Single'
        self.env_planning = 'One_Shot'
        self.env_space = self.action_space_type = 'Discrete'
        self.mask_flag = True
        
        self.save_env_info_on_callback = False
        
        
        ## Get the scenario config and update fenv_config
        if scenarios_dict is None: scenarios_dict = FEnvScenarios().get_scenarios()
        for k, v in scenarios_dict.items():
            setattr(self, k, v)
            
        
        ## Reward
        self.positive_final_reward = 1000 # self.stop_ep_time_step
        self.positive_done_reward = 100 # 0.1*self.stop_ep_time_step
        self.negative_badly_stop_reward = -200 # -0.2 * self.stop_ep_time_step
        
        self.weight_for_entrance_edge_diff_in_reward = 1
        self.weight_for_missing_corridor_living_room_connection = 10
        
        self.last_good_reward_scalar = 1 # I hope this could help to prioritize adjacency over episode_len
        self.last_good_reward_threshold = 500
        self.last_good_reward_high_coeff = 3
        self.last_good_reward_low_val = 100
        
        
        ## Constraints
        self.area_tolerance = 15 if self.resolution == 'Low' else 20
        
        self.n_facades_blocked = 1
        
        # self.min_desired_proportion = 0.5
        # self.max_desired_proportion = 5
        self.desired_aspect_ratio = 1
        self.aspect_ratios_tolerance = 4
            
        
        ## Objective
        self.area_diff_in_reward_flag = False #True if self.is_area_considered else False
        self.proportion_diff_in_reward_flag = False #True if self.is_proportion_considered else False  
        
        self.edge_diff_min = 0
        self.edge_diff_max = 30 if 'Smooth' in self.rewarding_method_name else 20
        
        self.linear_reward_coeff = 50
        self.quad_reward_coeff = 2.49
        self.exp_reward_coeff = 50
        self.exp_reward_temperature = 144.7
        
        
        ## Plan components
        self.maximum_num_masked_rooms = 4
        self.num_of_facades = 4
        self.num_entrance = 1
        self.maximum_num_real_rooms = 9 
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
        
        self.number_of_walls_range = list(range(3, 9))
        self.number_of_rooms_range = list(range(4, 10))
        
        # self.wall_alphabetic = [f"wall_{chr(i)}" for i in range(ord('A'), ord('A')+self.cell_id_max)]
        # self.wall_numeric = [f"wall_{i}" for i in range(1, self.cell_id_max)]
        
        # self.waroom_alphabetic = [f"waroom_{chr(i)}" for i in range(ord('A'), ord('A')+self.cell_id_max)]
        # self.waroom_numeric = [f"waroom_{i}" for i in range(1, self.cell_id_max)]
        
        # self.room_alphabetic = [f"room_{chr(i)}" for i in range(ord('A'), ord('A')+self.cell_id_max)]
        # self.room_numeric = [f"room_{i}" for i in range(1, self.maximum_num_real_plus_masked_rooms+1)]
        
        self.room_desired_area_dict_template = {f'room_{i}':0 for i in range(self.min_fake_room_id, self.cell_id_max)}
        self.room_achieved_area_dict_template = {f'room_{i}':0 for i in range(self.min_fake_room_id, self.cell_id_max)}
        self.room_delta_area_dict_template = {f'room_{i}':0 for i in range(self.min_fake_room_id, self.cell_id_max)}
        
        self.wall_important_points_names = ['start_of_wall', 'before_anchor', 'anchor', 'after_anchor', 'end_of_wall']
        self.wall_coords_template = {k: [-1, -1] for k in self.wall_important_points_names}
        
        self.treat_facade_as_wall = True
        
        
        ## Render
        self.show_render_flag = False
        self.so_thick_flag = False
        self.show_graph_on_plan_flag = False
        self.graph_line_style = 'hanging' # bezier hanging stright
        self.show_room_dots_flag = False
        self.save_render_flag = False
        self.show_edges_for_fake_rooms_flag = False
        
        self.learn_room_size_category_order_flag = True
        
        self.random_agent_flag = False
        
        self.phase = phase

        
        ## Observation 
        ## TODO how the agent knows that what is the corriror and what is the living room id? we need to embed it in the observation for generalization - Done!
        # self.len_state_vec_for_walls = 4465 # 4465 # 2340
        # self.len_state_vec_for_rooms = (self.num_plan_components + 0) * 3 #self.maximum_num_real_plus_masked_rooms * 3 # = 57 # 39
        # self.len_state_vec_for_walls_rooms = self.len_state_vec_for_walls + self.len_state_vec_for_rooms # = 4522 # 2389
        # self.len_state_vec_for_edges = (self.num_plan_components + 0)**2 # 361 #289
        # self.len_state_vec = self.len_state_vec_for_walls_rooms + self.len_state_vec_for_edges # = 4883 # 2678
        self.len_feature_state_vec = 4465
        # self.len_plan_state_vec = self.maximum_num_masked_rooms + self.num_of_facades * 2 + self.maximum_num_real_rooms * 2 + self.maximum_num_masked_rooms * 2 + 8 * 1 # the last 8 refers to the num of elements in entrance_coords
        self.len_plan_state_vec_one_hot = self.maximum_num_masked_rooms + self.num_of_facades * 2 + self.maximum_num_real_rooms * 3
        self.len_plan_state_vec_continous =  + self.maximum_num_masked_rooms * 2 + 8 * 1 # the last 8 refers to the num of elements in entrance_coords
        self.len_plan_state_vec = self.len_plan_state_vec_one_hot + self.len_plan_state_vec_continous
        self.len_area_state_vec = self.num_plan_components * 2
        self.len_proportion_state_vec = self.maximum_num_real_rooms * 2
        self.len_adjacency_state_vec = np.sum(list(range(1, self.maximum_num_real_rooms))) * 2 + self.maximum_num_real_rooms
        
        self.len_meta_state_vec = self.len_plan_state_vec + self.len_area_state_vec + self.len_proportion_state_vec + self.len_adjacency_state_vec
        
        self.len_feature_plan_state_vec = self.len_feature_state_vec + self.len_plan_state_vec
        self.len_feature_plan_area_state_vec = self.len_feature_plan_state_vec + self.len_area_state_vec
        self.len_feature_plan_area_prop_state_vec = self.len_feature_plan_area_state_vec + self.len_proportion_state_vec
        
        self.len_plan_area_state_vec = self.len_plan_state_vec + self.len_area_state_vec
        self.len_plan_area_prop_state_vec = self.len_plan_area_state_vec + self.len_proportion_state_vec
        self.len_plan_area_prop_adjacency_state_vec = self.len_plan_area_state_vec + self.len_proportion_state_vec + self.len_adjacency_state_vec
        
        self.len_state_vec = self.len_feature_plan_area_prop_state_vec + self.len_adjacency_state_vec # = 4672
        

        ## normalization factor
        self.plan_normalizer_factor = 1 / self.cell_id_max # it should be indeed max_x. even better to have two separate factors for length/width and coordinates
        self.wall_normalizer_factor = 1 / self.cell_id_max
        self.room_normalizer_factor = 1/self.cell_id_max
        self.cnn_obs_normalization_factor = 1 / self.cell_id_max
        self.proportion_normalizer_factor = 1. / self.aspect_ratios_tolerance
        
        
        ##### data model
        ## Plan data
        self.min_x = 0
        self.max_x = 22
        self.min_y = 0
        self.max_y = 22
        if self.resolution == 'High':
            self.max_x = 44
            self.max_y = 44
            
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
        self.corners = {'corner_00': [self.min_x, self.min_y],
                        'corner_01': [self.min_x, self.max_y],
                        'corner_10': [self.max_x, self.max_y],
                        'corner_11': [self.max_x, self.max_y]}

        self.n_rows = self.max_y + 1
        self.n_cols = self.max_x + 1
        
        self.north_anchor_coord = [self.max_x//2, self.max_y]
        self.south_anchor_coord = [self.max_x//2, self.min_y]
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
        
        
        ### Some obs config
        self.include_wall_in_area_flag = True
        self.use_areas_info_into_observation_flag = True
        self.use_edge_info_into_observation_flag = True
        
        self.n_shuffles_for_data_augmentation = 0 # TODO: set it to zero to avoid data augmentation.
        
        
        ### Actions
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
                [[ 1, 0], [ 0, 0], [ 0, 1]],
                [[ 0, 1], [ 0, 0], [-1, 0]],
                [[-1, 0], [ 0, 0], [ 0,-1]],
                [[ 0,-1], [ 0, 0], [ 1, 0]],
                
                [[-1, 0], [ 0, 0], [ 1, 0]],
                [[ 0,-1], [ 0, 0], [ 0, 1]],
                ]
            
            self.wall_type = {0: 'angeled', 1: 'angeled', 2: 'angeled', 3: 'angeled',
                              4: 'horizental', 5: 'vertical'}
            
            self.room_size_category = {0: 'large', 
                                       1: 'medium',
                                       2: 'small'}
            
            self.action_to_acts_tuple_dic = {}
            # self.acts_tuple_to_action_dic = {}
            a = 0
            for c in range(len(self.room_size_category)):
                for i in range(len(self.wall_lib)):
                    for j in range(2, self.max_x-1, 2):
                        for k in range(2, self.max_y-1, 2):
                            self.action_to_acts_tuple_dic.update({a: (c, i, j, k)})
                            # self.acts_tuple_to_action_dic.update({(c, i, j, k): a})
                            a += 1
            
            self.n_actions = len(self.action_to_acts_tuple_dic) 
            
            self.n_actions = len(self.action_to_acts_tuple_dic) 
            
            
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
            

        ### CNN obs
        self.color_map = {
                         0: 'cyan', #'blue', 
                         
                         1: 'dimgray', #'red', 
                         2: 'gray', #'magenta', 
                         3: 'gainsboro', #'darkblue', 
                         4: 'silver', #'darkred', 
                         
                         5: 'darkblue', 
                         6: 'darkgreen', # 'mediumvioletred', 
                         7: 'darkred', # 'orange', 
                         8: 'pink', # 'magenta', #'aqua', 
                         9: 'darkseagreen', # 'darkorchid', 
                         10: 'darkorchid', # 'skyblue', 
                         11: 'blue', # 'pink', 
                         12: 'darkviolet', # 'gold', 
                         13: 'darkmagenta', # 'beige', 
                         
                         14: 'lime', #'silver', # white
                         
                         15: 'antiquewhite', # 'yellow',
                         16: 'antiquewhite', # 'purple',  navajowhite
                         17: 'antiquewhite', # 'teal',  blanchedalmond
                         18: 'antiquewhite', # 'violet',  papayawhip
                         19: 'moccasin', # chocolate
                         
                         20: 'hotpink',
                         }
    
    
        ### Directories and paths
        self.root_dir = root_dir
        print('fenv_config.py -> housing_design_dir:', root_dir)
        self.storage_dir = os.path.join(self.root_dir, 'storage')
        self.rnd_agents_storage_dir = os.path.join(self.storage_dir, "rnd_agents_storage")
        self.plan_path = os.path.join(self.rnd_agents_storage_dir, 'plans.csv')
        self.ana_agents_storage_dir = os.path.join(self.storage_dir, "ana_agents_storage")
        self.rl_agents_storage_dir = os.path.join(self.storage_dir, "rl_agents_storage")
        self.rlb_agents_storage_dir = os.path.join(self.storage_dir, "rlb_agents_storage")
        self.rlb_agents_tunner_dir = os.path.join(self.rlb_agents_storage_dir, "tunner")
        self.rlb_agents_trainer_dir = os.path.join(self.rlb_agents_storage_dir, "trainer")
        self.off_agents_storage_dir = os.path.join(self.storage_dir, "off_agents_storage")
        self.trl_agents_storage_dir = os.path.join(self.storage_dir, "trl_agents_storage")
        
        if phase == "very_first_debug":
            if not os.path.exists(self.storage_dir):
                os.makedirs(self.storage_dir)
            if not os.path.exists(self.rnd_agents_storage_dir):
                os.makedirs(self.rnd_agents_storage_dir)
            if not os.path.exists(self.ana_agents_storage_dir):
                os.makedirs(self.ana_agents_storage_dir)
            if not os.path.exists(self.rl_agents_storage_dir):
                os.makedirs(self.rl_agents_storage_dir)
            if not os.path.exists(self.rlb_agents_storage_dir):
                os.makedirs(self.rlb_agents_storage_dir)
            if not os.path.exists(self.rlb_agents_tunner_dir):
                os.makedirs(self.rlb_agents_tunner_dir)
            if not os.path.exists(self.rlb_agents_trainer_dir):
                os.makedirs(self.rlb_agents_trainer_dir)
            if not os.path.exists(self.off_agents_storage_dir):
                os.makedirs(self.off_agents_storage_dir)
            if not os.path.exists(self.trl_agents_storage_dir):
                os.makedirs(self.trl_agents_storage_dir)
            
        # self.offline_datasets_dir = os.path.join(self.storage_dir, 'offline_datasets')
        # if not os.path.exists(self.offline_datasets_dir):
        #     os.makedirs(self.offline_datasets_dir)
        # self.context_mdoel_path = os.path.join(self.offline_datasets_dir, 'model.pt')    



    def get_config(self):
        return self.__dict__




# %%
if __name__ == '__main__':
    self = LaserWallConfig()
    fenv_config = self.get_config()
