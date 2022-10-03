# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 00:58:16 2021

@author: RK
"""

# %%
import os
from pathlib import Path

agent_dir = os.path.join(os.path.realpath(Path(os.getcwd()).parents[2]), 'agents_floorplan') \
    if __name__ == "__main__" else os.path.realpath(Path(os.getcwd()))


# %%
import ast
import numpy as np
import pandas as pd
from gym_floorplan.envs.fenv_scenarios import FEnvScenarios


# %%
class LaserWallConfig:
    def __init__(self):
        
        ## Get the scenario config and update fenv_config
        scenarios_dict = FEnvScenarios().get_scenarios()
        for k, v in scenarios_dict.items():
            setattr(self, k, v)
        
        self.load_valid_plans_flag = True if self.plan_config_source_name in ['load_fixed_config', 'load_random_config', 'offline_mode'] else False
        
            
        if self.only_final_reward_flag:
            self.is_adjacency_considered = True # 
            
        if (self.is_area_considered and self.is_proportion_considered and self.is_adjacency_considered):
            self.stop_time_step = 2000
        elif (self.is_area_considered and self.is_adjacency_considered):
            self.stop_time_step = 1000
        elif (self.is_area_considered and self.is_proportion_considered):
            self.stop_time_step = 1000
        else:
            self.stop_time_step = 1000
            
        self.number_of_walls_range = list(range(3, 9))
        self.number_of_rooms_range = list(range(4, 10))
        
        self.positive_final_reward = 100

        self.area_tolerance = 5

        self.min_desired_proportion = 0.5
        self.max_desired_proportion = 5
        self.desired_aspect_ratio = 1
        self.aspect_ratios_tolerance = 3
        
        self.zero_accepted_reward = 0
        self.positive_done_reward = 10
        self.positive_action_reward = 1
        self.negative_action_reward = -1
        self.negative_wrong_area_reward = -1
        self.negative_wrong_proportion_reward = -1
        self.negative_wrong_area_and_proportion_reward = -1
        self.negative_rejected_by_room_reward = -1
        self.negative_rejected_by_canvas_reward = -1
        self.negative_wrong_blocked_cells_reward = -1
        self.negative_wrong_odd_anchor_coord = -1
        self.negative_missing_room = -1
        
        if self.reward_shaping_flag:
            self.positive_done_reward = 50
            self.positive_action_reward = 10
            self.area_tolerance = 4

        self.show_render_flag = False
        self.so_thick_flag = False
        self.show_graph_on_plan_flag = False
        self.show_room_dots_flag = False
        self.save_render_flag = False
        
        self.fenv_config_verbose = 0
        self.only_straight_vertical_walls = False
        self.save_fixed_walls_sets_flag = False
        
        self.learn_room_size_category_order_flag = True
        
        self.random_agent_flag = False
        


        self.len_state_vec_for_walls = 2340
        self.len_state_vec_for_rooms = self.maximum_num_real_rooms * 2 # = 18
        self.len_state_vec_for_walls_rooms = self.len_state_vec_for_walls + self.len_state_vec_for_rooms # =2358
        self.len_state_vec_for_edges = int(self.maximum_num_real_rooms * (self.maximum_num_real_rooms - 1) / 2) * 2 # =72
        self.len_state_vec = self.len_state_vec_for_walls_rooms + self.len_state_vec_for_edges # =2430



        ##### data model
        ## Plan data
        self.min_x = 0
        self.max_x = 20
        self.min_y = 0
        self.max_y = 20
        if self.resolution == 'High':
            self.max_x = 40
            self.max_y = 40
            
        self.n_channels = 2
        self.scaling_factor = 1

        self.corners = {'corner_00': [self.min_x, self.min_y],
                        'corner_01': [self.min_x, self.max_y],
                        'corner_10': [self.max_x, self.max_y],
                        'corner_11': [self.max_x, self.max_y]}

        self.n_rows = self.max_y + 1
        self.n_cols = self.max_x + 1
        
        self.total_area = self.n_rows * self.n_cols
        
        self.seg_length = 2
        
        self.wall_pixel_value = 10
        self.mask_pixel_value = -100
        
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
            self.acts_tuple_to_action_dic = {}
            a = 0
            for c in range(len(self.room_size_category)):
                for i in range(len(self.wall_lib)):
                    for j in range(2, self.max_x-1, 2):
                        for k in range(2, self.max_y-1, 2):
                            self.action_to_acts_tuple_dic.update({a: (c, i, j, k)})
                            self.acts_tuple_to_action_dic.update({(c, i, j, k): a})
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

        ### CNN obs
        self.color_map = {-3: 'gray',
                          -2: 'lime',  # (0,0,0), #
                          -1: 'white',  # (255,255,255), #
                          0: 'black',  # (255,0,0), #
                          1: 'red',  # (0,255,0), #
                          2: 'green',  # (0,0,255), #
                          3: 'blue',  # (255,255,0), #
                          4: 'purple',  # (0,255,255), #
                          5: 'magenta',  # (255,0,255), #
                          6: 'maroon',  # (128,0,0), #
                          7: 'yellow',  # (128,128,0), #
                          8: 'cyan',  # (0,128,0), #
                          9: 'olive',  # (128,0,128), #
                          10: 'teal',  # (0,128,128), #
                          11: 'navy',  # (0,0,128), #
                          }

    
        ### Directories and paths
        self.root_dir = os.path.normpath("/home/rdbt/ETHZ/dbt_python/housing_design_making-general-env/")
        
        agent_dir = os.path.join(self.root_dir, 'agents_floorplan')

        self.storage_dir = os.path.join(self.root_dir, 'agents_floorplan/storage')
        
        #assert os.path.join(self.root_dir, 'agents_floorplan') == agent_dir
        
        self.sb_tb_log_dir = os.path.join(self.storage_dir, 'sb_tb_log')
        if not os.path.exists(self.sb_tb_log_dir):
            os.makedirs(self.sb_tb_log_dir)
        
        self.generated_plans_dir = os.path.join(self.root_dir, "agents_floorplan/storage/generated_plans")
        if not os.path.exists(self.generated_plans_dir):
            os.makedirs(self.generated_plans_dir)
            
        self.offline_datasets_dir = os.path.join(self.storage_dir, 'offline_datasets')
        if not os.path.exists(self.offline_datasets_dir):
            os.makedirs(self.offline_datasets_dir)
            
        self.context_mdoel_path = os.path.join(self.offline_datasets_dir, 'model.pt')    
            
        self.scenario_dir_for_storing_data = f"{self.generated_plans_dir}" # "/{self.scenario_name}"
        if not os.path.exists(self.scenario_dir_for_storing_data):
            os.makedirs(self.scenario_dir_for_storing_data)
            
        if self.load_valid_plans_flag:
            if self.is_area_considered and self.is_adjacency_considered and self.is_proportion_considered:
                if self.resolution == 'High':
                    self.plan_path = f"{self.generated_plans_dir}/plan_values__{self.room_count_str}__area_proportion_adjacency_hr.csv"
                else:
                    self.plan_path = f"{self.generated_plans_dir}/plan_values__{self.room_count_str}__area_proportion_adjacency_train.csv"
            elif self.is_area_considered and self.is_adjacency_considered:
                if self.resolution == 'High':
                    self.plan_path = f"{self.generated_plans_dir}/plan_values__{self.room_count_str}__area_adjacency_hr.csv"
                else:
                    self.plan_path = f"{self.generated_plans_dir}/plan_values__{self.room_count_str}__area_adjacency_train_100.csv"
            elif self.is_area_considered and self.is_proportion_considered:
                if self.resolution == 'High':
                    self.plan_path = f"{self.generated_plans_dir}/plan_values__{self.room_count_str}__area_proportion_hr.csv"
                else:
                    self.plan_path = f"{self.generated_plans_dir}/plan_values__{self.room_count_str}__area_proportion.csv"
            elif self.is_area_considered:
                if self.resolution == 'High':
                    self.plan_path = f"{self.generated_plans_dir}/plan_values__{self.room_count_str}__area_hr.csv"
                else:
                    self.plan_path = f"{self.generated_plans_dir}/plan_values__{self.room_count_str}__area.csv"
            else:
                if self.resolution == 'High':
                    self.plan_path = f"{self.generated_plans_dir}/plan_values__{self.room_count_str}__proportion_hr.csv"
                else:
                    self.plan_path = f"{self.generated_plans_dir}/plan_values__{self.room_count_str}__proportion.csv"
            
        #### Other dirs 
        # self.generated_gif_dir = f"{self.storage_dir}/generated_gif"
        # if not os.path.exists(self.generated_gif_dir):
        #     os.makedirs(self.generated_gif_dir)
        
        # self.area_plots_dir = f"{self.storage_dir}/area_plots"
        # if not os.path.exists(self.area_plots_dir):
        #     os.makedirs(self.area_plots_dir)
    
        # self.area_gif_dir = f"{self.storage_dir}/area_gif"
        # if not os.path.exists(self.area_gif_dir):
        #     os.makedirs(self.area_gif_dir)
    

    def get_config(self):
        return self.__dict__


# %%
if __name__ == '__main__':
    self = LaserWallConfig()
    fenv_config = self.get_config()