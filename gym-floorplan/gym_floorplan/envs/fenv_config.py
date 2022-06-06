# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 00:58:16 2021

@author: Reza Kakooee
"""

# %%
import os
from pathlib import Path

agent_dir = os.path.join(os.path.realpath(Path(os.getcwd()).parents[2]), 'agents_floorplan') \
    if __name__ == "__main__" else os.path.realpath(Path(os.getcwd()))




# %%
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
        
        self.custom_model_flag = True if self.net_arch in ['fccnn', 'cnnfc'] else False
        self.load_valid_plans_flag = True if self.plan_config_source_name in ['load_fixed_config', 'load_random_config'] else False
        
        if self.fixed_fc_observation_space:
            self.num_of_fixed_walls_for_masked = 14
        
        self.nrows_of_env_data_csv = 650
        self.stop_time_step = 1000
        
        self.min_desired_proportion = 0.5
        self.max_desired_proportion = 2
        self.area_tolerance = 0
        self.positive_done_reward = 10
        self.positive_action_reward = 1
        self.negative_action_reward = -1
        self.negative_wrong_area_reward = -1
        self.negative_rejected_by_room_reward = -1
        self.negative_rejected_by_canvas_reward = -1
        if self.reward_shaping_flag:
            self.positive_done_reward = 50
            self.positive_action_reward = 10
            self.area_tolerance = 4

        self.save_last_walls_falg = False
        self.shuffle_walls_sequence = False
        self.wall_correction_method = 'sequential'
        self.show_render_flag = False
        self.save_render_flag = False
        self.fenv_config_verbose = 0
        self.save_valid_plans_flag = True
        self.only_straight_vertical_walls = False
        self.number_of_fixed_walls_sets = False
        self.generate_fixed_walls_sets_flag = False
        self.save_fixed_walls_sets_flag = False
        self.load_fixed_walls_sets_flag = False

        ##### data model
        ## Plan data
        self.min_x = 0
        self.max_x = 20
        self.min_y = 0
        self.max_y = 20
        if 'High_Resolution_Plan' in self.scenario_name:
            self.max_x = 100
            self.max_y = 100
            
        self.n_channels = 1
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

        if self.only_straight_vertical_walls:
            self.action_dict = {
                0: 'move_right',
                1: 'move_left',
                2: 'no_action',
            }
            
        if self.env_planning == 'One_Shot':
            self.wall_set = [
                [[ 1, 0], [ 0, 0], [ 0, 1]],
                [[ 0, 1], [ 0, 0], [-1, 0]],
                [[-1, 0], [ 0, 0], [ 0,-1]],
                [[ 0,-1], [ 0, 0], [ 1, 0]],
                
                [[-1, 0], [ 0, 0], [ 1, 0]],
                [[ 0,-1], [ 0, 0], [ 0, 1]],
                ]
            self.n_actions = len(self.wall_set) * (self.max_x-2) * (self.max_y-2)
            
            self.action_to_acts_tuple_dic = {}
            self.acts_tuple_to_action_dic = {}
            a = 0
            for i in range(len(self.wall_set)):
                for j in range(1, self.max_x-1):
                    for k in range(1, self.max_y-1):
                        self.action_to_acts_tuple_dic.update({a: (i, j, k)})
                        self.acts_tuple_to_action_dic.update({(i, j, k): a})
                        a += 1
        else:
            self.n_actions = self.n_walls * len(self.action_dict)

            self.action_to_acts_tuple_dic = {}
            self.acts_tuple_to_action_dic = {}
            a = 0
            for i in range(self.n_walls):
                for j in range(len(self.action_dict)):
                    self.action_to_acts_tuple_dic.update({a: (i, j)})
                    self.acts_tuple_to_action_dic.update({(i, j): a})
                    a += 1

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
        self.storage_dir = f"{agent_dir}/storage/"
        self.sb_tb_log_dir = f"{self.storage_dir}/sb_tb_log"
        if not os.path.exists(self.sb_tb_log_dir):
            os.makedirs(self.sb_tb_log_dir)
            
        self.generated_plans_dir = f"{self.storage_dir}/generated_plans"
        if not os.path.exists(self.generated_plans_dir):
            os.makedirs(self.generated_plans_dir)
        self.generated_gif_dir = f"{self.storage_dir}/generated_gif"
        if not os.path.exists(self.generated_gif_dir):
            os.makedirs(self.generated_gif_dir)
    
        self.scenario_dir_for_storing_data = f"{self.generated_plans_dir}" # "/{self.scenario_name}"
        if not os.path.exists(self.scenario_dir_for_storing_data):
            os.makedirs(self.scenario_dir_for_storing_data)
            
        if self.load_valid_plans_flag:
            if self.is_area_considered and self.is_proportion_considered:
                self.plan_values_path = f"{self.generated_plans_dir}/plan_values__{self.n_walls:02}_walls__area_proportion.csv"
            elif self.is_area_considered:
                self.plan_values_path = f"{self.generated_plans_dir}/plan_values__{self.n_walls:02}_walls__area.csv"
            else:
                self.plan_values_path = f"{self.generated_plans_dir}/plan_values__{self.n_walls:02}_walls__proportion.csv"
                
        if self.mask_flag:
            if self.plan_config_source_name == 'test_config':
                self.mask_numbers = 4
                self.masked_corners = ['corner_11', 'corner_00', 'corner_01', 'corner_10']# ['corner_00', 'corner_11'] # ['corner_00'] # ['corner_10', 'corner_00', 'corner_11', 'corner_01']# ['corner_11', 'corner_10', 'corner_00'] # ['corner_01']# ['corner_11', 'corner_10'] # ['corner_00', 'corner_01', 'corner_10'] # ['corner_11', 'corner_10', 'corner_00']  
                
                self.mask_lengths = [3, 1, 1, 1]#[6, 4] # [5] # [6, 4, 2, 1] # [5, 8, 4] # [3] # [1, 2] # [4, 7, 9] #[5, 9, 9] 
                self.mask_widths = [9, 8, 9, 6]# [9, 4] # [2] # [3, 8, 3, 7] # [3, 7, 9] # [6] # [6, 9] # [2, 1, 7] #[6, 2, 5] 
                
                self.masked_area = np.sum([(L+1)*(W+1) for L, W in zip(self.mask_lengths, self.mask_widths)])
                
                fixed_desired_areas = [16.0, 35.0, 38.0, 18.0, 32.0, 18.0, 35.0, 43.0, 26.0, 21.0, 67.0]#[38.0, 48.0, 50.0, 23.0, 57.0, 37.0, 28.0, 160.0] # [33.0, 19.0, 23.0, 46.0, 43.0, 48.0, 47.0, 26.0, 24.0, 39.0, 75.0] # [67.0, 41.0, 44.0, 38.0, 34.0, 58.0, 58.0] # [41.0, 35.0, 32.0, 38.0, 26.0, 55.0, 68.0] # [77.0, 104.0, 103.0, 157.0] # [126.0, 83.0, 65.0, 167.0] # [71.0, 66.0, 50.0, 254.0] #[76.0, 72.0, 81.0,  80.0]
                self.sampled_desired_areas = {f"room_{self.mask_numbers+1+i}": area for i, area in enumerate(fixed_desired_areas)}
            
            elif self.plan_config_source_name in ['load_fixed_config', 'load_random_config']:
                env_data_df = pd.read_csv(self.plan_values_path, nrows=self.nrows_of_env_data_csv)
                sampled_plan = env_data_df.sample()
                self.mask_numbers = int(sampled_plan['mask_numbers'].values.tolist()[0])
                masked_corners = list(map(str, sampled_plan['masked_corners'].tolist()[0][1:-1].split(',') ) )
                self.masked_corners = [cor.replace("'","").strip() for cor in masked_corners]
                self.mask_lengths = np.array(list(map(float, sampled_plan['mask_lengths'].tolist()[0][1:-1].split(','))), dtype=np.int32).tolist()
                self.mask_widths = np.array(list(map(float, sampled_plan['mask_widths'].tolist()[0][1:-1].split(','))), dtype=np.int32).tolist()
                self.masked_area = np.sum([(L+1)*(W+1) for L, W in zip(self.mask_lengths, self.mask_widths)])
                sampled_desired_areas = list(map(float, sampled_plan['desired_areas'].tolist()[0][1:-1].split(',')))
                self.sampled_desired_areas = {f"room_{i+1+self.mask_numbers}":a for i, a in enumerate(sampled_desired_areas)}
            
            elif self.plan_config_source_name in ['create_fixed_config', 'create_random_config']:
                while True:
                    self.mask_numbers = np.random.randint(4) + 1
                    self.masked_corners = np.random.choice(list(self.corners.keys()), size=self.mask_numbers, replace=False)    
                    
                    self.mask_lengths = np.random.randint(1, int(self.max_x/2), size=self.mask_numbers)
                    self.mask_widths = np.random.randint(1, int(self.max_y/2), size=self.mask_numbers)
                    
                    self.masked_area = np.sum([(L+1)*(W+1) for L, W in zip(self.mask_lengths, self.mask_widths)])
                    if self.masked_area <= self.total_area / 2:
                        break
                self.sampled_desired_areas = 0 # be careful 
            else:
                raise ValueError('Wrong plan_config_source_name!')
                
        else:
            self.mask_numbers = 0
            self.sampled_desired_areas = 0 # be careful 
            

        #### area_plots
        self.area_plots_dir = f"{self.storage_dir}/area_plots"
        if not os.path.exists(self.area_plots_dir):
            os.makedirs(self.area_plots_dir)
    
        self.area_gif_dir = f"{self.storage_dir}/area_gif"
        if not os.path.exists(self.area_gif_dir):
            os.makedirs(self.area_gif_dir)
    
        self.walls_data_dir = f"{self.storage_dir}/walls_data"
        self.walls_data_path = f"{self.walls_data_dir}/walls_coords.p"
        # self.last_walls_data_path = f"{self.walls_data_dir}/last_walls_coords.p"
        self.fixed_walls_path = f"{self.walls_data_dir}/fixed_walls.p"
        if not os.path.exists(self.walls_data_dir):
            os.makedirs(self.walls_data_dir)
    
    def get_config(self):
        return self.__dict__


# %%
if __name__ == '__main__':
    self = LaserWallConfig()
    fenv_config = self.get_config()