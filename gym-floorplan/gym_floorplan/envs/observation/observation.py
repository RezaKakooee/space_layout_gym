# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 23:34:14 2021

@author: RK
"""
#%%

import gym
import copy
import itertools
import numpy as np
import pandas as pd
from collections import deque
from webcolors import name_to_rgb

from gym_floorplan.base_env.observation.base_observation import BaseObservation

from gym_floorplan.envs.observation.wall_generator import WallGenerator
from gym_floorplan.envs.observation.geometry import Outline, Plan
from gym_floorplan.envs.observation.sequential_drawing import SequentialDrawing
from gym_floorplan.envs.observation.wall_transform import WallTransform
from gym_floorplan.envs.observation.room_extractor import RoomExtractor


#%%
class Observation(BaseObservation):
    def __init__(self, fenv_config:dict={}):
        super().__init__()
        self.fenv_config = fenv_config
        self.painter = SequentialDrawing(fenv_config=fenv_config)
        self.rextractor = RoomExtractor(fenv_config=self.fenv_config)

    @property
    def observation_space(self): 
        self._creat_observation_space_variables()
        
        _observation_space_cnn = gym.spaces.Box(low=self.low_cnn, 
                                                high=self.high_cnn,
                                                shape=self.shape_cnn,
                                                dtype=np.uint8)
        _observation_space_fc = gym.spaces.Box(low=self.low_fc,
                                               high=self.high_fc, 
                                               shape=self.shape_fc, #(self.shape_fc,), #
                                               dtype=np.float)    
        
        if self.fenv_config['net_arch'] == 'cnn': 
            self._observation_space = _observation_space_cnn
            
        elif self.fenv_config['net_arch'] == 'fc':
            self._observation_space = _observation_space_fc
            
        elif self.fenv_config['net_arch'] == 'cnnfc':
            self._observation_space = gym.spaces.Tuple((_observation_space_cnn,
                                                        _observation_space_fc))

        elif self.fenv_config['net_arch'] == 'fccnn':
            self._observation_space = gym.spaces.Tuple((_observation_space_fc,
                                                        _observation_space_cnn))
        return self._observation_space
    
    
    def _creat_observation_space_variables(self):
        if self.fenv_config['fixed_fc_observation_space']:
            n_walls = self.fenv_config['num_of_fixed_walls_for_masked']
        else:
            n_walls = self.fenv_config['n_walls']
            
        if self.fenv_config['use_areas_info_into_observation_flag']:
            if n_walls > 1:
                self.len_state_vec = np.cumsum([i * 5 * 5 for i in range(1, n_walls)])[-1]  # 5=n_wall_points
            else:
                self.len_state_vec = 0
            self.len_state_vec += n_walls * 5 * 2  # 5=n_wall_points, 2=xy
            self.len_state_vec += n_walls * 5 * 4  # 4=n_corners
            self.len_state_vec += (n_walls + 1) * 3  # 3=desired_areas, delta_areas, acheived_areas
            # TOTO: I might need to adjust the following line w.r.t masked_areas
            lowest_obs_val_fc = -self.fenv_config['total_area'] #max(list(self.fenv_config['areas_config'].values()))
            highest_obs_val_fc = self.fenv_config['total_area'] #self.fenv_config['n_rows '] * self.fenv_config['n_cols']
            
        else:
            self.len_state_vec = n_walls * 5 * 2 # x y coordinates of the walls for 5 important points
            self.len_state_vec += n_walls * 5 * 4  # distance between each of 5 wall's important poitns and 4 corners
            if n_walls > 1:
                self.len_state_vec += np.cumsum([i * 5 * 5 for i in range(1, n_walls)])[-1] # distane between 5 important points of walls
            else:
                self.len_state_vec = 0
            lowest_obs_val_fc = self.fenv_config['min_x']
            highest_obs_val_fc = 30

        if (self.fenv_config['env_planning'] == 'One_Shot') and (self.fenv_config['net_arch'] in ['fc', 'cnnfc', 'fccnn']):
            if self.fenv_config['use_areas_info_into_observation_flag']:
                lowest_obs_val_fc = -self.fenv_config['n_rows'] * self.fenv_config['n_cols']
                highest_obs_val_fc = self.fenv_config['n_rows'] * self.fenv_config['n_cols']
            else:
                lowest_obs_val_fc = -1
                highest_obs_val_fc = 40

        self.low_fc = lowest_obs_val_fc * np.ones(self.len_state_vec, dtype=float)
        self.high_fc = highest_obs_val_fc * np.ones(self.len_state_vec, dtype=float)
        self.shape_fc = (self.len_state_vec,)
        
        self.low_cnn = 0
        self.high_cnn = 255
        self.shape_cnn = (self.fenv_config['n_rows'] * self.fenv_config['scaling_factor'],
                          self.fenv_config['n_cols'] * self.fenv_config['scaling_factor'],
                          self.fenv_config['n_channels'])
        
    
    def obs_reset(self):
        plan_data_dict = self._setup_plan() # creates an empty plan
        if self.fenv_config['mask_flag']:
            plan_data_dict = self._mask_plan(plan_data_dict) # only gets the false walls. updating happens afew line later
            if self.fenv_config['load_valid_plans_flag']:
                self.desired_areas = copy.deepcopy(self.sampled_desired_areas)
            else:
                self.desired_areas = self._configure_areas(plan_data_dict['masked_area'])
            plan_data_dict['desired_areas'] = self.desired_areas
            
            ## update the plan_data_dict according to false walls
            for fwall_name , fwalls_coord in plan_data_dict['fwalls_coords'].items():
                plan_data_dict = self._setup_plan(plan_data_dict=plan_data_dict, walls_coords=plan_data_dict['fwalls_coords'], active_wall_name=fwall_name)
                plan_data_dict = self.painter.updata_obs_mat(plan_data_dict, fwall_name)
                plan_data_dict = self.rextractor.update_room_dict(plan_data_dict, fwall_name)
        else:
            self.mask_numbers = 0
            self.desired_areas = self._configure_areas(masked_area=0)
            plan_data_dict['desired_areas'] = self.desired_areas
        
        plan_data_dict['room_wall_occupied_positions'].extend(np.argwhere(plan_data_dict['moving_ones']==1).tolist())
        plan_data_dict['obs_mat_mask'] = 1 - plan_data_dict['moving_ones']
        # plan_data_dict = self._adjust_fenv_config(plan_data_dict)
        
        if self.fenv_config['env_planning'] == 'Dynamic':
            plan_data_dict = self._setup_walls(plan_data_dict)
            
            for iwall_name , iwalls_coord in plan_data_dict['iwalls_coords'].items():
                plan_data_dict = self._setup_plan(plan_data_dict=plan_data_dict, walls_coords=plan_data_dict['iwalls_coords'], active_wall_name=iwall_name)
                plan_data_dict = self.painter.updata_obs_mat(plan_data_dict, iwall_name)
                plan_data_dict = self.rextractor.update_room_dict(plan_data_dict, iwall_name)
        
        self.wall_names_deque = deque([f"wall_{i+1}" for i in range(self.mask_numbers, 
                                                                    self.fenv_config['n_walls'] + self.mask_numbers)], 
                                          maxlen=self.fenv_config['n_walls'])
        
        self.plan_data_dict = copy.deepcopy(plan_data_dict)
        
        self.observation = self._make_observation(observation_matrix=self.plan_data_dict['moving_labels'])
        self.plan_data_dict.update({'obs_arr_conv': self.observation})
        
        self.done = False
        return self.observation
        
        
    def update(self, active_wall_name=None, action=None):
        self.active_wall_name = active_wall_name
        if self.fenv_config['env_planning'] == "One_Shot":
            self.active_wall_status, new_walls_coords = self._select_wall(action=action)
        elif self.fenv_config['env_planning'] == "Dynamic":
            shifted_wall_names_deque = self.wall_names_deque.copy()
            shifted_wall_names_deque.append(shifted_wall_names_deque[0]) # move the current agent to the last index. Because, we first need to draw the other agent, and then the current agent acts according to them
            previous_wall_id = f"wall_{shifted_wall_names_deque[-2].split('_')[1]}"
            
            self.active_wall_status, new_walls_coords = self._transform_walls(action, previous_wall_id) # for asp, actions only include the current agent. So, we only transform the corressponding walls
        
        if self.active_wall_status == "check_room_area":
            plan_data_dict = copy.deepcopy(self.plan_data_dict)
            plan_data_dict = self._setup_plan(plan_data_dict=plan_data_dict, walls_coords=new_walls_coords, active_wall_name=self.active_wall_name)
            plan_data_dict = self.painter.updata_obs_mat(plan_data_dict, active_wall_name) # here we update plan_data_dict based on the wall order
            plan_data_dict = self.rextractor.update_room_dict(plan_data_dict, active_wall_name)
            self.active_wall_status = self._get_active_wall_status(plan_data_dict, active_wall_name)
            
            if self.active_wall_status == "accepted":
                plan_data_dict['room_wall_occupied_positions'].extend(np.argwhere(plan_data_dict['moving_ones']==1).tolist())
                self.plan_data_dict = copy.deepcopy(plan_data_dict)
                self.wall_names_deque.append(self.wall_names_deque[0])
            
            self.observation = self._make_observation(observation_matrix=self.plan_data_dict['moving_labels'])
            self.plan_data_dict.update({'obs_arr_conv': self.observation})
        
            self.done, self.active_wall_status = self._check_terminate(active_wall_name, self.active_wall_status)
        
        return self.observation
    
    
    def _setup_plan(self, plan_data_dict=None, walls_coords=None, active_wall_name=None):
        outline_dict = Outline(fenv_config=self.fenv_config).get_outline_data()
        plan_dict = Plan(outline_dict=outline_dict, walls_coords=walls_coords).get_plan_data()
        if walls_coords is None:
            obs_mat = np.zeros((self.fenv_config['n_rows'], self.fenv_config['n_cols']), dtype=int)
            plan_data_dict = {'outline_segments': plan_dict['segments_dict']['outline'],
                              'inline_segments': {},
                              'walls_segments': {},
                              'walls_coords': {},
                              
                              'base_obs_mat': obs_mat,
                              'base_obs_mat_w': obs_mat,
                              'base_obs_mat_for_dot_prod': 1 - obs_mat,
                              
                              'obs_mat': obs_mat,
                              'obs_mat_w': obs_mat,
                              'obs_mat_for_dot_prod': 1 - obs_mat,
                              
                              'moving_labels': obs_mat,
                              'moving_ones': obs_mat,
                              
                              'canvas_mat': obs_mat,
                              'outside_positions': [],
                              'room_wall_occupied_positions': [],
                              
                              'rooms_dict': {},
                              'desired_areas': {},
                              'areas': {},
                              'delta_areas': {},
                              'number_of_walls': self.fenv_config['n_walls'],
                              
                              'mask_numbers': 0,
                              'masked_area': 0,
                              }
            
        else:
            # plan_data_dict = copy.deepcopy(self.plan_data_dict)
            plan_data_dict['inline_segments'].update({k: v for k, v in plan_dict['segments_dict']['inline'].items() if active_wall_name in k})
            plan_data_dict['walls_segments'].update({k: v for k, v in plan_dict['walls_segments_dict'].items() if active_wall_name in k})
            plan_data_dict['walls_coords'].update({active_wall_name: plan_dict['walls_dict'][active_wall_name]})
            
        return plan_data_dict
    
    
    def _mask_plan(self, plan_data_dict):
        self.blackbox_min_length = 1
        self.blackbox_max_length = int(self.fenv_config['max_x']/2)
        self.blackbox_min_width = 1
        self.blackbox_max_width = int(self.fenv_config['max_y']/2)
        
        if self.fenv_config['plan_config_source_name'] in ['test_config', 'load_fixed_config', 'create_fixed_config']:
            self.mask_numbers = self.fenv_config['mask_numbers']
            self.masked_corners = self.fenv_config['masked_corners']
            
            self.mask_lengths = self.fenv_config['mask_lengths']
            self.mask_widths = self.fenv_config['mask_widths']
            self.masked_area = self.fenv_config['masked_area'] 
            self.sampled_desired_areas = self.fenv_config['sampled_desired_areas']
            
        elif self.fenv_config['plan_config_source_name'] == 'load_random_config':
                env_data_df = pd.read_csv(self.fenv_config['env_data_csv_path'], nrows=self.fenv_config['nrows_of_env_data_csv'])
                sampled_plan = env_data_df.sample()
                self.mask_numbers = int(sampled_plan['mask_numbers'].values.tolist()[0])
                masked_corners = list(map(str, sampled_plan['masked_corners'].tolist()[0][1:-1].split(',') ) )
                self.masked_corners = [cor.replace("'","").strip() for cor in masked_corners]
                self.mask_lengths = np.array(list(map(float, sampled_plan['mask_lengths'].tolist()[0][1:-1].split(','))), dtype=np.int32).tolist()
                self.mask_widths = np.array(list(map(float, sampled_plan['mask_widths'].tolist()[0][1:-1].split(','))), dtype=np.int32).tolist()
                self.masked_area = np.sum([(L+1)*(W+1) for L, W in zip(self.mask_lengths, self.mask_widths)])
                sampled_desired_areas = list(map(float, sampled_plan['desired_areas'].tolist()[0][1:-1].split(',')))
                self.sampled_desired_areas = {f"room_{i+1+self.mask_numbers}":a for i, a in enumerate(sampled_desired_areas)}
            
        else: # self.plan_config_source_name == 'create_random_config':
            while True:
                self.mask_numbers = np.random.randint(4) + 1
                self.masked_corners = np.random.choice(list(self.fenv_config['corners'].keys()), size=self.mask_numbers, replace=False)    
                
                self.mask_lengths = np.random.randint(self.blackbox_min_length, self.blackbox_max_length, size=self.mask_numbers)
                self.mask_widths = np.random.randint(self.blackbox_min_width, self.blackbox_max_width, size=self.mask_numbers)
                
                self.masked_area = np.sum([(L+1)*(W+1) for L, W in zip(self.mask_lengths, self.mask_widths)])
                if self.masked_area <= self.fenv_config['total_area'] / 2:
                    break
        
        fwalls_coords = {}
        rectangles_vertex = [] #{}
        for i, mask_cor in enumerate(self.masked_corners):
            if mask_cor == 'corner_00':
                rect_vertex = (0, 0)
                fwall_anchor_coord = (self.mask_lengths[i], self.mask_widths[i])
                fwall_back_coord = (fwall_anchor_coord[0]-1, fwall_anchor_coord[1])
                fwall_front_coord = (fwall_anchor_coord[0], fwall_anchor_coord[1]-1)

            elif mask_cor == 'corner_01':
                rect_vertex = (self.fenv_config['max_x']-self.mask_lengths[i], 0)
                fwall_anchor_coord = (self.fenv_config['max_x']-self.mask_lengths[i], self.mask_widths[i])
                fwall_back_coord = (fwall_anchor_coord[0]+1, fwall_anchor_coord[1])
                fwall_front_coord = (fwall_anchor_coord[0], fwall_anchor_coord[1]-1)
                
            elif mask_cor == 'corner_10':
                rect_vertex = (0, self.fenv_config['max_y']-self.mask_widths[i])
                fwall_anchor_coord = (self.mask_lengths[i], self.fenv_config['max_y']-self.mask_widths[i])
                fwall_back_coord = (fwall_anchor_coord[0], fwall_anchor_coord[1]+1)
                fwall_front_coord = (fwall_anchor_coord[0]-1, fwall_anchor_coord[1])
                
            elif mask_cor == 'corner_11':
                rect_vertex = (self.fenv_config['max_x']-self.mask_lengths[i], self.fenv_config['max_y']-self.mask_widths[i])
                fwall_anchor_coord = (self.fenv_config['max_x']-self.mask_lengths[i], self.fenv_config['max_y']-self.mask_widths[i])
                fwall_back_coord = (fwall_anchor_coord[0], fwall_anchor_coord[1]+1)
                fwall_front_coord = (fwall_anchor_coord[0]+1, fwall_anchor_coord[1])
            
            rectangles_vertex.append(rect_vertex) # rectangles_vertex[mask_cor] =  rect_vertex
            
            fwalls_coords.update({f'wall_{i+1}': {'anchor_coord': fwall_anchor_coord,
                                                  'back_open_coord': fwall_back_coord,
                                                  'front_open_coord': fwall_front_coord}})
        
        valid_points_for_sampling = self._get_valid_points_for_sampling(plan_data_dict)
        for fwall_name, fwall_coord in fwalls_coords.items():
            fwall_coords = WallGenerator(self.fenv_config).make_walls(valid_points_for_sampling, fwall_coord, wall_name=fwall_name)
            fwalls_coords[fwall_name] = fwall_coords[fwall_name]
        
        # desired_areas_ = copy.deepcopy(plan_data_dict['desired_areas'])
        plan_data_dict.update({
        'mask_numbers': self.mask_numbers,
        'mask_lengths': self.mask_lengths,
        'mask_widths': self.mask_widths,
        'fwalls_coords': fwalls_coords,
        'masked_area': self.masked_area,
        'rectangles_vertex': rectangles_vertex,
        'obs_mat_mask': np.ones((self.fenv_config['n_rows'], self.fenv_config['n_cols']), dtype=int),
        'number_of_walls': self.fenv_config['n_walls'] + self.mask_numbers,
        })
        
        return plan_data_dict
    
    
    def _configure_areas(self, masked_area):
        if self.fenv_config['plan_config_source_name'] in ['create_fixed_config', 'create_random_config']:
            free_area = self.fenv_config['total_area'] - masked_area
            middle_area = np.floor(free_area / (self.fenv_config['n_walls']+1))
            min_area = np.floor(middle_area/2)
            max_area = min_area + middle_area
            areas_config = {f'room_{i+1}': list(np.random.randint(min_area, max_area , 1)/1.0)[0] for i in range(self.mask_numbers, self.fenv_config['n_walls']+self.mask_numbers)}
            sum_areas_except_last_room = np.sum(list(areas_config.values()))
            areas_config.update({f"room_{self.fenv_config['n_walls']+self.mask_numbers+1}": free_area - sum_areas_except_last_room}) # areas_config.update({f"room_{self.fenv_config['n_walls']+self.mask_numbers+1}": self.fenv_config['total_area'] - sum_areas_except_last_room})
        else:
            areas_config = self.fenv_config['sampled_desired_areas']
        return areas_config
        
        
    def _adjust_fenv_config(self, plan_data_dict):
        masked_area = np.sum(plan_data_dict['moving_ones'])
        print('masked_area', masked_area)
        # total_area = self.fenv_config['max_x'] * self.fenv_config['max_y']
        areas_config = self.fenv_config['areas_config']
        adjusting_rate = 1 - (masked_area / self.fenv_config['total_area'])
        areas_config_ = {room_name:0 for room_name in areas_config.keys()}
        sum_desired_areas = 0
        non_last_room_names = list(areas_config.keys())[:-1]
        last_room_name = list(areas_config.keys())[-1]
        for room_name in non_last_room_names:
            adjusted_room_area = int(adjusting_rate * areas_config[room_name])
            areas_config_[room_name] = adjusted_room_area
            sum_desired_areas += adjusted_room_area
        
        areas_config_[last_room_name] = self.fenv_config['total_area'] - areas_config[last_room_name] - sum_desired_areas
        plan_data_dict['desired_areas'] = {f'room_{i}': a for i, a in enumerate(list(areas_config_.values()), plan_data_dict['mask_numbers']+1)}
        return plan_data_dict

    
    def _setup_walls(self, plan_data_dict):
        valid_points_for_sampling = self._get_valid_points_for_sampling(plan_data_dict)
        walls_coords = WallGenerator(fenv_config=self.fenv_config).make_walls(valid_points_for_sampling)
        if not isinstance(walls_coords, dict):
            raise ValueError("-.- observation: walls_coords must be dict!")
        plan_data_dict['iwalls_coords'] = walls_coords
        return plan_data_dict
    
    
    def _select_wall(self, action=None):
        current_walls_coords = self.plan_data_dict['walls_coords']
        new_walls_coords = copy.deepcopy(current_walls_coords)
        num_current_walls = len(current_walls_coords)
        
        ### select a new wall
        new_wall_name = f"wall_{num_current_walls+1}"
        w_i, x, y = self.fenv_config['action_to_acts_tuple_dic'][action]
        w_coords = np.array(self.fenv_config['wall_set'][w_i])
        w_coords[:,0] += x
        w_coords[:,1] += y
        
        new_wall = {'back_open_coord': w_coords[0],
                    'anchor_coord': w_coords[1],
                    'front_open_coord': w_coords[2]}
        
        back_position = self._cartesian2image_coord(new_wall['back_open_coord'][0], new_wall['back_open_coord'][1], self.fenv_config['max_y'])
        anchor_position = self._cartesian2image_coord(new_wall['anchor_coord'][0], new_wall['anchor_coord'][1], self.fenv_config['max_y'])
        front_position = self._cartesian2image_coord(new_wall['front_open_coord'][0], new_wall['front_open_coord'][1], self.fenv_config['max_y'])
        
        if self.fenv_config['mask_flag']:
            if list(anchor_position) in self.plan_data_dict['room_wall_occupied_positions']:
                active_wall_status = "rejected_by_room"
            elif tuple(back_position) in self.plan_data_dict['outside_positions']:
                active_wall_status = "rejected_by_canvas"
            elif tuple(anchor_position) in self.plan_data_dict['outside_positions']:
                active_wall_status = "rejected_by_canvas"
            elif tuple(front_position) in self.plan_data_dict['outside_positions']:
                active_wall_status = "rejected_by_canvas"
            else:
                active_wall_status = "check_room_area"
        else:
            if list(anchor_position) in self.plan_data_dict['room_wall_occupied_positions']:
                active_wall_status = "rejected_by_room"
            else:
                active_wall_status = "check_room_area"
        
        if active_wall_status == "check_room_area":
            valid_points_for_sampling = self._get_valid_points_for_sampling(self.plan_data_dict)
            new_wall_coords = WallGenerator(self.fenv_config).make_walls(self.plan_data_dict, new_wall, wall_name=new_wall_name)
            new_walls_coords.update(new_wall_coords)
        
        return active_wall_status, new_walls_coords
     
     
    def _get_valid_points_for_sampling(self, plan_data_dict):
        valid_points_for_sampling = np.argwhere(plan_data_dict['moving_ones']==0)
        valid_points_for_sampling = [[r, c] for r, c in valid_points_for_sampling if (r%2==0 and c%2==0 and r!=0 and c!=0 and r!=self.fenv_config['max_x'] and c!=self.fenv_config['max_y'])]
       
        return np.array(valid_points_for_sampling)
       
   
    def _transform_walls(self, actions, previous_wall_id): # in asp actions includes the current wall, so we only trasnform the current wall
        current_walls_coords = self.plan_data_dict[0]['walls_coords'] # current_walls_coords means the walls before transformation
        new_walls_coords = copy.deepcopy(current_walls_coords)
        current_moving_wall_names = [f"wall_{agent_name.split('_')[1]}" for agent_name in actions.keys()]
        current_moving_wall_coords = {wall_name:current_walls_coords[wall_name] for wall_name in current_moving_wall_names}
        for (wall_name, wall_coords), (action_for_agent, action) in zip(current_moving_wall_coords.items(), actions.items()):
            if self.fenv_config['action_dict'][action] == 'no_action':
                pass
            else:
                n_wall = WallTransform(wall_name, wall_coords, action, self.plan_data_dict, previous_wall_id, self.fenv_config).transform()
                new_walls_coords[wall_name] = list(n_wall.values())[0]
        self.__check_new_walls_coords(new_walls_coords)      
        active_wall_status = "check_room_area"
        return active_wall_status, new_walls_coords
    
        
    def _make_observation(self, observation_matrix):
        def __get_cnn_obs(observation_matrix):
            if self.fenv_config['n_channels'] == 3:
                self.obs_arr_conv = self._get_obs_arr_for_conv(observation_matrix)
            else:
                K = 250 // (self.plan_data_dict['number_of_walls']+1) # for normalization
                self.obs_arr_conv = np.cast['uint8'](np.expand_dims(observation_matrix*K, axis=2))
        
            return copy.deepcopy(self.obs_arr_conv)
        
        def __get_fc_obs(observation_matrix):
            if self.fenv_config['env_planning'] == "One_Shot":
                vector_state_representation = np.array(self._wall_data_extractor_for_single_agent(self.plan_data_dict))
            else:
                vector_state_representation = np.array(self._wall_data_extractor(self.plan_data_dict))
                
            if self.fenv_config['use_areas_info_into_observation_flag']:
                desired_areas_dict = self.plan_data_dict['desired_areas']
                achieved_areas_dict = copy.deepcopy(self.plan_data_dict['desired_areas'])
                delta_areas_dict = self.fenv_config['areas_config']
                
                for room_name, achieved_area in self.plan_data_dict['areas'].items():
                    achieved_areas_dict[room_name] = achieved_area
                for room_neam, delta_area in self.plan_data_dict['delta_areas'].items():
                    delta_areas_dict[room_name] = delta_area
                  
                desired_areas_list = list(desired_areas_dict.values())
                achieved_areas_list = list(achieved_areas_dict.values())
                delta_areas_list = list(delta_areas_dict.values())
                
                all_area_related_info = np.array(desired_areas_list + achieved_areas_list + delta_areas_list)
                all_area_related_info_normalized = all_area_related_info/self.fenv_config['max_y']
                
                vector_state_representation = np.concatenate((vector_state_representation, all_area_related_info_normalized))
            
            if self.fenv_config['fixed_fc_observation_space']:
                if self.fenv_config['net_arch'] in ['fccnn', 'cnnfc']:
                    fc_len = self.observation_space[0].shape[0]
                else:
                    fc_len = self.observation_space.shape[0]
                additional_vec_fc = -1 * np.ones(fc_len-len(vector_state_representation))
                vector_state_representation = np.concatenate((vector_state_representation, additional_vec_fc), axis=-1)
            
            return vector_state_representation.astype(np.float)
            
        if self.fenv_config['net_arch'] == 'cnn':
            self.observation_cnn = __get_cnn_obs(observation_matrix)            
        elif self.fenv_config['net_arch'] == 'fc':
            self.observation_fc = __get_fc_obs(observation_matrix)
        else:
            self.observation_cnn = __get_cnn_obs(observation_matrix)
            self.observation_fc = __get_fc_obs(observation_matrix)
            
        ## CnnFc
        if self.fenv_config['net_arch'] == 'cnn':
            observation = copy.deepcopy(self.observation_cnn)
            
        elif self.fenv_config['net_arch'] == 'fc':
            observation = copy.deepcopy(self.observation_fc)
            
        elif self.fenv_config['net_arch'] == 'cnnfc':
            observation = (self.observation_cnn, self.observation_fc)

        elif self.fenv_config['net_arch'] == 'fccnn':
            observation = (self.observation_fc, self.observation_cnn)
            
            # assert np.all(self.observation_fc.shape == self.observation_space[0].shape)
            # assert np.all(self.observation_cnn.shape == self.observation_space[1].shape)
        return observation
    
    
    def _get_active_wall_status(self, plan_data_dict, active_wall_name):
        wall_i = int(active_wall_name.split('_')[1])
        room_name = f"room_{wall_i}"
        
        def __check_area_status():
            active_wall_abs_delta_area = abs(plan_data_dict['delta_areas'][room_name])
            if active_wall_abs_delta_area <= self.fenv_config['area_tolerance']:
                return True
            else:
                return False
        
        def __check_proportion_status():
            proportions = plan_data_dict['rooms_dict'][room_name]['proportions']
            if (min(proportions) >= self.fenv_config['min_desired_proportion']) and \
                (max(proportions) <= self.fenv_config['max_desired_proportion']):
                return True
            else:
                return False
        
        # if __check_area_status() and __check_proportion_status():
        #     active_wall_status = 'accepted'
        # elif not __check_area_status() and not __check_proportion_status():
        #     active_wall_status = 'rejected_by_both_area_and_proportion'
        # elif __check_area_status() and not __check_proportion_status():
        #         active_wall_status = 'rejected_by_proportion_but_accepted_by_area'
        # elif not __check_area_status() and __check_proportion_status():
        #         active_wall_status = 'rejected_by_area_but_accepted_by_proportion'
                
                
        if self.fenv_config['is_area_considered'] and self.fenv_config['is_proportion_considered']:
            if __check_area_status() and __check_proportion_status():
                return 'accepted'
            else:
                return 'rejected_by_both_area_and_proportion'
        elif self.fenv_config['is_area_considered'] and not self.fenv_config['is_proportion_considered']:
            if __check_area_status():
                return 'accepted'
            else:
                return 'rejected_by_area'
        elif not self.fenv_config['is_area_considered'] and self.fenv_config['is_proportion_considered']:
            if __check_proportion_status():
                return 'accepted'
            else:
                return 'rejected_by_proportion'
        else:
            return 'accepted'

        # return active_wall_status
    
    
    def _distance(self, coord_s, coord_e):
        return np.linalg.norm(np.array(coord_s)-np.array(coord_e))
        
    
    def _wall_data_extractor_for_single_agent(self, plan_data_dict):
        important_points = ['start_of_wall', 'before_anchor', 'anchor', 'after_anchor', 'end_of_wall']
        def __add_wall_important_points(plan_data_dict):
            wall_coords_template = {k: [-1, -1] for k in important_points}
            wall_important_points_dict = {f'wall_{i}': copy.deepcopy(wall_coords_template) for i in range(1, self.plan_data_dict['number_of_walls']+1)}
            
            for wall_name, wall_data in plan_data_dict['walls_coords'].items():
                back_segment = wall_data['back_segment']
                front_segment = wall_data['front_segment']
                wall_important_points_dict[wall_name].update({ 'start_of_wall': list(back_segment['reflection_coord']),
                                                               'before_anchor': list(back_segment['end_coord']),
                                                               'anchor':        list(back_segment['start_coord']),
                                                               'after_anchor':  list(front_segment['end_coord']),
                                                               'end_of_wall':   list(front_segment['reflection_coord']) })
            return wall_important_points_dict
    
        def __add_walls_to_corners_distance(plan_data_dict, wall_important_points_dict):
            # 5 wall's important points to 4 corners
            walls_to_corners_distance_dict = {wall_name: list(-1*np.ones((5*4),dtype=float)) for wall_name in wall_important_points_dict.keys()}
            
            for wall_name in walls_to_corners_distance_dict.keys(): # 3
                wall_to_corner_dist_vec = []
                for point_name in important_points: #5
                    for corner_name, corner_coord in self.fenv_config['corners'].items(): # 4
                        d = self._distance(wall_important_points_dict[wall_name][point_name], corner_coord)
                        wall_to_corner_dist_vec.append(d)
            
                walls_to_corners_distance_dict[wall_name] = copy.deepcopy(wall_to_corner_dist_vec)
            return walls_to_corners_distance_dict
            
        def __add_walls_to_walls_distance(plan_data_dict, wall_important_points_dict):
            n_walls = self.plan_data_dict['number_of_walls']
            mask_numbers = self.plan_data_dict['mask_numbers'] if self.fenv_config['mask_flag'] else 0
            walls_to_walls_distance_dict = {}
            for i in range(mask_numbers, n_walls):
                for j in range(i+1, n_walls):
                    walls_to_walls_distance_dict.update({
                        f"wall_{i+1}_to_wall_{j+1}": list(-1*np.ones((5*5),dtype=float))
                        })
                    
            for wall_name_s in wall_important_points_dict.keys():
                wall_s_i = int(wall_name_s.split('_')[-1])
                for wall_name_e in wall_important_points_dict.keys():
                    wall_e_i = int(wall_name_e.split('_')[-1])
                    if wall_e_i > wall_s_i:
                        wall_to_wall_dist_vec = []
                        for point_name_s in important_points:
                            for point_name_e in important_points:
                                d = self._distance(wall_important_points_dict[wall_name_s][point_name_s],
                                                   wall_important_points_dict[wall_name_e][point_name_e])
                                wall_to_wall_dist_vec.append(d)      
                        walls_to_walls_distance_dict[f"{wall_name_s}_to_{wall_name_e}"] = copy.deepcopy(wall_to_wall_dist_vec)
            return walls_to_walls_distance_dict
            
        wall_important_points_dict = __add_wall_important_points(plan_data_dict)
        wall_names = list(wall_important_points_dict.keys())
        if self.fenv_config['mask_flag']:
            if self.fenv_config['net_arch'] != 'fc':
                wall_names = wall_names[self.plan_data_dict['mask_numbers']:]
        wall_important_points_dict = {wall_name: wall_important_points_dict[wall_name] for wall_name in wall_names}       
        
        walls_to_corners_distance_dict = __add_walls_to_corners_distance(plan_data_dict, wall_important_points_dict)
        walls_to_walls_distance_dict = __add_walls_to_walls_distance(plan_data_dict, wall_important_points_dict)
        
        
        wall_important_coords = [coord for point in wall_important_points_dict.values() for coord in point.values()]
        walls_to_corners_distance = [ds for ds in walls_to_corners_distance_dict.values()]
        walls_to_walls_distance = [ds for ds in walls_to_walls_distance_dict.values()]
        
        wall_important_coords_vec = list(itertools.chain.from_iterable(wall_important_coords))
        walls_to_corners_distance_vec = list(itertools.chain.from_iterable(walls_to_corners_distance))
        walls_to_walls_distance_vec = list(itertools.chain.from_iterable(walls_to_walls_distance))
        
        state_vec = wall_important_coords_vec + walls_to_corners_distance_vec + walls_to_walls_distance_vec
        
        return np.array(state_vec)
        
    
    def _distance_calculator(self, wall_state_dict):
        wall_to_wall_dist_vec = []
        wall_to_wall_dist_dict = {}
        walls_name = list(wall_state_dict.keys())
        n_walls = len(walls_name)
        points_name = list(wall_state_dict[walls_name[0]].keys())
        n_points = len(points_name)
        for i in range(n_walls):
            wall_name_s = walls_name[i]
            for j in range(i+1, n_walls):
                wall_name_e = walls_name[j]
                for point_name_s in points_name:
                    for point_name_e in points_name:
                        d = self._distance(wall_state_dict[wall_name_s][point_name_s],
                                           wall_state_dict[wall_name_e][point_name_e])
                        wall_to_wall_dist_dict[f"{wall_name_s}_{wall_name_e}_{point_name_s}_{point_name_e}"] = d
                        wall_to_wall_dist_vec.append(d)       
        
        wall_to_corner_dist_vec = []
        wall_to_corner_dist_dict = {}
        for wall_name in walls_name:
            for point_name in points_name:
                for corner_name, corner_coord in self.fenv_config['corners'].items():
                    d = self._distance(wall_state_dict[wall_name][point_name], corner_coord)
                    wall_to_corner_dist_dict[f"{wall_name}_{point_name}_{corner_name}"] = d
                    wall_to_corner_dist_vec.append(d)
        
        all_dist_values = wall_to_wall_dist_vec + wall_to_corner_dist_vec
        return all_dist_values
    
    
    def _wall_data_extractor(self, plan_data_dict):
        walls_state_vec = []
        wall_state_dict = {}
        for i in range(1, self.plan_data_dict['number_of_walls']+1):
            wall_name = f"wall_{i}"
            back_segment = plan_data_dict['walls_coords'][wall_name]['back_segment']
            front_segment = plan_data_dict['walls_coords'][wall_name]['front_segment']
            this_wall_state = {'start_of_wall': list(back_segment['reflection_coord']),
                               'before_anchor': list(back_segment['end_coord']),
                               'anchor':        list(back_segment['start_coord']),
                               'after_anchor':  list(front_segment['end_coord']),
                               'end_of_wall':   list(front_segment['reflection_coord'])}
            
            walls_state_vec.extend(list(this_wall_state.values()))
            wall_state_dict[wall_name] = this_wall_state
        
        if self.fenv_config['only_straight_vertical_walls']:
            anchor_x_list = []
            for w_name, w_coord in wall_state_dict.items():
                anchor_x = w_coord['anchor'][0]
                anchor_x_list.append(anchor_x)
            state_vec = anchor_x_list
        else:
            walls_state_vec = list(itertools.chain.from_iterable(walls_state_vec))
            
            all_dist_values = self._distance_calculator(wall_state_dict)
            
            state_vec = walls_state_vec + all_dist_values
        
        return np.array(state_vec)
    
    
    def _get_obs_arr_for_conv(self, observation_matrix):
        obs_arr_conv = np.zeros((self.fenv_config['n_rows']*self.fenv_config['scaling_factor'], 
                                 self.fenv_config['n_cols']*self.fenv_config['scaling_factor'], 
                                 self.fenv_config['n_channels']), dtype=np.uint8)
        obs_arr_for_conv = np.zeros_like(obs_arr_conv) 
        obs_mat_scaled = np.kron(observation_matrix,
                                  np.ones((self.fenv_config['scaling_factor'], self.fenv_config['scaling_factor']), 
                                          dtype=observation_matrix.dtype))
        for r in range(self.fenv_config['n_rows']*self.fenv_config['scaling_factor']):
            for c in range(self.fenv_config['n_cols']*self.fenv_config['scaling_factor']):
                for v in list(self.fenv_config['color_map'].keys()): #[:self.fenv_config['n_rooms']+2]:
                    if obs_mat_scaled[r,c] == v:
                        obs_arr_for_conv[r,c,:] = list(  name_to_rgb(self.fenv_config['color_map'][v])  )
        return obs_arr_for_conv
    
    
    def _check_terminate(self, active_wall_name, active_wall_status):
        if active_wall_name == f"wall_{self.plan_data_dict['number_of_walls']}":
            if active_wall_status == 'accepted':        
                active_wall_status = 'finished'
                return True, active_wall_status
        return False, active_wall_status
    
    
    @staticmethod
    def _cartesian2image_coord(x, y, max_y):
        return max_y-y, x
    
    
#%% 
if __name__ == '__main__':
    from gym_floorplan.envs.fenv_config import LaserWallConfig
    fenv_config = LaserWallConfig().get_config()
    self = Observation(fenv_config)
    observation = self.obs_reset()
    active_wall_name = f"wall_{1+self.mask_numbers}"
    for action in [992, 714, 874, 1688, 930, 1742, 635]:
        self.update(active_wall_name=active_wall_name, action=action)
        
        if self.active_wall_status == "accepted":
            active_wall_name = f"wall_{int(self.active_wall_name.split('_')[1])+1}"
            
    plan_data_dict = self.plan_data_dict