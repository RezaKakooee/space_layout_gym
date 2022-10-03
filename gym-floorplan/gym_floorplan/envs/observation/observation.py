# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 23:34:14 2021

@author: RK
"""
#%%

import gym
import copy
# import ast
# import itertools
import numpy as np
# import pandas as pd
from collections import deque, defaultdict
# from webcolors import name_to_rgb

from gym_floorplan.base_env.observation.base_observation import BaseObservation

from gym_floorplan.envs.observation.wall_generator import WallGenerator
# from gym_floorplan.envs.observation.geometry import Outline, Plan
from gym_floorplan.envs.observation.sequential_painter import SequentialPainter
from gym_floorplan.envs.observation.room_extractor import RoomExtractor
from gym_floorplan.envs.observation.plan_construcror import PlanConstructor
from gym_floorplan.envs.observation.state_composer import StateComposer
from gym_floorplan.envs.observation.action_parser import ActionParser
from gym_floorplan.envs.layout_graph import LayoutGraph


#%%
class Observation:
    def __init__(self, fenv_config:dict={}):
        # super().__init__()
        self.fenv_config = fenv_config
        self.painter = SequentialPainter(fenv_config=self.fenv_config)
        self.rextractor = RoomExtractor(fenv_config=self.fenv_config)
        self.plan_construcror = PlanConstructor(fenv_config=self.fenv_config)
        self.state_composer = StateComposer(fenv_config=self.fenv_config)
        self.action_parser = ActionParser(fenv_config=self.fenv_config)
        
        self.observation_space = self._get_observation_space()


    # @property
    def _get_observation_space(self): # def observation_space(self): 
        self.state_data_dict = self.state_composer.creat_observation_space_variables()
        
        _observation_space_cnn = gym.spaces.Box(low=self.state_data_dict['low_cnn'], 
                                                high=self.state_data_dict['high_cnn'],
                                                shape=self.state_data_dict['shape_cnn'],
                                                dtype=np.uint8)
        _observation_space_fc = gym.spaces.Box(low=self.state_data_dict['low_fc'],
                                               high=self.state_data_dict['high_fc'], 
                                               shape=self.state_data_dict['shape_fc'], #(self.state_data_dict['shape_fc'],), #
                                               dtype=float)
        
        _observation_space_gcn = gym.spaces.Box(low=self.state_data_dict['low_gcn'],
                                               high=self.state_data_dict['high_gcn'], 
                                               shape=self.state_data_dict['shape_gcn'], 
                                               dtype=float)
        
        if self.fenv_config['action_masking_flag']:
            _observation_space_fc = gym.spaces.Dict({
                                'action_mask': gym.spaces.Box(low=0,
                                                              high=1, 
                                                              shape=(self.fenv_config['n_actions'],), 
                                                              dtype=float),
                                'action_avail': gym.spaces.Box(low=0,
                                                               high=1, 
                                                               shape=(self.fenv_config['n_actions'],), 
                                                               dtype=float),
                                'real_obs': _observation_space_fc,
                                })
        
        
            _observation_space_gcn = gym.spaces.Dict({
                                'action_mask': gym.spaces.Box(low=0,
                                                              high=1, 
                                                              shape=(self.fenv_config['n_actions'],), 
                                                              dtype=float),
                                'action_avail': gym.spaces.Box(low=0,
                                                               high=1, 
                                                               shape=(self.fenv_config['n_actions'],), 
                                                               dtype=float),
                                'real_obs': _observation_space_gcn,
                                })
        
        
        
        if self.fenv_config['net_arch'] == 'Cnn': 
            self._observation_space = _observation_space_cnn
            
        elif self.fenv_config['net_arch'] == 'Fc':
            self._observation_space = _observation_space_fc
            
        # elif self.fenv_config['net_arch'] == 'CnnFc':
        #     self._observation_space = gym.spaces.Tuple((_observation_space_cnn,
        #                                                 _observation_space_fc))

        elif self.fenv_config['net_arch'] == 'FcCnn':
            self._observation_space = gym.spaces.Tuple((_observation_space_fc,
                                                        _observation_space_cnn))

        elif self.fenv_config['net_arch'] == 'CnnGcn':
            if self.fenv_config['gcn_obs_method'] == 'embedded_image_graph':
                self._observation_space = _observation_space_gcn
            elif self.fenv_config['gcn_obs_method'] == 'image':
                self._observation_space = _observation_space_cnn
            elif self.fenv_config['gcn_obs_method'] == 'dummy_vector':
                self._observation_space = _observation_space_gcn
            else:
                raise ValueError('Invalid gcn_obs_method!')
                
        elif self.fenv_config['net_arch'] == 'MetaFc':
            self._observation_space = _observation_space_fc
            
        else:
            raise ValueError(f"{self.fenv_config['net_arch']} net_arch does not exist")
            
        return self._observation_space
    
    
    
    def obs_reset(self, episode):
        plan_data_dict = self.plan_construcror.get_plan_data_dict(episode=episode)
        self.input_plan_data_dict = copy.deepcopy(plan_data_dict)
        
        self.wall_names_deque = deque([f"wall_{i+1}" for i in range(plan_data_dict['mask_numbers'], plan_data_dict['mask_numbers']+plan_data_dict['n_walls'])], maxlen=plan_data_dict['n_walls'])
        self.room_names_deque = deque([f"room_{i+1}" for i in range(plan_data_dict['mask_numbers'], plan_data_dict['mask_numbers']+plan_data_dict['n_rooms'])], maxlen=plan_data_dict['n_rooms'])

        self.active_wall_name = None  
        self.active_wall_status = None 
        
        self.observation, plan_data_dict = self._make_observation(plan_data_dict)
        
        if self.fenv_config['action_masking_flag']:
            self.observation = {
                'action_mask': self.action_parser.get_masked_actions(plan_data_dict),
                'action_avail': np.ones(self.fenv_config['n_actions'], dtype=np.int16),
                'real_obs': self.observation}
        
        plan_data_dict.update({'obs_arr_conv': self.observation})
        
        self.done = False
        
        plan_data_dict = self.state_composer.update_block_cells(plan_data_dict)
        
        self.plan_data_dict = copy.deepcopy(plan_data_dict)
        
        return self.observation
        
    
    
    def update(self, episode, action, time_step):
        self.episode = episode
        if time_step > self.fenv_config['stop_time_step']:
            print('wait in update of obervation')
            raise ValueError('time_step went over then the limit!')
            
        plan_data_dict = copy.deepcopy(self.plan_data_dict)
        if self.fenv_config['env_planning'] == "One_Shot":
            if self.fenv_config['learn_room_size_category_order_flag']:
                self.decoded_action_dict = self.action_parser.decode_action(plan_data_dict, action)
            
            if self.decoded_action_dict['action_status'] is not None:
                self.active_wall_status, new_walls_coords = self.action_parser.select_wall(plan_data_dict, 
                                                                                           self.decoded_action_dict)
            else:
                self.active_wall_status = 'rejected_by_missing_room'
            
        elif self.fenv_config['env_planning'] == "Dynamic":
            shifted_wall_names_deque = self.wall_names_deque.copy()
            shifted_wall_names_deque.append(shifted_wall_names_deque[0]) # move the current agent to the last index. Because, we first need to draw the other agent, and then the current agent acts according to them
            previous_wall_id = f"wall_{shifted_wall_names_deque[-2].split('_')[1]}"
            
            self.active_wall_status, new_walls_coords = self.action_parser.transform_walls(action, previous_wall_id) # for asp, actions only include the current agent. So, we only transform the corressponding walls
        
        if self.active_wall_status == "check_room_area":
            self.active_wall_name = self.decoded_action_dict['active_wall_name']
            active_wall_i = self.decoded_action_dict['active_wall_i']
            
            try:
                assert active_wall_i <= plan_data_dict['number_of_total_rooms'], 'active_wall_i is bigger than total_number_of_rooms'
            except:
                print('wait in update of observation')
                raise ValueError('Probably sth need to be match with n_corners')
            
            plan_data_dict = self.plan_construcror.update_plan_with_active_wall(plan_data_dict=plan_data_dict, 
                                                                                walls_coords=new_walls_coords, 
                                                                                active_wall_name=self.active_wall_name)
            
            plan_data_dict = self.painter.updata_obs_mat(plan_data_dict, self.active_wall_name) # here we update plan_data_dict based on the wall order
            
            plan_data_dict = self.rextractor.update_room_dict(plan_data_dict, self.active_wall_name)
            
            self.active_wall_status = self._get_active_wall_status(plan_data_dict, self.active_wall_name)
            
            if self.active_wall_status == "accepted":
                ### Note: active_wall_status will probably change in this section
                plan_data_dict['accepted_actions'].append(action)
                plan_data_dict['wall_types'].update({self.decoded_action_dict['active_wall_name']: self.decoded_action_dict['wall_type']})
                plan_data_dict['room_wall_occupied_positions'].extend(np.argwhere(plan_data_dict['moving_ones']==1).tolist())
                self.wall_names_deque.append(self.wall_names_deque[0])
                
                del plan_data_dict['room_i_per_size_category'][self.decoded_action_dict['room_size_cat_name']][0]
                del plan_data_dict['room_area_per_size_category'][self.decoded_action_dict['room_size_cat_name']][0]   
            
                
                self.done, self.active_wall_status = self._check_terminate(plan_data_dict, 
                                                                           self.active_wall_name, 
                                                                           self.active_wall_status,
                                                                           time_step)
            
            
                self.observation, plan_data_dict = self._make_observation(plan_data_dict,
                                                                          self.active_wall_name,
                                                                          self.active_wall_status)
            
            
                if self.fenv_config['action_masking_flag']:
                    self.observation = {'action_mask': self.action_parser.get_masked_actions(plan_data_dict),
                                        'action_avail': np.ones(self.fenv_config['n_actions'], dtype=np.int16),
                                        'real_obs': self.observation}
                
                plan_data_dict.update({'obs_arr_conv': self.observation})
                plan_data_dict.update({'observation': self.observation})
                
                plan_data_dict = self.state_composer.update_block_cells(plan_data_dict) # Aug 02: not sure if I need to bring it out of 'if'
            
                if self.active_wall_status in ['badly_stopped', 'well_finished']:
                    self.edge_list = self._extract_edge_list(plan_data_dict)
                    plan_data_dict.update({'edge_list': self.edge_list})
            
                self.plan_data_dict = copy.deepcopy(plan_data_dict) # only in this situation I change the self.plan_data_dict
                
            else:
                self.done, self.active_wall_status = self._is_time_over(self.active_wall_status, time_step)
                
        else:
            self.done, self.active_wall_status = self._is_time_over(self.active_wall_status, time_step)
                
        
        return self.observation
    
    
    
    def _get_active_wall_status(self, plan_data_dict, active_wall_name):
        wall_i = int(active_wall_name.split('_')[1])
        room_name = f"room_{wall_i}"

        if len(plan_data_dict['areas']) <= plan_data_dict['number_of_total_walls']:#wall_i != plan_data_dict['number_of_total_walls']:
            if self.fenv_config['is_area_considered'] and self.fenv_config['is_proportion_considered']:
                if self.__check_area_status(plan_data_dict, room_name) and self.__check_proportion_status(plan_data_dict, room_name):
                    return 'accepted'
                else:
                    return 'rejected_by_both_area_and_proportion'
            elif self.fenv_config['is_area_considered'] and not self.fenv_config['is_proportion_considered']:
                if self.__check_area_status(plan_data_dict, room_name):
                    return 'accepted'
                else:
                    return 'rejected_by_area'
            elif not self.fenv_config['is_area_considered'] and self.fenv_config['is_proportion_considered']:
                if self.__check_proportion_status(plan_data_dict, room_name):
                    return 'accepted'
                else:
                    return 'rejected_by_proportion'
            else:
                raise ValueError('No design constraints have been considered!')
                # return 'accepted'
        
        elif len(plan_data_dict['areas']) == plan_data_dict['number_of_total_walls']+1:
            last_room_name = plan_data_dict['last_room']['last_room_name']  
            if self.fenv_config['is_area_considered'] and self.fenv_config['is_proportion_considered']:
                if self.__check_area_status(plan_data_dict, room_name) and self.__check_proportion_status(plan_data_dict, room_name) and \
                    self.__check_area_status(plan_data_dict, last_room_name) and self.__check_proportion_status(plan_data_dict, last_room_name):
                    return 'accepted'
                else:
                    return 'rejected_by_both_area_and_proportion'
            elif self.fenv_config['is_area_considered'] and not self.fenv_config['is_proportion_considered']:
                if self.__check_area_status(plan_data_dict, room_name) and self.__check_area_status(plan_data_dict, last_room_name):
                    return 'accepted'
                else:
                    return 'rejected_by_area'
            elif not self.fenv_config['is_area_considered'] and self.fenv_config['is_proportion_considered']:
                if self.__check_proportion_status(plan_data_dict, room_name) and self.__check_proportion_status(plan_data_dict, last_room_name):
                    return 'accepted'
                else:
                    return 'rejected_by_proportion'
            else:
                raise ValueError('No design constraints have been considered!')
                # return 'accepted'
            
        else:
            raise ValueError('All required walls already have been drawn!')
            
            
    
    def __check_area_status(self, plan_data_dict, room_name):
        # print(f"_get_active_wall_status of observation room_name :{room_name}")
        try:
            active_wall_abs_delta_area = abs(plan_data_dict['delta_areas'][room_name])
        except :
            print("wait in _get_active_wall_status of observation")
            raise ValueError('room_name does not exist!')
        if active_wall_abs_delta_area <= self.fenv_config['area_tolerance']:
            return True
        else:
            return False
        
        
        
    def __check_proportion_status(self, plan_data_dict, room_name):
        # try:
        room_shape = plan_data_dict['rooms_dict'][room_name]['room_shape']
        # if room_shape == 'nonrectangular':
        #     print(f"-- room_shape: {room_shape}")
        delta_aspect_ratio = plan_data_dict['rooms_dict'][room_name]['delta_aspect_ratio']
        # print(f" --- delta_aspect_ratio: {delta_aspect_ratio}")
        
        if delta_aspect_ratio <= self.fenv_config['aspect_ratios_tolerance']:
        # except:
            # print(f"some thing does not work correctly in __check_proportion_status of observation: room_name: {room_name}")
        # if (min(proportions) >= self.fenv_config['min_desired_proportion']) and \
        #     (max(proportions) <= self.fenv_config['max_desired_proportion']):
            return True
        else:
            return False
    
    
    
    def _check_terminate(self, plan_data_dict, active_wall_name, active_wall_status, time_step):
        if len(plan_data_dict['areas']) == plan_data_dict['number_of_total_rooms']:
            done = True
            active_wall_status = 'well_finished'
        
        elif len(plan_data_dict['areas']) < plan_data_dict['number_of_total_walls']+1:
            done, active_wall_status = self._is_time_over(active_wall_status, time_step)
        
        else:
            raise ValueError('n_rooms cannot be bigger than num_desired_rooms')
                
        return done, active_wall_status
    
    
    
    def _is_time_over(self, active_wall_status, time_step):
        done = False
        # if self.fenv_config['random_agent_flag']:
        if time_step >= self.fenv_config['stop_time_step']-1:
            done = True
            active_wall_status = 'badly_stopped'
        return done, active_wall_status
            
    
    
    def _make_observation(self, plan_data_dict, active_wall_name=None, active_wall_status=None):
        if active_wall_name is not None:
            active_room_i = int(active_wall_name.split('_')[1])
            active_room_name = f"room_{active_room_i}"
        
        if self.fenv_config['net_arch'] == 'Cnn':
            self.observation_cnn = self.__get_cnn_obs(plan_data_dict) 
            
        elif self.fenv_config['net_arch'] == 'Fc':
            self.observation_fc = self.__get_fc_obs(plan_data_dict, active_wall_name)
            
        elif self.fenv_config['net_arch'] == 'FcCnn':
            self.observation_cnn = self.__get_cnn_obs(plan_data_dict)
            self.observation_fc = self.__get_fc_obs(plan_data_dict, active_wall_name)
            
        elif self.fenv_config['net_arch'] == 'CnnGcn':
            active_room_data_dict = self.__get_active_room_data_dict(plan_data_dict, active_wall_name)
            
            if active_wall_name is not None:
                plan_data_dict.update({'active_rooms_data_dict': {active_room_name: active_room_data_dict}})
                
                if plan_data_dict['last_room']['last_room_name'] is not None:
                    last_room_name = plan_data_dict['last_room']['last_room_name']
                    last_room_data_dict = copy.deepcopy(active_room_data_dict)
                    last_room_data_dict['current_area'] = plan_data_dict['areas'][last_room_name]
                    last_room_data_dict['delta_area'] = plan_data_dict['delta_areas'][last_room_name]
                    plan_data_dict['active_rooms_data_dict'].update({last_room_name: last_room_data_dict})
                    
                    plan_data_dict = self.__update_graph_data_numpy(plan_data_dict, active_room_name, last_room_name)
                    
                else:
                    plan_data_dict = self.__update_graph_data_numpy(plan_data_dict, active_room_name)
                    
                    
            else:
                plan_data_dict.update({'active_rooms_data_dict': {}})

            self.observation_cnn = self.__get_cnn_obs(plan_data_dict) 
            plan_data_dict['plan_canvas_arr_old'] = copy.deepcopy(plan_data_dict['plan_canvas_arr'])
            plan_data_dict['plan_canvas_arr'] = copy.deepcopy(self.observation_cnn)
            
            if self.fenv_config['gcn_obs_method'] == 'embedded_image_graph':
                self.observation_gcn = self.__get_gcn_obs(plan_data_dict)
            
        elif self.fenv_config['net_arch'] == 'MetaFc':
            self.observation_fc = self.__get_fc_obs(plan_data_dict, active_wall_name)
            
        else:
            raise ValueError(f"{self.fenv_config['net_arch']} net_arch does not exist")
            
            
        if self.fenv_config['net_arch'] == 'Cnn':
            observation = copy.deepcopy(self.observation_cnn)
            
        elif self.fenv_config['net_arch'] == 'Fc':
            observation = copy.deepcopy(self.observation_fc)
            
        elif self.fenv_config['net_arch'] == 'CnnFc':
            observation = (self.observation_cnn, self.observation_fc)

        elif self.fenv_config['net_arch'] == 'FcCnn':
            observation = (self.observation_fc, self.observation_cnn)
            
        elif self.fenv_config['net_arch'] == 'CnnGcn':
            if self.fenv_config['gcn_obs_method'] == 'embedded_image_graph':
                observation = copy.deepcopy(self.observation_gcn)
            elif self.fenv_config['gcn_obs_method'] == 'image':
                observation = copy.deepcopy(self.observation_cnn)
            elif self.fenv_config['gcn_obs_method'] == 'dummy_vector':
                if self.fenv_config['action_masking_flag']:
                    observation = np.zeros(self._observation_space['real_obs'].shape)
                else:   
                    observation = np.zeros(self._observation_space.shape)
            else:
                raise ValueError("gcn_obs_method is unvalid or not-implemented yet!")
              
        elif self.fenv_config['net_arch'] == 'MetaFc':
            observation = copy.deepcopy(self.observation_fc)
            
        else:
            raise ValueError(f"{self.fenv_config['net_arch']} net_arch does not exist")
            
            # assert np.all(self.observation_fc.shape == self.observation_space[0].shape)
            # assert np.all(self.observation_cnn.shape == self.observation_space[1].shape)
        return observation, plan_data_dict
    
    
    
    def __get_cnn_obs(self, plan_data_dict):
        observation_matrix = plan_data_dict['moving_labels']
        canvas_cnn = self.___update_canvas_cnn(plan_data_dict)
        return canvas_cnn
        # if self.fenv_config['n_channels'] == 3:
        #     self.obs_arr_conv = self.state_composer.ta_get_obs_arr_for_conv(observation_matrix)
        # else:
        #     K = 250 // (plan_data_dict['number_of_total_walls']+1) # for normalization
        #     self.obs_arr_conv = np.cast['uint8'](np.expand_dims(observation_matrix*K, axis=2))
    
        # return copy.deepcopy(self.obs_arr_conv)
        
        
        
    def ___update_canvas_cnn(self, plan_data_dict):
        canvas_cnn_room_channel = copy.deepcopy(plan_data_dict['canvas_cnn'])
        moving_labels = copy.deepcopy(plan_data_dict['moving_labels'])
        
        temp_mv = copy.deepcopy(moving_labels)
        mask_numbers = plan_data_dict['mask_numbers']
        for room_i in range(plan_data_dict['number_of_total_rooms'], 0, -1):
            if room_i <= mask_numbers:
                temp_mv[temp_mv == room_i] += 4
            else:
                temp_mv[temp_mv == room_i] += 10
                
        canvas_cnn_room_channel[1:-1, 1:-1] = temp_mv
        
        canvas_cnn_masked_channel = np.zeros(canvas_cnn_room_channel.shape)
        if self.active_wall_status != 'well_finished':
            canvas_cnn_masked_channel[1:-1, 1:-1] = 1 - plan_data_dict['moving_ones']
        
        canvas_cnn = np.zeros((*canvas_cnn_room_channel.shape, 2))
        canvas_cnn[:, :, 0] = canvas_cnn_room_channel
        canvas_cnn[:, :, 1] = canvas_cnn_masked_channel
        return canvas_cnn
    


    def __get_fc_obs(self, plan_data_dict, active_wall_name):
        vector_state_representation = -1 * np.ones(self.state_data_dict['len_state_vec'])
        
        if self.fenv_config['env_planning'] == "One_Shot":
            walls_state_vector, _ = np.array(self.state_composer.wall_data_extractor_for_single_agent(plan_data_dict, 
                                                                                                     active_wall_name), dtype=object)
            
            walls_state_vector = self._normalize_walls_state_vector(walls_state_vector)
            
            vector_state_representation[0:len(walls_state_vector)] = walls_state_vector
            
        else:
            vector_state_representation = np.array(self._wall_data_extractor(plan_data_dict))
            
        if self.fenv_config['use_areas_info_into_observation_flag']:
            desired_areas_dict = plan_data_dict['desired_areas']
            desired_room_names = list(desired_areas_dict.keys()) #  the achived iare 0, but after some timeseteps they might change. see two lines later
            achieved_areas_dict = {room_name:0 for room_name in desired_room_names} # in the begining achieved areas are 0. but a bit later they might be different. see a few lines below
            delta_areas_dict = copy.deepcopy(desired_areas_dict) # in the degining the achieved areas are 0, so the delta = desired
            
            achieved_areas_dict.update({room_name:area for room_name, area in plan_data_dict['areas'].items() if room_name in desired_room_names})
            
            delta_areas_dict.update({room_name:desired_areas_dict[room_name]-achieved_areas_dict[room_name] for room_name in desired_room_names})
            
            desired_areas_list = list(desired_areas_dict.values())
            achieved_areas_list = list(achieved_areas_dict.values())
            delta_areas_list = list(delta_areas_dict.values())
            
            all_area_related_info = np.array(desired_areas_list + achieved_areas_list) # + delta_areas_list)
            all_area_related_info_normalized = all_area_related_info/self.fenv_config['max_y']
            
            areas_state_vector = -1 * np.ones(self.state_data_dict['len_state_vec_for_rooms'])
            try:
                areas_state_vector[:len(all_area_related_info)] = all_area_related_info
            except:
                print('wait in __get_fc_obs of observation')
                raise('the dimentions do not match!')
            
            areas_state_vector = self._normalize_areas_state_vector(areas_state_vector)
            
            vector_state_representation[self.state_data_dict['len_state_vec_for_walls']: self.state_data_dict['len_state_vec_for_walls_rooms']] = areas_state_vector #np.concatenate((vector_state_representation, all_area_related_info))
            
        
        if self.fenv_config['use_edge_info_into_observation_flag']:
            edge_state_vector = self._extract_adj_as_vector(plan_data_dict)
            
            vector_state_representation[self.state_data_dict['len_state_vec_for_walls_rooms']: self.state_data_dict['len_state_vec']] = edge_state_vector
            
        return vector_state_representation.astype(float)

    

    def _normalize_walls_state_vector(self, walls_state_vector):
        normalized = np.array([val/30 if val != -1 else -1 for val in walls_state_vector])
        return normalized
    
    
    
    def _normalize_areas_state_vector(self, areas_state_vector):
        normalized = np.array([val/200 if val != -1 else -1 for val in areas_state_vector])
        return normalized
        


    def __get_active_room_data_dict(self, plan_data_dict, active_wall_name):
        if self.fenv_config['env_planning'] == "One_Shot":
            _, active_room_data_dict = np.array(self.state_composer.wall_data_extractor_for_single_agent(plan_data_dict, 
                                                                                                         active_wall_name))
        
        if active_room_data_dict is not None:
            active_room_name = f"room_{active_wall_name.split('_')[1]}"
            active_room_data_dict['current_area'] = plan_data_dict['areas'][active_room_name]
            active_room_data_dict['delta_area'] = plan_data_dict['delta_areas'][active_room_name]
        return active_room_data_dict
    
    
    
    def __update_graph_data_numpy(self, plan_data_dict, active_room_name, last_room_name=None):
        graph_data_numpy = copy.deepcopy(plan_data_dict['graph_data_numpy'])
        
        ### features
        partially_current_graph_features_dict_numpy = copy.deepcopy(graph_data_numpy['graph_features_numpy']['partially_current_graph_features_dict_numpy'])
        fully_current_graph_features_dict_numpy = copy.deepcopy(graph_data_numpy['graph_features_numpy']['fully_current_graph_features_dict_numpy'])
        
        
        try:
            active_room_data_dict = plan_data_dict['active_rooms_data_dict'][active_room_name]
        except :
            print('wait in __update_graph_data_numpy of observation')
            raise ValueError('There should be sth wrong with last_room')
        
        partially_current_graph_features_dict_numpy[active_room_name]['status'] = 1
        fully_current_graph_features_dict_numpy[active_room_name]['status'] = 1
        
        for k, v in active_room_data_dict.items():
            if 'area' in k:
                partially_current_graph_features_dict_numpy[active_room_name][k] = v
            fully_current_graph_features_dict_numpy[active_room_name][k] = v
            
            
        if last_room_name is not None:
            last_room_data_dict = plan_data_dict['active_rooms_data_dict'][last_room_name]
        
            partially_current_graph_features_dict_numpy[last_room_name]['status'] = 1
            fully_current_graph_features_dict_numpy[last_room_name]['status'] = 1
            
            for k, v in last_room_data_dict.items():
                if 'area' in k:
                    partially_current_graph_features_dict_numpy[last_room_name][k] = v
                fully_current_graph_features_dict_numpy[last_room_name][k] = v
            
        
        graph_data_numpy['graph_features_numpy']['partially_current_graph_features_dict_numpy'] = copy.deepcopy(partially_current_graph_features_dict_numpy)
        graph_data_numpy['graph_features_numpy']['fully_current_graph_features_dict_numpy'] = copy.deepcopy(fully_current_graph_features_dict_numpy)
        
        ### edges
        current_edge_list = self._extract_edge_list(plan_data_dict)
        
        if current_edge_list:
            graph_data_numpy['graph_edge_list_numpy']['partially_current_graph_edge_list_numpy'] = current_edge_list
            graph_data_numpy['graph_edge_list_numpy']['fully_current_graph_edge_list_numpy'] = current_edge_list
        
        
        plan_data_dict['graph_data_numpy_old'] = copy.deepcopy(plan_data_dict['graph_data_numpy'])
        
        plan_data_dict['graph_data_numpy'] = copy.deepcopy(graph_data_numpy)
        
        return plan_data_dict
    
    
    
    def __get_gcn_obs(self, plan_data_dict):
        normalized_graph_data_numpy_old = self.state_composer.graph_normalization(plan_data_dict['graph_data_numpy_old'])
        normalized_graph_data_numpy = self.state_composer.graph_normalization(plan_data_dict['graph_data_numpy'])
        normalized_plan_canvas_arr_old = self.state_composer.image_normalization(plan_data_dict['plan_canvas_arr_old'])
        normalized_plan_canvas_arr = self.state_composer.image_normalization(plan_data_dict['plan_canvas_arr'])
        
        context = {'graph_data_numpy_old': normalized_graph_data_numpy_old,
                   'graph_data_numpy': normalized_graph_data_numpy,
                   'plan_canvas_arr_old': normalized_plan_canvas_arr_old,
                   'plan_canvas_arr': normalized_plan_canvas_arr}
        
        embeded_observation = self.state_composer.get_embeded_observation(context)
        # embeded_observation = np.zeros((self.fenv_config['shape_gcn']))
        return embeded_observation
        
      
        
    def _extract_edge_list(self, plan_data_dict):
        layout_graph = LayoutGraph(plan_data_dict)
        num_nodes, edge_list = layout_graph.extract_graph_data()
        edge_list = [[edge[0], edge[1]] for edge in edge_list]
        edge_list = np.array(edge_list).astype(float).tolist()
        edge_list = [[min(edge), max(edge)] for edge in edge_list]
        return edge_list
        
    
    
    def _extract_adj_as_vector(self, plan_data_dict):
        edge_list = np.array(self._extract_edge_list(plan_data_dict), dtype=int)
        adj_mat_achieved = np.zeros((self.fenv_config['maximum_num_real_rooms'], self.fenv_config['maximum_num_real_rooms']))
        try:
            if len(edge_list) > 0:
                for edge in edge_list:
                    adj_mat_achieved[edge[0]-1][edge[1]-1] = 1
                    adj_mat_achieved[edge[1]-1][edge[0]-1] = 1
        except:
            print('wait in _extract_adj_as_vector of observation')
            raise ValueError('Probably edge_list contains room_name larger then the number of rooms')
            
        adj_vec_achieved = []
        for i, row in enumerate(list(adj_mat_achieved)):
            adj_vec_achieved.extend(row[i+1:])
        adj_vec_achieved = np.array(adj_vec_achieved)
        
        if len(plan_data_dict['adj_vec_desired']) == 0:
            if self.fenv_config['plan_config_source_name'] != 'create_random_config':
                raise ValueError("adj_vec_desired cannot be empty when create_random_config=True")
            else:
                plan_data_dict['adj_vec_desired'] = copy.deepcopy(adj_vec_achieved)
            
        adj_vec = np.concatenate((plan_data_dict['adj_vec_desired'], adj_vec_achieved), axis=-1)
        return adj_vec
        
    
        
    @staticmethod
    def __augment_cnn_observation(observation_arr):
            observation_arr_090 = np.rot90(observation_arr, k=1, axes=(0, 1))
            observation_arr_180 = np.rot90(observation_arr, k=2, axes=(0, 1))
            observation_arr_270 = np.rot90(observation_arr, k=3, axes=(0, 1))
            
            observation_arr_flr = np.fliplr(observation_arr)
            observation_arr_fud = np.flipud(observation_arr)
            
            observation_arr_090_flr = np.fliplr(observation_arr_090)
            observation_arr_090_fud = np.flipud(observation_arr_090)
            
            observation_arr_180_flr = np.fliplr(observation_arr_180)
            observation_arr_180_fud = np.flipud(observation_arr_180)
            
            observation_arr_270_flr = np.fliplr(observation_arr_270)
            observation_arr_270_fud = np.flipud(observation_arr_270)
            
            
            observation_arr = np.concatenate([observation_arr, 
                                              observation_arr_flr,
                                              observation_arr_fud,
                                              
                                              observation_arr_090,
                                              observation_arr_090_flr,
                                              observation_arr_090_fud,
                                               
                                              observation_arr_180,
                                              observation_arr_180_flr,
                                              observation_arr_180_fud,
                                              
                                              observation_arr_270,
                                              observation_arr_270_flr,
                                              observation_arr_270_fud,                                              
                                              ], axis=2)
            return observation_arr
        
        
        
    @staticmethod
    def _cartesian2image_coord(x, y, max_y):
        return max_y-y, x
    
    
#%% 
if __name__ == '__main__':
    from gym_floorplan.envs.fenv_config import LaserWallConfig
    fenv_config = LaserWallConfig().get_config()
    self = Observation(fenv_config)
    episode = None
    observation = self.obs_reset(episode)
    active_wall_name = f"wall_{1+self.plan_data_dict['mask_numbers']}"
    for time_step, action in enumerate([992, 714, 874, 1456, 930, 134, 635]):
        self.update(episode, action, time_step)
        
        if self.active_wall_status == "accepted":
            active_wall_name = f"wall_{int(self.active_wall_name.split('_')[1])+1}"
            
    plan_data_dict = self.plan_data_dict