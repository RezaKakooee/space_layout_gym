#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 17:48:00 2022

@author: Reza Kakooee
"""

#%%

import os
import copy
import inspect
import numpy as np

from gym_floorplan.envs.observation.wall_generator import WallGenerator
from gym_floorplan.envs.observation.wall_transform import WallTransform



#%%
class ActionMapper:
    def __init__(self, zero_index_action_size, wall_lib_length, max_x, max_y):
        self.zero_index_action_size = zero_index_action_size
        self.wall_lib_length = wall_lib_length
        self.max_x = max_x
        self.max_y = max_y
        
    def action_to_id(self, c, i, j, k):
        """Converts an action tuple to a unique identifier."""
        max_x_values = self.max_x // 2 - 1
        max_y_values = self.max_y // 2 - 1
        
        a = (((c * self.wall_lib_length + i) * max_x_values + (j // 2 - 1)) * max_y_values + (k // 2 - 1))
        return a
    
    def id_to_action(self, a):
        """Converts a unique identifier back to an action tuple."""
        max_x_values = self.max_x // 2 - 1
        max_y_values = self.max_y // 2 - 1
        
        k_part = a % max_y_values
        jk_part = a // max_y_values
        j_part = jk_part % max_x_values
        ci_part = jk_part // max_x_values
        i = ci_part % self.wall_lib_length
        c = ci_part // self.wall_lib_length
        
        j = (j_part + 1) * 2
        k = (k_part + 1) * 2
        return (c, i, j, k)
    
    
    
class ActionParser:
    def __init__(self, fenv_config):
        self.fenv_config = fenv_config
        self.action_mapper = ActionMapper(fenv_config['zero_index_action_size'], 
                                          fenv_config['wall_lib_length'], 
                                          fenv_config['max_x'], 
                                          fenv_config['max_y'])
    
    
    
    def get_masked_actions(self, plan_data_dict):
        action_mask_status = True
        action_mask = np.ones(self.fenv_config['n_actions'], dtype=np.int8)
        a = 0
        for c in range( len(self.fenv_config['room_size_category']) ):
            for i in range( len(self.fenv_config['wall_lib']) ):
                for x in range(2, self.fenv_config['max_x']-1, 2):
                    for y in range(2, self.fenv_config['max_y']-1, 2):
                        j, k = self._cartesian2image_coord(x, y, self.fenv_config['max_y'])
                        if plan_data_dict['obs_blocked_cells_by_shift'][j][k] == 1:
                            action_mask[a] = 0
                        a += 1
        
        if sum(action_mask) == 0:
            np.save(f"plan_data_dict__{os.path.basename(__file__)}_{self.__class__.__name__}_{inspect.currentframe().f_code.co_name}_1.npy", self.plan_data_dict)
            action_mask_status = False
            # raise RuntimeError("no action left to be selected!")
            
        return action_mask#, action_mask_status
    
    
    
    def decode_action_from_direct_order_learning(self, plan_data_dict, action):
        decoded_action_dict = {'action': action,
                               'action_status': None,
                               'active_room_i': None,
                               'active_room_area': None,
                               'active_wall_i': None,
                               'active_wall_name': None,
                               'active_w_coords': None,
                               'wall_type': None}
        
        try:
            if not isinstance(action, int):
                action = int(action)
            # r_i, w_i, x, y = self.fenv_config['action_to_acts_tuple_dic'][action]
            r_i, w_i, x, y = self.action_mapper.id_to_action(action)
            # print(r_i, w_i, x, y)
        except:
            np.save(f"plan_data_dict__{__file__}_{self.__class__.__name__}_{inspect.currentframe().f_code.co_name}_0.npy", plan_data_dict)
            raise ValueError(f"There is something wrong with the action of {action}. plan_id is: {plan_data_dict['plan_id']}")
        
        largest_room_count = plan_data_dict['n_rooms']
        if self.fenv_config['removing_lv_room_from_action_set']:
            largest_room_count -= 1
        if r_i < largest_room_count: # if r_i > plan_data_dict['n_rooms'], the action has to be discarded
            r_i += self.fenv_config['min_room_id'] 
            if self.fenv_config['removing_lv_room_from_action_set']:
                r_i += 1
            room_name = f"room_{r_i}"
            if room_name not in plan_data_dict['areas_achieved']:
                decoded_action_dict['action_status'] = 'check'
                
                decoded_action_dict['active_room_i'] = r_i
                decoded_action_dict['active_room_area'] = room_name
                
                decoded_action_dict['active_wall_i'] = decoded_action_dict['active_room_i']
                decoded_action_dict['active_wall_name'] = f"wall_{decoded_action_dict['active_room_i']}"
                
                # print(f"_decode_aciotn of observation: active_wall_name: {decoded_action_dict['active_wall_name']} ")
                
                decoded_action_dict['active_w_coords'] = np.array(self.fenv_config['wall_lib'][w_i])
                decoded_action_dict['active_w_coords'][:,0] += x
                decoded_action_dict['active_w_coords'][:,1] += y
                
                decoded_action_dict['wall_type'] = self.fenv_config['wall_type'][w_i]
        #     else:
        #         print(f"Room {r_i} is selcted, while it has been already drawn {plan_data_dict['areas_achieved'].keys()}")
        # else:
        #     print(f"The selected action have r_i > largest_room_count: {r_i}>{largest_room_count}")
            
        return decoded_action_dict
            
        
        
    def decode_action(self, plan_data_dict, action):
        decoded_action_dict = {'action': action,
                               'action_status': None,
                               'room_size_cat_name': None,
                               'active_room_i': None,
                               'active_room_area': None,
                               'active_wall_i': None,
                               'active_wall_name': None,
                               'active_w_coords': None,
                               'wall_type': None}
        
        try:
            if not isinstance(action, int):
                action = int(action)
            # c_i, w_i, x, y = self.fenv_config['action_to_acts_tuple_dic'][action]
            c_i, w_i, x, y = self.action_mapper.id_to_action(action)
            # print(c_i, w_i, x, y)
        except:
            np.save(f"plan_data_dict__{__file__}_{self.__class__.__name__}_{inspect.currentframe().f_code.co_name}_1.npy", self.plan_data_dict)
            raise ValueError(f"There is something wrong with the action of {action}. plan_id is: {plan_data_dict['plan_id']}")
        
        room_size_cat_name = self.fenv_config['room_size_category'][c_i]
        decoded_action_dict['room_size_cat_name'] = room_size_cat_name
        
        if len(plan_data_dict['room_area_per_size_category'][room_size_cat_name]) >= 1:
            decoded_action_dict['action_status'] = 'check'
            
            decoded_action_dict['active_room_i'] = plan_data_dict['room_i_per_size_category'][room_size_cat_name][0]
            decoded_action_dict['active_room_area'] = plan_data_dict['room_area_per_size_category'][room_size_cat_name][0]
            
            decoded_action_dict['active_wall_i'] = decoded_action_dict['active_room_i']
            decoded_action_dict['active_wall_name'] = f"wall_{decoded_action_dict['active_room_i']}"
            
            # print(f"_decode_aciotn of observation: active_wall_name: {decoded_action_dict['active_wall_name']} ")
            
            decoded_action_dict['active_w_coords'] = np.array(self.fenv_config['wall_lib'][w_i])
            decoded_action_dict['active_w_coords'][:,0] += x
            decoded_action_dict['active_w_coords'][:,1] += y
            
            decoded_action_dict['wall_type'] = self.fenv_config['wall_type'][w_i]
            
        return decoded_action_dict
            
    
    
    def select_wall(self, plan_data_dict, decoded_action_dict=None):
        current_walls_coords = plan_data_dict['walls_coords']
        new_walls_coords = copy.deepcopy(current_walls_coords)
        num_current_walls = len(current_walls_coords)
        
        ### select a new wall
        new_wall_name = decoded_action_dict['active_wall_name'] # f"wall_{num_current_walls+1}"
        w_coords = decoded_action_dict['active_w_coords']
        
        try:
            new_wall = {'back_open_coord': w_coords[0],
                        'anchor_coord': w_coords[1],
                        'front_open_coord': w_coords[2]}
        except: 
            np.save(f"plan_data_dict__{os.path.basename(__file__)}_{self.__class__.__name__}_{inspect.currentframe().f_code.co_name}_1.npy", self.plan_data_dict)
            raise ValueError(f"The w_coords is wrogn: {w_coords}. plan_id is: plan_data_dict['plan_id']")
            
        back_position = self._cartesian2image_coord(new_wall['back_open_coord'][0], 
                                                    new_wall['back_open_coord'][1], 
                                                    self.fenv_config['max_y'])
        
        anchor_position = self._cartesian2image_coord(new_wall['anchor_coord'][0], 
                                                      new_wall['anchor_coord'][1], 
                                                      self.fenv_config['max_y'])
        
        front_position = self._cartesian2image_coord(new_wall['front_open_coord'][0], 
                                                     new_wall['front_open_coord'][1], 
                                                     self.fenv_config['max_y'])
        
        if self.fenv_config['mask_flag']:
            if list(anchor_position) in plan_data_dict['room_wall_occupied_positions']:
                active_wall_status = "rejected_by_room"
            elif tuple(back_position) in plan_data_dict['outside_positions']:
                active_wall_status = "rejected_by_canvas"
            elif tuple(anchor_position) in plan_data_dict['outside_positions']:
                active_wall_status = "rejected_by_canvas"
            elif tuple(front_position) in plan_data_dict['outside_positions']:
                active_wall_status = "rejected_by_canvas"
            elif plan_data_dict['obs_blocked_cells'][anchor_position] == 1: 
                active_wall_status = "rejected_by_other_blocked_cells"
            elif (anchor_position[0] % 2 != 0) or (anchor_position[1] % 2 != 0):
                active_wall_status = "rejected_by_odd_anchor_coord"
            else:
                active_wall_status = "check_room_area"
        else:
            if list(anchor_position) in plan_data_dict['room_wall_occupied_positions']:
                active_wall_status = "rejected_by_room"
            elif plan_data_dict['obs_blocked_cells'][anchor_position] == 1: 
                active_wall_status = "rejected_by_other_blocked_cells"
            elif (anchor_position[0] % 2 != 0) or (anchor_position[1] % 2 != 0):
                active_wall_status = "rejected_by_odd_anchor_coord"
            else:
                active_wall_status = "check_room_area"
        
        if active_wall_status == "check_room_area":
            # valid_points_for_sampling = self._get_valid_points_for_sampling(plan_data_dict)
            new_wall_coords = WallGenerator(self.fenv_config).make_walls(plan_data_dict, 
                                                                         new_wall, 
                                                                         wall_name=new_wall_name)
            new_walls_coords.update(new_wall_coords)
        
        return active_wall_status, new_walls_coords
    
    
    
    def transform_walls(self, plan_data_dict, actions, previous_wall_id): # in asp actions includes the current wall, so we only trasnform the current wall
        current_walls_coords = plan_data_dict[0]['walls_coords'] # current_walls_coords means the walls before transformation
        new_walls_coords = copy.deepcopy(current_walls_coords)
        current_moving_wall_names = [f"wall_{agent_name.split('_')[1]}" for agent_name in actions.keys()]
        current_moving_wall_coords = {wall_name:current_walls_coords[wall_name] for wall_name in current_moving_wall_names}
        for (wall_name, wall_coords), (action_for_agent, action) in zip(current_moving_wall_coords.items(), actions.items()):
            if self.fenv_config['action_dict'][action] == 'no_action':
                pass
            else:
                n_wall = WallTransform(wall_name, wall_coords, action, plan_data_dict, previous_wall_id, self.fenv_config).transform()
                new_walls_coords[wall_name] = list(n_wall.values())[0]
        self.__check_new_walls_coords(new_walls_coords)      
        active_wall_status = "check_room_area"
        return active_wall_status, new_walls_coords
        
        
        
    def setup_walls(self, plan_data_dict):
        valid_points_for_sampling = self._get_valid_points_for_sampling(plan_data_dict)
        walls_coords = WallGenerator(fenv_config=self.fenv_config).make_walls(valid_points_for_sampling)
        if not isinstance(walls_coords, dict):
            raise TypeError(f"In setup_walls of observation, the walls_coords must be dict, while the current type is: {type(walls_coords)}")
        plan_data_dict['iwalls_coords'] = walls_coords
        return plan_data_dict
    
    
    
    def _get_valid_points_for_sampling(self, plan_data_dict):
        valid_points_for_sampling = np.argwhere(plan_data_dict['obs_moving_ones']==0)
        valid_points_for_sampling = [[r, c] for r, c in valid_points_for_sampling if (r%2==0 and c%2==0 and r!=0 and c!=0 and r!=self.fenv_config['max_x'] and c!=self.fenv_config['max_y'])]
       
        return np.array(valid_points_for_sampling)
    
    
    
    @staticmethod
    def _cartesian2image_coord(x, y, max_y):
        xx = copy.deepcopy(x)
        yy = copy.deepcopy(y)
        return max_y-yy, xx