#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 19:20:46 2023

@author: Reza Kakooee
"""

import os
import inspect
import numpy as np



#%%
class RewardSensor:
    def __init__(self, fenv_config, plan_data_dict, active_wall_name, active_wall_status, done):
        self.fenv_config = fenv_config
        self.plan_data_dict = plan_data_dict
        self.active_wall_name = active_wall_name
        self.active_wall_status = active_wall_status
        self.done = done
        
        
        
    def inspect(self):
        inspection_output_dict = {'geometry': {}, 'topology': {}}
        def _inpect_a_room_geometry(room_name):
            active_room_desired_area = self.plan_data_dict['areas_desired'][room_name]
            active_room_achieved_area = self.plan_data_dict['areas_achieved'][room_name]
            active_room_achieved_aspect_ratio = self.plan_data_dict['rooms_dict'][room_name]['delta_aspect_ratio'] + self.fenv_config['desired_aspect_ratio']
            active_room_desired_aspect_ratio = self.fenv_config['desired_aspect_ratio']
            
            delta_area = abs(active_room_desired_area - active_room_achieved_area)
            delta_aspect_ratio = abs(active_room_desired_aspect_ratio - active_room_achieved_aspect_ratio)
            
            inspection_output_dict['geometry'].update({
                room_name: {
                    'desired_area': active_room_desired_area,
                    'achieved_area': active_room_achieved_area,
                    'achieved_aspect_ratio': active_room_achieved_aspect_ratio,
                    'desired_aspect_ratio': self.fenv_config['desired_aspect_ratio'],
                    'delta_area': delta_area,
                    'delta_aspect_ratio': delta_aspect_ratio,
                }
            })

        if not self.done:
                if 'reject' not in self.active_wall_status:
                    active_room_name = f"room_{self.active_wall_name.split('_')[1]}"
                    try:
                        _inpect_a_room_geometry(active_room_name)
                    except:
                        np.save(f"plan_data_dict__{os.path.basename(__file__)}_{self.__class__.__name__}_{inspect.currentframe().f_code.co_name}.npy", self.plan_data_dict)
                        raise ValueError(f"active_room_name of {active_room_name} dose not exist!")
        else:
            for room_name in self.plan_data_dict['areas_achieved'].keys():
                room_i = int(room_name.split('_')[1])
                if room_i in self.fenv_config['real_room_id_range']:
                    _inpect_a_room_geometry(room_name)

            def _inspect_topology():
                lvr_ent_edge = [self.plan_data_dict['lvroom_id'], self.fenv_config['entrance_cell_id']]
                ent_lvr_edge = [self.fenv_config['entrance_cell_id'], self.plan_data_dict['lvroom_id']]
                
                edge_list_room_desired = self.plan_data_dict['edge_list_room_desired']
                
                edge_color_data_dict_room_green = self.plan_data_dict['edge_color_data_dict_room']['green']
                edge_color_data_dict_room_red = self.plan_data_dict['edge_color_data_dict_room']['red']
                
                edge_list_facade_desired_str = self.plan_data_dict['edge_list_facade_desired_str']
                edge_color_data_dict_facade_green = self.plan_data_dict['edge_color_data_dict_facade']['green']
                edge_color_data_dict_facade_red = self.plan_data_dict['edge_color_data_dict_facade']['red']
                if self.fenv_config['adaptive_window']:
                    edge_color_data_dict_facade_adaptive_rooms = self.plan_data_dict['edge_color_data_dict_facade']['adaptive_rooms']
                    edge_color_data_dict_facade_blind_rooms = self.plan_data_dict['edge_color_data_dict_facade']['blind_rooms']
                    edge_color_data_dict_facade_sighted_rooms = self.plan_data_dict['edge_color_data_dict_facade']['sighted_rooms']
                
                
                n_desired_room_room_connections = len(edge_list_room_desired)  # 6
                if self.fenv_config['is_entrance_lvroom_connection_a_constraint']: 
                   if  ( lvr_ent_edge in edge_list_room_desired or 
                         ent_lvr_edge in edge_list_room_desired):
                       n_desired_room_room_connections -= 1 # subtracted by 1 to exclude # lvroom-entrance connection
                else:
                    n_desired_room_room_connections += self.fenv_config['weight_for_missing_entrance_lvroom_connection'] - 1 # it is automatically adjusted as weight can be 1 and 1-1=0
                
                
                if self.fenv_config['adaptive_window']:
                    n_desired_room_facade_connections = len(edge_color_data_dict_facade_adaptive_rooms)
                    if ( self.fenv_config['is_entrance_lvroom_connection_a_constraint'] and 
                         self.plan_data_dict['lvroom_id'] in edge_color_data_dict_facade_adaptive_rooms):
                        n_desired_room_facade_connections -= 1
                else:
                    n_desired_room_facade_connections = len(edge_list_facade_desired_str) 
                    if not self.fenv_config['lv_mode'] and self.fenv_config['is_entrance_lvroom_connection_a_constraint']:
                        for ed in self.plan_data_dict['edge_list_facade_desired_str']:
                            if ( self.plan_data_dict['lvroom_id'] in ed or
                                 'd' in ed):
                               n_desired_room_facade_connections -= 1
                
                n_desired_entrance_lvroom_connections = 0 if self.fenv_config['is_entrance_lvroom_connection_a_constraint'] else 1
                
                n_desired_connections = n_desired_room_room_connections + n_desired_room_facade_connections + n_desired_entrance_lvroom_connections
                ## so far we count the whole number of desired connections.  
                
                
                n_missed_room_room_connections = len(edge_color_data_dict_room_red)
                
                if self.fenv_config['adaptive_window']:
                    n_missed_room_facade_connections = len(edge_color_data_dict_facade_blind_rooms)
                else:
                    n_missed_room_facade_connections = len(edge_color_data_dict_facade_red)
                    
                
                if self.fenv_config['is_entrance_lvroom_connection_a_constraint']:
                    if len(self.plan_data_dict['edge_color_data_dict_entrance']['red']) >= 1: # the only possibility is that: maybe entrance-lvroom connection missed; no matter what self.fenv_config['is_entrance_lvroom_connection_a_constraint'] is
                        n_missed_entrance_lvroom_connections = 1 
                    else:
                        n_missed_entrance_lvroom_connections = 0
                else:
                    n_missed_entrance_lvroom_connections = 0
                    
                
                n_missed_connections = n_missed_room_room_connections + n_missed_room_facade_connections + n_missed_entrance_lvroom_connections
                ## so far we count the whole number of missed connections. 
                
                
                n_nicely_achieved_room_room_connections = len(edge_color_data_dict_room_green)
                if self.fenv_config['is_entrance_lvroom_connection_a_constraint']: 
                   if  ( lvr_ent_edge in edge_list_room_desired or 
                         ent_lvr_edge in edge_list_room_desired): 
                       n_nicely_achieved_room_room_connections -= 1 
                
                if self.fenv_config['adaptive_window']:
                    sighted_real_rooms = [r for r in edge_color_data_dict_facade_sighted_rooms if r in self.fenv_config['real_room_id_range']]
                    n_nicely_achieved_room_facade_connections = len(sighted_real_rooms)
                    if ( self.fenv_config['is_entrance_lvroom_connection_a_constraint'] and 
                         self.plan_data_dict['lvroom_id'] in edge_color_data_dict_facade_sighted_rooms):
                        n_nicely_achieved_room_facade_connections -= 1

                else:
                    n_nicely_achieved_room_facade_connections = len(edge_color_data_dict_facade_green)
                    if not self.fenv_config['lv_mode'] and self.fenv_config['is_entrance_lvroom_connection_a_constraint']:
                        for ed in self.plan_data_dict['edge_color_data_dict_facade']['green']:
                            try:
                                if ( self.plan_data_dict['lvroom_id'] in ed or
                                     'd' in ed):
                                   n_nicely_achieved_room_facade_connections -= 1
                            except:
                                raise TypeError("TypeError: 'in <string>' requires string as left operand, not int")
                                
                if len(self.plan_data_dict['edge_color_data_dict_entrance']['green']) >= 1:                     
                    n_nicely_achieved_entrance_lvroom_connections = 0 if self.fenv_config['is_entrance_lvroom_connection_a_constraint'] else self.fenv_config['weight_for_missing_entrance_lvroom_connection']
                else:
                    n_nicely_achieved_entrance_lvroom_connections = 0
                    
                
                n_nicely_achieved_connections = n_nicely_achieved_room_room_connections + n_nicely_achieved_room_facade_connections + n_nicely_achieved_entrance_lvroom_connections
                

                try:
                    if self.plan_data_dict['active_wall_status'] == 'well_finished' and self.fenv_config['plan_config_source_name'] != 'create_random_config':
                        assert n_nicely_achieved_connections == n_desired_connections - n_missed_connections, "Sth is wrong in calculating the adj performance"
                except Exception as e:
                    np.save(f"plan_data_dict__{os.path.basename(__file__)}_{self.__class__.__name__}_{inspect.currentframe().f_code.co_name}.npy", self.plan_data_dict)
                    raise ValueError(f"""Sth is wrong in calculating the adj performance for badly_finished situation
                                      n_nicely_achieved_connections: {n_nicely_achieved_connections}, 
                                      n_desired_connections: {n_desired_connections}, 
                                      n_missed_connections: {n_missed_connections}, 
                                      plan_id: {self.plan_data_dict['plan_id']}, 
                                      """) from e
                    
                inspection_output_dict['topology'].update({
                    'n_desired_connections': n_desired_connections,
                    'n_nicely_achieved_connections': n_nicely_achieved_connections,
                    'n_missed_connections': n_missed_connections,
                })
                
            _inspect_topology()
        return inspection_output_dict