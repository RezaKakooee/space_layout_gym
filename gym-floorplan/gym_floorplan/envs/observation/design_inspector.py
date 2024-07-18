#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 16:11:14 2023

@author: Reza Kakooee
"""

import os
import copy
import inspect
import itertools
import numpy as np

from skimage import graph
from skimage import filters

from gym_floorplan.envs.observation.layout_graph import LayoutGraph



#%%
class DesignInspector:
    def __init__(self, fenv_config):
        self.fenv_config = fenv_config
        
    
    
    def inspect_constraints(self, plan_data_dict, active_wall_name):
        active_wall_status = self._get_active_wall_status(plan_data_dict, active_wall_name)
        # print(f"active_wall_status: {active_wall_status}")
        return active_wall_status
    
    
    
    def inspect_objective(self, plan_data_dict):
        plan_data_dict = self._extract_edge_list(plan_data_dict)
        return plan_data_dict
    
        

    def _get_active_wall_status(self, plan_data_dict, active_wall_name):
        wall_i = int(active_wall_name.split('_')[1])
        room_name = f"room_{wall_i}"
        
        if len(plan_data_dict['areas_achieved']) <= plan_data_dict['number_of_total_walls']:
            if self.fenv_config['is_area_a_constraint'] and self.fenv_config['is_proportion_a_constraint']:
                if self._check_area_status(plan_data_dict, room_name) and self._check_proportion_status(plan_data_dict, room_name):
                    w_status = 'accepted' 
                else:
                    w_status = 'rejected_by_both_area_and_proportion'
            elif self.fenv_config['is_area_a_constraint'] and not self.fenv_config['is_proportion_a_constraint']:
                if self._check_area_status(plan_data_dict, room_name):
                    w_status = 'accepted'
                else:
                    w_status = 'rejected_by_area' 
            elif not self.fenv_config['is_area_a_constraint'] and self.fenv_config['is_proportion_a_constraint']:
                if self._check_proportion_status(plan_data_dict, room_name):
                    w_status = 'accepted'
                else:
                    w_status = 'rejected_by_proportion'
            else:
                if self.fenv_config['zc_has_geom_tolerance'] and self._check_geom_in_zc(plan_data_dict, room_name):
                    w_status = 'rejected_by_both_area_or_proportion'
                else:
                    w_status = 'accepted' 
                # raise ValueError('No design constraints have been considered!')
        
        elif len(plan_data_dict['areas_achieved']) == plan_data_dict['number_of_total_walls']+1:
            last_room_name = plan_data_dict['last_room']['last_room_name']  
            if self.fenv_config['lv_mode'] and last_room_name == 'room_11':
                if self.fenv_config['is_area_a_constraint'] and self.fenv_config['is_proportion_a_constraint']:
                    if self._check_area_status(plan_data_dict, room_name) and self._check_proportion_status(plan_data_dict, room_name):
                        w_status = 'accepted'
                    else:
                        w_status = 'rejected_by_both_area_and_proportion'
                elif self.fenv_config['is_area_a_constraint'] and not self.fenv_config['is_proportion_a_constraint']:
                    if self._check_area_status(plan_data_dict, room_name):
                        w_status = 'accepted'
                    else:
                        w_status = 'rejected_by_area'
                elif not self.fenv_config['is_area_a_constraint'] and self.fenv_config['is_proportion_a_constraint']:
                    if self._check_proportion_status(plan_data_dict, room_name):
                        w_status = 'accepted'
                    else:
                        w_status = 'rejected_by_proportion'
                else:
                    if self.fenv_config['zc_has_geom_tolerance'] and self._check_geom_in_zc(plan_data_dict, room_name):
                        w_status = 'rejected_by_both_area_or_proportion'
                    else:
                        w_status = 'accepted' # TODO : this needs caution
                    # raise ValueError('No design constraints have been considered!')
            else:
                if self.fenv_config['is_area_a_constraint'] and self.fenv_config['is_proportion_a_constraint']:
                    if self._check_area_status(plan_data_dict, room_name) and self._check_proportion_status(plan_data_dict, room_name) and \
                        self._check_area_status(plan_data_dict, last_room_name) and self._check_proportion_status(plan_data_dict, last_room_name):
                        w_status = 'accepted'
                    else:
                        w_status = 'rejected_by_both_area_and_proportion'
                elif self.fenv_config['is_area_a_constraint'] and not self.fenv_config['is_proportion_a_constraint']:
                    if self._check_area_status(plan_data_dict, room_name) and self._check_area_status(plan_data_dict, last_room_name):
                        w_status = 'accepted'
                    else:
                        w_status = 'rejected_by_area'
                elif not self.fenv_config['is_area_a_constraint'] and self.fenv_config['is_proportion_a_constraint']:
                    if self._check_proportion_status(plan_data_dict, room_name) and self._check_proportion_status(plan_data_dict, last_room_name):
                        w_status = 'accepted'
                    else:
                        w_status = 'rejected_by_proportion'
                else:
                    if self.fenv_config['zc_has_geom_tolerance'] and self._check_geom_in_zc(plan_data_dict, room_name, last_room_name):
                        w_status = 'rejected_by_both_area_or_proportion'
                    else:
                        w_status = 'accepted' # TODO : this needs caution
                    # raise ValueError('No design constraints have been considered!')
            
        else:
            np.save(f"plan_data_dict__{os.path.basename(__file__)}_{self.__class__.__name__}_{inspect.currentframe().f_code.co_name}.npy", self.plan_data_dict)
            message = f"""All required walls already have been drawn!
            The current room_name is: {room_name}, 
            The areas_achieved are: {plan_data_dict['areas_achieved']}, 
            number_of_total_walls is: {plan_data_dict['number_of_total_walls']}, 
            plan_id is: {plan_data_dict['plan_id']}            
            """
            raise ValueError(message)
           
        if 'rejected' not in w_status:
            if self.fenv_config['is_entrance_a_constraint']:
                if not self._check_entrance_status(plan_data_dict, active_wall_name):
                    w_status = 'rejected_by_entrance'
                    return w_status
            
            if ( self.fenv_config['is_entrance_adjacency_a_constraint'] or 
                 self.fenv_config['is_entrance_lvroom_connection_a_constraint'] ):
                tmp_edge_list = self._get_tmp_edge_list(plan_data_dict)
            
                if ( self.fenv_config['is_entrance_adjacency_a_constraint']):# and 
                     # self.fenv_config['is_adjacency_considered']): # TODO
                    if not self._check_entrance_adjacency_status(plan_data_dict, tmp_edge_list):
                        w_status = 'rejected_by_entrance'
                        
                # this checks if lvroom is connected to the entrance
                # this is required, as the above alone cannot check the entrance-lvroom connecion
                if ( self.fenv_config['is_entrance_lvroom_connection_a_constraint'] and 
                      self.fenv_config['lvroom_id'] in plan_data_dict['obs_moving_labels'] ):
                    if not self._check_lvroom_adjacency_status(plan_data_dict, tmp_edge_list): 
                        w_status = 'rejected_by_lvroom'
            
        return w_status
            
      
        
    def _check_geom_in_zc(self, plan_data_dict, room_name, last_room_name=None):
        room_data = plan_data_dict['rooms_dict'][room_name]
        asp_ratio = room_data['room_aspect_ratio'] if room_data['room_shape'] == 'rectangular' else room_data['aspect_ratio']
        goem_status = (abs(plan_data_dict['areas_delta'][room_name]) > self.fenv_config['area_tolerance_for_zero_constraint'] or 
               abs(room_data['delta_aspect_ratio']) > self.fenv_config['aspect_ratios_tolerance_for_zero_constraint'] or 
               plan_data_dict['areas_achieved'][room_name] < self.fenv_config['min_acceptable_area'] or 
               asp_ratio > self.fenv_config['max_acceptable_aspect_ratio'])
        
        if last_room_name is not None:
            last_room_data = plan_data_dict['rooms_dict'][last_room_name]
            goem_status = ( goem_status or 
                            (abs(plan_data_dict['areas_delta'][last_room_name]) > self.fenv_config['area_tolerance_for_zero_constraint'] or 
                             abs(last_room_data['delta_aspect_ratio']) > self.fenv_config['aspect_ratios_tolerance_for_zero_constraint'] or
                             plan_data_dict['areas_achieved'][last_room_name] < self.fenv_config['min_acceptable_area'])
                        )
         
        return goem_status
      
        
    
    def _get_tmp_edge_list(self, plan_data_dict):
        obs_moving_labels_modified = copy.deepcopy(np.array(plan_data_dict['obs_moving_labels']))
        # obs_moving_labels_modified[obs_moving_labels_modified == 0] = np.max(obs_moving_labels_modified)+1
        obs_moving_labels_modified = obs_moving_labels_modified * plan_data_dict['obs_mat_for_dot_prod']
        obs_moving_labels_modified = obs_moving_labels_modified - plan_data_dict['obs_mat_w']
        for r, c in plan_data_dict['entrance_positions']:
            obs_moving_labels_modified[r][c] = self.fenv_config['entrance_cell_id']
            
        img = np.expand_dims(obs_moving_labels_modified, axis=-1)
        edge_map = filters.sobel(img)
        edge_map = edge_map[:,:,0]
        rag = graph.rag_boundary(obs_moving_labels_modified.astype(int), edge_map, connectivity=1)
        tmp_edge_list = list(rag.edges)
        return tmp_edge_list
        
    
    
    def _check_entrance_adjacency_status(self, plan_data_dict, tmp_edge_list):
        edge_list_only_connected_to_entrance = list(filter(lambda x: self.fenv_config['entrance_cell_id'] in x, tmp_edge_list))
        rooms_connected_to_entrance = set(itertools.chain(*edge_list_only_connected_to_entrance)).difference({self.fenv_config['entrance_cell_id']})
        real_room_connected_to_entrance = [room_i for room_i in rooms_connected_to_entrance if room_i in range(self.fenv_config['min_room_id'], self.fenv_config['max_room_id']+1)]
        for room_i in real_room_connected_to_entrance:
            if room_i != self.fenv_config['lvroom_id']:
                return False
        return True
    
    
    
    def _check_lvroom_adjacency_status(self, plan_data_dict, tmp_edge_list):
        edge_list_only_connected_to_lvroom = list(filter(lambda x: self.fenv_config['lvroom_id'] in x, tmp_edge_list))
        rooms_connected_to_lvroom = set(itertools.chain(*edge_list_only_connected_to_lvroom))
        if self.fenv_config['entrance_cell_id'] not in rooms_connected_to_lvroom:
            return False
        return True
        
        
        
    def _check_entrance_status(self, plan_data_dict, active_wall_name):
        active_wall_anchor_coord = plan_data_dict['walls_coords'][active_wall_name]['anchor_coord']
        active_wall_front_reflection_coord = plan_data_dict['walls_coords'][active_wall_name]['front_segment']['reflection_coord']
        active_wall_back_reflection_coord = plan_data_dict['walls_coords'][active_wall_name]['back_segment']['reflection_coord']
        
        if (list(active_wall_anchor_coord) in plan_data_dict['extended_entrance_coords'] or
            list(active_wall_front_reflection_coord) in plan_data_dict['extended_entrance_coords'] or
            list(active_wall_back_reflection_coord) in plan_data_dict['extended_entrance_coords']):
           return False
        
        active_wall_anchor_position = self._cartesian2image_coord(active_wall_anchor_coord[0], active_wall_anchor_coord[1], self.fenv_config['max_y'])
        active_wall_front_reflection_position = self._cartesian2image_coord(active_wall_front_reflection_coord[0], active_wall_front_reflection_coord[1], self.fenv_config['max_y'])
        active_wall_back_reflection_position = self._cartesian2image_coord(active_wall_back_reflection_coord[0], active_wall_back_reflection_coord[1], self.fenv_config['max_y'])
        
        if (list(active_wall_anchor_position) in plan_data_dict['extended_entrance_positions'] or
            list(active_wall_front_reflection_position) in plan_data_dict['extended_entrance_positions'] or
            list(active_wall_back_reflection_position) in plan_data_dict['extended_entrance_positions']):
           return False
        
        return True
                
                
                
    def _check_area_status(self, plan_data_dict, room_name):
        active_wall_area = plan_data_dict['areas_achieved'][room_name]
        if active_wall_area < self.fenv_config['min_acceptable_area']:
            return False
        try:
            active_wall_abs_delta_area = abs(plan_data_dict['areas_delta'][room_name])
        except :
            print("wait in _get_active_wall_status of observation")
            np.save('plan_data_dict__in__design_inspector__check_area_status.npy', plan_data_dict)
            messsage = f"""
            room_name of {room_name} does noe exist.
            plan_id is: {plan_data_dict['plan_id']}
            
            """
            raise ValueError(messsage)
        if active_wall_abs_delta_area <= self.fenv_config['area_tolerance']:
            return True
        else:
            return False
        
        
        
    def _check_proportion_status(self, plan_data_dict, room_name):
        room_i = int(room_name.split('_')[1])
        if ('aspect_ratio_desired' in plan_data_dict.keys() and room_i > self.fenv_config['lvroom_id']):
            room_shape = plan_data_dict['rooms_dict'][room_name]['room_shape']
            if room_shape == 'rectangular':
                prop = plan_data_dict['rooms_dict'][room_name]['room_aspect_ratio']
            else:
                prop = plan_data_dict['rooms_dict'][room_name]['aspect_ratio']
        if prop > self.fenv_config['max_acceptable_aspect_ratio']:
            return False
        #     
        #     pdelta = abs(prop - plan_data_dict['aspect_ratio_desired'][room_name])
        #     return pdelta <= <= self.fenv_config['aspect_ratios_tolerance'] 
        delta_aspect_ratio = plan_data_dict['rooms_dict'][room_name]['delta_aspect_ratio']
        return delta_aspect_ratio <= self.fenv_config['aspect_ratios_tolerance']    
    
    
    
    def _extract_edge_list(self, plan_data_dict, end_of_episode=True):
        layout_graph = LayoutGraph(plan_data_dict, self.fenv_config)
        edge_dict = layout_graph.extract_graph_data()
        
        edge_color_data_dict_room, edge_color_data_dict_facade, edge_color_data_dict_entrance = self._get_edge_colors(plan_data_dict,
                                                                                                                      edge_dict['edge_list_room_achieved'], 
                                                                                                                      edge_dict['edge_list_facade_achieved_str'], 
                                                                                                                      edge_dict['edge_list_entrance_achieved_str'])
        
        if not end_of_episode:
            pass
        
        
        
        
        
        
        
        
        
        plan_data_dict.update({'edge_list_room_achieved': edge_dict['edge_list_room_achieved'],
                               'edge_list_facade_achieved': edge_dict['edge_list_facade_achieved'],
                               'edge_list_facade_achieved_str': edge_dict['edge_list_facade_achieved_str'],
                               'edge_list_entrance_achieved': edge_dict['edge_list_entrance_achieved'],
                               'edge_list_entrance_achieved_str': edge_dict['edge_list_entrance_achieved_str'],
                               'state_adj_matrix': edge_dict['state_adj_matrix'],
                               'edge_dict': edge_dict})
        
        plan_data_dict.update({'edge_color_data_dict_room': edge_color_data_dict_room})
        plan_data_dict.update({'edge_color_data_dict_facade': edge_color_data_dict_facade})
        plan_data_dict.update({'edge_color_data_dict_entrance': edge_color_data_dict_entrance})
                        
        return plan_data_dict
    
    
    
    def _get_edge_colors(self, plan_data_dict, edge_list_room_achieved, edge_list_facade_achieved_str, edge_list_entrance_achieved_str):
        edge_list_room_desired = (np.array(plan_data_dict['edge_list_room_desired'], dtype=int)).tolist()
        edge_list_room_achieved = (np.array(edge_list_room_achieved, dtype=int)).tolist()
        edge_color_data_dict_room = self._get_edge_color_data_dict(edge_list_room_desired, edge_list_room_achieved)
                
        edge_list_facade_desired_str = [[e[0], e[1]] for e in plan_data_dict['edge_list_facade_desired_str']]
        edge_list_facade_achieved_str = [[e[0], e[1]] for e in edge_list_facade_achieved_str]
        if self.fenv_config['adaptive_window']:
            edge_color_data_dict_facade = self._get_edge_color_data_dict_facade_for_adaptive_window(plan_data_dict, edge_list_facade_desired_str, edge_list_facade_achieved_str)
        else:
            edge_color_data_dict_facade = self._get_edge_color_data_dict(edge_list_facade_desired_str, edge_list_facade_achieved_str)
            edge_color_data_dict_facade.update({
                'sighted_rooms': [],
                'blind_rooms': [],
                'adaptive_rooms': [],
                })
            
        edge_list_entrance_desired_str = [[e[0], e[1]] for e in plan_data_dict['edge_list_entrance_desired_str']]
        edge_list_entrance_achieved_str = [[e[0], e[1]] for e in edge_list_entrance_achieved_str]
        edge_color_data_dict_entrance = self._get_edge_color_data_dict(edge_list_entrance_desired_str, edge_list_entrance_achieved_str, sort=False)
        
        ## refine the dicts
        ent_lvr_edge_numeric = [self.fenv_config['entrance_cell_id'], self.fenv_config['lvroom_id']]
        ent_lvr_edge_string = ['d', self.fenv_config['lvroom_id']]
        if ( ent_lvr_edge_numeric in edge_color_data_dict_room['red'] and
             ent_lvr_edge_string in edge_color_data_dict_entrance['green'] ):
           edge_color_data_dict_room['red'].remove(ent_lvr_edge_numeric)
           edge_color_data_dict_room['green'].append(ent_lvr_edge_numeric)
           
        facade_ent_edge_numeric = [plan_data_dict['entrance_is_on_facade'], self.fenv_config['entrance_cell_id']]
        facade_ent_edge_string = ['d', plan_data_dict['entrance_is_on_facade']]
        if ( facade_ent_edge_numeric in edge_color_data_dict_facade['red'] and
             facade_ent_edge_string in edge_color_data_dict_entrance['green'] ):
           edge_color_data_dict_facade['red'].remove(facade_ent_edge_numeric)
           edge_color_data_dict_facade['green'].append(facade_ent_edge_numeric)
        
        # edge_color_data_dict_room = self.remove_non_real_rooms(edge_color_data_dict_room) 
        # edge_color_data_dict_facade = self.remove_non_real_rooms(edge_color_data_dict_facade)
        # edge_color_data_dict_entrance = self.remove_non_real_rooms(edge_color_data_dict_entrance)

        return edge_color_data_dict_room, edge_color_data_dict_facade, edge_color_data_dict_entrance
    
    

    def remove_non_real_rooms(self, lst):
        if len(lst) == 0:
            return lst
        for ed in lst:
            if ed[0] not in self.fenv_config['real_room_id_range']:
                lst.remove(ed)
            elif ed[1] not in self.fenv_config['real_room_id_range']:
                lst.remove(ed)
        return lst



    def _get_edge_color_data_dict(self, desired, achieved, sort=True):
        edges = desired + achieved
        edge_color_data_dict = {'green': [], 'blue': [], 'red': []}
        for ed in edges:
            if (ed in desired) and (ed in achieved):
                if ed not in edge_color_data_dict['green']: 
                    edge_color_data_dict['green'].append(ed)
            elif (ed in desired) and (ed not in achieved):
                if ed not in edge_color_data_dict['red']: 
                    edge_color_data_dict['red'].append(ed)
            elif (ed not in desired) and (ed in achieved):
                if ed not in edge_color_data_dict['blue']: 
                    edge_color_data_dict['blue'].append(ed)
        if sort:
            edge_color_data_dict['green'].sort()
            edge_color_data_dict['red'].sort()
            edge_color_data_dict['blue'].sort()
        return edge_color_data_dict
    
    
    
    def _get_edge_color_data_dict_facade_for_adaptive_window(self, plan_data_dict, desired, achieved, sort=True):
        # edges = desired + achieved
        
        adaptive_rooms = set(i for i in range(self.fenv_config['min_room_id'], self.fenv_config['min_room_id']+plan_data_dict['n_rooms']))
        
        edge_color_data_dict = {'green': [], 'blue': [], 'red': []}
        for ed in desired:
            if ed[0] not in plan_data_dict['facades_blocked']:
                if ed in achieved:
                    if ed not in edge_color_data_dict['green']: 
                        edge_color_data_dict['green'].append(ed)
                    if ed[1] in adaptive_rooms:
                        adaptive_rooms.remove(ed[1])
                else:
                    if ed not in edge_color_data_dict['red']: 
                        edge_color_data_dict['red'].append(ed)
                
        sighted_rooms = set()
        for ed in achieved:
            if ed[0] not in plan_data_dict['facades_blocked']:
                sighted_rooms.add(ed[1])
                
        blind_rooms = adaptive_rooms.difference(sighted_rooms)
    
        edge_color_data_dict.update({
            'sighted_rooms': sighted_rooms,
            'blind_rooms': blind_rooms,
            'adaptive_rooms': adaptive_rooms,
            })
        
        if sort:
            edge_color_data_dict['green'].sort()
            edge_color_data_dict['red'].sort()
            edge_color_data_dict['blue'].sort()
        return edge_color_data_dict
    
    
    
    @staticmethod
    def _cartesian2image_coord(x, y, max_y):
        return max_y-y, x