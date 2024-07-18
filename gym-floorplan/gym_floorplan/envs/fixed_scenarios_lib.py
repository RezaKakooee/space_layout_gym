#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 'n':47:29 2023

@author: Reza Kakooee
"""

import numpy as np


#%%
def get_fixed_scenario(n_rooms):
    
    if n_rooms == 4:
        mask_numbers = 1
        masked_corners = ['corner_11']
        mask_lengths = [6]
        mask_widths = [8]
    
        areas_desired_fixed = [124, 93, 83, 78]
        
        edge_list_room_desired = [[10, 14], [11, 12], [11, 13], [11, 14]] # Note: better to keep entrance and corridor connection in this list.
        edge_list_facade_desired_str = [['w', 10]]
        edge_list_entrance_desired_str = [['d', 'w'], ['d', 14]]
        extended_entrance_positions = [[9, 0], [8, 0], [9, 1], [8, 1]]
        extended_entrance_coords = [[0, 13], [0, 14], [1, 13], [1, 14]]
        facades_blocked = ['n']  
        sample_action_sequence = [1017, 1543, 1721]
     
    
    if n_rooms == 5:
        mask_numbers = 2
        masked_corners = ['corner_10', 'corner_11']
        mask_lengths = [6, 4]
        mask_widths = [8, 4]
    
        areas_desired_fixed = [103, 85, 70, 55, 40]
        
        edge_list_room_desired = [[10, 15], [11, 12], [11, 13], [11, 14], [11, 15]]
        edge_list_facade_desired_str = [['e', 10]] 
        edge_list_entrance_desired_str = [['d', 'e'], ['d', 15]] 
        extended_entrance_positions = [[1, 18], [2, 18], [1, 17], [2, 17]] 
        extended_entrance_coords = [[18, 21], [18, 20], [17, 21], [17, 20]] 
        facades_blocked = ['n']
        sample_action_sequence = [1275, 853, 847, 1149]


    if n_rooms == 6:
        mask_numbers = 1
        masked_corners = ['corner_10']
        mask_lengths = [2]
        mask_widths = [2]
    
        areas_desired_fixed = [107, 90, 85, 65, 45, 40] 
        
        edge_list_room_desired = [[10, 16], [11, 12], [11, 13], [11, 14], [11, 15], [11, 16]]
        edge_list_facade_desired_str = [['w', 10]]
        edge_list_entrance_desired_str = [['d', 'w'], ['d', 16]] 
        extended_entrance_positions = [[9, 0], [8, 0], [9, 1], [8, 1]] 
        extended_entrance_coords = [[0, 13], [0, 14], [1, 13], [1, 14]] 
        facades_blocked = []
        sample_action_sequence = [647, 1617, 1313, 464, 553]
        
        
    if n_rooms == 7:
        mask_numbers = 4
        masked_corners = ['corner_00', 'corner_01', 'corner_10', 'corner_11']
        mask_lengths = [4, 6, 6, 4]
        mask_widths = [6, 4, 4, 6]
    
        areas_desired_fixed = [66, 45, 44, 43, 42, 41, 20]
        
        edge_list_room_desired = [[10, 17], [11, 12], [11, 13], [11, 14], [11, 15], [11, 16], [11, 17]] 
        edge_list_facade_desired_str = [['w', 10]] 
        edge_list_entrance_desired_str = [['d', 'w'], ['d', 17]] 
        extended_entrance_positions = [[19, 4], [18, 4], [19, 5], [18, 5]]
        extended_entrance_coords = [[4, 3], [4, 4], [5, 3], [5, 4]] 
        facades_blocked = []
        sample_action_sequence = [1183, 1068, 766, 474, 1151, 1320]
        
    
    if n_rooms == 8:
        mask_numbers = 1
        masked_corners = ['corner_00']
        mask_lengths = [4]
        mask_widths = [2]
    
        areas_desired_fixed = [73, 68, 55, 50, 50, 45, 45, 40]
        
        edge_list_room_desired = [[10, 18], [11, 12], [11, 13], [11, 14], [11, 15], [11, 16], [11, 17], [11, 18]]
        edge_list_facade_desired_str = [['s', 10]]
        edge_list_entrance_desired_str = [['d', 's'], ['d', 18]]
        extended_entrance_positions = [[22, 5], [22, 6], [21, 5], [21, 6]]
        extended_entrance_coords = [[5, 0], [6, 0], [5, 1], [6, 1]]
        facades_blocked = []
        sample_action_sequence = [747, 1112, 1434, 941, 964, 44, 1086]
        
        
    if n_rooms == 9: # pretty plan
        mask_numbers = 1
        masked_corners = ['corner_10']
        mask_lengths = [8]	
        mask_widths = [8]
    
        areas_desired_fixed = [95, 50, 45, 40, 35, 30, 25, 20, 20] 
        
        edge_list_room_desired = [[10, 19], [11, 12], [11, 13], [11, 14], [11, 15], [11, 16], [11, 17], [11, 19]]
        edge_list_facade_desired_str = [['e', 10]]
        edge_list_entrance_desired_str = [['d', 'e'], ['d', 19]]
        extended_entrance_positions = [[20, 14], [21, 14], [20, 13], [21, 13]]
        extended_entrance_coords = [[14, 2], [14, 1], [13, 2], [13, 1]]
        facades_blocked = []
        sample_action_sequence = [736, 832, 1441, 1651, 1125, 286, 1167, 989]
        
        
        
    n_corners = 4
    very_min_fake_room_id = 2
    min_room_id = 11
    corner_to_mask_room_id = {'corner_00':2, 'corner_01':3, 'corner_10':4, 'corner_11':5}
    
    areas_masked_list = [(L+1)*(W+1) for L, W in zip(mask_lengths, mask_widths)] 
    area_masked = sum(areas_masked_list)
    
    areas_masked = {f"room_{corner_to_mask_room_id[coord]}":area for coord, area in zip(masked_corners, areas_masked_list)}
    
    areas_desired_fixed = np.sort(areas_desired_fixed)[::-1]
    areas_desired = {f"room_{min_room_id+i}": area for i, area in enumerate(areas_desired_fixed)}
    
    edge_list_room_desired = [[int(min(edge)), int(max(edge))] for edge in edge_list_room_desired]
    edge_list_room_desired.sort()
    
    edge_list_facade_desired_str.sort()
    
    for edge in edge_list_entrance_desired_str:
        for n in edge:
            if (isinstance(n, str) and n != 'd'):
                entrance_is_on = n
                
    entrance_positions = extended_entrance_positions[:2]
    entrance_coords = extended_entrance_coords[:2]
    corridor_id = [room for edge in edge_list_entrance_desired_str for room in edge if isinstance(room, int)][0]
    living_room_id = 11 # TODO check this mayber later

    fixed_scenario_config = {
        'n_corners': n_corners, # always = 4
        'plan_config_source_name': 'fixed_test_configs',
        'mask_numbers': mask_numbers,
        'number_of_total_rooms': n_rooms + n_corners, # TODO
        'masked_corners': masked_corners,
        'mask_lengths': mask_lengths,
        'mask_widths': mask_widths,
        'areas_masked': areas_masked,
        'area_masked': area_masked,
        'areas_desired': areas_desired,
        'edge_list_room_desired': edge_list_room_desired,   
        'edge_list_facade_desired_str': edge_list_facade_desired_str,
        'edge_list_entrance_desired_str': edge_list_entrance_desired_str,
        'entrance_is_on_facade': entrance_is_on,
        'entrance_positions': entrance_positions,
        'entrance_coords': entrance_coords,
        'extended_entrance_positions': extended_entrance_positions,
        'extended_entrance_coords': extended_entrance_coords,
        'facades_blocked': facades_blocked,
        'n_facades_blocked': len(facades_blocked),
        'corridor_id': corridor_id,
        'living_room_id': living_room_id, 
        }
    
    return fixed_scenario_config




#%%
if __name__ == '__main__':
    def _image_coords2cartesian(r, c, n_rows=23):
        return c, n_rows-1-r 
    
    extended_entrance_positions = [[22, 5], [22, 6], [21, 5], [21, 6]]
    
    extended_entrance_coords = []
    for r, c in extended_entrance_positions:
        x, y = _image_coords2cartesian(r, c)
        extended_entrance_coords.append([x, y])
    
    
    print(extended_entrance_coords)