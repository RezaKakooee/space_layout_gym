#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 'n':47:29 2023

@author: Reza Kakooee
"""

import numpy as np


#%%
def get_fixed_scenario(n_rooms):
    lvroom_id = 11
    entrance_id = 10
    
    if n_rooms == 44:
        mask_numbers = 1
        masked_corners = ['corner_11']
        mask_lengths = [6]
        mask_widths = [8]
    
        areas_desired_fixed = [124, 93, 83, 78]
        
        edge_list_room_desired =  [[entrance_id, lvroom_id], [11, 12], [11, 13], [11, 14]]
        edge_list_facade_desired_str = [['w', entrance_id]]
        edge_list_entrance_desired_str = [['d', 'w'], ['d', lvroom_id]]
        extended_entrance_positions = [[9, 0], [8, 0], [9, 1], [8, 1]]
        extended_entrance_coords = [[0, 13], [0, 14], [1, 13], [1, 14]]
        facades_blocked = [] # ['n']  
        sample_action_sequence = [1017, 1543, 1721]
        plan_id = 'Base_4_Room_Plan'
        
     
    
    if n_rooms == 55:
        mask_numbers = 2
        masked_corners = ['corner_10', 'corner_11']
        mask_lengths = [6, 4]
        mask_widths = [8, 4]
    
        areas_desired_fixed = [103, 85, 70, 55, 40]
        
        edge_list_room_desired = [[entrance_id, lvroom_id], [11, 12], [11, 13], [11, 14], [11, 15]]
        edge_list_facade_desired_str = [['e', entrance_id]] 
        edge_list_entrance_desired_str = [['d', 'e'], ['d', lvroom_id]] 
        extended_entrance_positions = [[1, 18], [2, 18], [1, 17], [2, 17]] 
        extended_entrance_coords = [[18, 21], [18, 20], [17, 21], [17, 20]] 
        facades_blocked = ['n']
        sample_action_sequence = [1275, 853, 847, 1149]
        plan_id = 'Base_5_Room_Plan'
        

    
    if n_rooms == 66:
        mask_numbers = 1
        masked_corners = ['corner_10']
        mask_lengths = [2]
        mask_widths = [2]
    
        areas_desired_fixed = [107, 90, 85, 65, 45, 40] 
        
        edge_list_room_desired = [[entrance_id, lvroom_id], [11, 12], [11, 13], [11, 14], [11, 15], [11, 16]]
        edge_list_facade_desired_str = [['w', entrance_id]]
        edge_list_entrance_desired_str = [['d', 'w'], ['d', lvroom_id]] 
        extended_entrance_positions = [[9, 0], [8, 0], [9, 1], [8, 1]] 
        extended_entrance_coords = [[0, 13], [0, 14], [1, 13], [1, 14]] 
        facades_blocked = []
        sample_action_sequence = [647, 1617, 1313, 464, 553]
        plan_id = 'Base_6_Room_Plan'
        
        
        
    if n_rooms == 77:
        mask_numbers = 4
        masked_corners = ['corner_00', 'corner_01', 'corner_10', 'corner_11']
        mask_lengths = [4, 6, 6, 4]
        mask_widths = [6, 4, 4, 6]
    
        areas_desired_fixed = [66, 45, 44, 43, 42, 41, 20]
        
        edge_list_room_desired = [[entrance_id, lvroom_id], [11, 12], [11, 13], [11, 14], [11, 15], [11, 16], [11, 17]] 
        edge_list_facade_desired_str = [['w', entrance_id]] 
        edge_list_entrance_desired_str = [['d', 'w'], ['d', lvroom_id]] 
        extended_entrance_positions = [[19, 4], [18, 4], [19, 5], [18, 5]]
        extended_entrance_coords = [[4, 3], [4, 4], [5, 3], [5, 4]] 
        facades_blocked = []
        sample_action_sequence = [1183, 1068, 766, 474, 1151, 1320]
        plan_id = 'Base_7_Room_Plan'
        
    
    if n_rooms == 88:
        mask_numbers = 1
        masked_corners = ['corner_00']
        mask_lengths = [4]
        mask_widths = [2]
    
        areas_desired_fixed = [73, 68, 55, 50, 50, 45, 45, 40]
        
        edge_list_room_desired = [[entrance_id, lvroom_id], [11, 12], [11, 13], [11, 14], [11, 15], [11, 16], [11, 17], [11, 18]]
        edge_list_facade_desired_str = [['s', entrance_id]]
        edge_list_entrance_desired_str = [['d', 's'], ['d', lvroom_id]]
        extended_entrance_positions = [[22, 5], [22, 6], [21, 5], [21, 6]]
        extended_entrance_coords = [[5, 0], [6, 0], [5, 1], [6, 1]]
        facades_blocked = []
        sample_action_sequence = [747, 1112, 1434, 941, 964, 44, 1086]
        plan_id = 'Base_8_Room_Plan'
        
        
    if n_rooms == 99: # pretty plan
        mask_numbers = 1
        masked_corners = ['corner_10']
        mask_lengths = [8]	
        mask_widths = [8]
    
        areas_desired_fixed = [95, 50, 45, 40, 35, 30, 25, 20, 20] 
        
        edge_list_room_desired = [[entrance_id, lvroom_id], [11, 12], [11, 13], [11, 14], [11, 15], [11, 16], [11, 17], [11, 18], [11, 19]]
        edge_list_facade_desired_str = [['e', entrance_id]]
        edge_list_entrance_desired_str = [['d', 'e'], ['d', lvroom_id]]
        extended_entrance_positions = [[20, 14], [21, 14], [20, 13], [21, 13]]
        extended_entrance_coords = [[14, 2], [14, 1], [13, 2], [13, 1]]
        facades_blocked = []
        sample_action_sequence = [736, 832, 1441, 1651, 1125, 286, 1167, 989]
        plan_id = 'Base_9_Room_Plan'
        
        
    if n_rooms == 20:
        mask_numbers = 1
        masked_corners = ['corner_11']
        mask_lengths = [18]
        mask_widths = [20]

        areas_desired_fixed = [500, 65, 63, 60, 55, 54, 51, 50, 48, 46, 43, 43, 41, 39, 38, 38, 36, 34, 33, 30] # [105, 102,  95,  88,  86,  86,  84,  81,  81,  79,  77,  69,  69, 66,  55,  53,  48,  47,  40,  39]

        edge_list_room_desired = [[entrance_id, lvroom_id], [11, 12], [11, 13], [11, 14], [11, 15], [11, 16], [11, 17], [11, 18], [11, 19], [11, 20], [11, 21], [11, 22], [11, 23], [11, 24], [11, 25], [11, 26], [11, 27], [11, 28], [11, 29], [11, 30]]
        edge_list_facade_desired_str = [['e', entrance_id]]
        edge_list_entrance_desired_str = [['d', 'e'], ['d', lvroom_id]]
        extended_entrance_positions = [[30, 44], [31, 44], [30, 43], [31, 43]]
        extended_entrance_coords = [[44, 14], [44, 13], [43, 14], [43, 13]]
        facades_blocked = ['w']
        sample_action_sequence = [6889, 518, 1950, 6281, 952, 5616, 3740, 2766, 6228, 558, 1600, 5635, 1852, 4632, 1018, 7901, 616, 2506, 2441]
        plan_id = 'Base_20_Room_Plan'
    


    if n_rooms == 4:
        mask_numbers = 2
        masked_corners = ['corner_10', 'corner_11']
        mask_lengths = [2, 10]
        mask_widths = [6, 2]

        areas_desired_fixed = [251, 94, 40, 24]
        aspect_ratio_desired = [1.3, 1.6, 3.0, 6.0]
        edge_list_room_desired = [[11, 13], [13, 14], [11, 12]] # [[11, 13], [13, 14]] #[[11, 13], [12, 13], [12, 14], [13, 14]] # [[11, 12], [11, 13], [11, 14], [13, 14]]
        edge_list_facade_desired_str = [['s', entrance_id]] ## [['e', 12], ['e', 13], ['e', 14], ['n', 11], ['n', 12], ['n', 13], ['s', 11], ['s', 14], ['w', 11], ['w', 12]] ## [['s', entrance_id]]
        edge_list_entrance_desired_str = [['d', 's'], ['d', lvroom_id]]
        extended_entrance_positions = [[22, 14], [22, 15], [21, 14], [21, 15]]
        extended_entrance_coords = [[14, 0], [15, 0], [14, 1], [15, 1]]
        facades_blocked = []
        sample_action_sequence = [156, 684, 1782]
        plan_id = 'rplan_1736_r4'
        
        
    if n_rooms == 5:
        mask_numbers = 2
        masked_corners = ['corner_01', 'corner_11']
        mask_lengths = [10, 4]
        mask_widths = [2, 10]

        areas_desired_fixed = [139, 96, 72, 60, 14]
        aspect_ratio_desired = [2.0, 1.6, 1.3, 1.8, 7.0]
        edge_list_room_desired = [[11, 12], [11, 13], [12, 13]] # [[11, 12], [11, 13], [11, 14], [11, 15], [12, 13]]
        edge_list_facade_desired_str = [['n', entrance_id]] ## [['e', 11], ['e', 13], ['e', 15], ['n', 11], ['n', 14], ['n', 15], ['s', 12], ['s', 13], ['w', 11], ['w', 12], ['w', 14], ['w', 15]] ## [['s', entrance_id]]
        edge_list_entrance_desired_str = [['d', 'n'], ['d', lvroom_id]]
        extended_entrance_positions = [[10, 19], [10, 20], [11, 19], [11, 20]]
        extended_entrance_coords = [[19, 12], [20, 12], [19, 11], [20, 11]]
        facades_blocked = []
        sample_action_sequence = [253, 1083, 1346, 2269]
        plan_id = 'rplan_390_r5'
        
        
    if n_rooms == 6:
        mask_numbers = 3
        masked_corners = ['corner_00', 'corner_10', 'corner_01']
        mask_lengths = [4, 6, 8]
        mask_widths = [6, 2, 2]

        areas_desired_fixed = [177, 80, 60, 30, 30, 12] 
        aspect_ratio_desired = [3.3, 1.3, 1.8, 1.0, 2.3, 5.0]
        edge_list_room_desired = [[11, 16], [11, 12], [11, 13]] # [[11, 16], [11, 12], [11, 13]] # [[11, 12], [11, 13], [11, 14], [11, 15], [11, 16], [12, 13], [12, 14], [13, 15]]
        edge_list_facade_desired_str = [['n', entrance_id]] ## [['e', 11], ['e', 15], ['n', 11], ['n', 14], ['n', 16], ['s', 12], ['s', 13], ['s', 15], ['w', 12], ['w', 13], ['w', 14], ['w', 16]] ## [['n', 11], ['n', 16], ['s', 13], ['s', 15], ['e', 15], ['e', 11], ['w', 14], ['w', 12]] ## [['n', entrance_id]] #
        edge_list_entrance_desired_str = [['d', 'n'], ['d', lvroom_id]]
        extended_entrance_positions = [[0, 17], [0, 18], [1, 17], [1, 18]] 
        extended_entrance_coords = [[17, 22], [18, 22], [17, 21], [18, 21]]
        facades_blocked = []
        sample_action_sequence = [862, 2282, 246, 1727, 2569]
        plan_id = 'rplan_158_r6'
        
        
    if n_rooms == 77: 
        mask_numbers = 3
        masked_corners = ['corner_00', 'corner_10', 'corner_11']
        mask_lengths = [4, 8, 4]
        mask_widths = [8, 2, 2]

        areas_desired_fixed = [173, 64, 40, 36, 36, 20, 16] 
        edge_list_room_desired = [[11, 12], [11, 13], [11, 14], [11, 16], [11, 17], [12, 14], [12, 15], [12, 17], [13, 16], [13, 17], [14, 15]]
        edge_list_facade_desired_str = [['n', entrance_id]] # [['n', 11], ['n', 16], ['e', 13], ['e', 17], ['s', 15], ['s', 12], ['w', 14], ['w', 11]] #  [['n', entrance_id]] #
        edge_list_entrance_desired_str = [['d', 'n'], ['d', lvroom_id]]
        extended_entrance_positions = [[0, 7], [0, 8], [1, 7], [1, 8]] 
        extended_entrance_coords = [[7, 22], [8, 22], [7, 21], [8, 21]]
        facades_blocked = []
        sample_action_sequence = [964, 2841, 2243, 3049, 1257, 4186]
        plan_id = 'rplan_21_r7'
        
        
    if n_rooms == 7:
        mask_numbers = 3
        masked_corners = ['corner_00', 'corner_01', 'corner_11']
        mask_lengths = [12, 6, 2]
        mask_widths = [4, 2, 10]

        areas_desired_fixed = [149, 56, 50, 40, 40, 16, 10] 
        aspect_ratio_desired = [2.5, 1.5, 1.8, 3.0, 3.0, 7.0, 5.0]
        edge_list_room_desired = [[11, 12], [11, 17], [12, 15], [12, 14], [11, 15], [11, 16]] #[[11, 12], [11, 17], [12, 15], [12, 14]] # [[11, 12], [11, 13], [11, 14], [11, 15], [11, 16], [11, 17], [12, 14], [12, 15], [13, 15], [14, 17]]
        edge_list_facade_desired_str = [['s', entrance_id]] ## [['n', 12], ['n', 15], ['e', 13], ['e', 16], ['s', 11], ['s', 17], ['w', 11], ['w', 17]] ## [['s', entrance_id]] #
        edge_list_entrance_desired_str = [['d', 's'], ['d', lvroom_id]]
        extended_entrance_positions = [[18, 8], [18, 9], [17, 8], [17, 9]] 
        extended_entrance_coords = [[8, 4], [9, 4], [8, 5], [9, 5]]
        facades_blocked = []
        sample_action_sequence = [1314, 145, 1965, 1085, 2491, 3503]
        plan_id = 'rplan_24_r7'

        
    if n_rooms == 8:
        mask_numbers = 3
        masked_corners = ['corner_00','corner_01', 'corner_11']
        mask_lengths = [10, 6, 2]
        mask_widths = [2, 2, 10]

        areas_desired_fixed = [159, 66, 48, 36, 36, 20, 12, 12]
        aspect_ratio_desired = [7.5, 2.0, 1.4, 1.0, 1.0, 1.7, 5.0, 5.0]
        edge_list_room_desired = [[11, 16], [11, 12], [12, 13], [14, 16], [11, 14], [11, 15], [11, 17]] # [[11, 16], [11, 12], [12, 13], [14, 16]] # [[11, 12], [11, 13], [11, 14], [11, 15], [11, 16], [11, 17], [12, 13], [12, 15], [12, 18], [13, 17], [14, 15], [14, 16]]
        edge_list_facade_desired_str = [['s', entrance_id]] ## [['e', 11], ['e', 12], ['e', 13], ['e', 17], ['n', 12], ['n', 15], ['n', 17], ['n', 18], ['s', 11], ['s', 16], ['w', 11], ['w', 14], ['w', 15], ['w', 16], ['w', 18]] ## [['s', entrance_id]]
        edge_list_entrance_desired_str = [['d', 's'], ['d', lvroom_id]]
        extended_entrance_positions = [[19, 6], [19, 7], [20, 6], [20, 7]]
        extended_entrance_coords = [[6, 2], [7, 2], [6, 3], [7, 3]]
        facades_blocked = []
        sample_action_sequence = [3092, 3759, 1926, 1323, 2912, 457, 654]
        plan_id = 'rplan_377_r8'



    ##        
    n_corners = 4
    min_room_id = lvroom_id
    corner_to_mask_room_id = {'corner_00':2, 'corner_01':3, 'corner_10':4, 'corner_11':5}
    
    areas_masked_list = [(L+1)*(W+1) for L, W in zip(mask_lengths, mask_widths)] 
    area_masked = sum(areas_masked_list)
    
    areas_masked = {f"room_{corner_to_mask_room_id[coord]}":area for coord, area in zip(masked_corners, areas_masked_list)}
    
    areas_desired_fixed = np.sort(areas_desired_fixed)[::-1]
    areas_desired = {f"room_{min_room_id+i}": area for i, area in enumerate(areas_desired_fixed)}
    aspect_ratio_desired = {f"room_{min_room_id+i}": p for i, p in enumerate(aspect_ratio_desired)}
    
    edge_list_room_desired = [[int(min(edge)), int(max(edge))] for edge in edge_list_room_desired]
    edge_list_room_desired.sort()
    
    edge_list_facade_desired_str.sort()
    
    for edge in edge_list_entrance_desired_str:
        for n in edge:
            if (isinstance(n, str) and n != 'd'):
                entrance_is_on = n
                
    entrance_positions = extended_entrance_positions[:2]
    entrance_coords = extended_entrance_coords[:2]
    # lvroom_id = [room for edge in edge_list_entrance_desired_str for room in edge if isinstance(room, int)][0]

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
        'aspect_ratio_desired': aspect_ratio_desired,
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
        'lvroom_id': lvroom_id,
        'plan_id': plan_id,
        'sample_action_sequence': sample_action_sequence,
        }
    
    return fixed_scenario_config




#%% This is only for testing and debugging
    def _image_coords2cartesian(r, c, n_rows=23):
        return c, n_rows-1-r 
    extended_entrance_positions = [[22, 5], [22, 6], [21, 5], [21, 6]]
    extended_entrance_coords = []
    for r, c in extended_entrance_positions:
        x, y = _image_coords2cartesian(r, c)
        extended_entrance_coords.append([x, y])
    print(extended_entrance_coords)
    get_fixed_scenario(n_rooms=4)