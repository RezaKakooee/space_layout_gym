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
        mask_numbers = 3
        masked_corners = ['corner_00', 'corner_10', 'corner_11']
        mask_lengths = [10, 16, 20]
        mask_widths = [12, 10, 18]
    
        areas_desired_fixed = [581, 272, 200, 156]
        aspect_ratio_desired = [5.0, 1.1333333333333333, 1.8888888888888888, 5.0]  
    
        edge_list_room_desired = [[11, 12], [11, 13], [11, 14]]#, [12, 13], [13, 14]]
        edge_list_facade_desired_str = [['w', entrance_id]] #[['e', 11], ['e', 12], ['e', 13], ['n', 12], ['n', 13], ['n', 14], ['s', 11], ['s', 12], ['w', 11], ['w', 13], ['w', 14]]
        edge_list_entrance_desired_str = [['d', 'w'], ['d', lvroom_id]]
        extended_entrance_positions = [[41, 10], [40, 10], [41, 11], [40, 11]]
        extended_entrance_coords = [[10, 3], [10, 4], [11, 3], [11, 4]]
        facades_blocked = ['n']
        sample_action_sequence = [256, 6234, 4510]
        plan_id = 'Base_04_High_Resolution_Sep02'
        
        
    if n_rooms == 444:
        mask_numbers = 3
        masked_corners = ['corner_00', 'corner_10', 'corner_11']
        mask_lengths = [10, 16, 20]
        mask_widths = [12, 10, 18]
    
        areas_desired_fixed = [431, 322, 250, 206]
        aspect_ratio_desired = [5.0, 1.1333333333333333, 1.8888888888888888, 5.0]  
    
        edge_list_room_desired = [[11, 12], [11, 13], [11, 14]]#, [12, 13], [13, 14]]
        edge_list_facade_desired_str = [['w', entrance_id]] #[['e', 11], ['e', 12], ['e', 13], ['n', 12], ['n', 13], ['n', 14], ['s', 11], ['s', 12], ['w', 11], ['w', 13], ['w', 14]]
        edge_list_entrance_desired_str = [['d', 'w'], ['d', lvroom_id]]
        extended_entrance_positions = [[41, 10], [40, 10], [41, 11], [40, 11]]
        extended_entrance_coords = [[10, 3], [10, 4], [11, 3], [11, 4]]
        facades_blocked = ['n']
        sample_action_sequence = [256, 6234, 4510]
        plan_id = 'Base_04_High_Resolution_Sep02'
        
        
    if n_rooms == 4:
        mask_numbers = 4
        masked_corners = ['corner_00', 'corner_01', 'corner_10', 'corner_11']
        mask_lengths = [20, 18, 20, 12]
        mask_widths = [8, 8, 8, 8]
        
        areas_desired_fixed = [545, 270, 264, 210]
        aspect_ratio_desired = [1.625, 2.0, 4.333333333333333, 2.3333333333333335]    
        
        edge_list_room_desired = [[11, 12], [11, 13], [11, 14]]#[[11, 12], [11, 13], [11, 14], [12, 13], [13, 14]]
        edge_list_facade_desired_str = [['e', entrance_id]] # [['e', 11], ['e', 12], ['e', 13], ['n', 11], ['n', 12], ['n', 14], ['s', 12], ['s', 13], ['w', 11], ['w', 13], ['w', 14]]
        edge_list_entrance_desired_str = [['d', 'e'], ['d', lvroom_id]]
        extended_entrance_positions = [[6, 32], [7, 32], [6, 31], [7, 31]]
        extended_entrance_coords = [[32, 38], [32, 37], [31, 38], [31, 37]]
        facades_blocked = ['n']
        sample_action_sequence = [3933, 7595, 2530]
        # plan_id = 'Base_04_High_Resolution_Sep24'
        # plan_id = 'Base_04_High_Resolution_Oct01_OA'
        plan_id = 'Base_04_High_Resolution_Oct01_OS'
        
        
    
    if n_rooms == 55:
        mask_numbers = 3
        masked_corners = ['corner_00', 'corner_01', 'corner_11']
        mask_lengths = [20, 14, 10]
        mask_widths = [16, 16, 14]
    
        areas_desired_fixed = [701, 120, 120, 114, 110] # [497, 184, 184, 168, 132]
        aspect_ratio_desired = [6.6, 2.0, 1.2222222222222223, 3.8, 1.2222222222222223]
    
        edge_list_room_desired = [[11, 12], [11, 13], [11, 14], [11, 15]]#, [12, 14], [13, 15]]
        edge_list_facade_desired_str = [['e', entrance_id]] #  [['e', 11], ['e', 12], ['e', 14], ['n', 11], ['n', 13], ['n', 14], ['n', 15], ['s', 12], ['s', 13], ['s', 15], ['w', 11], ['w', 12], ['w', 13], ['w', 15]]
        edge_list_entrance_desired_str = [['d', 'e'], ['d', lvroom_id]]
        extended_entrance_positions = [[9, 34], [10, 34], [9, 33], [10, 33]]
        extended_entrance_coords = [[34, 35], [34, 34], [33, 35], [33, 34]]
        facades_blocked = ['w']
        sample_action_sequence = [6493, 4659, 10238, 640]
        plan_id = 'Base_05_High_Resolution_Sep02'
        
      
    if n_rooms == 555:
        mask_numbers = 3
        masked_corners = ['corner_00', 'corner_01', 'corner_11']
        mask_lengths = [20, 14, 10]
        mask_widths = [16, 16, 14]
    
        areas_desired_fixed = [301, 220, 220, 214, 210] # [497, 184, 184, 168, 132]
        aspect_ratio_desired = [6.6, 2.0, 1.2222222222222223, 3.8, 1.2222222222222223]
    
        edge_list_room_desired = [[11, 12], [11, 13], [11, 14], [11, 15]]#, [12, 14], [13, 15]]
        edge_list_facade_desired_str = [['e', entrance_id]] #  [['e', 11], ['e', 12], ['e', 14], ['n', 11], ['n', 13], ['n', 14], ['n', 15], ['s', 12], ['s', 13], ['s', 15], ['w', 11], ['w', 12], ['w', 13], ['w', 15]]
        edge_list_entrance_desired_str = [['d', 'e'], ['d', lvroom_id]]
        extended_entrance_positions = [[9, 34], [10, 34], [9, 33], [10, 33]]
        extended_entrance_coords = [[34, 35], [34, 34], [33, 35], [33, 34]]
        facades_blocked = ['w']
        sample_action_sequence = [6493, 4659, 10238, 640]
        plan_id = 'Base_05_High_Resolution_Sep02'
        
        
    if n_rooms == 5:
        mask_numbers = 3
        masked_corners = ['corner_00', 'corner_01', 'corner_10']
        mask_lengths = [10, 20, 8]
        mask_widths = [14, 10, 12]
    
        areas_desired_fixed = [539, 252, 224, 206, 192]
        aspect_ratio_desired = [10.0, 1.3076923076923077, 1.1538461538461537, 6.666666666666667, 1.3636363636363635]    
    
        edge_list_room_desired = [[11, 12], [11, 13], [11, 14], [11, 15]]#[[11, 12], [11, 13], [11, 14], [11, 15], [12, 14], [13, 14]]
        edge_list_facade_desired_str = [['s', entrance_id]]  #[['e', 11], ['e', 12], ['n', 12], ['n', 14], ['s', 11], ['s', 15], ['w', 11], ['w', 13], ['w', 14], ['w', 15]]
        edge_list_entrance_desired_str = [['d', 's'], ['d', lvroom_id]]
        extended_entrance_positions = [[32, 37], [32, 38], [31, 37], [31, 38]]
        extended_entrance_coords = [[37, 12], [38, 12], [37, 13], [38, 13]]
        facades_blocked = ['e']
        sample_action_sequence = [306, 7196, 9077, 3241]
        # plan_id = 'Base_05_High_Resolution_Sep24'
        plan_id = 'Base_05_High_Resolution_Oct01_OA'
        # plan_id = 'Base_05_High_Resolution_Oct01_OS'
        
        
    if n_rooms == 66:
        mask_numbers = 4
        masked_corners = ['corner_00', 'corner_01', 'corner_10', 'corner_11']
        mask_lengths = [12, 16, 14, 6]
        mask_widths = [12, 18, 12, 16]
    
        areas_desired_fixed = [429, 180, 156, 130, 130, 128]
        aspect_ratio_desired = [3.75, 1.8888888888888888, 1.1818181818181819, 1.0, 4.0, 2.142857142857143]    
    
        edge_list_room_desired = [[11, 12], [11, 13], [11, 14], [11, 15], [11, 16]]#, [12, 14], [13, 15]] #  [14, 16]]
        edge_list_facade_desired_str = [['s', entrance_id]] # [['e', 11], ['e', 14], ['e', 16], ['n', 11], ['n', 12], ['n', 13], ['n', 14], ['n', 16], ['s', 11], ['s', 13], ['s', 15], ['s', 16], ['w', 12], ['w', 13], ['w', 15]]
        edge_list_entrance_desired_str = [['d', 's'], ['d', lvroom_id]]
        extended_entrance_positions = [[44, 26], [44, 27], [43, 26], [43, 27]]
        extended_entrance_coords = [[26, 0], [27, 0], [26, 1], [27, 1]]
        facades_blocked = ['n']
        sample_action_sequence = [705, 12279, 7407, 4965, 9016]
        plan_id = 'Base_06_High_Resolution_Sep02'
        
        
    if n_rooms == 666:
        mask_numbers = 4
        masked_corners = ['corner_00', 'corner_01', 'corner_10', 'corner_11']
        mask_lengths = [12, 16, 14, 6]
        mask_widths = [12, 18, 12, 16]
    
        areas_desired_fixed = [279, 210, 186, 160, 160, 158]
        aspect_ratio_desired = [3.75, 1.8888888888888888, 1.1818181818181819, 1.0, 4.0, 2.142857142857143]    
    
        edge_list_room_desired = [[11, 12], [11, 13], [11, 14], [11, 15], [11, 16]]#, [12, 14], [13, 15]] #  [14, 16]]
        edge_list_facade_desired_str = [['s', entrance_id]] # [['e', 11], ['e', 14], ['e', 16], ['n', 11], ['n', 12], ['n', 13], ['n', 14], ['n', 16], ['s', 11], ['s', 13], ['s', 15], ['s', 16], ['w', 12], ['w', 13], ['w', 15]]
        edge_list_entrance_desired_str = [['d', 's'], ['d', lvroom_id]]
        extended_entrance_positions = [[44, 26], [44, 27], [43, 26], [43, 27]]
        extended_entrance_coords = [[26, 0], [27, 0], [26, 1], [27, 1]]
        facades_blocked = ['n']
        sample_action_sequence = [705, 12279, 7407, 4965, 9016]
        plan_id = 'Base_06_High_Resolution_Sep02'
        

    if n_rooms == 6:
        mask_numbers = 4
        masked_corners = ['corner_00', 'corner_01', 'corner_10', 'corner_11']
        mask_lengths = [12, 16, 14, 6]
        mask_widths = [12, 18, 12, 16]
        
        areas_desired_fixed = [429, 180, 156, 130, 130, 128]
        aspect_ratio_desired = [3.75, 1.8888888888888888, 1.1818181818181819, 1.0, 4.0, 2.142857142857143]    
        
        edge_list_room_desired = [[11, 12], [11, 13], [11, 14], [11, 15], [11, 16]]#[[11, 12], [11, 13], [11, 14], [11, 15], [11, 16], [12, 14], [13, 15], [14, 16]]
        edge_list_facade_desired_str = [['s', entrance_id]] # [['e', 11], ['e', 14], ['e', 16], ['n', 11], ['n', 12], ['n', 13], ['n', 14], ['n', 16], ['s', 11], ['s', 13], ['s', 15], ['s', 16], ['w', 12], ['w', 13], ['w', 15]]
        edge_list_entrance_desired_str = [['d', 's'], ['d', lvroom_id]]
        extended_entrance_positions = [[44, 26], [44, 27], [43, 26], [43, 27]]
        extended_entrance_coords = [[26, 0], [27, 0], [26, 1], [27, 1]]
        facades_blocked = ['n']
        sample_action_sequence = [705, 12279, 7407, 4965, 9016]
        plan_id = 'Base_06_High_Resolution_Sep24'
        # plan_id = 'Base_06_High_Resolution_Oct01_OA'
        # plan_id = 'Base_06_High_Resolution_Oct01_OS'
            

    if n_rooms == 77:
        mask_numbers = 4
        masked_corners = ['corner_00', 'corner_01', 'corner_10', 'corner_11']
        mask_lengths = [4, 18, 20, 18]
        mask_widths = [8, 18, 6, 12]
    
        areas_desired_fixed = [493, 120, 114, 110, 108, 108, 104]
        aspect_ratio_desired = [5.666666666666667, 3.8, 2.2, 1.3333333333333333, 4.0, 6.666666666666667, 2.0]    
    
        edge_list_room_desired = [[11, 12], [11, 13], [11, 14], [11, 15], [11, 16], [11, 17]] # [[11, 12], [11, 13], [11, 14], [11, 16], [11, 17], [12, 14], [12, 15], [13, 17], [14, 15], [14, 17]]
        edge_list_facade_desired_str = [['n', entrance_id]] # [['e', 13], ['e', 15], ['e', 16], ['e', 17], ['n', 11], ['n', 13], ['n', 16], ['s', 12], ['s', 14], ['s', 15], ['s', 17], ['w', 11], ['w', 12], ['w', 16]]
        edge_list_entrance_desired_str = [['d', 'n'], ['d', lvroom_id]]
        extended_entrance_positions = [[18, 1], [18, 2], [19, 1], [19, 2]]
        extended_entrance_coords = [[1, 26], [2, 26], [1, 25], [2, 25]]
        facades_blocked = []
        sample_action_sequence = [1012, 3029, 14877, 9097, 11312, 6684]
        plan_id = 'Base_07_High_Resolution_Sep02'
        
        
    if n_rooms == 777:
        mask_numbers = 4
        masked_corners = ['corner_00', 'corner_01', 'corner_10', 'corner_11']
        mask_lengths = [4, 18, 20, 18]
        mask_widths = [8, 18, 6, 12]
    
        areas_desired_fixed = [313, 150, 144, 140, 138, 138, 134]
        aspect_ratio_desired = [5.666666666666667, 3.8, 2.2, 1.3333333333333333, 4.0, 6.666666666666667, 2.0]    
    
        edge_list_room_desired = [[11, 12], [11, 13], [11, 14], [11, 15], [11, 16], [11, 17]] # [[11, 12], [11, 13], [11, 14], [11, 16], [11, 17], [12, 14], [12, 15], [13, 17], [14, 15], [14, 17]]
        edge_list_facade_desired_str = [['n', entrance_id]] # [['e', 13], ['e', 15], ['e', 16], ['e', 17], ['n', 11], ['n', 13], ['n', 16], ['s', 12], ['s', 14], ['s', 15], ['s', 17], ['w', 11], ['w', 12], ['w', 16]]
        edge_list_entrance_desired_str = [['d', 'n'], ['d', lvroom_id]]
        extended_entrance_positions = [[18, 1], [18, 2], [19, 1], [19, 2]]
        extended_entrance_coords = [[1, 26], [2, 26], [1, 25], [2, 25]]
        facades_blocked = []
        sample_action_sequence = [1012, 3029, 14877, 9097, 11312, 6684]
        plan_id = 'Base_07_High_Resolution_Sep02'
        
    
    if n_rooms == 7:
        mask_numbers = 3
        masked_corners = ['corner_00', 'corner_01', 'corner_11']
        mask_lengths = [14, 14, 14]
        mask_widths = [6, 14, 14]
    
        areas_desired_fixed = [453, 184, 184, 156, 156, 120, 120]
        aspect_ratio_desired = [11.666666666666666, 3.2857142857142856, 2.7142857142857144, 6.0, 5.0, 1.2222222222222223, 3.8]    
    
        edge_list_room_desired = [[11, 12], [11, 13], [11, 14], [11, 15], [11, 16], [11, 17]] # [[11, 12], [11, 13], [11, 14], [11, 15], [11, 16], [11, 17], [12, 14], [13, 16], [13, 17], [15, 17]]
        edge_list_facade_desired_str = [['s', entrance_id]] # [['e', 11], ['e', 13], ['e', 15], ['e', 16], ['n', 11], ['n', 12], ['n', 13], ['n', 15], ['n', 16], ['n', 17], ['s', 11], ['s', 12], ['s', 14], ['w', 11], ['w', 12], ['w', 14]]
        edge_list_entrance_desired_str = [['d', 's'], ['d', lvroom_id]]
        extended_entrance_positions = [[44, 37], [44, 38], [43, 37], [43, 38]]
        extended_entrance_coords = [[37, 0], [38, 0], [37, 1], [38, 1]]
        facades_blocked = ['e']
        sample_action_sequence = [10768, 8317, 2272, 13549, 6492, 2881]
        # plan_id = 'Base_07_High_Resolution_Sep24'
        plan_id = 'Base_07_High_Resolution_Oct01_OA'
        # plan_id = 'Base_07_High_Resolution_Oct01_OS'
    
        
    if n_rooms == 88:
        mask_numbers = 2
        masked_corners = ['corner_00', 'corner_01']
        mask_lengths = [4, 12]
        mask_widths = [8, 20]
    
        areas_desired_fixed = [679, 156, 140, 132, 120, 120, 120, 110]
        aspect_ratio_desired = [4.714285714285714, 5.0, 1.4444444444444444, 1.75, 1.2222222222222223, 3.8, 1.2222222222222223, 2.0]    
    
        edge_list_room_desired = [[11, 12], [11, 13], [11, 14], [11, 15], [11, 16], [11, 17], [11, 18]]
        edge_list_facade_desired_str =  [['n', entrance_id]] # [['e', 11], ['e', 14], ['e', 16], ['e', 18], ['n', 11], ['n', 17], ['s', 12], ['s', 13], ['s', 16], ['w', 11], ['w', 12], ['w', 13], ['w', 17]]
        edge_list_entrance_desired_str = [['d', 'n'], ['d', lvroom_id]]
        extended_entrance_positions = [[20, 6], [20, 7], [21, 6], [21, 7]]
        extended_entrance_coords = [[6, 24], [7, 24], [6, 23], [7, 23]]
        facades_blocked = []
        sample_action_sequence = [1178, 12183, 3661, 6961, 17551, 13850, 9079]
        plan_id = 'Base_08_High_Resolution_Sep02'
        
    
    if n_rooms == 888:
        mask_numbers = 2
        masked_corners = ['corner_00', 'corner_01']
        mask_lengths = [4, 12]
        mask_widths = [8, 20]
    
        areas_desired_fixed = [299, 216, 210, 192, 180, 170, 160, 150]
        aspect_ratio_desired = [4.714285714285714, 5.0, 1.4444444444444444, 1.75, 1.2222222222222223, 3.8, 1.2222222222222223, 2.0]    
    
        edge_list_room_desired = [[11, 12], [11, 13], [11, 14], [11, 15], [11, 16], [11, 17], [11, 18]]
        edge_list_facade_desired_str = [['n', entrance_id]] # [['e', 11], ['e', 14], ['e', 16], ['e', 18], ['n', 11], ['n', 17], ['s', 12], ['s', 13], ['s', 16], ['w', 11], ['w', 12], ['w', 13], ['w', 17]]
        edge_list_entrance_desired_str = [['d', 'n'], ['d', lvroom_id]]
        extended_entrance_positions = [[20, 6], [20, 7], [21, 6], [21, 7]]
        extended_entrance_coords = [[6, 24], [7, 24], [6, 23], [7, 23]]
        facades_blocked = []
        sample_action_sequence = [1178, 12183, 3661, 6961, 17551, 13850, 9079]
        plan_id = 'Base_08_High_Resolution_Sep02'
        
        
    if n_rooms == 8:
        mask_numbers = 3
        masked_corners = ['corner_00', 'corner_01', 'corner_10']
        mask_lengths = [6, 10, 12]
        mask_widths = [14, 12, 16]
    
        areas_desired_fixed = [543, 162, 146, 144, 132, 120, 106, 100]
        aspect_ratio_desired = [29.0, 5.4, 4.0, 4.6, 4.0, 1.2222222222222223, 1.5, 1.0]    
    
        edge_list_room_desired =  [[11, 12], [11, 13], [11, 14], [11, 15], [11, 16], [11, 17], [11, 18]] ## [[11, 12], [11, 13], [11, 14], [11, 15], [11, 16], [11, 17], [11, 18], [13, 17], [14, 15], [14, 17], [15, 16], [16, 18]]
        edge_list_facade_desired_str = [['e', entrance_id]] # [['e', 11], ['e', 12], ['n', 11], ['n', 12], ['n', 14], ['n', 15], ['n', 16], ['n', 18], ['s', 11], ['s', 12], ['s', 13], ['s', 14], ['w', 13], ['w', 14], ['w', 15], ['w', 17]]
        edge_list_entrance_desired_str = [['d', 'e'], ['d', lvroom_id]]
        extended_entrance_positions = [[40, 32], [41, 32], [40, 31], [41, 31]]
        extended_entrance_coords = [[32, 4], [32, 3], [31, 4], [31, 3]]
        facades_blocked = []
        sample_action_sequence = [2595, 3824, 5800, 8513, 11271, 16669, 14287]
        plan_id = 'Base_08_High_Resolution_Sep24'
        # plan_id = 'Base_08_High_Resolution_Oct01_OA'
        # plan_id = 'Base_08_High_Resolution_Oct01_OS'
        
        
    if n_rooms == 99:
        mask_numbers = 1
        masked_corners = ['corner_11']
        mask_lengths = [10]
        mask_widths = [20]

        areas_desired_fixed = [650, 158, 146, 136, 122, 110, 108, 105, 104]
        aspect_ratio_desired = [17.0, 5.4, 5.0, 5.0, 1.5, 1.2222222222222223, 3.4, 3.3333333333333335, 1.0]    

        edge_list_room_desired = [[11, 12], [11, 13], [11, 14], [11, 15], [11, 18], [11, 19], [16, 18], [17, 19]]
        edge_list_facade_desired_str = [['n', entrance_id]] #[['e', 12], ['e', 13], ['e', 15], ['e', 16], ['e', 18], ['n', 11], ['n', 15], ['n', 16], ['n', 17], ['n', 19], ['s', 11], ['s', 12], ['w', 11], ['w', 14], ['w', 17]]
        edge_list_entrance_desired_str = [['d', 'n'], ['d', lvroom_id]]
        extended_entrance_positions = [[0, 16], [0, 17], [1, 16], [1, 17]]
        extended_entrance_coords = [[16, 44], [17, 44], [16, 43], [17, 43]]
        facades_blocked = []
        sample_action_sequence = [1472, 4142, 13771, 13131, 5778, 17486, 8182, 19101]
        plan_id = 'Base_09_High_Resolution_Aug_31'


    if n_rooms == 99:
        mask_numbers = 1
        masked_corners = ['corner_00']
        mask_lengths = [6]
        mask_widths = [18]
    
        areas_desired_fixed = [695, 168, 150, 144, 140, 120, 120, 104, 100]
        aspect_ratio_desired = [9.5, 1.1818181818181819, 5.0, 1.0, 1.4444444444444444, 1.2222222222222223, 1.2222222222222223, 1.8571428571428572, 1.0]    
    
        edge_list_room_desired = [[11, 12], [11, 13], [11, 14], [11, 15], [11, 16], [11, 17], [11, 18], [11, 19]]#, [12, 15], [12, 16], [13, 19], [14, 18], [16, 17], [17, 19]]
        edge_list_facade_desired_str = [['e', entrance_id]] # [['e', 11], ['e', 13], ['e', 14], ['n', 11], ['n', 12], ['n', 14], ['n', 15], ['n', 18], ['s', 12], ['s', 13], ['s', 17], ['s', 19], ['w', 12], ['w', 16], ['w', 17]]
        edge_list_entrance_desired_str = [['d', 'e'], ['d', lvroom_id]]
        extended_entrance_positions = [[19, 44], [20, 44], [19, 43], [20, 43]]
        extended_entrance_coords = [[44, 25], [44, 24], [43, 25], [43, 24]]
        facades_blocked = []
        sample_action_sequence = [2262, 4290, 14284, 11643, 20562, 8562, 5642, 16123]
        plan_id = 'Base_09_High_Resolution_Sep02'
        
        
    if n_rooms == 999:
        mask_numbers = 1
        masked_corners = ['corner_00']
        mask_lengths = [6]
        mask_widths = [18]
    
        areas_desired_fixed = [435, 228, 190, 178, 150, 146, 140, 140, 134]
        aspect_ratio_desired = [9.5, 1.1818181818181819, 5.0, 1.0, 1.4444444444444444, 1.2222222222222223, 1.2222222222222223, 1.8571428571428572, 1.0]    
    
        edge_list_room_desired = [[11, 12], [11, 13], [11, 14], [11, 15], [11, 16], [11, 17], [11, 18], [11, 19]]#, [12, 15], [12, 16], [13, 19], [14, 18], [16, 17], [17, 19]]
        edge_list_facade_desired_str = [['e', entrance_id]] # [['e', 11], ['e', 13], ['e', 14], ['n', 11], ['n', 12], ['n', 14], ['n', 15], ['n', 18], ['s', 12], ['s', 13], ['s', 17], ['s', 19], ['w', 12], ['w', 16], ['w', 17]]
        edge_list_entrance_desired_str = [['d', 'e'], ['d', lvroom_id]]
        extended_entrance_positions = [[19, 44], [20, 44], [19, 43], [20, 43]]
        extended_entrance_coords = [[44, 25], [44, 24], [43, 25], [43, 24]]
        facades_blocked = []
        sample_action_sequence = [2262, 4290, 14284, 11643, 20562, 8562, 5642, 16123]
        plan_id = 'Base_09_High_Resolution_Sep02'
        
        
    if n_rooms == 9:
        mask_numbers = 2
        masked_corners = ['corner_00', 'corner_11']
        mask_lengths = [6, 10]
        mask_widths = [12, 14]
    
        areas_desired_fixed = [543, 160, 152, 150, 144, 144, 120, 112, 112]
        aspect_ratio_desired = [15.0, 1.6666666666666667, 4.666666666666667, 5.0, 4.6, 1.0, 3.8, 1.8571428571428572, 1.8571428571428572]    
    
        edge_list_room_desired = [[11, 12], [11, 13], [11, 14], [11, 15], [11, 16], [11, 17], [11, 18], [11, 19]] # [[11, 12], [11, 13], [11, 14], [11, 15], [11, 16], [11, 18], [11, 19], [12, 13], [13, 17], [14, 15], [14, 16]]
        edge_list_facade_desired_str = [['w', entrance_id]] # [['e', 11], ['e', 12], ['e', 16], ['e', 17], ['n', 11], ['n', 12], ['n', 13], ['n', 17], ['n', 19], ['s', 14], ['s', 16], ['s', 18], ['w', 11], ['w', 14], ['w', 15], ['w', 18], ['w', 19]]
        edge_list_entrance_desired_str = [['d', 'w'], ['d', lvroom_id]]
        extended_entrance_positions = [[10, 0], [9, 0], [10, 1], [9, 1]]
        extended_entrance_coords = [[0, 34], [0, 35], [1, 34], [1, 35]]
        facades_blocked = []
        sample_action_sequence = [16815, 12227, 7121, 9119, 342, 13496, 19106, 2866]
        # plan_id = 'Base_09_High_Resolution_Sep24'
        plan_id = 'Base_09_High_Resolution_Oct01_OA'
        # plan_id = 'Base_09_High_Resolution_Oct01_OS'
    

    if n_rooms == 9999:
        mask_numbers = 2
        masked_corners = ['corner_00', 'corner_11']
        mask_lengths = [6, 10]
        mask_widths = [12, 14]
    
        areas_desired_fixed = [383, 180, 172, 170, 164, 164, 140, 132, 132]
        aspect_ratio_desired = [15.0, 1.6666666666666667, 4.666666666666667, 5.0, 4.6, 1.0, 3.8, 1.8571428571428572, 1.8571428571428572]    
    
        edge_list_room_desired = [[11, 12], [11, 13], [11, 14], [11, 15], [11, 16], [11, 17], [11, 18], [11, 19]] # [[11, 12], [11, 13], [11, 14], [11, 15], [11, 16], [11, 18], [11, 19], [12, 13], [13, 17], [14, 15], [14, 16]]
        edge_list_facade_desired_str = [['w', entrance_id]] # [['e', 11], ['e', 12], ['e', 16], ['e', 17], ['n', 11], ['n', 12], ['n', 13], ['n', 17], ['n', 19], ['s', 14], ['s', 16], ['s', 18], ['w', 11], ['w', 14], ['w', 15], ['w', 18], ['w', 19]]
        edge_list_entrance_desired_str = [['d', 'w'], ['d', lvroom_id]]
        extended_entrance_positions = [[10, 0], [9, 0], [10, 1], [9, 1]]
        extended_entrance_coords = [[0, 34], [0, 35], [1, 34], [1, 35]]
        facades_blocked = []
        sample_action_sequence = [16815, 12227, 7121, 9119, 342, 13496, 19106, 2866]
        plan_id = 'Base_09_High_Resolution_Sep24'
        
        
    # if n_rooms == 9: # pretty plan
    #     mask_numbers = 0
    #     masked_corners = []
    #     mask_lengths = []	
    #     mask_widths = []
    
    #     areas_desired_fixed = [95, 50, 45, 40, 35, 30, 25, 20, 20] 
        
    #     edge_list_room_desired = [[entrance_id, lvroom_id], [11, 12], [11, 13], [11, 14], [11, 15], [11, 16], [11, 17], [11, 18], [11, 19]]
    #     edge_list_facade_desired_str = [['e', entrance_id]]
    #     edge_list_entrance_desired_str = [['d', 'e'], ['d', lvroom_id]]
    #     extended_entrance_positions = [[20, 14], [21, 14], [20, 13], [21, 13]]
    #     extended_entrance_coords = [[14, 2], [14, 1], [13, 2], [13, 1]]
    #     facades_blocked = []
    #     sample_action_sequence = [736, 832, 1441, 1651, 1125, 286, 1167, 989]
    #     plan_id = 'Base_9_Room_Plan_Low_Res'
        
        
    if n_rooms == 9: # pretty plan
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


    ##        
    n_corners = 4
    min_room_id = lvroom_id
    corner_to_mask_room_id = {'corner_00':2, 'corner_01':3, 'corner_10':4, 'corner_11':5}
    
    areas_masked_list = [(L+1)*(W+1) for L, W in zip(mask_lengths, mask_widths)] 
    area_masked = sum(areas_masked_list)
    
    areas_masked = {f"room_{corner_to_mask_room_id[corner]}":area for corner, area in zip(masked_corners, areas_masked_list)}
    
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
    get_fixed_scenario(n_rooms=4)