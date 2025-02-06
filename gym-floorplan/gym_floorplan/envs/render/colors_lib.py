#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 23:49:11 2023

@author: Reza Kakooee
"""


#%%
def get_color_dict(fenv_config):
    phase = fenv_config['phase']
    
    color_map = {
        1: 'silver', 
        
        2: 'silver', 
        3: 'silver',
        4: 'silver', 
        5: 'silver',
        
        6: 'skyblue', 
        7: 'aqua',
        8: 'beige', 
        9: 'pink',
        
        10: 'darkorchid',
        
        11: 'red', 
        12: 'blue', 
        13: 'green', # 'magenta',  
        14: 'deeppink', 
        15: 'magenta', 
        16: 'gold',
        17: 'darkcyan', #'mediumvioletred', 
        18: 'orange', 
        19: 'aqua', 
        20: 'yellow',
        
        21: 'darkslategray',
        22: 'darkolivegreen',
        23: 'midnightblue',
        24: 'darkred',
        25: 'darkslateblue',
        26: 'darkgoldenrod',
        27: 'saddlebrown',
        28: 'indigo',
        29: 'maroon',
        30: 'darkmagenta',
        
        }
    
    outline_color = 'silver'

    wall_colors = {
        'outline_color': outline_color,
        
        'left_segment': outline_color,
        'down_segment': outline_color,
        'right_segment': outline_color,
        'up_segment': outline_color,
        }
    
    wall_colors.update({
        f"wall_{i}_back_segment": outline_color for i in range(2, fenv_config['entrance_cell_id'])
        })
    
    wall_colors.update({
        f"wall_{i}_front_segment": outline_color for i in range(2, fenv_config['entrance_cell_id'])
        })
        
    room_colors = {
        'room_n': 'purple',
        'room_s': 'turquoise',
        'room_e': 'coral',
        'room_w': 'lightblue',
        
        'room_d': 'black',
        }
    
    if phase in ['test', 'bc_evaluate']:
        room_colors = {
            'room_n': 'silver',
            'room_s': 'silver',
            'room_e': 'silver',
            'room_w': 'silver',
            
            'room_d': 'black',
            'room_10': 'black',
            }
    
    room_colors.update({
        f"room_{i}": outline_color for i in range(2, fenv_config['entrance_cell_id'])
        })
    
    
    for i in fenv_config['real_room_id_range']:
        if phase in ['train', 'debug']: # TODO, Maybe remove the test
            wall_colors.update({f"wall_{i}_back_segment": color_map[i],
                                f"wall_{i}_front_segment": color_map[i]})
            
            room_colors.update({f"room_{i}": color_map[i]})
        elif phase in ['test', 'bc_evaluate']:
            wall_colors.update({f"wall_{i}_back_segment": outline_color,
                                f"wall_{i}_front_segment": outline_color})
            
            room_colors.update({f"room_{i}": color_map[i]}) # room_colors.update({f"room_{i}": outline_color})
        
    
    cmaps = ['CMRmap', 'CMRmap_r', 'Paired', 'Paired_r', 
             'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 
             'Spectral', 'Spectral_r', 'binary', 'binary_r', 'bone', 'bone_r', 
             'brg', 'brg_r', 'bwr', 'bwr_r', 
             'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r',  
             'hsv_r', 'jet', 'jet_r', 'nipy_spectral', 'plasma']
    
    
    return wall_colors, room_colors



def outline_color():
    return 'black'



#%%
if __name__ == '__main__':
    wall_colors, room_colors = get_color_dict()