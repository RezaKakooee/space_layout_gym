# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 14:42:29 2021

@author: Reza Kakooee
"""

# %%

import copy
import numpy as np

# %%

class SequentialDrawing:
    def __init__(self, fenv_config:dict=None):
        self.fenv_config = fenv_config
        
        
    @staticmethod
    def _cartesian2image_coord(x, y, max_y):
        return max_y-y, x


    def updata_obs_mat(self, plan_data_dict:dict=None, active_wall_name:str=None):
        outline_segments = copy.deepcopy(plan_data_dict['outline_segments'])
        inline_segments = copy.deepcopy(plan_data_dict['inline_segments'])
        
        wall_coords_type_name = 'walls_coords'
            
        wall_coords = copy.deepcopy(plan_data_dict[wall_coords_type_name])
        
        obs_mat = copy.deepcopy(plan_data_dict['obs_mat'])
        # obs_mat_mask = copy.deepcopy(plan_data_dict['obs_mat_mask'])
        obs_mat_w = copy.deepcopy(plan_data_dict['obs_mat_w'])
        obs_mat_for_dot_prod = copy.deepcopy(plan_data_dict['obs_mat_for_dot_prod']) 
        
        base_obs_mat = copy.deepcopy(plan_data_dict['base_obs_mat'])
        base_obs_mat_w = copy.deepcopy(plan_data_dict['base_obs_mat_w'])
        base_obs_mat_for_dot_prod = copy.deepcopy(plan_data_dict['base_obs_mat_for_dot_prod'])
        
        base_coords = wall_coords[active_wall_name]['base_coords']
        wall_i = int(active_wall_name.split('_')[1])
        
        for coord in base_coords:
            r, c = self._cartesian2image_coord(coord[0], coord[1], self.fenv_config['max_y'])
            
            obs_mat[r, c] = self.fenv_config['wall_pixel_value']
            obs_mat_w[r, c] = -wall_i 
            obs_mat_for_dot_prod[r, c] = 0
            
            base_obs_mat[r, c] = self.fenv_config['wall_pixel_value']
            base_obs_mat_w[r, c] = -wall_i 
            base_obs_mat_for_dot_prod[r, c] = 0
    
        plan_data_dict['base_obs_mat'] = copy.deepcopy(base_obs_mat)
        plan_data_dict['base_obs_mat_w'] = copy.deepcopy(base_obs_mat_w)
        plan_data_dict['base_obs_mat_for_dot_prod'] = copy.deepcopy(base_obs_mat_for_dot_prod)
        
        segment_names = ['front_segment', 'back_segment']
        wall_segment_names = [f"{active_wall_name}_front_segment", f"{active_wall_name}_back_segment"]
        for seg_name, wall_seg_name, in zip(segment_names, wall_segment_names):
            wall_i = int(wall_seg_name.split('_')[1])
            
            seg_val = inline_segments[wall_seg_name]
            
            anchor_coord =  seg_val['start_coord']
            radiation_coord = seg_val['end_coord']
            radiation_direction = seg_val['direction']
            
            x = radiation_coord[0]
            y = radiation_coord[1]
            
            r, c = self._cartesian2image_coord(x, y, self.fenv_config['max_y'])
            if plan_data_dict['moving_ones'][r, c] == 0:
            
                if radiation_direction == 'east':
                    if  x <= self.fenv_config['max_x']-self.fenv_config['scaling_factor']:
                        while True:
                            x += 1
                            if x <= self.fenv_config['max_x']:
                                r, c = self._cartesian2image_coord(x, y, self.fenv_config['max_y'])
                                # if (obs_mat[r, c] == 0) and (obs_mat_mask[r, c] == 1):
                                if obs_mat[r, c] == 0:
                                    obs_mat[r, c] = self.fenv_config['wall_pixel_value']
                                    obs_mat_w[r, c] = -wall_i 
                                    obs_mat_for_dot_prod[r, c] = 0
                                else:
                                    break
                            else:
                                x = self.fenv_config['max_x']
                                break
                                
                elif radiation_direction == 'west':
                    if  x >= self.fenv_config['scaling_factor']:
                        while True:
                            x -= 1
                            if x >= 0:
                                r, c = self._cartesian2image_coord(x, y, self.fenv_config['max_y'])
                                # if (obs_mat[r, c] == 0) and (obs_mat_mask[r, c] == 1):
                                if obs_mat[r, c] == 0:
                                    obs_mat[r, c] = self.fenv_config['wall_pixel_value']
                                    obs_mat_w[r, c] = -wall_i 
                                    obs_mat_for_dot_prod[r, c] = 0
                                else:
                                    break
                            else:
                                x = 0
                                break
                                
                elif radiation_direction == 'south':
                    if  y >= self.fenv_config['scaling_factor']:
                        while True:
                            y -= 1
                            if y >= 0:
                                r, c = self._cartesian2image_coord(x, y, self.fenv_config['max_y'])
                                # if (obs_mat[r, c] == 0) and (obs_mat_mask[r, c] == 1):
                                if obs_mat[r, c] == 0:
                                    obs_mat[r, c] = self.fenv_config['wall_pixel_value']
                                    obs_mat_w[r, c] = -wall_i 
                                    obs_mat_for_dot_prod[r, c] = 0
                                else:
                                    break
                            else:
                                y = 0
                                break
                                
                elif radiation_direction == 'north':
                    if  y <= self.fenv_config['max_y']-self.fenv_config['scaling_factor']:
                        while True:
                            y += 1
                            if y <= self.fenv_config['max_y']:
                                r, c = self._cartesian2image_coord(x, y, self.fenv_config['max_y'])
                                # if (obs_mat[r, c] == 0) and (obs_mat_mask[r, c] == 1):
                                if obs_mat[r, c] == 0:
                                    obs_mat[r, c] = self.fenv_config['wall_pixel_value']
                                    obs_mat_w[r, c] = -wall_i 
                                    obs_mat_for_dot_prod[r, c] = 0
                                else:
                                    break
                            else:
                                y = self.fenv_config['max_y']
                                break
                
                reflection_coord = [x, y]
            
            else:
                reflection_coord = [x, y]
                
            inline_segments[wall_seg_name]['reflection_coord'] = copy.deepcopy(reflection_coord)
            plan_data_dict[wall_coords_type_name][active_wall_name][seg_name].update({'reflection_coord': copy.deepcopy(reflection_coord)})
            
        this_wall_positions = np.argwhere(obs_mat_w == -wall_i).tolist()
        plan_data_dict[wall_coords_type_name][active_wall_name].update({'wall_positions': copy.deepcopy(this_wall_positions)})
            
        plan_data_dict['inline_segments'] = copy.deepcopy(inline_segments)
        plan_data_dict['obs_mat'] = copy.deepcopy(obs_mat)
        plan_data_dict['obs_mat_w'] = copy.deepcopy(obs_mat_w)
        plan_data_dict['canvas_mat'] = copy.deepcopy(obs_mat)
        plan_data_dict['obs_mat_for_dot_prod'] = copy.deepcopy(obs_mat_for_dot_prod)
        
        wall_names = plan_data_dict[wall_coords_type_name].keys()
        for wall_name in wall_names:
            other_walls_positions = []
            for w_n in wall_names:
                if w_n != wall_name:
                    other_walls_positions.extend(plan_data_dict[wall_coords_type_name][w_n]['wall_positions'])
            plan_data_dict[wall_coords_type_name][wall_name].update({'other_walls_positions': other_walls_positions})    
            wall_positions = plan_data_dict[wall_coords_type_name][wall_name]['wall_positions']
            obs_mat_witout_this_wall = copy.deepcopy(obs_mat)
            for position in wall_positions:
                obs_mat_witout_this_wall[position[0], position[1]] = 0
                plan_data_dict[wall_coords_type_name][wall_name].update({'obs_mat_witout_this_wall': obs_mat_witout_this_wall})

        return plan_data_dict


    def _test_odd_coords(self, wall_positions):
        points = [p for pp in wall_positions for p in pp]
        for p in points:
            if p%2 != 0:
                print(p)