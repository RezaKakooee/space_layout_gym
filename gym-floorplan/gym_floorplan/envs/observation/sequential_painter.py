# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 14:42:29 2021

@author: Reza Kakooee
"""

# %%

import copy
import numpy as np



# %%

class SequentialPainter:
    def __init__(self, fenv_config:dict=None):
        self.fenv_config = fenv_config
        
        

    def update_obs_mat(self, plan_data_dict:dict=None, active_wall_name:str=None):
        wall_outline_segments = copy.deepcopy(plan_data_dict['wall_outline_segments'])
        wall_inline_segments = copy.deepcopy(plan_data_dict['wall_inline_segments'])
        
        wall_coords_type_name = 'walls_coords'
            
        wall_coords = copy.deepcopy(plan_data_dict[wall_coords_type_name])
        
        obs_blocked_cells = copy.deepcopy(plan_data_dict['obs_blocked_cells'])
        obs_mat = copy.deepcopy(plan_data_dict['obs_mat'])
        # obs_mat_mask = copy.deepcopy(plan_data_dict['obs_mat_mask'])
        obs_mat_w = copy.deepcopy(plan_data_dict['obs_mat_w'])
        obs_mat_for_dot_prod = copy.deepcopy(plan_data_dict['obs_mat_for_dot_prod']) 
        
        obs_mat_base = copy.deepcopy(plan_data_dict['obs_mat_base'])
        obs_mat_base_w = copy.deepcopy(plan_data_dict['obs_mat_base_w'])
        obs_mat_base_for_dot_prod = copy.deepcopy(plan_data_dict['obs_mat_base_for_dot_prod'])
        
        
        if self.fenv_config['env_planning'] == 'One_Shot': 
            base_coords = wall_coords[active_wall_name]['base_coords']
            wall_i = int(active_wall_name.split('_')[1])
            
            for coord in base_coords:
                r, c = self._cartesian2image_coord(coord[0], coord[1], self.fenv_config['max_y'])
                
                obs_mat[r, c] = self.fenv_config['wall_pixel_value']
                obs_mat_w[r, c] = -wall_i 
                obs_mat_for_dot_prod[r, c] = 0  
                
                obs_mat_base[r, c] = self.fenv_config['wall_pixel_value']
                obs_mat_base_w[r, c] = -wall_i 
                obs_mat_base_for_dot_prod[r, c] = 0
        else:
            new_walls_coords_dyp = plan_data_dict.get('new_walls_coords_dyp', plan_data_dict['walls_coords'])
            for wall_name in new_walls_coords_dyp.keys():
                base_coords = new_walls_coords_dyp[wall_name]['base_coords']
                wall_i = int(wall_name.split('_')[1])
                
                for coord in base_coords:
                    r, c = self._cartesian2image_coord(coord[0], coord[1], self.fenv_config['max_y'])
                    
                    obs_mat[r, c] = self.fenv_config['wall_pixel_value']
                    obs_mat_w[r, c] = -wall_i 
                    obs_mat_for_dot_prod[r, c] = 0  
                    
                    obs_mat_base[r, c] = self.fenv_config['wall_pixel_value']
                    obs_mat_base_w[r, c] = -wall_i 
                    obs_mat_base_for_dot_prod[r, c] = 0
        
        plan_data_dict['obs_mat_base'] = copy.deepcopy(obs_mat_base)
        plan_data_dict['obs_mat_base_w'] = copy.deepcopy(obs_mat_base_w)
        plan_data_dict['obs_mat_base_for_dot_prod'] = copy.deepcopy(obs_mat_base_for_dot_prod)
        
        segment_names = ['front_segment', 'back_segment']
        wall_segment_names = [f"{active_wall_name}_front_segment", f"{active_wall_name}_back_segment"]
        for seg_name, wall_seg_name, in zip(segment_names, wall_segment_names):
            wall_i = int(wall_seg_name.split('_')[1])
            
            seg_val = wall_inline_segments[wall_seg_name]
            
            anchor_coord =  seg_val['start_coord']
            radiation_coord = seg_val['end_coord']
            radiation_direction = seg_val['direction']
            
            x = radiation_coord[0]
            y = radiation_coord[1]
            
            delta_x_or_y = 1 if self.fenv_config['so_thick_flag'] else 0
            r, c = self._cartesian2image_coord(x, y, self.fenv_config['max_y'])
            if plan_data_dict['obs_moving_ones'][r, c] == 0:
            
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
                                    x -= delta_x_or_y
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
                                    x += delta_x_or_y
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
                                    y += delta_x_or_y
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
                                    y -= delta_x_or_y
                                    break
                            else:
                                y = self.fenv_config['max_y']
                                break
                
                reflection_coord = [x, y]
            
            else:
                reflection_coord = [x, y]
                
            wall_inline_segments[wall_seg_name]['reflection_coord'] = copy.deepcopy(reflection_coord)
            plan_data_dict[wall_coords_type_name][active_wall_name][seg_name].update({'reflection_coord': copy.deepcopy(reflection_coord)})
            
        this_wall_positions = np.argwhere(obs_mat_w == -wall_i).tolist()
        plan_data_dict[wall_coords_type_name][active_wall_name].update({'wall_positions': copy.deepcopy(this_wall_positions)})
            
        plan_data_dict['wall_inline_segments'] = copy.deepcopy(wall_inline_segments)
        plan_data_dict['obs_mat'] = copy.deepcopy(obs_mat)
        plan_data_dict['obs_mat_w'] = copy.deepcopy(obs_mat_w)
        plan_data_dict['obs_canvas_mat'] = copy.deepcopy(obs_mat)
        plan_data_dict['obs_mat_for_dot_prod'] = copy.deepcopy(obs_mat_for_dot_prod)
        
        wall_names = plan_data_dict[wall_coords_type_name].keys()
        for wall_name in wall_names:
            other_walls_positions = []
            for w_n in wall_names:
                if w_n != wall_name:
                    other_walls_positions.extend(plan_data_dict[wall_coords_type_name][w_n]['wall_positions'])
            plan_data_dict[wall_coords_type_name][wall_name].update({'other_walls_positions': other_walls_positions})    
            wall_positions = plan_data_dict[wall_coords_type_name][wall_name]['wall_positions']
            obs_mat_without_this_wall = copy.deepcopy(obs_mat)
            for position in wall_positions:
                obs_mat_without_this_wall[position[0], position[1]] = 0
                plan_data_dict[wall_coords_type_name][wall_name].update({'obs_mat_without_this_wall': obs_mat_without_this_wall})

        return plan_data_dict
    
    
    
    def update_obs_mat_for_dyp_step_size(self, input_plan_data_dict, plan_data_dict, agent_names_deque):
        for agent_name in agent_names_deque:
            wall_name = agent_name.replace('agent', 'wall')
            wall_i = int(wall_name.split('_')[1])
            for coord in plan_data_dict['walls_coords'][wall_name]['base_coords']:
                r, c = self._cartesian2image_coord(coord[0], coord[1], self.fenv_config['max_y'])
                plan_data_dict['obs_mat_base_w'][r, c] = -wall_i
        
        dummy_obs_mat_w = np.zeros_like(plan_data_dict['obs_mat_w']) + input_plan_data_dict['obs_mat_w'] + np.where(plan_data_dict['obs_mat_base_w'] <= -12, plan_data_dict['obs_mat_base_w'], 0)
        dummy_wall_inline_segments = copy.deepcopy(plan_data_dict['wall_inline_segments'])
        stop_flag_list = []
        while True:
            old_dummy_obs_mat_w = copy.deepcopy(dummy_obs_mat_w)
            for i, ag_name in enumerate(agent_names_deque):
                inwall_i = int(ag_name.split('_')[1])
                inwall_name = f"wall_{inwall_i}"
                dummy_obs_mat_w, dummy_wall_inline_segments, stop_flag_list = self._take_one_step_ahead(dummy_obs_mat_w, dummy_wall_inline_segments, inwall_name, stop_flag_list)
                
            if np.all(np.equal(old_dummy_obs_mat_w, dummy_obs_mat_w)):
                break
        dummy_obs_mat = np.where(old_dummy_obs_mat_w < 0, 10, 0)
        dummy_obs_mat_for_dot_prod = np.where(old_dummy_obs_mat_w < 0, 1, 0)
        plan_data_dict['obs_mat'] = copy.deepcopy(dummy_obs_mat)
        plan_data_dict['obs_mat_w'] = copy.deepcopy(dummy_obs_mat_w)
        plan_data_dict['obs_canvas_mat'] = copy.deepcopy(dummy_obs_mat)
        plan_data_dict['obs_mat_for_dot_prod'] = copy.deepcopy(dummy_obs_mat_for_dot_prod)
        
        for agent_name in agent_names_deque:
            wall_name = agent_name.replace('agent', 'wall')
            wall_i = int(wall_name.split('_')[1])
            this_wall_positions = np.argwhere(dummy_obs_mat_w == -wall_i).tolist()
            plan_data_dict['walls_coords'][wall_name].update({'wall_positions': copy.deepcopy(this_wall_positions)})
            
            segment_names = ['front_segment', 'back_segment']
            wall_segment_names = [f"{wall_name}_front_segment", f"{wall_name}_back_segment"]
            for seg_name, wall_seg_name, in zip(segment_names, wall_segment_names):
                wall_i = int(wall_seg_name.split('_')[1])
                dummy_wall_inline_segments[wall_seg_name]['reflection_coord'] = dummy_wall_inline_segments[wall_seg_name]['reflection_coord']
                plan_data_dict['walls_coords'][wall_name][seg_name].update({'reflection_coord': copy.deepcopy(dummy_wall_inline_segments[wall_seg_name]['reflection_coord'])})
        
        wall_names = plan_data_dict['walls_coords'].keys()
        for wall_name in wall_names:
            other_walls_positions = []
            for w_n in wall_names:
                if w_n != wall_name:
                    other_walls_positions.extend(plan_data_dict['walls_coords'][w_n]['wall_positions'])
            plan_data_dict['walls_coords'][wall_name].update({'other_walls_positions': other_walls_positions})    
            wall_positions = plan_data_dict['walls_coords'][wall_name]['wall_positions']
            obs_mat_without_this_wall = copy.deepcopy(dummy_obs_mat)
            for position in wall_positions:
                obs_mat_without_this_wall[position[0], position[1]] = 0
                plan_data_dict['walls_coords'][wall_name].update({'obs_mat_without_this_wall': obs_mat_without_this_wall})
        
        
        plan_data_dict.update({
            'wall_inline_segments': copy.deepcopy(dummy_wall_inline_segments)
        })
        
        return plan_data_dict
    
    
    
    def _take_one_step_ahead(self, obs_mat_w, wall_inline_segments, active_wall_name, stop_flag_list):
        wall_i = int(active_wall_name.split('_')[1])
        segment_names = ['front_segment', 'back_segment']
        wall_segment_names = [f"{active_wall_name}_front_segment", f"{active_wall_name}_back_segment"]
        for seg_name, wall_seg_name, in zip(segment_names, wall_segment_names):
            key_name = active_wall_name + '_' + seg_name
            
            if key_name in stop_flag_list:
                continue
            
            wall_i = int(wall_seg_name.split('_')[1])
            radiation_direction = wall_inline_segments[wall_seg_name]['direction']
            if 'reflection_coord' not in wall_inline_segments[wall_seg_name].keys():
                x = wall_inline_segments[wall_seg_name]['end_coord'][0]
                y = wall_inline_segments[wall_seg_name]['end_coord'][1]
            else:
                x = wall_inline_segments[wall_seg_name]['reflection_coord'][0]
                y = wall_inline_segments[wall_seg_name]['reflection_coord'][1]
                
            r, c = self._cartesian2image_coord(x, y, self.fenv_config['max_y'])
            stop_flag = 0
            if radiation_direction == 'east':
                if  x <= self.fenv_config['max_x']-self.fenv_config['scaling_factor']:
                    x += 1
                    if x <= self.fenv_config['max_x']:
                        r, c = self._cartesian2image_coord(x, y, self.fenv_config['max_y'])
                        if obs_mat_w[r, c] == 0:
                            obs_mat_w[r, c] = -wall_i 
                        else:
                            stop_flag = 1
                    else:
                        x = self.fenv_config['max_x']
                            
            elif radiation_direction == 'west':
                if  x >= self.fenv_config['scaling_factor']:
                    x -= 1
                    if x >= 0:
                        r, c = self._cartesian2image_coord(x, y, self.fenv_config['max_y'])
                        if obs_mat_w[r, c] == 0:
                            obs_mat_w[r, c] = -wall_i 
                        else:
                            stop_flag = 1
                    else:
                        x = 0
                            
            elif radiation_direction == 'south':
                if  y >= self.fenv_config['scaling_factor']:
                    y -= 1
                    if y >= 0:
                        r, c = self._cartesian2image_coord(x, y, self.fenv_config['max_y'])
                        if obs_mat_w[r, c] == 0:
                            obs_mat_w[r, c] = -wall_i 
                        else:
                            stop_flag = 1
                    else:
                        y = 0
                            
            elif radiation_direction == 'north':
                if  y <= self.fenv_config['max_y']-self.fenv_config['scaling_factor']:
                    y += 1
                    if y <= self.fenv_config['max_y']:
                        r, c = self._cartesian2image_coord(x, y, self.fenv_config['max_y'])
                        if obs_mat_w[r, c] == 0:
                            obs_mat_w[r, c] = -wall_i 
                        else:
                            stop_flag = 1
                    else:
                        y = self.fenv_config['max_y']
            
            reflection_coord = [x, y]
            wall_inline_segments[wall_seg_name].update({'reflection_coord': reflection_coord})
            if stop_flag:
                stop_flag_list.append(key_name)
            
        return obs_mat_w, wall_inline_segments, stop_flag_list



    def _test_odd_coords(self, wall_positions):
        points = [p for pp in wall_positions for p in pp]
        for p in points:
            if p%2 != 0:
                print(p)
                
                
                
    def update_dummy_obs_mat_for_dyp(self, obs_mat, obs_mat_w, obs_moving_ones, wall_inline_segments, active_wall_name):
        wall_i = int(active_wall_name.split('_')[1])
        
        segment_names = ['front_segment', 'back_segment']
        wall_segment_names = [f"{active_wall_name}_front_segment", f"{active_wall_name}_back_segment"]
        for seg_name, wall_seg_name, in zip(segment_names, wall_segment_names):
            wall_i = int(wall_seg_name.split('_')[1])
            
            seg_val = wall_inline_segments[wall_seg_name]
            radiation_coord = seg_val['end_coord']
            radiation_direction = seg_val['direction']
            x = radiation_coord[0]
            y = radiation_coord[1]
            
            delta_x_or_y = 1 if self.fenv_config['so_thick_flag'] else 0
            r, c = self._cartesian2image_coord(x, y, self.fenv_config['max_y'])
            if obs_moving_ones[r, c] == 0:
            
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
                                else:
                                    x -= delta_x_or_y
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
                                else:
                                    x += delta_x_or_y
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
                                else:
                                    y += delta_x_or_y
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
                                else:
                                    y -= delta_x_or_y
                                    break
                            else:
                                y = self.fenv_config['max_y']
                                break
                
            reflection_coord = [x, y]
        
        return obs_mat, obs_mat_w
    
        
    @staticmethod
    def _cartesian2image_coord(x, y, max_y):
        return int(max_y-y), int(x)