# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 18:32:31 2021

@author: Reza Kakooee
"""

#%%
import pickle
import numpy as np



#%%
class WallGenerator:
    def __init__(self, fenv_config:dict=None):
        self.fenv_config = fenv_config
            

    def make_walls(self, valid_points_for_sampling, walls_coords=None, wall_name=None):
        if walls_coords is None:
            walls_coords = self._generate_random_wall_coords(valid_points_for_sampling)

        else:
            if not isinstance(walls_coords, dict):
                    raise ValueError(f"In make_walls of wall_generator: walls_coords must be dict! The current type is: {type(walls_coords)}")
            walls_coords = {wall_name: walls_coords} # wall_1 is a temp name
        
        walls_coords = self._make_walls_properties(walls_coords)
        return walls_coords 
    
    
    def _make_walls_properties(self, walls_coords):
        for w_name, w_coords in walls_coords.items():
            front_ori, front_dir, front_pos = self.__fragment_identifier(start_coord=w_coords['anchor_coord'], 
                                                                       end_coord=w_coords['front_open_coord'])
            back_ori, back_dir, back_pos = self.__fragment_identifier(start_coord=w_coords['anchor_coord'], 
                                                                    end_coord=w_coords['back_open_coord'])
            
            walls_coords[w_name].update({'front_segment': {'orientation': front_ori,
                                                           'direction': front_dir,
                                                           'location': front_pos},
                                         'back_segment':  {'orientation': back_ori,
                                                           'direction': back_dir,
                                                           'location': back_pos} })
        for wall_name in walls_coords.keys():
            walls_coords[wall_name].update({'base_coords' : [walls_coords[wall_name]['back_open_coord'], 
                                                  walls_coords[wall_name]['anchor_coord'],
                                                  walls_coords[wall_name]['front_open_coord']]})
            
        if self.fenv_config['seg_length'] >= 2:
            segment_names = ['front_segment', 'back_segment']
            for wall_name, wall_data in walls_coords.items():
                for seg_nam in segment_names:
                    direction = wall_data[seg_nam]['direction']
                    if direction == 'east':
                        for u in range(1, self.fenv_config['seg_length']):
                            walls_coords[wall_name]['base_coords'].append([ walls_coords[wall_name]['anchor_coord'][0]+u, walls_coords[wall_name]['anchor_coord'][1]])
                    elif direction == 'west':
                        for u in range(1, self.fenv_config['seg_length']):
                            walls_coords[wall_name]['base_coords'].append([ walls_coords[wall_name]['anchor_coord'][0]-u, walls_coords[wall_name]['anchor_coord'][1]])        
                    elif direction == 'north':
                        for u in range(1, self.fenv_config['seg_length']):
                            walls_coords[wall_name]['base_coords'].append([ walls_coords[wall_name]['anchor_coord'][0], walls_coords[wall_name]['anchor_coord'][1]+u])
                    elif direction == 'south':
                        for u in range(1, self.fenv_config['seg_length']):
                            walls_coords[wall_name]['base_coords'].append([ walls_coords[wall_name]['anchor_coord'][0], walls_coords[wall_name]['anchor_coord'][1]-u])
        
        return walls_coords
        
        
    def _generate_random_wall_coords(self, valid_poinsts_for_sampling=None):
        seg_length = self.fenv_config['seg_length']
        if valid_poinsts_for_sampling is None:
            anchor_points_Xs = list(seg_length * 
                                    np.random.randint(
                                        0+1, int(self.fenv_config['max_x']/seg_length) -1, 
                                        self.fenv_config['n_walls']))
            anchor_points_Ys = list(seg_length * 
                                    np.random.randint(
                                        0+1, int(self.fenv_config['max_y']/seg_length) -1, 
                                        self.fenv_config['n_walls']))
            anchor_points = [[x,y] for x, y in zip(anchor_points_Xs, anchor_points_Ys)]
        
        else:
            anchor_points = valid_poinsts_for_sampling[np.random.choice(valid_poinsts_for_sampling.shape[0], 3, replace=False), :].tolist()
        
        walls_coords = {}
        for i, anchor_point in enumerate(anchor_points):
            neighborhood_points = [[anchor_point[0]+self.fenv_config['seg_length'], 
                                            anchor_point[1]],
                                   [anchor_point[0]-self.fenv_config['seg_length'], 
                                            anchor_point[1]],
                                   [anchor_point[0], 
                                            anchor_point[1]+self.fenv_config['seg_length']],
                                   [anchor_point[0],
                                            anchor_point[1]-self.fenv_config['seg_length']]]
                                   
            n_neighborhood = len(neighborhood_points)
            choices = np.random.choice(range(0,n_neighborhood), 2, replace=False)
            w_coords = {'anchor_coord': anchor_point,
                        'back_open_coord': neighborhood_points[choices[0]],
                        'front_open_coord': neighborhood_points[choices[1]]}
            
            walls_coords.update({f"wall_{i+1+self.fenv_config['mask_numbers']}": w_coords})
            
        return walls_coords
            
    
    def __fragment_identifier(self, start_coord=None, end_coord=None):
        if (start_coord is None) or (end_coord is None):
            raise ValueError('Input coords cannot be None')
            
        max_x = self.fenv_config['max_x']
        max_y = self.fenv_config['max_y']
        
        xs = start_coord[0]
        ys = start_coord[1]
        xe = end_coord[0]
        ye = end_coord[1]
        
        in_condition = (np.max([xs, xe]) <= max_x) and (np.max([ys, ye]) <= max_y) \
            and (np.min([xs, xe]) >= 0) and (np.min([ys, ye]) >= 0)
        if in_condition:
            location = 'in'
        else:
            location = 'out'
            print(f"xs: {xs}, ys: {ys}, xe: {xe}, ye: {ye}")
            raise ValueError('Invalid wall coords')
            
        dx = xe - xs
        dy = ye - ys
        
        if (dx == 0) and (dy == 0):
            raise ValueError("Not a valid fragment")
        elif (dx == 0) and (dy != 0):
            if dy > 0:
                orientation = 'axial'
                direction = 'north'
            else:
                orientation = 'axial'
                direction = 'south'
        elif (dx != 0) and (dy == 0):
            if dx > 0:
                orientation = 'axial'
                direction = 'east'
            else:
                orientation = 'axial'
                direction = 'west'   
        else:
            if dx > 0:
                if dy > 0:
                    orientation = 'diagonal'
                    direction = 'north_east'
                else:
                    orientation = 'diagonal'
                    direction = 'south_east'
            else:
                if dy > 0:
                    orientation = 'diagonal'
                    direction = 'north_west'
                else:
                    orientation = 'diagonal'
                    direction = 'south_west'
                    
        return orientation, direction, location