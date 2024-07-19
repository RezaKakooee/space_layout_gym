# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 17:30:10 2021

@author: Reza Kakooee
"""
#%%
import copy
import numpy as np

from gym_floorplan.envs.observation.wall_generator import WallGenerator



#%%
class WallTransform:
    def __init__(self, wall_name:str=None, wall_coords:dict=None, action_i:int=None, plan_data_dict=None, fenv_config:dict=None):
        self.wall_name = wall_name
        self.wall_coords = wall_coords
        self.transformed_wall_coords = copy.deepcopy(wall_coords)
        self.back_open_coord  = wall_coords['back_open_coord']
        self.anchor_coord =  wall_coords['anchor_coord']
        self.front_open_coord = wall_coords['front_open_coord']
        self.wall_only_coords = [self.back_open_coord, self.anchor_coord, self.front_open_coord]
        self.front_coords = [self.anchor_coord, self.front_open_coord]
        self.back_coords = [self.anchor_coord, self.back_open_coord]
        
        self.fenv_config = fenv_config
        self.action_i = action_i
        self.action_name = fenv_config['action_dict'][action_i]
        self.plan_data_dict = plan_data_dict
        
    def transform(self):
        self.dynamic_coords = self._get_dynamic_coords()
        self.transformer_matrix = self._get_transformer(self.action_name)
        
        if np.size(self.transformer_matrix) != 0:
            if 'front' in self.action_name:
                dynamic_coords = self.front_coords
                transformed_dynamic_coords = self._apply_transformer(dynamic_coords)
                new_wall = [self.back_open_coord, transformed_dynamic_coords[0], transformed_dynamic_coords[1]]
                
            elif 'back' in self.action_name:
                dynamic_coords = self.back_coords
                transformed_dynamic_coords = self._apply_transformer(dynamic_coords)
                new_wall = [transformed_dynamic_coords[1], transformed_dynamic_coords[0], self.front_open_coord]
                
            else:
                dynamic_coords = self.wall_only_coords
                new_wall = self._apply_transformer(dynamic_coords)
            
            on_top_flag = False
            for new_potential_point in new_wall:
                if self._is_on_top_of(new_potential_point):
                    on_top_flag = True
                
            if not on_top_flag:
                self.transformed_wall_coords = self._make_wall_coords(new_wall)
            
                self._test_odd_coords(self.transformed_wall_coords[self.wall_name])
            else:
                self.transformed_wall_coords = {self.wall_name: self.wall_coords}
            
            return self.transformed_wall_coords
        
        else:
            
            if self.action_name == 'grow_front_seg':
                self.new_front_open_coord = self._grow_segment(self.front_open_coord, 
                                                           self.wall_coords['front_segment']['direction'], 
                                                           self.fenv_config['seg_length'])
                new_wall = [self.back_open_coord, self.anchor_coord, self.new_front_open_coord]
                
            elif self.action_name == 'cut_front_seg':
                self.new_front_open_coord = self._cut_segment(self.front_open_coord, 
                                                           self.wall_coords['front_segment']['direction'], 
                                                           self.fenv_config['seg_length'])
                new_wall = [self.back_open_coord, self.anchor_coord, self.new_front_open_coord]
                
            elif self.action_name == 'grow_back_seg':
                self.new_back_open_coord = self._grow_segment(self.back_open_coord, 
                                                           self.wall_coords['back_segment']['direction'], 
                                                           self.fenv_config['seg_length'])
                new_wall = [self.new_back_open_coord, self.anchor_coord, self.front_open_coord]
                
            elif self.action_name == 'cut_back_seg':
                self.new_back_open_coord = self._cut_segment(self.back_open_coord, 
                                                           self.wall_coords['back_segment']['direction'], 
                                                           self.fenv_config['seg_length'])
                new_wall = [self.new_back_open_coord, self.anchor_coord, self.front_open_coord]
                
            else:   
                raise ValueError('Invalid action selected')
            
            
            if self._is_valid(new_wall):
                self.transformed_wall_coords = self._make_wall_coords(new_wall)
                return self.transformed_wall_coords
            else:
                return {self.wall_name: self.wall_coords}
    
    
    def _is_on_top_of(self, new_point_coord):
        r, c = self._cartesian2image_coord(new_point_coord[0], new_point_coord[1], self.fenv_config['max_y'])
        obs_mat_w = -1 * self.plan_data_dict['obs_mat_w']
        wall_i = int(self.wall_name.split('_')[1])
        obs_mat_w[np.where(obs_mat_w == wall_i)] = 0
        
        flag = False
        if int(abs(obs_mat_w[r][c])) != 0:
            flag = True

        if new_point_coord[0] == 0 or \
            new_point_coord[1] == 0 or \
                new_point_coord[0] == self.fenv_config['max_x'] or \
                    new_point_coord[1] == self.fenv_config['max_y']:
            flag = True
        
        return flag
    
    
    @staticmethod
    def _test_odd_coords(wall_coords):
        points = np.zeros((6))
        points[0], points[1] = wall_coords['back_open_coord'][0], wall_coords['back_open_coord'][1]
        points[2], points[3] = wall_coords['anchor_coord'][0], wall_coords['anchor_coord'][1]
        points[4], points[5] = wall_coords['front_open_coord'][0], wall_coords['front_open_coord'][1]
        points = points.tolist()
        for p in points:
            if p%2 != 0:
                print(p)
                raise ValueError(f"Wall_transform, there is something wrong here. The wall_coords is: {wall_coords}")
        
    def _get_dynamic_coords(self):
        if 'front' in self.action_name:
            dynamic_coords = self.front_coords
        elif 'back' in self.action_name:
            dynamic_coords = self.back_coords
        else:
            dynamic_coords = self.wall_only_coords
        return dynamic_coords
    
    
    def _get_transformer(self, action_name:str=None):
        if 'move' in self.action_name:
            tx, ty = self.fenv_config['translation_mat_dict'][action_name]
            M = self._get_translation_matrix(tx, ty, self.fenv_config['seg_length'])
            return M
        
        elif 'flip' in self.action_name:
            ax, ay = self.fenv_config['flip_mat_dict'][action_name]
            M = self._get_flip_matrix(ax, ay)
            return M
            
        elif 'rotate' in self.action_name:
            angle = self.fenv_config['rotation_mat_dict'][action_name]
            M = self._get_rotation_matrix(angle)
            return M
        else:
            return []
        
    
    @staticmethod  
    def _get_translation_matrix(tx:int, ty:int, step_size:int):
        T = np.identity(3)
        T[:2, 2] = (tx*step_size, ty*step_size)
        return T
    
    
    @staticmethod
    def _get_flip_matrix(ax:int, ay:int):
        F = np.identity(3)
        F[0, 0] = ax
        F[1, 1] = ay
        return F
        
    
    @staticmethod
    def _get_rotation_matrix(angle:float):
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([ [c, -s, 0],
                       [s,  c, 0],
                       [0,  0, 1] ])
        return R
    
    
    def _apply_transformer(self, dynamic_coords:dict):
        originated_dynamic_coords = self._shift_to_origin(dynamic_coords, self.anchor_coord)
        originated_dynamic_coords = self._to_homogeneous(originated_dynamic_coords)
        transformed_originated_dynamic_coords = self.transformer_matrix @ originated_dynamic_coords
        transformed_originated_dynamic_coords = self._exit_homogeneous(transformed_originated_dynamic_coords)
        transformed_originated_dynamic_coords = np.asarray(transformed_originated_dynamic_coords, dtype=int)
        transformed_dynamic_coords = self._return_to_anchor(transformed_originated_dynamic_coords, self.anchor_coord)
        
        if self._is_valid(transformed_dynamic_coords):
            return transformed_dynamic_coords
        else: 
            return dynamic_coords
        
        
    @staticmethod
    def _shift_to_origin(coords, anchor_coord):
        if isinstance(coords, list):
            coords = np.array(coords)
        if isinstance(anchor_coord, list):
            anchor_coord = np.array(anchor_coord) 
        
        new_coords = coords - anchor_coord
        return new_coords
    
    
    @staticmethod
    def _return_to_anchor(coords, anchor_coord):
        if isinstance(coords, list):
            coords = np.array(coords)
        if isinstance(anchor_coord, list):
            anchor_coord = np.array(anchor_coord) 
        
        new_coords = coords + anchor_coord
        return new_coords
        
    
    @staticmethod 
    def _to_homogeneous(coords):
        coords = np.hstack((coords, np.ones((coords.shape[0], 1), dtype=coords.dtype)))
        return coords.T
    
    
    @staticmethod 
    def _exit_homogeneous(coords):
        coords /= coords[-1, :]
        return coords[:-1, :].T
    
    
    def _make_wall_coords(self, new_wall:list):
        """ Making sure if for some reason two wall's segments lie on 
        each other, we covert the transformed wall to the original wall """
        if np.allclose(new_wall[0], new_wall[1]) or \
           np.allclose(new_wall[0], new_wall[2]) or \
           np.allclose(new_wall[1], new_wall[2]):
               
              new_wall = [self.back_open_coord, self.anchor_coord, self.front_open_coord]
            
        self.transformed_wall_coords['back_open_coord'] = new_wall[0]
        self.transformed_wall_coords['anchor_coord'] = new_wall[1]
        self.transformed_wall_coords['front_open_coord'] = new_wall[2]
        self.transformed_wall_coords = WallGenerator(self.fenv_config).make_walls(self.transformed_wall_coords, wall_name=self.wall_name)
        
        return self.transformed_wall_coords
        
        
    @staticmethod
    def _grow_segment(coord, direction, step_size):
        if direction == 'north':
            return [coord[0], coord[1]+step_size]
        elif direction == 'south':
            return [coord[0], coord[1]-step_size]
        elif direction == 'east':
            return [coord[0]+step_size, coord[1]]
        elif direction == 'west':
            return [coord[0]-step_size, coord[1]]
        else:
            raise ValueError('Invalid direction')
        
            
    @staticmethod
    def _cut_segment(coord, direction, step_size):
        if direction == 'north':
            return [coord[0], coord[1]-step_size]
        elif direction == 'south':
            return [coord[0], coord[1]+step_size]
        elif direction == 'east':
            return [coord[0]-step_size, coord[1]]
        elif direction == 'west':
            return [coord[0]+step_size, coord[1]]
        else:
            raise ValueError('Invalid direction')
    
    
    @staticmethod
    def _cartesian2image_coord(x, y, max_y):
        return max_y-y, x
    
    
    def _is_valid(self, new_coords):
        if isinstance(new_coords, list):
            new_coords = np.array(new_coords)
        if ( (np.min(new_coords[:,0]) >= self.fenv_config['min_x']) and \
             (np.min(new_coords[:,1]) >= self.fenv_config['min_y']) and \
             (np.max(new_coords[:,0]) <= self.fenv_config['max_x']) and \
             (np.max(new_coords[:,1]) <= self.fenv_config['max_y']) ):
            return True
        else:
            return False
        
#%% This is only for testing and debugging
if __name__ == '__main__':
    from fenv_config import LaserWallConfig
    fenv_config = LaserWallConfig().get_config()
    action = 1
    
    wall_coords = {'anchor_coord': [12, 8],
      'back_open_coord': [12, 12],
      'front_open_coord': [8, 8],
      'front_segment': {'orientation': 'axial',
       'direction': 'west',
       'position': 'in'},
      'back_segment': {'orientation': 'axial',
       'direction': 'north',
       'position': 'in'}}       

    self = WallTransform('wall_1', wall_coords, action, fenv_config)
    new_wall = self.transform()