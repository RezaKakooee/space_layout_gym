# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 17:30:10 2021

@author: RK
"""
#%%
import copy
import numpy as np

from gym_floorplan.envs.observation.wall_generator import WallGenerator

#%%
class WallTransform:
    def __init__(self, wall_name:str=None, wall_coords:dict=None, action_i:int=None, plan_data_dict=None, fenv_config:dict=None):
        """
        Receives a wall, and and action, then transforms the wall based on the
        action. 

        Parameters
        ----------
        wall_name : str, mandatory
            The name of the input wall.
        wall_coords : dict, mandatory
            A dict includes input wall data.
        action : int, mandatory
            The action selected by the agent
        fenv_config : dict, mandatory
            A dict including env data.

        Returns
        -------
        A transformed wall if feasible. Transformation is not possible for all
        actions. Some actions are invalid, and some do not change the wall.

        """
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
        """
        The main method to transform the input wall

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        TYPE
            A transformed wall if possible.

        """
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
                raise ValueError('Wall_transform, there is something wrong here')
        
    def _get_dynamic_coords(self):
        """
        According to the action, determines which segment of the input wall 
        should be transfered: back, front or both.

        Returns
        -------
        dynamic_coords : TYPE
            A dict consists of the coords of the dynamic segments.

        """
        if 'front' in self.action_name:
            dynamic_coords = self.front_coords
        elif 'back' in self.action_name:
            dynamic_coords = self.back_coords
        else:
            dynamic_coords = self.wall_only_coords
        return dynamic_coords
    
    
    def _get_transformer(self, action_name:str=None):
        """
        Based on the input actions define the transformation matrix

        Parameters
        ----------
        action_name : str, mandatory
            The of the action selected by the agent

        Returns
        -------
        M: list
            The transformation correspods to the selected action.
            If action is not move, flip, or rotate, means that the action is 
            resizing the wall's segement. In this case there is not 
            transformation matrix. So, the methods return an empty list
        """
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
        """
        The action selected by the agent is 'move'. So, this method generates
        a translation matrix corresponds to the selected action.
        Parameters
        ----------
        tx : int
            The amount of movment along the x axis, can be -1, 0, +1
        ty : int
            The amount of movment along the y axis, can be -1, 0, +1
        step_size : float or int
            The step size of a single movement

        Returns
        -------
        T : list
            A 3*3 affine translation matrix

        """
        T = np.identity(3)
        T[:2, 2] = (tx*step_size, ty*step_size)
        return T
    
    
    @staticmethod
    def _get_flip_matrix(ax:int, ay:int):
        """
        The action selected by the agent is 'flip'. So, this method generates
        a translation matrix corresponds to the selected action.
        
        Parameters
        ----------
        ax : int
            Flipping coefficient w.r.t the y axis
        ay : int
            Flipping coefficient w.r.t the x axis

        Returns
        -------
        F : list
            A 3*3 fliping matrix

        """
        F = np.identity(3)
        F[0, 0] = ax
        F[1, 1] = ay
        return F
        
    
    @staticmethod
    def _get_rotation_matrix(angle:float):
        """
        The action selected by the agent is 'rotate'. So, this method generates
        a rotation matrix corresponds to the selected action.

        Parameters
        ----------
        angle : float
            THe rotation angle in radian

        Returns
        -------
        R : list
            A 3*3 rotation matrix

        """
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([ [c, -s, 0],
                       [s,  c, 0],
                       [0,  0, 1] ])
        return R
    
    
    def _apply_transformer(self, dynamic_coords:dict):
        """
        Transforms the wall according to the selected action, or better to say,
        according to the extracted affine transformation matrix. 

        Parameters
        ----------
        dynamic_coords : dict
            A dict consists of the coords needed to be transfered. 

        Returns
        -------
        transformed_dynamic_coords: dict
            This methds returns the transformed coords. Or in the case, the 
            transformed coords is invalid, it returns the input coords without 
            any transformation. 

        """
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
        """
        For applying the transformation, first we shift all coords to the
        main origin. And then gets them back to the current coords. Indeed,
        we shift the anchor_coord to (0, 0), and shift the other coords 
        according to it. 

        Parameters
        ----------
        coords : list or nparray
            An array of the dynamic coords that we want to shift.
        anchor_coord : list or nparray
            The current origin of the segment/wall, which is not (0, 0)

        Returns
        -------
        new_coords : nparray
            The new coords shiftted to the main origin (0, 0)

        """
        if isinstance(coords, list):
            coords = np.array(coords)
        if isinstance(anchor_coord, list):
            anchor_coord = np.array(anchor_coord) 
        
        new_coords = coords - anchor_coord
        return new_coords
    
    
    @staticmethod
    def _return_to_anchor(coords, anchor_coord):
        """
        After applying the transformation, we shift back the dynamic coords
        to the initil coord.

        Parameters
        ----------
        coords : list or nparray
            An array of the dynamic coords that we want to shift.
        anchor_coord : list or nparray
            The current origin of the segment/wall, which is not (0, 0).

        Returns
        -------
        new_coords : nparray
            The new coords shiftted to the current coord

        """
        if isinstance(coords, list):
            coords = np.array(coords)
        if isinstance(anchor_coord, list):
            anchor_coord = np.array(anchor_coord) 
        
        new_coords = coords + anchor_coord
        return new_coords
        
    
    @staticmethod 
    def _to_homogeneous(coords):
        """
        For transformation we use homogeneous coorinates, as it is more 
        convinient to work with. 

        Parameters
        ----------
        coords : list or nparray
            An array of the coords that we want convert them to homogeneous 
            coorinates.

        Returns
        -------
        nparray
            The homogeneous coorinates of the input normal coords.

        """
        coords = np.hstack((coords, np.ones((coords.shape[0], 1), dtype=coords.dtype)))
        return coords.T
    
    
    @staticmethod 
    def _exit_homogeneous(coords):
        """
        Returns the coords from homogeneous coorinates to normal coorinates

        Parameters
        ----------
        coords : list or nparray
            An array of the coords that we want convert them to normal 
            coorinates from homogeneous coorinates.

        Returns
        -------
        nparray
            The normal coorinates of the input homogeneous coords.

        """
        coords /= coords[-1, :]
        return coords[:-1, :].T
    
    
    def _make_wall_coords(self, new_wall:list):
        """
        Receives the new_wall and sends it to 'make_walls' in order to get
        wall's properties

        Parameters
        ----------
        new_wall : list
            A list consists of 3 coords: [back_open_coord, anchor_coord, 
                                          front_open_coord] of the new wall

        Returns
        -------
        dict
            The new wall coords consists of back, front, and anchor coords, 
            and also, the properties of the two back and front segments of the
            new wall. 

        """
        
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
        """
        Grow a segment according to its current direction, and step_size

        Parameters
        ----------
        coord : list or nparray
            The coords of the segment we want to grwo
        direction : str
            The direction of the segment
        step_size : int
            The step size for grwoing the segment

        Raises
        ------
        ValueError
            'Invalid direction'. 
        
        Returns
        -------
        list
            a list consists of the coords of the grown segment.

        """
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
        """
        Cut a segment according to its current direction, and step_size

        Parameters
        ----------
        coord : list or nparray
            DESCRIPTION.
        direction : str
            The direction of the segment
        step_size : int
            The step size for grwoing the segment

        Raises
        ------
        ValueError
            'Invalid direction'.

        Returns
        -------
        list
            a list consists of the coords of the shortened segment.

        """
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
        """
        Check if the coords, particularly the new coords are valid or not.
        If the coords are out of the plan boundries, we consider them as 
        invalid coords. 

        Parameters
        ----------
        new_coords : nparray
            an array consists of new coords

        Returns
        -------
        bool
            returns True if the new_coords in valid, otherwise it returns False

        """
        if isinstance(new_coords, list):
            new_coords = np.array(new_coords)
        if ( (np.min(new_coords[:,0]) >= self.fenv_config['min_x']) and \
             (np.min(new_coords[:,1]) >= self.fenv_config['min_y']) and \
             (np.max(new_coords[:,0]) <= self.fenv_config['max_x']) and \
             (np.max(new_coords[:,1]) <= self.fenv_config['max_y']) ):
            return True
        else:
            return False
        
        
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