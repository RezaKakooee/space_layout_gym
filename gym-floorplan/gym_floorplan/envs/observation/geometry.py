# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 18:03:15 2021

@author: Reza Kakooee
"""

#%%
import numpy as np

#%%
class Segment:
    def __init__(self, start_coord:np.array=None, 
                       end_coord:np.array=None,
                       orientation:str='diagonal',
                       direction:str='north_east',
                       location:str='in',
                       name:str=None):
        
        self.start_coord = start_coord
        self.end_coord = end_coord
        self.orientation = orientation
        self.direction = direction
        self.location = location
        self.name = name
    
        self.slop = self._slop()
        self.angle = self._angle()
        self.linear_equation = self._linear_equation()


    def _slop(self):
        if (self.direction == 'east') or (self.direction == 'west') or (self.direction == 'horizental'):
            m = 0
        elif (self.direction == 'north') or (self.direction == 'south') or (self.direction == 'vertical'):
            m = np.inf
        else:
            m = (self.end_coord[1]-self.start_coord[1]) / \
               (self.end_coord[0]-self.start_coord[0] + np.finfo(float).eps)
        return m
    
    
    def _angle(self):
        return np.arctan(self.slop)
    
    
    def _linear_equation(self): # 1y -m*x -1 (y_0 - m*x_0) = 0!
        m = self.slop
        x0 = self.start_coord[0]
        y0 = self.start_coord[1]
        if (self.direction == 'east') or (self.direction == 'west') or (self.direction == 'horizental'):
            m = 0
            line_eq = [1, 0, -y0] # [1, -m, -(y0 -m*x0)] -> y=-[2], where [2] referes to index 2
        elif (self.direction == 'north') or (self.direction == 'south') or (self.direction == 'vertical'):
            m = np.inf
            line_eq = [0, 1, -x0] # -[(1/m) -1 -((1/m)*y0-x0)] => x=-[2]
        else:
            line_eq = [1, -m, -(y0 - m*x0)] # y-ax-b=0 -> a=-[1] b=-[2]
            
        return line_eq
    
    
    def get_segment_data(self):
        return self.__dict__


#%%
class Wall:
    def __init__(self, wall_coords):
        
        self.anchor_coord = wall_coords['anchor_coord']
        self.front_open_coord = wall_coords['front_open_coord']
        self.back_open_coord = wall_coords['back_open_coord']
        self.base_coords = wall_coords['base_coords']
        
        self.front_segment = self._create_segment(start_coord=self.anchor_coord,
                                                  end_coord=self.front_open_coord,
                                                  orientation=wall_coords['front_segment']['orientation'],
                                                  direction=wall_coords['front_segment']['direction'],
                                                  location=wall_coords['front_segment']['location'],
                                                  name='front')
        
        self.back_segment = self._create_segment(start_coord=self.anchor_coord,
                                                 end_coord=self.back_open_coord,
                                                 orientation=wall_coords['back_segment']['orientation'],
                                                 direction=wall_coords['back_segment']['direction'],
                                                 location=wall_coords['back_segment']['location'],
                                                 name='back')
    @staticmethod    
    def _create_segment(start_coord, end_coord, 
                        orientation, direction, location,
                        name):
        
        return Segment(start_coord=start_coord,
                       end_coord=end_coord,
                       orientation=orientation,
                       direction=direction,
                       location=location,
                       name=name).get_segment_data()
    
    def get_wall_data(self):
        return self.__dict__


#%%
class Outline:
    def __init__(self, fenv_config:dict=None):
        coords = {'left':  [  [ fenv_config['min_x'], fenv_config['min_y'] ], 
                              [ fenv_config['min_x'], fenv_config['max_y'] ]  ],
                  'down':  [  [ fenv_config['min_x'], fenv_config['min_y'] ], 
                              [ fenv_config['max_x'], fenv_config['min_y'] ]  ],
                  'right': [  [ fenv_config['max_x'], fenv_config['min_y'] ], 
                              [ fenv_config['max_x'], fenv_config['max_y'] ]  ],
                  'up':    [  [ fenv_config['min_x'], fenv_config['max_y'] ], 
                              [ fenv_config['max_x'], fenv_config['max_y'] ]  ]}
        ordered_directions = ['vertical', 'horizental', 'vertical', 'horizental']
        
        self.wall_outline_segments = {}
        for i, (key, val) in enumerate(coords.items()):   
            self.wall_outline_segments[f"{key}_segment"] = self._create_segment(start_coord=val[0],
                                                                           end_coord=val[1],
                                                                           orientation='axial',
                                                                           direction=ordered_directions[i],
                                                                           location=key,
                                                                           name=key)
        
    @staticmethod   
    def _create_segment(start_coord=None, 
                        end_coord=None, 
                        orientation=None, 
                        direction=None, 
                        location=None,
                        name=None):
        
          return  Segment(start_coord=start_coord,
                          end_coord=end_coord,
                          orientation=orientation,
                          direction=direction,
                          location=location,
                          name=name).get_segment_data()
     
    def get_outline_data(self):
         return self.__dict__


#%%   
class Plan:
    def __init__(self, outline_dict:dict=None, walls_coords:dict=None):
        wall_outline_segments = outline_dict['wall_outline_segments']
        
        if walls_coords is None:
            self.walls_dict = {}
            self.walls_segments_dict = wall_inline_segments = {}
        else:
            self.walls_dict = self._get_walls_dict(walls_coords=walls_coords)
            
            self.walls_segments_dict, wall_inline_segments = self._get_wall_inline_segments(walls_dict=self.walls_dict)
            
        self.segments_dict = self._get_segments_dict(wall_outline_segments=wall_outline_segments, 
                                                      wall_inline_segments=wall_inline_segments)
    
    
    @staticmethod
    def _get_walls_dict(walls_coords:dict=None):
        walls_dict = {}
        for name, wall_coords in walls_coords.items():
            # print(f"geometry:Plan: wall_coords: {wall_coords}")
            walls_dict[name] = Wall(wall_coords).get_wall_data()
        return walls_dict
    
    
    @staticmethod
    def _get_wall_inline_segments(walls_dict:dict=None):
        wall_inline_segments = {}
        for wall_name, wall_value in walls_dict.items():
          wall_inline_segments.update({f"{wall_name}_back_segment":wall_value['back_segment']})
          wall_inline_segments.update({f"{wall_name}_front_segment":wall_value['front_segment']}) 
          
        walls_segments_dict = {}
        for wall_name, wall_value in walls_dict.items():
                walls_segments_dict.update({wall_name: {'back_segment': wall_value['back_segment'],
                                             'front_segment': wall_value['front_segment']}})
        return walls_segments_dict, wall_inline_segments
            
    
    @staticmethod        
    def _get_segments_dict(wall_outline_segments:dict=None, wall_inline_segments:dict=None):
        segments_dict = {'outline': {key:val for key, val in wall_outline_segments.items()}}
        if wall_inline_segments is None:
            segments_dict.update({'inline': {}})
        else:
            segments_dict.update({'inline': {key:val for key, val in wall_inline_segments.items()}})
        return segments_dict


    def get_plan_data(self):
        return self.__dict__
