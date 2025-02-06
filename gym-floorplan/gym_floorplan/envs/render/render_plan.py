# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 14:27:30 2021

@author: Reza Kakooee
"""


#%%
import os
import inspect
import copy
import datetime
import numpy as np
from PIL import Image
from webcolors import name_to_rgb
from collections import defaultdict

import bezier
import matplotlib
from matplotlib import cm
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
# from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import matplotlib.patheffects as path_effects
from matplotlib.backends.backend_agg import FigureCanvasAgg

from scipy import ndimage as ndi
import matplotlib as mpl
from skimage.exposure import histogram
from skimage.filters import sobel
from skimage.segmentation import watershed
from skimage.measure import find_contours, approximate_polygon
from skimage.color import label2rgb

import shapely
from shapely.geometry import Polygon, LineString, LinearRing
from shapely.ops import unary_union, polygonize

from gym_floorplan.envs.render.colors_lib import get_color_dict



#%%
class RenderPlan:
    def __init__(self, fenv_config={}):
        self.fenv_config = copy.deepcopy(fenv_config)
        # self.fenv_config['phase'] = 'test'
        self.fenv_config['trial'] = 0 if 'trial' not in self.fenv_config.keys() else self.fenv_config['trial'] # type: ignore

        
        self._initialize_variables()

            
      
    def render(self, plan_data_dict, episode, ep_time_step):
        self.plan_data_dict = plan_data_dict
        self.episode = episode
        self.ep_time_step = ep_time_step + 1
        
        self.fig, self.ax = plt.subplots(figsize=self.figsize, dpi=self.dpi, frameon=False)
        self.fig.tight_layout()
        self.fig.patch.set_visible(False) # type: ignore
        
        self._draw_inlines()
    
        if self.fenv_config['show_room_dots_flag']:
            self._draw_dots()
    
        if ( self.fenv_config['show_graph_on_plan_flag'] and 
              len(plan_data_dict['wall_types']) == plan_data_dict['n_walls'] ):
            self.__get_all_gravity_coords()
            if not self.fenv_config['only_draw_room_gravity_points_flag']: 
                self.__draw_room_edges()
            if not self.fenv_config['only_draw_room_gravity_points_flag']: 
                self.__draw_entrance_edges()
            self.__draw_room_gravity_points()
        else:
            self.fenv_config['phase'] = 'debug'
            self.__get_all_gravity_coords()
        
    
        # ax.set_axis_off()
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['bottom'].set_visible(False)
        self.ax.spines['left'].set_visible(False)
        self.ax.get_xaxis().set_ticks([])
        self.ax.get_yaxis().set_ticks([])
        # fig.savefig("test.png")
        self.ax.grid(True)
    
        
        canvas = FigureCanvasAgg(self.fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        X = np.asarray(buf)
        im = Image.fromarray(X)
        
        # obs_mat = self._edit_obs_mat(X[:,:,0])
        
        if self.fenv_config['show_render_flag']:
            ## legend = self.ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.015), fancybox=True, shadow=False, ncol=4)
            # self.ax.legend(loc='lower left', bbox_to_anchor=(0.2, -0.15), fancybox=True, shadow=False, ncol=4)
            try:
                self.ax.set_title(f"plan_id: {self.plan_data_dict['plan_id']}")
            except:
                pass
            plt.show(block=False)
            
        if self.fenv_config['save_render_flag']: 
            plan_fig_path = self._save_the_plan(self.fig)
            return plan_fig_path
        

    def _initialize_variables(self):
        self.wall_colors, self.room_colors = get_color_dict(self.fenv_config)
        
        self.marker_dict = {'north': '^', 'south': 'v', 'east': '>', 'west': '<'}
    
        if self.fenv_config['so_thick_flag']:
            if self.fenv_config['resolution'] == 'Low':
                self.wall_linewidth = 15
                self.basewall_linewidth = 15
                self.outline_linewidth = 20
                self.masked_walls_linewidth = 15
                self.linewidth_of_graph = 2
                self.room_centroid_marker_size = 15
                self.patch_lw = 10
                self.dots_markersize = 15
                self.linestyle_of_graph = 'dashed'
                self.red_green_blue_color_map = {'red': 'tomato', 'green': 'limegreen', 'blue': 'dodgerblue'}
                
                
            elif self.fenv_config['resolution'] == 'High':
                self.wall_linewidth = 12
                self.basewall_linewidth = 4
                self.outline_linewidth = 13
                self.masked_walls_linewidth = 13
                self.linewidth_of_graph = 3
                self.room_centroid_marker_size = 15
                self.patch_lw = 10
                self.dots_markersize = 15
                self.linestyle_of_graph = 'dashed'
                self.red_green_blue_color_map = {'red': 'tomato', 'green': 'limegreen', 'blue': 'dodgerblue'}
                
        else:
            self.wall_linewidth = 5
            self.basewall_linewidth = 5
            self.outline_linewidth = 5
            self.masked_walls_linewidth = 6
            self.linewidth_of_graph = 2
            self.room_centroid_marker_size = 15
            self.patch_lw = 10
            self.dots_markersize = 15
            self.linestyle_of_graph = 'dashed'
            self.red_green_blue_color_map = {'red': 'tomato', 'green': 'limegreen', 'blue': 'dodgerblue'}
        
        
        if self.fenv_config['resolution'] == 'Low':
            self.outline_control_points = [
                [[-1, -1], [-1, 21]],
                [[-1, 21], [-1, -1]],
                [[21, 21], [-1, 21]],
                [[-1, 21], [21, 21]],
                ]
        else:
            self.outline_control_points = [
                [[-1, -1], [-1, 41]],
                [[-1, 41], [-1, -1]],
                [[41, 41], [-1, 41]],
                [[-1, 41], [41, 41]],
                ]
            
            
        if self.fenv_config['show_render_flag']:
            self.dpi = 300
            self.figsize = (8,8)
        else:
            self.dpi = 300
            self.figsize = (10,10)
            
            
            
    def _draw_inlines(self):
        for key, seg_data in self.plan_data_dict['wall_inline_segments'].items():
            x = [seg_data['start_coord'][0], seg_data['end_coord'][0]]
            y = [seg_data['start_coord'][1], seg_data['end_coord'][1]]
            
            if 'front' in key and int(key.split('_')[1]) > self.fenv_config['maximum_num_masked_rooms']:
                self.ax.plot(x,y, 
                    color=self.wall_colors[key],#wall_colors[key[:6]], 
                    # label="_".join(map(str, key.split('1s_')[:2])), 
                    linewidth=self.basewall_linewidth,
                    ) 
            else:
                self.ax.plot(x,y, 
                color=self.wall_colors[key],#wall_colors[key[:6]], 
                linewidth=self.basewall_linewidth,
                ) 
            
        show_only_base_wall = False
        for key, seg_data in self.plan_data_dict['wall_inline_segments'].items():
            if show_only_base_wall:
                wall_id = int(key.split('_')[1])
                if wall_id > self.fenv_config['entrance_cell_id']:
                    continue
            x = [seg_data['start_coord'][0], seg_data['reflection_coord'][0]]
            y = [seg_data['start_coord'][1], seg_data['reflection_coord'][1]]
            
            self.ax.plot(x,y, 
                color=self.wall_colors[key],#wall_colors[key[:6]], 
                # label=key, 
                linewidth=self.wall_linewidth,
                ) 
            
         
        
    def _draw_wall_segments(self):
        delta_l = 1 if self.fenv_config['so_thick_flag'] else 0 # delta length
    
        if self.fenv_config['so_thick_flag']:
            if self.fenv_config['mask_flag']:
                for j, (wname, seg_data) in enumerate(self.plan_data_dict['walls_segments'].items()):
                    room_i = int(wname.split('_')[1])
                    if room_i in self.fenv_config['fake_room_id_range']:
                        x1, y1, x2, y2 = self.__get_light_coords(seg_data, delta_l, j)     
                        self.ax.plot(x1, y1, 
                            color='white',#wall_colors[wname[:6]], 
                            # label=wname, 
                            linewidth=self.masked_walls_linewidth,
                            ) 
    
                        self.ax.plot(x2, y2, 
                            color='white',#wall_colors[wname[:6]], 
                            # label=wname, 
                            linewidth=self.masked_walls_linewidth,
                            ) 
            
        else:
            if self.fenv_config['mask_flag']:
                for j, (wname, seg_data) in enumerate(self.plan_data_dict['walls_segments'].items()):
                    room_i = int(wname.split('_')[1])
                    if room_i in self.fenv_config['fake_room_id_range']:
                        x1, y1, x2, y2 = self.__get_light_coords(seg_data, delta_l, j)                        
                        self.ax.plot(x1,y1, 
                            color='white',#wall_colors[wname[:6]], 
                            # label=wname, 
                            linewidth=self.masked_walls_linewidth,
                            ) 
                        
                        self.ax.plot(x2,y2, 
                            color='white',#wall_colors[wname[:6]], 
                            # label=wname, 
                            linewidth=self.masked_walls_linewidth,
                            ) 
        


    def _draw_dots(self):
        for room_name, room_data in self.plan_data_dict['rooms_dict'].items():
            room_i = int(room_name.split('_')[1])
            if room_i in self.fenv_config['real_room_id_range']:
                room_positions = np.argwhere(self.plan_data_dict['obs_moving_labels'] == room_i).tolist()
                room_coords = [
                        self.__image_coords2cartesian(p[0], p[1], self.fenv_config['n_rows']) for p in room_positions
                        ]
                for rc in room_coords:
                    self.ax.plot(rc[0]+1, rc[1]+1, 
                            color=self.room_colors[room_name], 
                            marker='.',
                            markersize=self.dots_markersize,
                            ) 
        
    
    
    def __get_light_coords(self, seg_data, delta_l, i):
        if self.plan_data_dict['masked_corners'][i] == 'corner_00':
            corner_xy = [self.fenv_config['min_x'], self.fenv_config['min_y']]
            for k, sd in seg_data.items():
                if sd['direction'] == 'south':
                    x1 = [corner_xy[0] , sd['reflection_coord'][0]-delta_l]
                    y1 = [corner_xy[1] , sd['reflection_coord'][1]]
                elif sd['direction'] == 'west':
                    x2 = [corner_xy[0] , sd['reflection_coord'][0]]
                    y2 = [corner_xy[1] , sd['reflection_coord'][1]-delta_l]
            
        elif self.plan_data_dict['masked_corners'][i] == 'corner_01':
            corner_xy = [self.fenv_config['min_x'], self.fenv_config['max_y']]
            for k, sd in seg_data.items():
                if sd['direction'] == 'north':
                    x1 = [corner_xy[0] , sd['reflection_coord'][0]-delta_l]
                    y1 = [corner_xy[1] , sd['reflection_coord'][1]]
                elif sd['direction'] == 'west':
                    x2 = [corner_xy[0] , sd['reflection_coord'][0]]
                    y2 = [corner_xy[1] , sd['reflection_coord'][1]+delta_l]
            
        elif self.plan_data_dict['masked_corners'][i] == 'corner_10':
            corner_xy = [self.fenv_config['max_x'], self.fenv_config['min_y']]
            for k, sd in seg_data.items():
                if sd['direction'] == 'south':
                    x1 = [corner_xy[0] , sd['reflection_coord'][0]+delta_l]
                    y1 = [corner_xy[1] , sd['reflection_coord'][1]]
                elif sd['direction'] == 'east':
                    x2 = [corner_xy[0] , sd['reflection_coord'][0]]
                    y2 = [corner_xy[1] , sd['reflection_coord'][1]-delta_l]
            
        elif self.plan_data_dict['masked_corners'][i] == 'corner_11':
            corner_xy = [self.fenv_config['max_x'], self.fenv_config['max_y']]
            for k, sd in seg_data.items():
                if sd['direction'] == 'north':
                    x1 = [corner_xy[0] , sd['reflection_coord'][0]+delta_l]
                    y1 = [corner_xy[1] , sd['reflection_coord'][1]]
                elif sd['direction'] == 'east':
                    x2 = [corner_xy[0] , sd['reflection_coord'][0]]
                    y2 = [corner_xy[1] , sd['reflection_coord'][1]+delta_l]
                    
        else:
            raise ValueError(f"Wrong corner to display: {self.plan_data_dict['masked_corners'][i]}")
            
        return x1, y1, x2, y2
    
    
    
    def __get_all_gravity_coords(self):
        if self.fenv_config['wall_identity_mode']:
            self.rooms_gravity_coord_dict = self.find_room_centers_median_for_dyp_identity_less_wall(self.plan_data_dict['obs_moving_labels'])
        else:
            self.rooms_gravity_coord_dict = self._get_rooms_gravity_coord(self.fenv_config, self.plan_data_dict)
        self.rooms_gravity_coord_dict.update(self.fenv_config['facade_coords'])
        
        x = (self.plan_data_dict['entrance_coords'][0][0] + self.plan_data_dict['entrance_coords'][1][0]) / 2.0
        y = (self.plan_data_dict['entrance_coords'][0][1] + self.plan_data_dict['entrance_coords'][1][1]) / 2.0
        self.rooms_gravity_coord_dict.update({'room_d': [x, y]})# self.plan_data_dict['entrance_coords'][0]})
        


    def __get_gravity(self, room_coords):
        room_coords = np.array(room_coords)
        median = np.median(room_coords,axis=0).tolist() #  np.array(np.median(room_coords,axis=0), dtype=int).tolist()
        dists = [np.linalg.norm(median-rc) for rc in room_coords]
        gravity_coord = room_coords[np.argmin(dists)]
        gravity_coord = [gravity_coord[0] + np.random.rand()/2 , gravity_coord[1] + np.random.rand()/2]
        return list(gravity_coord)
    
    
    
    def _get_rooms_gravity_coord(self, fenv_config, plan_data_dict):
        rooms_dict = self.plan_data_dict['rooms_dict']
        
        rooms_gravity_coord_dict = {}
        for room_name, this_room in rooms_dict.items():
            room_i = int(room_name.split('_')[1]) 
            if ( (room_i in self.fenv_config['fake_room_id_range']) or 
                 (room_i in self.fenv_config['real_room_id_range']) ):
                room_shape = this_room['room_shape']
                room_positions = this_room['room_positions']
                room_coords = [self.__gravity_image_coords2cartesians(p[0], p[1], self.fenv_config['max_y']) for p in room_positions]
                
                if room_shape == 'rectangular':
                    gravity_coord = self.__get_gravity(room_coords)
                    rooms_gravity_coord_dict[room_name] = gravity_coord
                    
                else:
                    sub_rects = this_room['sub_rects']
                    max_area_ind = np.argmax(sub_rects['areas_achieved'])+1
                    max_sub_rects_positions = sub_rects['all_rects_positions'][max_area_ind]
                    max_sub_rects_coords = [self.__gravity_image_coords2cartesians(p[0], p[1], self.fenv_config['max_y']) for p in max_sub_rects_positions]
                    gravity_coord = self.__get_gravity(max_sub_rects_coords)
                    rooms_gravity_coord_dict[room_name] = gravity_coord
                    
        return rooms_gravity_coord_dict
    
    
    
    def find_room_centers_median_for_dyp_identity_less_wall(self, matrix):
        arr = np.array(matrix)
        room_numbers = np.unique(arr[arr > 0])
        centers = {}
        
        for room in room_numbers:
            room_mask = arr == room
            room_indices = np.where(room_mask)
            
            median_y = np.median(room_indices[0])
            median_x = np.median(room_indices[1])
            
            while not room_mask[int(median_y), int(median_x)]:
                in_room_y, in_room_x = room_indices[0][0], room_indices[1][0]
                median_y += np.sign(in_room_y - median_y)
                median_x += np.sign(in_room_x - median_x)
            
            centers[room] = (int(median_y), int(median_x))
        
        processed_centers = {}
        for room_i, (r, c) in centers.items():
            if room_i != self.fenv_config['entrance_cell_id']:
                x, y = self.__gravity_image_coords2cartesians(r, c, self.fenv_config['max_y'])
                x_jittered, y_jittered = self.add_jitter(x, y)
                processed_centers[f'room_{room_i}'] = [x_jittered, y_jittered]
        return processed_centers
    


    @staticmethod
    def add_jitter(x, y, amount=0.5):
        return x + np.random.rand()/4, y + np.random.rand()/4
    


    def __gravity_image_coords2cartesians(self, r, c, n_rows):
            return c, n_rows-r
        
        
        
    def __image_coords2cartesian(self, r, c, n_rows):
        return c-1, n_rows-2-r
        
    
    
    def __cartesian2image_coord(self, x, y, max_y):
            return max_y-y, x
    
    
    
    def _conn_sig(self, coord_1, coord_2, beta=1):
        coord_1_ = np.linspace(-5, 5)
        coord_1_adj = np.linspace(*coord_1)
        sig = 1/(1 + np.exp(-coord_1_*beta))
        sig = (sig - sig.min()) / sig.max()
        sig_adj = sig * (coord_2[1] - coord_2[0]) + coord_2[0]
        return coord_1_adj, sig_adj
        
    
    
    def _hanging_line(self, coord_1, coord_2):
        den = np.cosh(coord_2[0]) - np.cosh(coord_1[0])
        
        # if den == 0:
        #     coord_2[0] += np.random.uniform(-1, 1)/2
        #     coord_1[0] += np.random.uniform(-1, 1)/2
        #     den = np.cosh(coord_2[0]) - np.cosh(coord_1[0])
        #     a = (coord_2[1] - coord_1[1])/( den ) # + np.finfo(float).eps)
        #     a = np.clip(a, -1, 1)
        #     b = coord_1[1] - a*np.cosh(coord_1[0])
        #     b = np.clip(b, 1, 15)
        
        a = (coord_2[1] - coord_1[1]) / den
        b = coord_1[1] - a*np.cosh(coord_1[0])
        # print(b)
        xx = np.linspace(coord_1[0], coord_2[0], 20)
        yy = a*np.cosh(xx) + b
        return xx, yy
        
    
    
    def _bezier(self, coord_1, coord_2):
        mx = np.mean([coord_1[0], coord_2[0]])
        my = np.max([coord_1[1], coord_2[1]]) + 1
        
        xx = [coord_1[0], mx, coord_2[0]] 
        yy = [coord_1[1], my, coord_2[1]]
        nodes = np.asfortranarray([xx, yy])
        return nodes


    
    def __draw_room_gravity_points(self):
        if self.fenv_config['adaptive_window']:
            blind_rooms_str = [str(br) for br in self.plan_data_dict['edge_color_data_dict_facade']['blind_rooms']]
        for room_name, room_gravity in self.rooms_gravity_coord_dict.items():
            room_id = room_name.split('_')[1]
            # if self.fenv_config['only_draw_room_gravity_points_flag'] and room_id in ['n', 'w', 'e', 's', '2', '3', '4', '5']:
            #     continue
            if room_id in self.plan_data_dict['facades_blocked']:
                marker = 'x'
                mew = 5
                ms = 15
                color = 'darkgray'
            elif self.fenv_config['adaptive_window'] and room_id in blind_rooms_str:
                marker = '+'
                mew = 5
                ms = 15
                color = 'darkgray'
            else:
                marker = '.'
                mew = 1
                ms = self.room_centroid_marker_size
                color = self.room_colors[room_name] 
                
            if self.fenv_config['only_draw_room_gravity_points_flag'] and room_id in ['n', 'w', 'e', 's', '2', '3', '4', '5']:
                color = 'white'
                
            if self.fenv_config['show_edges_for_fake_rooms_flag']:
                self.ax.plot([room_gravity[0]], [room_gravity[1]], 
                    color=color, 
                    label=room_name, 
                    # linewidth=masked_walls_linewidth,
                    marker=marker,
                    markersize=ms,
                    mew=mew,
                    ) 
            else:
                if room_id not in self.fenv_config['fake_room_id_range']:
                    self.ax.plot([room_gravity[0]], [room_gravity[1]], 
                        color=color, 
                        label=room_name, 
                        # linewidth=masked_walls_linewidth,
                        marker=marker,
                        markersize=ms,
                        mew=mew,
                        ) 
    
    
    
    def __draw_room_edges(self):
        for color, edges in self.plan_data_dict['edge_color_data_dict_room'].items():
            for ed in edges:
                if self.fenv_config['entrance_cell_id'] not in ed:
                    if color != 'blue':
                        from_room = f"room_{ed[0]}"
                        to_room = f"room_{ed[1]}"
                        # from_room = 'room_d' if from_room == 'room_10' else from_room
                        # to_room = 'room_d' if to_room == 'room_10' else to_room
                        coord_1 = self.rooms_gravity_coord_dict[from_room]
                        coord_2 = self.rooms_gravity_coord_dict[to_room]
                
                        if self.fenv_config['graph_line_style'] == 'hanging':
                            # coord_1_adj, sig_adj = self._conn_sig(coord_1, coord_2, beta=1)
                            xx, yy = self._hanging_line(coord_1, coord_2)
                            self.ax.plot(xx, yy, color=self.red_green_blue_color_map[color], linewidth=self.linewidth_of_graph, linestyle=self.linestyle_of_graph)
                        
                        elif self.fenv_config['graph_line_style'] == 'bezier':
                            nodes = self._bezier(coord_1, coord_2)
                            curve = bezier.Curve(nodes, degree=2)
                            nodes_array = curve.evaluate_multi(np.linspace(0.0, 1.0, 100))
                            self.ax.plot(nodes_array[0], nodes_array[1], color=self.red_green_blue_color_map[color], linewidth=self.linewidth_of_graph, linestyle=self.linestyle_of_graph)
                            # self.ax = curve.plot(100, color=self.red_green_blue_color_map[color], alpha=None, ax=self.ax)
                            
                        else:
                            xx = [coord_1[0], coord_2[0]]
                            yy = [coord_1[1], coord_2[1]]
                            self.ax.plot(xx, yy, 
                                    linewidth=self.linewidth_of_graph, color=self.red_green_blue_color_map[color],
                              path_effects=[path_effects.SimpleLineShadow(shadow_color='white', alpha=0.3, rho=0.01),
                                            path_effects.Normal()]
                              )
            
        
    
    def __draw_facade_edges(self):
        edge_color_data_dict_facade = {k:self.plan_data_dict['edge_color_data_dict_facade'][k] for k in ['green', 'blue', 'red']}
        for color, edges in edge_color_data_dict_facade.items():
            for ed in edges:
                if color != 'blue':
                    from_room = f"room_{ed[0]}"
                    to_room = f"room_{ed[1]}"
                    from_room = 'room_d' if from_room == 'room_10' else from_room
                    to_room = 'room_d' if to_room == 'room_10' else to_room
                    # print(f"__draw_facade_edges from_room: {from_room}")
                    # print(f"__draw_facade_edges to_room: {to_room}")
                    
                    try:
                        from_room_id = from_room.split('_')[1]
                        if from_room_id not in self.plan_data_dict['facades_blocked']:
                        
                            coord_1 = self.rooms_gravity_coord_dict[from_room]
                            coord_2 = self.rooms_gravity_coord_dict[to_room]
                    
                            if self.fenv_config['graph_line_style'] == 'hanging':
                                # coord_1_adj, sig_adj = self._conn_sig(coord_1, coord_2, beta=1)
                                xx, yy = self._hanging_line(coord_1, coord_2)
                                self.ax.plot(xx, yy, color=self.red_green_blue_color_map[color], linewidth=self.linewidth_of_graph, linestyle=self.linestyle_of_graph)
                            
                            elif self.fenv_config['graph_line_style'] == 'bezier':
                                nodes = self._bezier(coord_1, coord_2)
                                curve = bezier.Curve(nodes, degree=2)
                                nodes_array = curve.evaluate_multi(np.linspace(0.0, 1.0, 100))
                                self.ax.plot(nodes_array[0], nodes_array[1], color=self.red_green_blue_color_map[color], linewidth=self.linewidth_of_graph, linestyle=self.linestyle_of_graph)
                                # self.ax = curve.plot(100, color=self.red_green_blue_color_map[color], alpha=None, ax=self.ax)
                                
                            else:
                                xx = [coord_1[0], coord_2[0]]
                                yy = [coord_1[1], coord_2[1]]
                                self.ax.plot(xx, yy, 
                                        linewidth=self.linewidth_of_graph, color=self.red_green_blue_color_map[color],
                                  path_effects=[path_effects.SimpleLineShadow(shadow_color='white', alpha=0.3, rho=0.01),
                                                path_effects.Normal()]
                                  )
                    except:
                        time = datetime.now().strftime("%Y%m%d_%H%M%S") # type: ignore
                        np.save(f"{self.fenv_config['root_dir']}/storage_nobackup/plan_data_dict_storage/plan_data_dict__{os.path.basename(__file__)}_{self.__class__.__name__}_{inspect.currentframe().f_code.co_name}_{time}.npy", self.plan_data_dict)
                        raise ValueError("In render_plan.py edge seems to have node 0. plan_id is: {self.plan_data_dict['plan_id']}")
                        
                        
                        
    def __draw_entrance_edges(self):
        for color, edges in self.plan_data_dict['edge_color_data_dict_entrance'].items():
            for ed in edges:
                if color != 'blue':
                    from_room = f"room_{ed[0]}"
                    to_room = f"room_{ed[1]}"
                    # if to_room == 'room_17':
                    #     continue
                    # print(f"__draw_entrance_edges from_room: {from_room}")
                    # print(f"__draw_entrance_edges to_room: {to_room}")
                    coord_1 = self.rooms_gravity_coord_dict[from_room]
                    coord_2 = self.rooms_gravity_coord_dict[to_room]
                    # print(f"__draw_entrance_edges coord_from: {coord_1}")
                    # print(f"__draw_entrance_edges coord_to: {coord_2}")
            
                    if self.fenv_config['graph_line_style'] == 'hanging':
                        # coord_1_adj, sig_adj = self._conn_sig(coord_1, coord_2, beta=4)
                        xx, yy = self._hanging_line(coord_1, coord_2)
                        self.ax.plot(xx, yy, color=self.red_green_blue_color_map[color], linewidth=self.linewidth_of_graph, linestyle=self.linestyle_of_graph)
                        # self.ax.plot(xx, yy, linewidth=4)
                    
                    elif self.fenv_config['graph_line_style'] == 'bezier':
                        nodes = self._bezier(coord_1, coord_2)
                        curve = bezier.Curve(nodes, degree=2)
                        nodes_array = curve.evaluate_multi(np.linspace(0.0, 1.0, 100))
                        self.ax.plot(nodes_array[0], nodes_array[1], color=self.red_green_blue_color_map[color], linewidth=self.linewidth_of_graph, linestyle=self.linestyle_of_graph)
                        # self.ax = curve.plot(100, color=self.red_green_blue_color_map[color], alpha=None, ax=self.ax)
                        
                    else:
                        xx = [coord_1[0], coord_2[0]]
                        yy = [coord_1[1], coord_2[1]]
                        self.ax.plot(xx, yy, 
                                linewidth=self.linewidth_of_graph, color=self.red_green_blue_color_map[color],
                          path_effects=[path_effects.SimpleLineShadow(shadow_color='white', alpha=0.3, rho=0.01),
                                        path_effects.Normal()]
                          )
                        
                        
                    
    def __crop_center(self, img, cropx, cropy):
        y,x = img.shape
        startx = x//2 - (cropx//2)
        starty = y//2 - (cropy//2)    
        return img[starty:starty+cropy,startx:startx+cropx]
    
    
    
    def _edit_obs_mat(self, obs_mat):
        obs_mat = self.__crop_center(obs_mat, 800, 800)
        obs_mat[0:10, :] = 20
        obs_mat[-10:, :] = 20
        obs_mat[:, 0:10] = 20
        obs_mat[:, -10:] = 20
        return obs_mat
    
    
    
    def _save_the_plan(self, fig, name='layout'):
        results_dir = self.fenv_config['results_dir']
        self.trial = 0 if 'trial' not in self.fenv_config.keys() else self.fenv_config['trial']
        # if 'plan_id' in self.plan_data_dict.keys():
        #     plan_id = self.plan_data_dict['plan_id']
        # else:
        plan_id = datetime.datetime.now().strftime('%Y_%m_%d_%H%M_%S_%f')[:22]  # Truncate to get milliseconds
        
        plan_fig_path = os.path.join(results_dir, f"pid_{plan_id}__{name}__tr_{self.trial}__ep_{self.episode:02}__ts_{self.ep_time_step:02}.png")
        # fig.savefig(plan_fig_path, bbox_extra_artists=(lgd,), bbox_inches='tight')
        fig.savefig(plan_fig_path, bbox_inches='tight') #, transparent=True, pad_inches=0, format='png')
        return plan_fig_path
        
        

    def portray(self, plan_data_dict, episode, ep_time_step):
        self.plan_data_dict = plan_data_dict
        self.episode = episode
        self.ep_time_step = ep_time_step + 1
        
        
        self.wall_colors, self.room_colors = get_color_dict(self.fenv_config)
        
        
        arr = copy.deepcopy(plan_data_dict['obs_rooms_cmap'])
        arr[arr<=5] = 0
        
        # Get unique rooms
        rooms = np.unique(arr)[1:]
        self.room_colors.update({
            'room_10': 'brown'
            })
        
        colors = [self.room_colors[f"room_{r}"] for r in rooms]
        
        arr = np.kron(arr, np.ones((32, 32), dtype=arr.dtype))
        

        # Generate color image  
        colored = label2rgb(arr, colors=colors, bg_label=0, bg_color=(1,1,1), alpha=0.5)
        # im = Image.fromarray(np.uint8(colored*255)).convert('RGB')
        # im.show()
        

        # Find contours for each room
        arr = np.transpose(arr)
        contours = []
        for room in rooms:
          contours.append(find_contours(arr==room, 0.99999)[0])
        
        # convert countours to polygons
        polygons = []
        for c in contours:
          # Close contour into ring
          ring = LinearRing(c) 
          # Simplify coordinates
          simplified = shapely.simplify(ring, tolerance=2)
          # Create polygon 
          poly = Polygon(simplified)
          polygons.append(poly)
        
        
        # Plot
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        ax.imshow(colored)
        for poly in polygons:
          x, y = poly.exterior.xy
          ax.plot(x, y, color='black', linewidth=2)

        ax.set_axis_off()
        plt.show()
        
        if self.fenv_config['save_render_flag']: self._save_the_plan(fig, name='portray')
        
        
        
        
#%%
class DisplayPlan:
    def __init__(self, fenv_config):
        self.fenv_config = fenv_config
        self.figsize = (8,8)
        
        self.trial = 0 if 'trial' not in self.fenv_config.keys() else self.fenv_config['trial']
        
        self.blocked_facade_coord = {
            'n': [self.fenv_config['max_x']//2, self.fenv_config['max_y']+0.25],
            's': [self.fenv_config['max_x']//2, self.fenv_config['min_y']+0.25],
            'e': [self.fenv_config['max_x'], self.fenv_config['max_y']//2],
            'w': [self.fenv_config['min_x'], self.fenv_config['max_y']//2],            
                  }
        
        self.color_dict = {
            2: '#FFFFFF',  # White
            3: '#FFFFFF',  # White
            4: '#FFFFFF',  # White
            5: '#FFFFFF',  # White
            
            6: '#000000',  # Black
            7: '#000000',  # Black
            8: '#000000',  # Black
            9: '#000000',  # Black
            
            10: '#D3D3D3',  # Light Gray
            
            11: '#D3D3D3',  # Light Gray
            
            12:'#E57373',  # Light Maroon
            13: '#20B2AA',  # Light Sea Green (Light Teal)
            14: '#BAE1FF',  # Light Blue
            15: '#E2BAFF',  # Light Purple '#FFFFBA',  # Light Yellow
            16: '#FFDFBA',  # Light Peach
            17: '#FF69B4',  # Hot Pink (reddish tone) # 
            18: '#BAF2FF',  # Light Cyan
            19: '#C4FF9E',  # Light Lime
        }
        
        
        if self.fenv_config['is_internal_walls_gray']:
            self.internal_wall_colors = {
                -11: '#333333',  # Dark Pink
                
                -12: '#333333',  # Dark Maroon
                -13: '#333333',  # Dark Green
                -14: '#333333',  # Dark Blue
                -15: '#333333',  # Dark Purple '#FFD700',  # Dark Yellow
                -16: '#333333',  # Dark Orange
                -17: '#333333',  # Dark Magenta (corresponding to Hot Pink) # 
                -18: '#333333',  # Dark Cyan
                -19: '#333333',  # Dark Lime
            }
            
        else:
            self.internal_wall_colors = {
                -11: '#FF69B4',  # Dark Pink
                
                -12: '#800000',  # Dark Maroon
                -13: '#006400',  # Dark Green
                -14: '#4169E1',  # Dark Blue
                -15: '#8A2BE2',  # Dark Purple '#FFD700',  # Dark Yellow
                -16: '#FF8C00',  # Dark Orange
                -17: '#8B008B',  # Dark Magenta (corresponding to Hot Pink) # 
                -18: '#008B8B',  # Dark Cyan
                -19: '#32CD32',  # Dark Lime
            }

    def value_to_color(self, val):
        return self.color_dict.get(val, '#FFFFFF')  # Default to white if value not in dict
    
    def dynamic_room_overlay(self, plan_data_dict, episode, ep_time_step):
        left_array = copy.deepcopy(plan_data_dict['obs_moving_labels'])
        right_array = copy.deepcopy(plan_data_dict['obs_mat_w'])

        def add_entrance_to_plan(left_array, right_array):
            for r, c in plan_data_dict['entrance_positions']:
                left_array[r][c] = self.fenv_config['entrance_cell_id']
                right_array[r][c] = self.fenv_config['entrance_cell_id']
            return left_array, right_array
        left_array, right_array = add_entrance_to_plan(left_array, right_array)

        self.episode = episode
        self.ep_time_step = ep_time_step + 1

        # Add base walls if show_base_wall_in_expose is True
        # show_base_wall_in_expose = False
        if self.fenv_config['show_base_wall_in_expose']:
            obs_mat_base_w = copy.deepcopy(plan_data_dict['obs_mat_base_w'])
            for i in range(obs_mat_base_w.shape[0]):
                for j in range(obs_mat_base_w.shape[1]):
                    if obs_mat_base_w[i, j] < 0:
                        right_array[i, j] = -20  # Use a special value to represent base walls

        # Create the base image from the right array (to include base walls)
        img = np.array([[self.value_to_color(val) for val in row] for row in right_array])

        # Convert color strings to RGBA
        img_rgba = np.array([mcolors.to_rgba(color) for color in img.flatten()]).reshape(img.shape + (4,))

        fig, ax = plt.subplots(figsize=(20, 10))

        # Color internal walls and outline
        unique_right_values = np.unique(right_array)
        available_walls = [val for val in unique_right_values if -19 <= val <= -12]
        for i in range(right_array.shape[0]):
            for j in range(right_array.shape[1]):
                if right_array[i, j] < 0:
                    if -19 <= right_array[i, j] <= -12:
                        img_rgba[i, j] = mcolors.to_rgba(self.internal_wall_colors[right_array[i, j]]) # type: ignore
                    elif right_array[i, j] == -20:
                        img_rgba[i, j] = mcolors.to_rgba('#333333')  # Very dark gray for base walls # type: ignore
                    else:
                        img_rgba[i, j] = mcolors.to_rgba('black')  # Outline walls # type: ignore

        # Add room colors
        for i in range(left_array.shape[0]):
            for j in range(left_array.shape[1]):
                if left_array[i, j] >= 10:
                    img_rgba[i, j] = mcolors.to_rgba(self.color_dict.get(left_array[i, j], '#FFFFFF')) # type: ignore

        im = ax.imshow(img_rgba)
        ax.axis('off')

        # Add a color bar for rooms
        unique_values = sorted(set(val for val in left_array.flatten() if val >= 10))
        colors = [self.color_dict.get(val, '#FFFFFF') for val in unique_values]
        color_map = mcolors.ListedColormap(colors)
        norm = mcolors.BoundaryNorm(unique_values + [max(unique_values) + 1], color_map.N)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=color_map),
                            cax=cax, orientation='vertical')
        cbar.set_ticks(np.array(unique_values) + 0.5)
        cbar.set_ticklabels(unique_values)
        cbar.set_label('Room IDs')

        # Add a color bar for walls on the left
        if not self.fenv_config['is_internal_walls_gray']:
            wall_colors = [self.internal_wall_colors[val] for val in available_walls] + ['#000000']
            wall_color_map = mcolors.ListedColormap(wall_colors)
            wall_values = available_walls + [-11]
            wall_norm = mcolors.BoundaryNorm(wall_values + [max(wall_values) + 1], wall_color_map.N)
            cax_legend = divider.append_axes("left", size="5%", pad=0.05)
            wall_cbar = plt.colorbar(plt.cm.ScalarMappable(norm=wall_norm, cmap=wall_color_map),
                                     cax=cax_legend, orientation='vertical')
            wall_cbar.set_ticks(np.array(wall_values) + 0.5)
            wall_cbar.set_ticklabels([f"{-val}" if val != -1 else "10" for val in wall_values])
            wall_cbar.set_label('Wall Types')
    
            # Adjust the position of tick labels to be centered within their corresponding color blocks
            cax_legend.yaxis.set_ticks_position('left')
            cax_legend.yaxis.set_label_position('left')
            wall_cbar.ax.invert_yaxis()

            # Manually set the ticks to ensure they are centered
            ticks = (np.array(wall_values) + 0.5).tolist()
            cax_legend.yaxis.set_major_locator(plt.FixedLocator(ticks))
            cax_legend.set_yticklabels([f"{-val}" if val != -1 else "10" for val in wall_values], rotation=0, va='center')
        
        plt.tight_layout()

        if self.fenv_config.get('save_render_flag', False):
            self._save_the_plan(fig, title='combined-map')

        plt.show()
        
        
    
    def dynamic_room_overlay_without_base_walls(self, plan_data_dict, episode, ep_time_step):
        left_array = copy.deepcopy(plan_data_dict['obs_moving_labels'])
        right_array = copy.deepcopy(plan_data_dict['obs_mat_w'])
    
        def add_entrance_to_plan(left_array, right_array):
            for r, c in plan_data_dict['entrance_positions']:
                left_array[r][c] = self.fenv_config['entrance_cell_id']
                right_array[r][c] = self.fenv_config['entrance_cell_id']
            return left_array, right_array
        left_array, right_array = add_entrance_to_plan(left_array, right_array)
    
        self.episode = episode
        self.ep_time_step = ep_time_step + 1
    
        # Create the base image from the left array
        img = np.array([[self.value_to_color(val) for val in row] for row in left_array])
    
        # Convert color strings to RGBA
        img_rgba = np.array([mcolors.to_rgba(color) for color in img.flatten()]).reshape(img.shape + (4,))
    
        fig, ax = plt.subplots(figsize=(20, 10))
    
        # Color internal walls and outline
        unique_right_values = np.unique(right_array)
        available_walls = [val for val in unique_right_values if -19 <= val <= -12]
        for i in range(right_array.shape[0]):
            for j in range(right_array.shape[1]):
                if right_array[i, j] < 0:
                    if -19 <= right_array[i, j] <= -12:
                        img_rgba[i, j] = mcolors.to_rgba(self.internal_wall_colors[right_array[i, j]])
                    else:
                        img_rgba[i, j] = mcolors.to_rgba('black')  # Outline walls
    
        im = ax.imshow(img_rgba)
        ax.axis('off')
    
        # Add a color bar for rooms
        unique_values = sorted(set(val for val in left_array.flatten() if val >= 10))
        colors = [self.color_dict.get(val, '#FFFFFF') for val in unique_values]
        color_map = mcolors.ListedColormap(colors)
        norm = mcolors.BoundaryNorm(unique_values + [max(unique_values) + 1], color_map.N)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=color_map),
                            cax=cax, orientation='vertical')
        cbar.set_ticks(np.array(unique_values) + 0.5)
        cbar.set_ticklabels(unique_values)
        cbar.set_label('Room IDs')
    
        # Add a color bar for walls on the left
        wall_colors = [self.internal_wall_colors[val] for val in available_walls] + ['#000000']
        wall_color_map = mcolors.ListedColormap(wall_colors)
        wall_values = available_walls + [-11]
        wall_norm = mcolors.BoundaryNorm(wall_values + [max(wall_values) + 1], wall_color_map.N)
        cax_legend = divider.append_axes("left", size="5%", pad=0.05)
        wall_cbar = plt.colorbar(plt.cm.ScalarMappable(norm=wall_norm, cmap=wall_color_map),
                                 cax=cax_legend, orientation='vertical')
        wall_cbar.set_ticks(np.array(wall_values) + 0.5)
        wall_cbar.set_ticklabels([f"{-val}" if val != -1 else "10" for val in wall_values])
        wall_cbar.set_label('Wall Types')
    
        # Adjust the position of tick labels to be centered within their corresponding color blocks
        cax_legend.yaxis.set_ticks_position('left')  # Ensure ticks are on the left side
        cax_legend.yaxis.set_label_position('left')  # Ensure label is on the left side
        # Invert y-axis to center labels properly within color blocks
        wall_cbar.ax.invert_yaxis()

        # Manually set the ticks to ensure they are centered
        ticks = (np.array(wall_values) + 0.5).tolist()
        cax_legend.yaxis.set_major_locator(plt.FixedLocator(ticks))
        cax_legend.set_yticklabels([f"{-val}" if val != -1 else "10" for val in wall_values], rotation=0, va='center')
        
        plt.tight_layout()
    
        if self.fenv_config.get('save_render_flag', False):
            self._save_the_plan(fig, title='combined-map')
    
        plt.show()





        
    def show_matrix_with_values(self, plan_data_dict, episode, ep_time_step):
        self.episode = episode
        self.ep_time_step = ep_time_step + 1
        
        room_matrix = plan_data_dict['obs_moving_labels']
        wall_matrix = plan_data_dict['obs_mat_w']
        
        fig, ax = plt.subplots(figsize=(15, 12))
        
        # Create a color-coded background for rooms
        im_room = ax.imshow(room_matrix, cmap='coolwarm', alpha=0.7)
        
        # Add text annotations for rooms
        for i in range(room_matrix.shape[0]):
            for j in range(room_matrix.shape[1]):
                if room_matrix[i, j] >= 10:  # Room numbers
                    ax.text(j, i, int(room_matrix[i, j]), # type: ignore
                            ha="center", va="center", color="w", fontweight='bold')
        
        # Add text annotations for walls
        for i in range(wall_matrix.shape[0]):
            for j in range(wall_matrix.shape[1]):
                if wall_matrix[i, j] < 0:  # Wall numbers (negative values)
                    ax.text(j, i, int(wall_matrix[i, j]), # type: ignore
                            ha="center", va="center", color="r", fontweight='bold')
        
        ax.set_title(f'Room and Wall Visualization - Episode {episode}, Step {ep_time_step}')
        
        # Add colorbar for room values
        cbar = plt.colorbar(im_room)
        cbar.set_label('Room IDs')
        
        plt.tight_layout()
        
        if self.fenv_config.get('save_render_flag', False):
            self._save_the_plan(fig, title='matrix_with_values')
            
        plt.show()
        
        
    
        
        
        
    def show_room_map(self, plan_data_dict, episode, ep_time_step):
        self.episode = episode
        self.ep_time_step = ep_time_step + 1
        
        # obw = copy.deepcopy(plan_data_dict['obs_mat_w'])
        # obw[obw == -self.fenv_config['north_facade_id']] = 0
        # obw[obw == -self.fenv_config['south_facade_id']] = 0
        # obw[obw == -self.fenv_config['east_facade_id']] = 0
        # obw[obw == -self.fenv_config['west_facade_id']] = 0
        # rw = plan_data_dict['obs_moving_labels']  - obw
        
        rw = copy.deepcopy(plan_data_dict['obs_rooms_cmap'])
        
        rw[rw <= 5] = 0
        for r, c in plan_data_dict['extended_entrance_positions'][2:]:
            rw[r][c] = self.fenv_config['entrance_cell_id'] * 2
            
        fig = plt.figure(figsize=self.figsize)
        plt.imshow(rw, cmap='binary')
        plt.title('Display: Room Lables')
        plt.axis('off')
        
        if self.fenv_config['save_render_flag']: self._save_the_plan(fig, title='room-lables')
        
        
        
    def _get_segmentation_map(self, plan_data_dict):
        obs_mat = copy.deepcopy(plan_data_dict['obs_mat'].astype(np.uint8))
        for r, c in plan_data_dict['entrance_positions']:
            obs_mat[r][c] = self.fenv_config['entrance_cell_id']
        hist, hist_centers = histogram(obs_mat)
        markers = np.zeros_like(obs_mat)
        markers[obs_mat == self.fenv_config['wall_pixel_value']] = 1
        markers[obs_mat != self.fenv_config['wall_pixel_value']] = 2
        elevation_map = sobel(obs_mat)
        segmentations_ = watershed(elevation_map, markers)
        segmentations = ndi.binary_fill_holes(segmentations_ - 1)
        labels, _ = ndi.label(segmentations)
        return obs_mat, labels
    
    
    
    def show_segmentation_map(self, plan_data_dict, episode, ep_time_step):
        self.episode = episode
        self.ep_time_step = ep_time_step + 1
        obs_mat, labels = self._get_segmentation_map(plan_data_dict)
        ep_time_step += 1
    
        mpl.rcParams['axes.linewidth'] = 0.1 #set the value globallys
        
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111)
        # ax.matshow(1-obs_mat, cmap='gray')
        plt.imshow(color.label2rgb(labels, image=obs_mat))
        # row, col = self.env_data_model.state_coords_for_ofbs_dict[f'S{start_state}']
        # plt.plot(col, row, '*')
        plt.title('Exibit: Segmentation Map')
        plt.axis('off')
        plt.show()
        if self.fenv_config['save_render_flag']: self._save_the_plan(fig, title='segmentation-map')
    

        
    def show_1d_obs(self, plan_data_dict, episode, ep_time_step):
        self.episode = episode
        self.ep_time_step = ep_time_step + 1
        obs_arr_conv = copy.deepcopy(plan_data_dict['obs_canvas_arr_1ch'])
        # obs_arr_conv = np.moveaxis(obs_arr_conv, 0, -1)
        ep_time_step += 1
        
        fig = plt.figure(figsize=self.figsize)
        plt.imshow(obs_arr_conv)
        plt.title('Illustrate: Canvas 1ch')
        plt.axis('off')
        plt.show()
        if self.fenv_config['save_render_flag']: self._save_the_plan(fig, title='1ch-canvas')
        

        
    def _get_canvas_cnn_3d(self, plan_data_dict):
        obs_moving_labels_refined = copy.deepcopy(plan_data_dict['obs_moving_labels_refined']) 
        obs_canvas_arr_3ch = np.zeros((obs_moving_labels_refined.shape[0], obs_moving_labels_refined.shape[1], 3), dtype=obs_moving_labels_refined.dtype)
            
        for r in range(obs_moving_labels_refined.shape[0]): #self.fenv_config['n_rows']*self.fenv_config['cnn_scaling_factor']):
            for c in range(obs_moving_labels_refined.shape[1]): #range(self.fenv_config['n_cols']*self.fenv_config['cnn_scaling_factor']):
                for v in list(self.fenv_config['color_map'].keys()):
                    if obs_moving_labels_refined[r, c] == v:
                        obs_canvas_arr_3ch[r, c, :] = list(  name_to_rgb(self.fenv_config['color_map'][v])  )
                        
        obs_canvas_arr_3ch = obs_canvas_arr_3ch.astype(np.uint8)
        return obs_canvas_arr_3ch
    
    
    
    def show_3d_obs(self, plan_data_dict, episode, ep_time_step):
        self.episode = episode
        self.ep_time_step = ep_time_step + 1
        ep_time_step += 1
        
        obs_canvas_arr_3ch = self._get_canvas_cnn_3d(plan_data_dict) / 255.0
        assert obs_canvas_arr_3ch.shape[-1] == 3, "For showing an image the 3rd dim should be color channel"
            
        fig = plt.figure(figsize=self.figsize)
        plt.imshow(obs_canvas_arr_3ch)
        
        for bf in plan_data_dict['facades_blocked']:
            plt.plot(self.blocked_facade_coord[bf][0]*2, self.blocked_facade_coord[bf][1]*2, marker='x', mew=5, ms=15, c='red')
            
        plt.title('Demonestrate: Canvas 3ch')
        plt.axis('off')
        # plt.grid("on")
        # plt.locator_params(nbins=23)
        plt.show()
        if self.fenv_config['save_render_flag']: self._save_the_plan(fig, title='3ch-canvas')
        
        
    
    def depict_3d_stacked(self, plan_data_dict, episode, ep_time_step):
        self.episode = episode
        self.ep_time_step = ep_time_step + 1
        stacked_3d = np.concatenate((plan_data_dict['obs_canvas_arr_1ch'], plan_data_dict['obs_rooms_cmap_1ch'], np.ones(plan_data_dict['obs_rooms_cmap_1ch'].shape, dtype=float)))
        # stacked_3d = np.moveaxis(stacked_3d, 0, -1)
        
        fig = plt.figure(figsize=self.figsize)
        plt.imshow(stacked_3d)
            
        plt.title('Depict: Stacked 3ch')
        plt.axis('off')
        plt.show()
        if self.fenv_config['save_render_flag']: self._save_the_plan(fig, title='3ch-stacked')
        
    
    
    def view_obs_mat(self, plan_data_dict, episode, ep_time_step):
        self.episode = episode
        self.ep_time_step = ep_time_step + 1
        obs_mat = copy.deepcopy(plan_data_dict['obs_mat'])
        for r, c in plan_data_dict['entrance_positions']:
            obs_mat[r][c] = self.fenv_config['entrance_cell_id']
            
        ep_time_step += 1
        
        fig = plt.figure(figsize=self.figsize)
        plt.imshow(obs_mat)
        plt.title('View: Observation Matrix')
        plt.axis('off')
        plt.show()
        if self.fenv_config['save_render_flag']: self._save_the_plan(fig, title='obs_mat')
        
        
        
    def _save_the_plan(self, fig, title):
        results_dir = self.fenv_config['results_dir']
        plan_fig_path = os.path.join(results_dir, f"trial_{self.trial}__fig_{title}__episode_{self.episode:02}__timestep_{self.ep_time_step:02}.png")
        # fig.savefig(plan_fig_path, bbox_extra_artists=(lgd,), bbox_inches='tight')
        fig.savefig(plan_fig_path, bbox_inches='tight')