# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 23:40:09 2021

@author: RK
"""
# %%

import math
import copy
import numpy as np
from scipy import ndimage as ndi

from scipy import ndimage
from skimage import feature, measure
from skimage import segmentation
from skimage.exposure import histogram
from skimage.filters import sobel
from skimage.segmentation import watershed

from gym_floorplan.envs.observation.partitioner import Partitioner

# %%

class RoomExtractor:
    def __init__(self, fenv_config:dict=None):
        self.fenv_config = fenv_config
        
        
    def update_room_dict(self, plan_data_dict, active_wall_name):
        room_i = wall_i = int(active_wall_name.split("_")[1])
        room_name = f"room_{room_i}"
        
        plan_data_dict['rooms_dict'].update({room_name: {}})
        
        
        obs_mat_label, labels = self._get_segmentation_map(plan_data_dict['obs_mat'])
        
        plan_data_dict = self._update_plan_data_dict_based_on_room_areas(plan_data_dict, labels, wall_i, room_i, room_name)     
        
        plan_data_dict = self._update_plan_data_dict_based_on_room_shape_properties(plan_data_dict, room_i, room_name)
        
        plan_data_dict = self._update_plan_data_dict_based_on_sub_rectangles(plan_data_dict, room_name)
        
        room_shape = plan_data_dict['rooms_dict'][room_name]['room_shape']
        
        if  room_shape == 'rectangular':    
            proportions = [plan_data_dict['rooms_dict'][room_name]['room_aspect_ratio']]
        else:
            proportions = plan_data_dict['rooms_dict'][room_name]['sub_rects']['aspect_ratios']
            
        plan_data_dict['rooms_dict'][room_name]['proportions'] = proportions
        
        if wall_i >= plan_data_dict['mask_numbers']:
            if len(proportions) == 0:
                print(proportions)
            
        return plan_data_dict
    
    
    def _update_plan_data_dict_based_on_room_areas(self, plan_data_dict, labels, wall_i, room_i, room_name):
        moving_labels = plan_data_dict['moving_labels']
        obs_mat_for_dot_prod = plan_data_dict['obs_mat_for_dot_prod']
        labels_ = copy.deepcopy(labels)
        cover0 = np.argwhere(moving_labels != 0)
        if len(cover0) >= 1:
            for i, j in cover0:
                labels[i,j] = 0
        
        if self.fenv_config['mask_flag']:
            obs_mat_mask = copy.deepcopy(plan_data_dict['obs_mat_mask'])
            labels = labels * obs_mat_mask
        room_unique_possible_ids = np.unique(labels[labels>0])
        possible_areas = [np.sum(labels==i) for i in room_unique_possible_ids]
        if len(possible_areas) == 0:
            possible_areas = [0]
        min_area = np.min(possible_areas)
        # print(f"possible_areas: {possible_areas}")
        min_room_unique_possible_id = room_unique_possible_ids[np.argmin(possible_areas)]
        moving_labels[labels==min_room_unique_possible_id] = room_i
        # moving_labels = np.multiply(moving_labels, obs_mat_for_dot_prod)
        this_wall_len = len(np.argwhere(plan_data_dict['obs_mat_w'] == -wall_i ))
        if self.fenv_config['include_wall_in_area_flag']:
            this_room_area = min_area + this_wall_len
        else:
            this_room_area = min_area
        plan_data_dict['areas'].update({room_name: this_room_area})
        if self.fenv_config['mask_flag']:
            if room_i > plan_data_dict['mask_numbers']:
                plan_data_dict['delta_areas'].update({room_name: this_room_area - plan_data_dict['desired_areas'][room_name]})
        else:
            plan_data_dict['delta_areas'].update({room_name: this_room_area - plan_data_dict['desired_areas'][room_name]})
        plan_data_dict['rooms_dict'].update({room_name: {}})
        plan_data_dict['rooms_dict'][room_name].update({"room_area": this_room_area})
        plan_data_dict['rooms_dict'][room_name].update({"room_pure_area": min_area})
        
        room_positions = np.argwhere(moving_labels == room_i).tolist()
        plan_data_dict['rooms_dict'][room_name].update({'room_positions': room_positions}) 
        
        if room_i == (plan_data_dict['number_of_walls']):
            last_room_name = f"room_{room_i+1}"
            last_room_i = room_i + 1
            plan_data_dict['rooms_dict'].update({last_room_name: {}})
            
            max_area = np.max(possible_areas)
            max_room_unique_possible_id = room_unique_possible_ids[np.argmax(possible_areas)]
            
            moving_labels[labels==max_room_unique_possible_id] = last_room_i
            # moving_labels = np.multiply(moving_labels, obs_mat_for_dot_prod)
            
            
            plan_data_dict['areas'].update({last_room_name: max_area})
            plan_data_dict['delta_areas'].update({last_room_name: max_area - plan_data_dict['desired_areas'][last_room_name]})
            plan_data_dict['rooms_dict'].update({last_room_name: {}})
            plan_data_dict['rooms_dict'][last_room_name].update({"room_area": max_area})
            
            room_positions = np.argwhere(moving_labels == last_room_i).tolist()
            plan_data_dict['rooms_dict'][last_room_name].update({'room_positions': room_positions})
            
        
        plan_data_dict['moving_labels'] = moving_labels
        return plan_data_dict
        
    
    def _update_plan_data_dict_based_on_room_shape_properties(self, plan_data_dict, room_i, room_name):
        room_positions = plan_data_dict['rooms_dict'][room_name]['room_positions']
        wall_positions = plan_data_dict['walls_coords'][f"wall_{room_i}"]['wall_positions']
        
        only_this_room_labels_mat = np.zeros((self.fenv_config['n_rows'], self.fenv_config['n_cols']))
        
        moving_ones = copy.deepcopy(plan_data_dict['moving_ones'])
        for r, c in room_positions:
            only_this_room_labels_mat[r,c] = room_i
            moving_ones[r,c] = 1
            
        for r, c in wall_positions:
            moving_ones[r,c] = 1
            
        only_this_room_labels_mat = self._add_outlines(only_this_room_labels_mat)
        
        contours = self._get_contours(only_this_room_labels_mat)
        contours_height, contours_width = self._get_contours_height_width(contours)
        room_height = contours_height -1
        room_width = contours_width - 1
        
        if plan_data_dict['rooms_dict'][room_name]['room_pure_area'] == (room_height * room_width): # it means that the new room has certanily rectangle shape
            room_shape = "rectangular"
            aspect_ratio = max(room_height, room_width) / min(room_height, room_width)
        else:
            ### note that a room could have non-rectangle shape. 
            ### But still we store its properties for now, and then 
            ### we partition it to sub-rectangles
            room_shape = "nonrectangular"
            aspect_ratio = None
        
        plan_data_dict['rooms_dict'][room_name].update({'room_shape': room_shape})
        plan_data_dict['rooms_dict'][room_name].update({'room_height': room_height})
        plan_data_dict['rooms_dict'][room_name].update({'room_width': room_width})
        plan_data_dict['rooms_dict'][room_name].update({'room_aspect_ratio': aspect_ratio})
        
        plan_data_dict['moving_ones'] = copy.deepcopy(moving_ones)
        return plan_data_dict


    def _add_outlines(self, arr):
        row = -10 * np.ones((self.fenv_config['n_cols']))
        col = -10 * np.ones((self.fenv_config['n_rows']+2,1))
        arr = np.vstack([row, arr])
        arr = np.vstack([arr, row])
        arr = np.hstack([col, arr])
        arr = np.hstack([arr, col])
        return arr
        
        
    def _update_plan_data_dict_based_on_sub_rectangles(self, plan_data_dict, room_name):
        room_data = plan_data_dict['rooms_dict'][room_name]
        if room_data['room_shape'] == "nonrectangular":
            room_positions = room_data['room_positions']
            ### in each step we only focus on those cells which are related 
            ### to the room called by the for loop
            temp_obs_mat = np.zeros((self.fenv_config['n_rows'], self.fenv_config['n_cols']))
            for r, c in room_positions:
                temp_obs_mat[r, c] = 1
            ### we partition this room
            all_rects = Partitioner(temp_obs_mat).get_rectangules()
            areas = [len(val['rows'])*np.shape(val['cols'])[1] for key, val in all_rects.items()]
            widths = [np.shape(val['cols'])[1] for key, val in all_rects.items()]
            heights = [np.shape(val['cols'])[0] for key, val in all_rects.items()]
            aspect_ratios = [max(h, w) / min(h, w) for w,h in zip(heights, widths)]
            plan_data_dict['rooms_dict'][room_name].update({'sub_rects': {'areas': areas, 
                                                      'widths': widths,
                                                      'heights': heights,
                                                      'aspect_ratios': aspect_ratios}})
        return plan_data_dict
    
    
    def _get_segmentation_map(self, obs_mat):
        obs_mat = obs_mat.astype(np.uint8)
        hist, hist_centers = histogram(obs_mat)
        
        markers = np.zeros_like(obs_mat)
        markers[obs_mat == self.fenv_config['wall_pixel_value']] = 1
        markers[obs_mat != self.fenv_config['wall_pixel_value']] = 2
        
        elevation_map = sobel(obs_mat)
        
        segmentations_ = watershed(elevation_map, markers)
        segmentations = ndi.binary_fill_holes(segmentations_ - 1)
        
        labels, _ = ndi.label(segmentations)
        
        return obs_mat, labels
    
    
    def _get_contours(self, only_new_room_labels_mat):
        contours = measure.find_contours(only_new_room_labels_mat, 0.01)
        return contours
    
    
    def _get_room_map(self, obs_mat):
        edges = feature.canny(obs_mat, sigma=0.5)
        dt = ndimage.distance_transform_edt(~edges)
        local_max = feature.peak_local_max(dt, indices=False, min_distance=1)
        markers = measure.label(local_max)
        labels = segmentation.watershed(-dt, markers)
        rooms_idx, rooms_area  = np.unique(labels, return_counts=True)
        num_rooms = len(rooms_idx)
        return labels
    
    
    def _get_contours_height_width(self, contour):
        if len(contour) == 0:
            return 0, 0
        else:
            contour = contour[0]
            ll, ur = np.min(contour, 0), np.max(contour, 0)
            wh = ur - ll
            contour_height, contour_width = math.ceil(wh[0]), math.ceil(wh[1])
            return contour_height, contour_width
    