#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 10:04:55 2024

@author: Reza Kakooee
"""
import os
import inspect
import copy
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import sobel
from skimage.segmentation import watershed
from datetime import datetime
from typing import List, Tuple, Dict
from itertools import permutations
from skimage import measure
import math
from gym_floorplan.envs.observation.partitioner import Partitioner


#%% ###########################################################################
########################                                  #####################
###############################################################################

def get_unique_integers(matrix: np.ndarray) -> List[int]:
    return list(set(matrix.flatten()) - {0})


def get_region_coordinates(matrix: np.ndarray, value: int) -> List[Tuple[int, int]]:
    return list(zip(*np.where(matrix == value)))


def calculate_similarity(region1: List[Tuple[int, int]], region2: List[Tuple[int, int]]) -> float:
    set1 = set(region1)
    set2 = set(region2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0


def find_best_matching_integers(left_matrix: np.ndarray, right_matrix: np.ndarray) -> Dict[int, Tuple[int, float]]:
    left_integers = get_unique_integers(left_matrix)
    right_integers = get_unique_integers(right_matrix)
    
    result = {}
    
    for left_value in left_integers:
        left_region = get_region_coordinates(left_matrix, left_value)
        max_similarity = 0
        best_match = None
        
        for right_value in right_integers:
            right_region = get_region_coordinates(right_matrix, right_value)
            similarity = calculate_similarity(left_region, right_region)
            
            if similarity > max_similarity:
                max_similarity = similarity
                best_match = right_value
        
        result[left_value] = (best_match, max_similarity)
    
    return result


def find_best_match_for_region(source_matrix: np.ndarray, source_value: int, target_matrix: np.ndarray) -> Tuple[int, float]:
    source_region = get_region_coordinates(source_matrix, source_value)
    target_integers = get_unique_integers(target_matrix)
    
    max_similarity = 0
    best_match = None
    
    for target_value in target_integers:
        target_region = get_region_coordinates(target_matrix, target_value)
        similarity = calculate_similarity(source_region, target_region)
        
        if similarity > max_similarity:
            max_similarity = similarity
            best_match = target_value
    
    return best_match, max_similarity


def find_best_match_across_matrices(left_matrix: np.ndarray, right_matrix: np.ndarray, new_matrix: np.ndarray, last_room=False) -> Tuple[int, int, float]:
    # Find best matches between left and right matrices
    initial_results = find_best_matching_integers(left_matrix, right_matrix)
    # print("Initial results:", initial_results)
    
    # Find the match with the highest similarity
    best_left_value = max(initial_results, key=lambda k: initial_results[k][1])
    best_right_value, _ = initial_results[best_left_value]
    
    last_room_value = None
    if last_room:
        # Find the least similar region for the living room
        right_integers = get_unique_integers(right_matrix)
        min_similarity = float('inf')
        for right_value in right_integers:
            if right_value != best_right_value:
                _, similarity = find_best_match_for_region(right_matrix, right_value, new_matrix)
                if similarity < min_similarity:
                    min_similarity = similarity
                    last_room_value = right_value
    
    # Find the best match for this region in the new matrix
    final_best_match, final_similarity = find_best_match_for_region(right_matrix, best_right_value, new_matrix)
    
    return best_right_value, last_room_value
    
    
    
def assign_rooms_initial_way(plan_data_dict_backup, plan_data_dict):
    obs_moving_labels_old = copy.deepcopy(plan_data_dict_backup['obs_moving_labels'])
    new_labels_complete = -1 * get_segmentation_map(plan_data_dict['obs_mat_w'])
    used_integer_from_new_labels_complete = set()
    
    obs_mat_w = copy.deepcopy(plan_data_dict['cleaned_obs_matrix'])
    obs_moving_labels = copy.deepcopy(plan_data_dict['obs_moving_labels'])
    for agent_name in plan_data_dict['shifted_agent_names_deque']:
        room_i = int(agent_name.split('_')[1])
        obs_mat_w += plan_data_dict['wall_matrices'][-room_i]
        obs_moving_labels = assign_one_room(plan_data_dict, obs_mat_w, obs_moving_labels, obs_moving_labels_old, new_labels_complete, room_i, used_integer_from_new_labels_complete, agent_name==plan_data_dict['shifted_agent_names_deque'][-1])
    return obs_mat_w, obs_moving_labels
        
def assign_one_room(plan_data_dict, obs_mat_w, obs_moving_labels, obs_moving_labels_old, new_labels_complete, room_i, used_integer_from_new_labels_complete, last_room=False):
    labels = get_segmentation_map(obs_mat_w)
    mask = obs_moving_labels != 0
    labels[mask] = 0
    lvroom_id = 11
    
    best_right_value, last_room_value = find_best_match_across_matrices(labels, new_labels_complete, obs_moving_labels_old, last_room)
    # print(f"Best match from right matrix: {best_right_value}")

    used_integer_from_new_labels_complete.add(best_right_value)
    
    mask = new_labels_complete == best_right_value
    obs_moving_labels[mask] = room_i
    
    if last_room and last_room_value is not None:
        # mask = new_labels_complete == last_room_value
        obs_moving_labels_with_walls = obs_moving_labels - plan_data_dict['obs_mat_w']
        mask = obs_moving_labels_with_walls == 0
        obs_moving_labels[mask] = lvroom_id
    
    return obs_moving_labels



#%% ###########################################################################
########################                                  #####################
###############################################################################
def get_nonrectangular_aspect_ratio(room_binary):
    all_rects = Partitioner(room_binary).get_rectangules()
    areas = [len(val['rows'])*np.shape(val['cols'])[1] for key, val in all_rects.items()]
    widths = [np.shape(val['cols'])[1] for key, val in all_rects.items()]
    heights = [np.shape(val['cols'])[0] for key, val in all_rects.items()]
    aspect_ratios = [max(h, w) / min(h, w) for w,h in zip(heights, widths)]
    aspect_ratio = max(np.array(aspect_ratios))
    
    
    all_rects_ = Partitioner(np.rot90(room_binary)).get_rectangules()
    areas_ = [len(val['rows'])*np.shape(val['cols'])[1] for key, val in all_rects_.items()]
    widths_ = [np.shape(val['cols'])[1] for key, val in all_rects_.items()]
    heights_ = [np.shape(val['cols'])[0] for key, val in all_rects_.items()]
    aspect_ratios_ = [max(h, w) / min(h, w) for w,h in zip(heights_, widths_)]
    aspect_ratio_ = max(np.array(aspect_ratios_))
    
    aspect_ratio = min(aspect_ratio, aspect_ratio_)
    
    all_rects_positions = {key: [] for key in all_rects}
    for sub_r, rc in all_rects.items():
        for r, cs in zip(rc['rows'], rc['cols']):
            for c in cs:
                all_rects_positions[sub_r].append([r, c])
                
    sub_rects = {'areas_achieved': areas, 
                'widths': widths,
                'heights': heights,
                'aspect_ratio': aspect_ratio,
                # 'delta_aspect_ratio': delta_aspect_ratio,
                'all_rects_positions': all_rects_positions}
    
    return aspect_ratio, sub_rects
    
    
def analyze_room_shapes(room_array):
    room_array[:, 0] = 0
    room_array[:, -1] = 0
    room_array[0, :] = 0
    room_array[-1, :] = 0
    min_room_id = 11
    def get_contours(room_labels_mat):
        contours = measure.find_contours(room_labels_mat, 0.5)
        return contours

    def get_contours_height_width(contours):
        if not contours:
            return 0, 0
        else:
            # Find the largest contour by area
            largest_contour = max(contours, key=lambda c: len(c))
            ll, ur = np.min(largest_contour, axis=0), np.max(largest_contour, axis=0)
            wh = ur - ll
            contour_height, contour_width = math.ceil(wh[0]), math.ceil(wh[1])
            return contour_height, contour_width

    room_ids = np.unique(room_array)
    room_ids = room_ids[room_ids != 0]  # Exclude 0 which typically represents walls or empty space
    room_ids = [i for i in room_ids if i >= min_room_id]
    room_analysis = {}

    for room_id in room_ids:
        # Create a binary map for the current room
        room_binary = (room_array == room_id).astype(float)
        
        contours = get_contours(room_binary)
        if not contours:
            room_shape  = "undefined"
            room_height = 0
            room_width = 0
            room_area = 0
            aspect_ratio = None
        else:
            contours_height, contours_width = get_contours_height_width(contours)
            contour_area = np.sum(room_binary)
            bounding_box_area = contours_height * contours_width
    
            if contour_area == bounding_box_area:
                room_shape = "rectangular"
                room_height = contours_height# - 1
                room_width = contours_width# - 1
                room_area = room_height * room_width
                assert room_area == np.sum(room_binary)
                aspect_ratio = max(room_height, room_width) / min(room_height, room_width)
            else:
                aspect_ratio_, sub_rects = get_nonrectangular_aspect_ratio(room_binary)
                room_shape = "nonrectangular"
                room_area = np.sum(room_binary)
                room_height = None
                room_width = None
                aspect_ratio = aspect_ratio_
                
        room_analysis[room_id] = {
            "shape": room_shape,
            "aspect_ratio": aspect_ratio,
            "height": room_height,
            "width": room_width,
            "area": room_area,
        }
        if room_shape == 'nonrectangular':
            room_analysis[room_id].update({'sub_rects': sub_rects})
        
    return room_analysis


    
def get_geometrical_properties(plan_data_dict):
    oml = copy.deepcopy(plan_data_dict['obs_moving_labels'])
    ## remove entrance with id=10 from the oml # this is added on Sep 25, 2024
    oml[oml == 10] = 0
    for k, mat in plan_data_dict['wall_matrices'].items():
        oml += -mat
    oml += -plan_data_dict['cleaned_obs_matrix']
    
    # this is correct, but we donot need oit
    # unique_labels, counts = np.unique(oml, return_counts=True)
    # room_areas = {}
    # for label, count in zip(unique_labels, counts):
    #     if label != 0:  # Exclude the label for walls/non-room areas
    #         room_areas[f"room_{label}"] = count
        
    room_data = analyze_room_shapes(oml) #(copy.deepcopy(plan_data_dict['obs_moving_labels']))
    
    areas_achieved = {f'room_{key}': value['area'] for key, value in room_data.items()}
    areas_delta = {}   
    for room_name, a_desired in plan_data_dict['areas_desired'].items():
        areas_delta[room_name] = abs(a_desired - areas_achieved[room_name])
        
    aspect_ratio_achieved = {f'room_{key}': value['aspect_ratio'] for key, value in room_data.items()}
    aspect_ratio_delta = {}
    for room_name, as_desired in plan_data_dict['aspect_ratio_desired'].items():
        aspect_ratio_delta[room_name] = abs(as_desired - aspect_ratio_achieved[room_name])
            
    plan_data_dict.update({
        'areas_achieved': areas_achieved,
        'areas_delta': areas_delta,
        'aspect_ratio_achieved': aspect_ratio_achieved,
        'aspect_ratio_delta': aspect_ratio_delta,
    })
    
    all_areas_delta = copy.deepcopy(areas_delta)
    all_areas_delta.update({room_name:0 for room_name in plan_data_dict['areas_masked'].keys()})
    
    all_aspect_ratio_achieved = copy.deepcopy(aspect_ratio_achieved)
    all_aspect_ratio_achieved.update({room_name:1 for room_name in plan_data_dict['areas_masked'].keys()})
    
    all_shapes = {room_name:'rectangular' for room_name in plan_data_dict['areas_masked'].keys()}
    all_shapes.update({f'room_{i}': d['shape'] for i, d in room_data.items()})
    
    all_rooms_positions = {}
    room_ids = np.unique(copy.deepcopy(plan_data_dict['obs_moving_labels']))
    room_ids = room_ids[(room_ids != 0) & (room_ids != 10)]
    array = copy.deepcopy(plan_data_dict['obs_moving_labels'])
    for room_i in room_ids:
        room_name = f'room_{room_i}'
        indices = [[i, j] for i in range(len(array)) for j in range(len(array[0])) if array[i][j] == room_i]
        all_rooms_positions[room_name] = indices
    
    all_areas_achieved = copy.deepcopy(areas_achieved)
    all_areas_achieved.update(plan_data_dict['areas_masked'])
    rooms_dict = {}
    for room_i in room_ids:
        room_name = f'room_{room_i}'
        try:
            rooms_dict[room_name] = {
                'room_area': all_areas_achieved[room_name],
                'room_positions': all_rooms_positions[room_name],
                'room_shape': all_shapes[room_name],
                'room_height':  None,
                'room_width': None,
                'room_aspect_ratio': all_aspect_ratio_achieved[room_name], 
                'delta_aspect_ratio': all_aspect_ratio_achieved[room_name],
                'proportions': None,
                'areas_delta': all_areas_delta[room_name],
            }
        except:
            print(f"room_name: {room_name}")
            raise
        if all_shapes[room_name] == 'nonrectangular':
            rooms_dict[room_name].update({'sub_rects': room_data[room_i]['sub_rects']})
    plan_data_dict.update({
        'rooms_dict': rooms_dict,
    })
    
    return plan_data_dict
    
    
    
def get_clean_matrices(obs_mat_w):
    min_room_id = 11
    rows, cols = obs_mat_w.shape
    cleaned_obs_matrix = obs_mat_w.copy()
    a, b = -min_room_id-1, np.min(obs_mat_w)
    cleaned_obs_matrix[(cleaned_obs_matrix <= a) & (cleaned_obs_matrix >= b)] = 0
    wall_matrices = {}
    for wall_id in range(b, -min_room_id):
        wall_matrix = np.zeros_like(obs_mat_w)
        wall_matrix[obs_mat_w == wall_id] = wall_id
        wall_matrices[wall_id] = wall_matrix
    return cleaned_obs_matrix, wall_matrices



def calculate_iou(region1, region2):
    intersection = np.logical_and(region1, region2).sum()
    union = np.logical_or(region1, region2).sum()
    return intersection / union if union > 0 else 0



def get_segmentation_map(matrix):
    labeled, _ = ndi.label(matrix == 0)
    return labeled
    

    
def get_segmentation_mapv0(obs_mat_w):
    obs_mat_uint8 = np.abs(obs_mat_w).astype(np.uint8)
    markers = np.zeros_like(obs_mat_uint8)
    markers[obs_mat_w != 0] = 1
    markers[obs_mat_w == 0] = 2
    elevation_map = sobel(obs_mat_uint8)
    segmentations = watershed(elevation_map, markers)
    segmentations = ndi.binary_fill_holes(segmentations - 1)
    labels, _ = ndi.label(segmentations)
    return labels



#%% ###########################################################################
########################                                  #####################
###############################################################################
def assign_room_ids_max_iou_stepwise(old_matrix, new_matrix, agent_order):
    new_labeled = get_segmentation_map(new_matrix)
    new_room_ids = np.unique(new_labeled[new_labeled != 0])
    old_room_ids = np.unique(old_matrix[old_matrix > 0])

    # Calculate IoU matrix
    iou_matrix = np.zeros((len(new_room_ids), len(old_room_ids)))
    for i, new_id in enumerate(new_room_ids):
        for j, old_id in enumerate(old_room_ids):
            iou_matrix[i, j] = calculate_iou(new_labeled == new_id, old_matrix == old_id)

    new_id_to_old_id = {}
    assigned_new_ids = set()

    # Step 1: Assign rooms 2 to 5 if they exist
    for room_id in range(2, 6):
        if room_id in old_room_ids:
            best_new_id = None
            best_iou = 0
            for i, new_id in enumerate(new_room_ids):
                if new_id not in assigned_new_ids:
                    iou = iou_matrix[i][list(old_room_ids).index(room_id)]
                    if iou > best_iou:
                        best_iou = iou
                        best_new_id = new_id
            if best_new_id is not None:
                new_id_to_old_id[best_new_id] = room_id
                assigned_new_ids.add(best_new_id)

    # Convert agent_order to room IDs
    room_order = [int(agent.split('_')[1]) for agent in agent_order]

    # Step 2: Assign rooms based on the agent_order, excluding room 11
    for room_id in room_order:
        if room_id != 11 and room_id in old_room_ids:
            best_new_id = None
            best_iou = 0
            for i, new_id in enumerate(new_room_ids):
                if new_id not in assigned_new_ids:
                    iou = iou_matrix[i][list(old_room_ids).index(room_id)]
                    if iou > best_iou:
                        best_iou = iou
                        best_new_id = new_id
            if best_new_id is not None:
                new_id_to_old_id[best_new_id] = room_id
                assigned_new_ids.add(best_new_id)

    # Step 3: Assign room 11 to the largest remaining unassigned region
    if 11 in old_room_ids:
        remaining_new_ids = set(new_room_ids) - assigned_new_ids
        if remaining_new_ids:
            largest_remaining_id = max(remaining_new_ids, key=lambda x: np.sum(new_labeled == x))
            new_id_to_old_id[largest_remaining_id] = 11
            assigned_new_ids.add(largest_remaining_id)

    # Create the result matrix
    result = np.zeros_like(new_matrix)
    for new_id, old_id in new_id_to_old_id.items():
        result[new_labeled == new_id] = old_id


    return result
                
    

#%% ###########################################################################
########################                                  #####################
###############################################################################
def assign_room_ids_max_iou_min_mismatch_stepwise(old_matrix, new_matrix, agent_order):
    new_labeled = get_segmentation_map(new_matrix)
    new_room_ids = np.unique(new_labeled[new_labeled != 0])
    old_room_ids = np.unique(old_matrix[old_matrix > 0])

    # Calculate IoU matrix
    iou_matrix = np.zeros((len(new_room_ids), len(old_room_ids)))
    for i, new_id in enumerate(new_room_ids):
        for j, old_id in enumerate(old_room_ids):
            iou_matrix[i, j] = calculate_iou(new_labeled == new_id, old_matrix == old_id)

    new_id_to_old_id = {}
    assigned_new_ids = set()

    # Step 1: Assign rooms 2 to 5 if they exist
    for room_id in range(2, 6):
        if room_id in old_room_ids:
            best_new_id = None
            best_iou = 0
            for i, new_id in enumerate(new_room_ids):
                if new_id not in assigned_new_ids:
                    iou = iou_matrix[i][list(old_room_ids).index(room_id)]
                    if iou > best_iou:
                        best_iou = iou
                        best_new_id = new_id
            if best_new_id is not None:
                new_id_to_old_id[best_new_id] = room_id
                assigned_new_ids.add(best_new_id)

    # Step 2: Assign rooms with IoU = 1
    for i, new_id in enumerate(new_room_ids):
        if new_id not in assigned_new_ids:
            perfect_matches = np.where(iou_matrix[i] == 1)[0]
            if len(perfect_matches) == 1:
                old_id = old_room_ids[perfect_matches[0]]
                new_id_to_old_id[new_id] = old_id
                assigned_new_ids.add(new_id)

    # Convert agent_order to room IDs and ensure room 11 is included
    room_order = [int(agent.split('_')[1]) for agent in agent_order]
    if 11 not in room_order and 11 in old_room_ids:
        room_order.append(11)

    # Step 3: Assign remaining rooms based on agent order and minimum mismatches
    for room_id in room_order:
        if room_id not in new_id_to_old_id.values():
            possible_assignments = []
            for new_id in new_room_ids:
                if new_id not in assigned_new_ids:
                    temp_result = np.zeros_like(new_matrix)
                    temp_result[new_labeled == new_id] = room_id
                    mismatches = np.sum(temp_result != old_matrix)
                    possible_assignments.append((new_id, mismatches))
            
            if possible_assignments:
                best_new_id = min(possible_assignments, key=lambda x: x[1])[0]
                new_id_to_old_id[best_new_id] = room_id
                assigned_new_ids.add(best_new_id)

    # Create the result matrix
    result = np.zeros_like(new_matrix)
    for new_id, old_id in new_id_to_old_id.items():
        result[new_labeled == new_id] = old_id

    return result
        
    
#%% ###########################################################################
########################                                  #####################
###############################################################################
def assign_room_ids_max_iou_min_mismatch_ordered_permutaion(old_matrix, new_matrix, agent_order):
    new_labeled = get_segmentation_map(new_matrix)
    new_room_ids = np.unique(new_labeled[new_labeled != 0])
    old_room_ids = np.unique(old_matrix[old_matrix > 0])

    # Calculate IoU matrix
    iou_matrix = np.zeros((len(new_room_ids), len(old_room_ids)))
    for i, new_id in enumerate(new_room_ids):
        for j, old_id in enumerate(old_room_ids):
            iou_matrix[i, j] = calculate_iou(new_labeled == new_id, old_matrix == old_id)

    new_id_to_old_id = {}
    assigned_new_ids = set()

    # Step 1: Assign rooms 2 to 5 if they exist
    for room_id in range(2, 6):
        if room_id in old_room_ids:
            best_new_id = None
            best_iou = 0
            for i, new_id in enumerate(new_room_ids):
                if new_id not in assigned_new_ids:
                    iou = iou_matrix[i][list(old_room_ids).index(room_id)]
                    if iou > best_iou:
                        best_iou = iou
                        best_new_id = new_id
            if best_new_id is not None:
                new_id_to_old_id[best_new_id] = room_id
                assigned_new_ids.add(best_new_id)

    # Step 2: Assign rooms with IoU = 1
    for i, new_id in enumerate(new_room_ids):
        if new_id not in assigned_new_ids:
            perfect_matches = np.where(iou_matrix[i] == 1)[0]
            if len(perfect_matches) == 1:
                old_id = old_room_ids[perfect_matches[0]]
                new_id_to_old_id[new_id] = old_id
                assigned_new_ids.add(new_id)

    # Convert agent_order to room IDs
    room_order = [int(agent.split('_')[1]) for agent in agent_order]

    # Step 3: Assign remaining rooms based on agent order and minimum mismatches
    for room_id in room_order:
        if room_id != 11 and room_id not in new_id_to_old_id.values():
            possible_assignments = []
            for new_id in new_room_ids:
                if new_id not in assigned_new_ids:
                    temp_result = np.zeros_like(new_matrix)
                    temp_result[new_labeled == new_id] = room_id
                    mismatches = np.sum(temp_result != old_matrix)
                    possible_assignments.append((new_id, mismatches))
            
            if possible_assignments:
                best_new_id = min(possible_assignments, key=lambda x: x[1])[0]
                new_id_to_old_id[best_new_id] = room_id
                assigned_new_ids.add(best_new_id)

    # Step 4: Assign room 11 to the largest remaining unassigned region
    if 11 in old_room_ids:
        remaining_new_ids = set(new_room_ids) - assigned_new_ids
        if remaining_new_ids:
            largest_remaining_id = max(remaining_new_ids, key=lambda x: np.sum(new_labeled == x))
            new_id_to_old_id[largest_remaining_id] = 11

    # Create the result matrix
    result = np.zeros_like(new_matrix)
    for new_id, old_id in new_id_to_old_id.items():
        result[new_labeled == new_id] = old_id
    return result
                
    
#%% ###########################################################################
########################                                  #####################
###############################################################################
def assign_room_ids_max_iou_min_mismatch_permutations(old_matrix, new_matrix, agent_order):
    new_labeled = get_segmentation_map(new_matrix)
    new_room_ids = np.unique(new_labeled[new_labeled != 0])
    old_room_ids = np.unique(old_matrix[old_matrix > 0])

    # Calculate IoU matrix
    iou_matrix = np.zeros((len(new_room_ids), len(old_room_ids)))
    for i, new_id in enumerate(new_room_ids):
        for j, old_id in enumerate(old_room_ids):
            iou_matrix[i, j] = calculate_iou(new_labeled == new_id, old_matrix == old_id)

    new_id_to_old_id = {}
    assigned_new_ids = set()

    # Step 1: Assign rooms 2 to 5 if they exist
    for room_id in range(2, 6):
        if room_id in old_room_ids:
            best_new_id = None
            best_iou = 0
            for i, new_id in enumerate(new_room_ids):
                if new_id not in assigned_new_ids:
                    iou = iou_matrix[i][list(old_room_ids).index(room_id)]
                    if iou > best_iou:
                        best_iou = iou
                        best_new_id = new_id
            if best_new_id is not None:
                new_id_to_old_id[best_new_id] = room_id
                assigned_new_ids.add(best_new_id)

    # Step 2: Assign rooms with IoU = 1
    for i, new_id in enumerate(new_room_ids):
        if new_id not in assigned_new_ids:
            perfect_matches = np.where(iou_matrix[i] == 1)[0]
            if len(perfect_matches) == 1:
                old_id = old_room_ids[perfect_matches[0]]
                new_id_to_old_id[new_id] = old_id
                assigned_new_ids.add(new_id)

    # Step 3: Assign remaining rooms using combinatorial approach
    remaining_new_ids = [id for id in new_room_ids if id not in assigned_new_ids]
    remaining_old_ids = [id for id in old_room_ids if id not in new_id_to_old_id.values()]

    if len(remaining_new_ids) == len(remaining_old_ids):
        best_assignment = None
        min_mismatches = float('inf')

        for perm in permutations(remaining_old_ids):
            temp_result = np.zeros_like(new_matrix)
            for new_id, old_id in zip(remaining_new_ids, perm):
                temp_result[new_labeled == new_id] = old_id
            
            # Count mismatched cells
            mismatches = np.sum(temp_result != old_matrix)
            
            if mismatches < min_mismatches:
                min_mismatches = mismatches
                best_assignment = list(zip(remaining_new_ids, perm))

        if best_assignment:
            for new_id, old_id in best_assignment:
                new_id_to_old_id[new_id] = old_id

    # Create the result matrix
    result = np.zeros_like(new_matrix)
    for new_id, old_id in new_id_to_old_id.items():
        result[new_labeled == new_id] = old_id
    return result
        


def assign_room_ids(old_matrix, new_matrix, wall_base_matrix, agent_order):
    def calculate_iou(region1, region2):
        intersection = np.logical_and(region1, region2).sum()
        union = np.logical_or(region1, region2).sum()
        return intersection / union if union > 0 else 0

    def get_segmentation_map(matrix):
        labeled, _ = ndi.label(matrix == 0)
        return labeled

    def get_connected_regions(wall_id, labeled_matrix):
        wall_mask = wall_base_matrix == wall_id
        dilated_wall = ndi.binary_dilation(wall_mask)
        connected_regions = np.unique(labeled_matrix[dilated_wall])
        return [r for r in connected_regions if r != 0]  # Exclude background (0)

    new_labeled = get_segmentation_map(new_matrix)
    new_room_ids = np.unique(new_labeled[(new_labeled != 0) & (new_matrix != 10)])
    old_room_ids = np.unique(old_matrix[(old_matrix > 0) & (old_matrix != 10)])

    # Calculate IoU matrix
    iou_matrix = np.zeros((len(new_room_ids), len(old_room_ids)))
    for i, new_id in enumerate(new_room_ids):
        for j, old_id in enumerate(old_room_ids):
            iou_matrix[i, j] = calculate_iou(new_labeled == new_id, old_matrix == old_id)

    new_id_to_old_id = {}
    assigned_new_ids = set()

    # Step 1: Assign rooms 2 to 5 if they exist
    for room_id in range(2, 6):
        if room_id in old_room_ids:
            best_new_id = None
            best_iou = 0
            for i, new_id in enumerate(new_room_ids):
                if new_id not in assigned_new_ids:
                    iou = iou_matrix[i][list(old_room_ids).index(room_id)]
                    if iou > best_iou:
                        best_iou = iou
                        best_new_id = new_id
            if best_new_id is not None:
                new_id_to_old_id[best_new_id] = room_id
                assigned_new_ids.add(best_new_id)

    # Step 2: Assign rooms with IoU = 1
    for i, new_id in enumerate(new_room_ids):
        if new_id not in assigned_new_ids:
            perfect_matches = np.where(iou_matrix[i] == 1)[0]
            if len(perfect_matches) == 1:
                old_id = old_room_ids[perfect_matches[0]]
                new_id_to_old_id[new_id] = old_id
                assigned_new_ids.add(new_id)

    # Convert agent_order to room IDs
    room_order = [int(agent.split('_')[1]) for agent in agent_order if int(agent.split('_')[1]) != 11]

    # Step 3: Assign remaining rooms based on wall connections and max IoU
    for room_id in room_order:
        if room_id not in new_id_to_old_id.values():
            wall_id = -room_id  # Assuming wall IDs are negative room IDs
            connected_regions = get_connected_regions(wall_id, new_labeled)
            potential_regions = [r for r in connected_regions if r not in assigned_new_ids]
            
            if potential_regions:
                best_new_id = max(potential_regions, key=lambda r: iou_matrix[list(new_room_ids).index(r)][list(old_room_ids).index(room_id)])
                new_id_to_old_id[best_new_id] = room_id
                assigned_new_ids.add(best_new_id)

    # Step 4: Assign leftover region to room 11
    if 11 in old_room_ids:
        remaining_new_ids = set(new_room_ids) - assigned_new_ids
        if remaining_new_ids:
            new_id_to_old_id[remaining_new_ids.pop()] = 11

    # Create the result matrix
    result = np.zeros_like(new_matrix)
    for new_id, old_id in new_id_to_old_id.items():
        result[new_labeled == new_id] = old_id

    return result



#%% ###########################################################################
########################                                  #####################
###############################################################################
def assign_room_ids_max_iou_by_base_wall_nice_version(old_matrix, new_matrix, wall_base_matrix, agent_order):
    def calculate_iou(region1, region2):
        intersection = np.logical_and(region1, region2).sum()
        union = np.logical_or(region1, region2).sum()
        return intersection / union if union > 0 else 0

    def get_segmentation_map(matrix):
        labeled, _ = ndi.label(matrix == 0)
        return labeled

    def get_connected_regions(wall_id, labeled_matrix):
        wall_mask = wall_base_matrix == wall_id
        dilated_wall = ndi.binary_dilation(wall_mask)
        connected_regions = np.unique(labeled_matrix[dilated_wall])
        return [r for r in connected_regions if r != 0]  # Exclude background (0)

    new_labeled = get_segmentation_map(new_matrix)
    new_room_ids = np.unique(new_labeled[new_labeled != 0])
    old_room_ids = np.unique(old_matrix[old_matrix > 0])

    # Calculate IoU matrix
    iou_matrix = np.zeros((len(new_room_ids), len(old_room_ids)))
    for i, new_id in enumerate(new_room_ids):
        for j, old_id in enumerate(old_room_ids):
            iou_matrix[i, j] = calculate_iou(new_labeled == new_id, old_matrix == old_id)

    new_id_to_old_id = {}
    assigned_new_ids = set()

    # Step 1: Assign rooms 2 to 5 if they exist
    for room_id in range(2, 6):
        if room_id in old_room_ids:
            best_new_id = None
            best_iou = 0
            for i, new_id in enumerate(new_room_ids):
                if new_id not in assigned_new_ids:
                    iou = iou_matrix[i][list(old_room_ids).index(room_id)]
                    if iou > best_iou:
                        best_iou = iou
                        best_new_id = new_id
            if best_new_id is not None:
                new_id_to_old_id[best_new_id] = room_id
                assigned_new_ids.add(best_new_id)

    # Step 2: Assign rooms with IoU = 1
    for i, new_id in enumerate(new_room_ids):
        if new_id not in assigned_new_ids:
            perfect_matches = np.where(iou_matrix[i] == 1)[0]
            if len(perfect_matches) == 1:
                old_id = old_room_ids[perfect_matches[0]]
                new_id_to_old_id[new_id] = old_id
                assigned_new_ids.add(new_id)

    # Convert agent_order to room IDs
    room_order = [int(agent.split('_')[1]) for agent in agent_order if int(agent.split('_')[1]) != 11]

    # Step 3: Assign remaining rooms based on wall connections and max IoU
    affected_rooms = []
    for room_id in room_order:
        if room_id not in new_id_to_old_id.values():
            affected_rooms.append(room_id)
            wall_id = -room_id  # Assuming wall IDs are negative room IDs
            connected_regions = get_connected_regions(wall_id, new_labeled)
            potential_regions = [r for r in connected_regions if r not in assigned_new_ids]
            
            if potential_regions:
                best_new_id = max(potential_regions, key=lambda r: iou_matrix[list(new_room_ids).index(r)][list(old_room_ids).index(room_id)])
                new_id_to_old_id[best_new_id] = room_id
                assigned_new_ids.add(best_new_id)

    # Step 4: Assign leftover region to room 11
    if 11 in old_room_ids:
        remaining_new_ids = set(new_room_ids) - assigned_new_ids
        if remaining_new_ids:
            new_id_to_old_id[remaining_new_ids.pop()] = 11

    # Create the result matrix
    result = np.zeros_like(new_matrix)
    for new_id, old_id in new_id_to_old_id.items():
        result[new_labeled == new_id] = old_id

    return result, affected_rooms



#%% ###########################################################################
########################                                  #####################
###############################################################################
def assign_room_ids_max_iou_by_base_wall(old_matrix, new_matrix, wall_base_matrix, agent_order):
    def calculate_iou(region1, region2):
        intersection = np.logical_and(region1, region2).sum()
        union = np.logical_or(region1, region2).sum()
        return intersection / union if union > 0 else 0

    def get_segmentation_map(matrix):
        labeled, _ = ndi.label(matrix == 0)
        return labeled

    def get_connected_regions(mask, labeled_matrix):
        dilated_mask = ndi.binary_dilation(mask)
        connected_regions = np.unique(labeled_matrix[dilated_mask])
        return [r for r in connected_regions if r != 0]  # Exclude background (0)

    new_labeled = get_segmentation_map(new_matrix)
    new_room_ids = np.unique(new_labeled[new_labeled != 0])
    old_room_ids = np.unique(old_matrix[old_matrix > 0])

    # Calculate IoU matrix
    iou_matrix = np.zeros((len(new_room_ids), len(old_room_ids)))
    for i, new_id in enumerate(new_room_ids):
        for j, old_id in enumerate(old_room_ids):
            iou_matrix[i, j] = calculate_iou(new_labeled == new_id, old_matrix == old_id)

    new_id_to_old_id = {}
    assigned_new_ids = set()

    # Step 1: Assign rooms 2 to 5 if they exist
    for room_id in range(2, 6):
        if room_id in old_room_ids:
            best_new_id = None
            best_iou = 0
            for i, new_id in enumerate(new_room_ids):
                if new_id not in assigned_new_ids:
                    iou = iou_matrix[i][list(old_room_ids).index(room_id)]
                    if iou > best_iou:
                        best_iou = iou
                        best_new_id = new_id
            if best_new_id is not None:
                new_id_to_old_id[best_new_id] = room_id
                assigned_new_ids.add(best_new_id)

    # New Step: Assign living room (room 11) connected to the entrance
    entrance_mask = new_matrix == 10
    living_room_regions = get_connected_regions(entrance_mask, new_labeled)
    if living_room_regions and 11 in old_room_ids:
        # Filter out regions already assigned to rooms 2-5
        available_living_room_regions = [r for r in living_room_regions if r not in assigned_new_ids]
        if available_living_room_regions:
            living_room_id = available_living_room_regions[0]  # Take the first available connected region
            new_id_to_old_id[living_room_id] = 11
            assigned_new_ids.add(living_room_id)

    # Step 2: Assign rooms with IoU = 1
    for i, new_id in enumerate(new_room_ids):
        if new_id not in assigned_new_ids:
            perfect_matches = np.where(iou_matrix[i] == 1)[0]
            if len(perfect_matches) == 1:
                old_id = old_room_ids[perfect_matches[0]]
                new_id_to_old_id[new_id] = old_id
                assigned_new_ids.add(new_id)

    # Convert agent_order to room IDs
    room_order = [int(agent.split('_')[1]) for agent in agent_order if int(agent.split('_')[1]) != 11]

    # Step 3: Assign remaining rooms based on wall connections and max IoU
    affected_rooms = []
    for room_id in room_order:
        if room_id not in new_id_to_old_id.values():
            affected_rooms.append(room_id)
            wall_id = -room_id  # Assuming wall IDs are negative room IDs
            connected_regions = get_connected_regions(wall_base_matrix == wall_id, new_labeled)
            potential_regions = [r for r in connected_regions if r not in assigned_new_ids]
            
            if potential_regions:
                best_new_id = max(potential_regions, key=lambda r: iou_matrix[list(new_room_ids).index(r)][list(old_room_ids).index(room_id)])
                new_id_to_old_id[best_new_id] = room_id
                assigned_new_ids.add(best_new_id)

    # Create the result matrix
    result = np.zeros_like(new_matrix)
    for new_id, old_id in new_id_to_old_id.items():
        result[new_labeled == new_id] = old_id

    # Preserve entrance
    result[new_matrix == 10] = 10

    return result, affected_rooms



#%% ###########################################################################
########################                                  #####################
###############################################################################
def assign_room_ids_for_identity_less_walls(old_matrix, new_matrix, desired_areas=[435, 228, 190, 178, 150, 146, 140, 140, 134]):
    def calculate_iou(region1, region2):
        intersection = np.logical_and(region1, region2).sum()
        union = np.logical_or(region1, region2).sum()
        return intersection / union if union > 0 else 0

    def get_segmentation_map(matrix):
        labeled, _ = ndi.label(matrix == 0)
        return labeled

    def get_connected_regions(mask, labeled_matrix):
        dilated_mask = ndi.binary_dilation(mask)
        connected_regions = np.unique(labeled_matrix[dilated_mask])
        return [r for r in connected_regions if r != 0]  # Exclude background (0)

    new_labeled = get_segmentation_map(new_matrix)
    new_room_ids = np.unique(new_labeled[new_labeled != 0])
    old_room_ids = np.unique(old_matrix[old_matrix > 0])

    # Calculate IoU matrix
    iou_matrix = np.zeros((len(new_room_ids), len(old_room_ids)))
    for i, new_id in enumerate(new_room_ids):
        for j, old_id in enumerate(old_room_ids):
            iou_matrix[i, j] = calculate_iou(new_labeled == new_id, old_matrix == old_id)

    new_id_to_old_id = {}
    assigned_new_ids = set()
    affected_rooms = []

    # Step 1: Assign rooms 2 to 5 if they exist
    for room_id in range(2, 6):
        if room_id in old_room_ids:
            best_new_id = None
            best_iou = 0
            for i, new_id in enumerate(new_room_ids):
                if new_id not in assigned_new_ids:
                    iou = iou_matrix[i][list(old_room_ids).index(room_id)]
                    if iou > best_iou:
                        best_iou = iou
                        best_new_id = new_id
            if best_new_id is not None:
                new_id_to_old_id[best_new_id] = room_id
                assigned_new_ids.add(best_new_id)
                affected_rooms.append(room_id)

    # New Step: Assign living room (room 11) connected to the entrance
    entrance_mask = new_matrix == 10
    living_room_regions = get_connected_regions(entrance_mask, new_labeled)
    if living_room_regions and 11 in old_room_ids:
        available_living_room_regions = [r for r in living_room_regions if r not in assigned_new_ids]
        if available_living_room_regions:
            living_room_id = available_living_room_regions[0]
            new_id_to_old_id[living_room_id] = 11
            assigned_new_ids.add(living_room_id)
            affected_rooms.append(11)

    # Step 3: Assign remaining rooms based on area matching
    remaining_new_ids = set(new_room_ids) - assigned_new_ids
    remaining_room_ids = set(range(12, 20))  # Rooms 12 to 19
    remaining_desired_areas = desired_areas[1:]  # Exclude living room area

    # Calculate areas of remaining regions
    new_areas = {new_id: np.sum(new_labeled == new_id) for new_id in remaining_new_ids}

    for desired_area in remaining_desired_areas:
        if not remaining_new_ids:
            break
        best_new_id = min(remaining_new_ids, key=lambda x: abs(new_areas[x] - desired_area))
        best_room_id = min(remaining_room_ids)
        new_id_to_old_id[best_new_id] = best_room_id
        assigned_new_ids.add(best_new_id)
        remaining_new_ids.remove(best_new_id)
        remaining_room_ids.remove(best_room_id)
        affected_rooms.append(best_room_id)

    # Create the result matrix
    result = np.zeros_like(new_matrix)
    for new_id, old_id in new_id_to_old_id.items():
        result[new_labeled == new_id] = old_id

    # Preserve entrance
    result[new_matrix == 10] = 10

    return result, affected_rooms
    