#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 13:58:10 2023

@author: Reza Kakooee
"""

# import torch
# from vqvae.vqvae_model import Model
# from vqvae.vqvae_config import VqVaeConfig


import copy
import numpy as np
# from scipy import ndimage
from scipy.spatial.distance import cdist


#%%
class CnnStateObserver:
    def __init__(self, fenv_config):
        self.fenv_config = fenv_config
        
        if self.fenv_config['encode_img_obs_by_vqvae_flag']:
            self.device = 'cpu'
            self.vqvae_config = VqVaeConfig()
            self.vqvae_model = Model(self.vqvae_config.num_hiddens, self.vqvae_config.num_residual_layers, self.vqvae_config.num_residual_hiddens,
                                     self.vqvae_config.num_embeddings, self.vqvae_config.embedding_dim, 
                                     self.vqvae_config.commitment_cost, self.vqvae_config.decay).to(self.device)
            self.vqvae_model.eval()
        
    
    
    def get_observation(self, plan_data_dict):
        obs_canvas_mat, obs_canvas_arr_1ch = self._get_canvas_cnn(plan_data_dict)
        obs_rooms_cmap, obs_rooms_cmap_1ch = self._get_rooms_color_map(plan_data_dict)
        obs_blocked_cells, obs_blocked_cells_1ch = self._get_obs_blocked_cells(plan_data_dict)
        
        if self.fenv_config['cnn_observation_name'] == 'canvas_1d':
            observation_cnn = obs_canvas_arr_1ch
        elif self.fenv_config['cnn_observation_name'] == 'rooms_cmap':
            observation_cnn = obs_rooms_cmap_1ch
        elif self.fenv_config['cnn_observation_name'] == 'stacked_3d':
             observation_cnn = np.concatenate((obs_canvas_arr_1ch, obs_rooms_cmap_1ch, obs_blocked_cells_1ch), axis=-1)
        else:
            raise ValueError("Invalid cnn_observation_name. The current one is: {self.fenv_config['cnn_observation_name']}")
        
        if self.fenv_config['distance_field_flag']:
            # observation_cnn = self._get_distance_field(plan_data_dict, observation_cnn)
            observation_cnn = self._add_distance_feature_mape_to_obs(plan_data_dict, observation_cnn)
        
        if self.fenv_config['non_squared_plan_flag']:
            obs_background_canvas = np.zeros((self.fenv_config['min_y_background'], self.fenv_config['max_x_background']))
            def center_overlay(observation_cnn, background):
                if observation_cnn.shape[0] > obs_background_canvas.shape[0] or observation_cnn.shape[1] > obs_background_canvas.shape[1]:
                    raise ValueError("observation_cnn array must be smaller than obs_background_canvas array in both dimensions.")

                center_y, center_x = obs_background_canvas.shape[0] // 2, obs_background_canvas.shape[1] // 2
                start_y = center_y - observation_cnn.shape[0] // 2
                start_x = center_x - observation_cnn.shape[1] // 2
                end_y, end_x = start_y + observation_cnn.shape[0], start_x + observation_cnn.shape[1]
                obs_background_canvas[start_y:end_y, start_x:end_x] = observation_cnn

                return obs_background_canvas
            observation_cnn = center_overlay(observation_cnn, obs_background_canvas)
            
        
        plan_data_dict.update({
            'obs_canvas_mat': obs_canvas_mat,
            'obs_canvas_arr_1ch': obs_canvas_arr_1ch,
            
            'obs_rooms_cmap': obs_rooms_cmap,
            'obs_rooms_cmap_1ch': obs_rooms_cmap_1ch,
            
            'obs_blocked_cells': obs_blocked_cells,
            'obs_blocked_cells_1ch': obs_blocked_cells_1ch,
            
            # 'obs_canvas_arr_3ch': obs_canvas_arr_3ch,
            
            'observation_cnn': observation_cnn,
            })
        
        if self.fenv_config['encode_img_obs_by_vqvae_flag']:
            with torch.no_grad():
                observation_latent_fc = (self.vqvae_model._encoder(torch.tensor(np.expand_dims(np.transpose(observation_cnn, (2, 0, 1)), axis=0)).float())).cpu().numpy()
            observation_latent_fc = (np.mean(observation_latent_fc, axis=(2, 3))).squeeze()
            plan_data_dict.update({'observation_latent_fc': observation_latent_fc})
            
        return plan_data_dict
    
    

    def _add_distance_feature_mape_to_obs(self, plan_data_dict, observation_cnn):
        img = plan_data_dict['obs_moving_labels_completed']
        img[0,:] = 0
        img[-1,:] = 0
        img[:,0] = 0
        img[:,-1] = 0

        def find_nearest_room_pixel(centroid, room_coords):
            distances = cdist([centroid], room_coords)
            nearest_index = np.argmin(distances)
            return room_coords[nearest_index]
        
        room_centroids = {}
        distance_maps = {}
        for room_id in np.unique(img):
            if room_id in [0, self.fenv_config['entrance_cell_id']]:  # Skip background and entrance
                continue
            room_coords = np.column_stack(np.where(img == room_id))
            centroid = np.mean(room_coords, axis=0)
            if not any(np.all(room_coords == centroid, axis=1)):
                centroid = find_nearest_room_pixel(centroid, room_coords)
            room_centroids[room_id] = centroid
            distances = np.linalg.norm(room_coords - centroid, axis=1)
            distance_map = np.full(img.shape, np.inf)
            distance_map[room_coords[:,0], room_coords[:,1]] = distances
            distance_maps[room_id] = distance_map
        combined_feature_map = np.full(img.shape, np.inf)

        for room_id, distance_map in distance_maps.items():
            room_mask = img == room_id
            combined_feature_map[room_mask] = distance_map[room_mask]
        non_room_areas = (img == 0) | (img == self.fenv_config['entrance_cell_id'])
        combined_feature_map[non_room_areas] = 0  # Set to 0 or another value for non-room areas

        outline_mask = img == 0
        outline_feature_map = combined_feature_map.copy()  # Start with the combined feature map
        for y, x in np.column_stack(np.where(outline_mask)):
            pixel = np.array([y, x])
            nearest_distance = np.inf
            for centroid in room_centroids.values():
                distance = np.linalg.norm(pixel - centroid)
                if distance < nearest_distance:
                    nearest_distance = distance
            outline_feature_map[y, x] = nearest_distance
        
        combined_feature_map[outline_mask] = outline_feature_map[outline_mask]

        if 11 in room_centroids:  # Check if living room centroid is calculated
            entrance_mask = img == self.fenv_config['entrance_cell_id']
            lvroom_centroid = room_centroids[self.fenv_config['lvroom_id']]
            entrance_coords = np.column_stack(np.where(entrance_mask))
            for i, (y, x) in enumerate(entrance_coords):
                combined_feature_map[y, x] = np.linalg.norm([y, x] - lvroom_centroid)
    
        observation_cnn = np.concatenate((observation_cnn, np.expand_dims(combined_feature_map, axis=-1)), axis=-1)
        return observation_cnn



    def _get_canvas_cnn(self, plan_data_dict):
        obs_canvas_mat = copy.deepcopy(plan_data_dict['obs_moving_labels_completed_projected']) 
            
        obs_canvas_mat_kroned = np.kron(obs_canvas_mat, 
                                    np.ones((self.fenv_config['cnn_scaling_factor'], self.fenv_config['cnn_scaling_factor']), dtype=obs_canvas_mat.dtype))
        
        obs_canvas_arr_1ch = np.expand_dims(obs_canvas_mat_kroned, axis=2) # obs_canvas_arr_1ch = np.expand_dims(obs_canvas_mat_kroned, axis=0) #  TODO
        obs_canvas_arr_1ch = obs_canvas_arr_1ch * self.fenv_config['cnn_obs_normalization_factor']
        # obs_canvas_arr_1ch = obs_canvas_arr_1ch.astype(np.uint8)
        
        return obs_canvas_mat, obs_canvas_arr_1ch
    
    
    
    def _get_rooms_color_map(self, plan_data_dict):
        obw = copy.deepcopy(plan_data_dict['obs_mat_w'])
        obw[obw == -self.fenv_config['north_facade_id']] = 0
        obw[obw == -self.fenv_config['south_facade_id']] = 0
        obw[obw == -self.fenv_config['east_facade_id']] = 0
        obw[obw == -self.fenv_config['west_facade_id']] = 0
        
        obs_rooms_cmap = plan_data_dict['obs_moving_labels'] - obw
        
        for r, c in plan_data_dict['extended_entrance_positions'][2:]:
            obs_rooms_cmap[r][c] = self.fenv_config['entrance_cell_id']
            
        obs_rooms_cmap_kroned = np.kron(obs_rooms_cmap, 
                                    np.ones((self.fenv_config['cnn_scaling_factor'], self.fenv_config['cnn_scaling_factor']), dtype=obs_rooms_cmap.dtype))
        
        obs_rooms_cmap_1ch = np.expand_dims(obs_rooms_cmap_kroned, axis=2) # obs_rooms_cmap_1ch = np.expand_dims(obs_rooms_cmap_kroned, axis=0) #  TODO
        obs_rooms_cmap_1ch = obs_rooms_cmap_1ch * self.fenv_config['cnn_obs_normalization_factor']
        # obs_rooms_cmap_1ch = obs_rooms_cmap_1ch.astype(np.uint8)
        
        return obs_rooms_cmap, obs_rooms_cmap_1ch
    
    
    
    def _get_obs_blocked_cells(self, plan_data_dict):
        obs_blocked_cells = copy.deepcopy(plan_data_dict['obs_blocked_cells_by_shift'])
        for r, c in plan_data_dict['extended_entrance_positions']:
            obs_blocked_cells[r][c] = 1
            
        obs_blocked_cells_kroned = np.kron(obs_blocked_cells, 
                                       np.ones((self.fenv_config['cnn_scaling_factor'], self.fenv_config['cnn_scaling_factor']), dtype=obs_blocked_cells.dtype))
        
        obs_blocked_cells_1ch = np.expand_dims(obs_blocked_cells, axis=2) #obs_blocked_cells_1ch = np.expand_dims(obs_blocked_cells, axis=0) #  TODO
        # obs_blocked_cells_1ch = obs_blocked_cells_1ch.astype(np.uint8)
        return obs_blocked_cells_kroned, obs_blocked_cells_1ch    
