#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 17:43:42 2022

@author: RK
"""
#%%
import copy
import itertools
import numpy as np
from webcolors import name_to_rgb

import torch
from torchvision.transforms import transforms

from torch_intermediate_layer_getter import IntermediateLayerGetter as MidGetter


# %%
class ToTensor(object):
    def __call__(self, image):
        image = image.transpose((2, 0, 1))
        transformed_sample = torch.from_numpy(image)
        return transformed_sample

transform = transforms.Compose([
                            ToTensor(), 
                            # Normalize(),
                            ])
    

#%%
class StateComposer:
    def __init__(self, fenv_config):
        self.fenv_config = fenv_config
        
        if (self.fenv_config['net_arch'] == 'CnnGcn') and (self.fenv_config['gcn_obs_method'] == 'embedded_image_graph'):
            from gym_floorplan.envs.observation.context_model import ContextModel
            net_config = {
                    '_in_channels_cnn': 2,
                    'dim_pd_features': 4,
                    'dim_pc_features': 4,
                    'dim_fc_features': 18,
                    }
            
            self.device = 'cpu' # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = ContextModel(net_config)
            self.model = torch.load(self.fenv_config['context_mdoel_path'], map_location ='cpu')
            self.model = self.model.to(self.device)
        
    def creat_observation_space_variables(self):
        num_walls_for_net = num_rooms_for_net = self.fenv_config['num_of_fixed_walls_for_masked']
        
        # num_coords_points = num_walls_for_net * 5 * 2
        # num_wall_to_wall_distances = (num_walls_for_net * (num_walls_for_net - 1))/2 * 25
        # num_wall_to_corner_distances = num_walls_for_net * 5 * 4
        # len_state_vec = len_state_vec_for_walls = int(num_coords_points + num_wall_to_wall_distances + num_wall_to_corner_distances)
                
        len_state_vec = 0
        len_state_vec_for_walls = num_walls_for_net * 5 * 2  # 5=n_wall_points, 2=xy # x y coordinates of the walls for 5 important points
        len_state_vec_for_walls += num_walls_for_net * 5 * 4  # 4=n_corners # distance between each of 5 wall's important poitns and 4 corners
        len_state_vec_for_walls += np.cumsum([i * 5 * 5 for i in range(1, num_walls_for_net)])[-1]  # 5=n_wall_points # distane between 5 important points of walls
        
        len_state_vec += len_state_vec_for_walls
        
        if self.fenv_config['use_areas_info_into_observation_flag']:
            len_state_vec_for_rooms = self.fenv_config['maximum_num_real_rooms'] * 2 # 2 id delta_area excluded else 3  # 3=desired_areas, acheived_areas , ### delta_areas
            len_state_vec += len_state_vec_for_rooms
            
                
        if self.fenv_config['use_edge_info_into_observation_flag']:
            len_state_vec_for_edges = int(self.fenv_config['maximum_num_real_rooms'] * (self.fenv_config['maximum_num_real_rooms'] - 1) / 2) * 2 # multiply by 2, becuase we want to have both desired and currend adj sub mat
            len_state_vec += len_state_vec_for_edges
    

        if self.fenv_config['env_planning'] == 'One_Shot': # we need the last list as it contains 'fc'
            if self.fenv_config['use_areas_info_into_observation_flag']:
                lowest_obs_val_fc = -self.fenv_config['total_area']
                highest_obs_val_fc = self.fenv_config['total_area']
            else:
                lowest_obs_val_fc = -1
                highest_obs_val_fc = 30


        assert len_state_vec_for_walls == self.fenv_config['len_state_vec_for_walls'], "These two should be equal!"
        assert len_state_vec_for_rooms == self.fenv_config['len_state_vec_for_rooms'], "These two should be equal!"
        assert len_state_vec_for_edges == self.fenv_config['len_state_vec_for_edges'], "These two should be equal!"
        assert len_state_vec == self.fenv_config['len_state_vec'], "These two should be equal!"
        


        low_fc = lowest_obs_val_fc #* np.ones(len_state_vec, dtype=float)
        high_fc = highest_obs_val_fc #* np.ones(len_state_vec, dtype=float)
        shape_fc = (len_state_vec,)
        
        low_cnn = 0
        high_cnn = 255
        shape_cnn = (self.fenv_config['n_rows']+2 * self.fenv_config['scaling_factor'],
                     self.fenv_config['n_cols']+2 * self.fenv_config['scaling_factor'],
                     self.fenv_config['n_channels'])
        
        low_gcn = -500
        high_gcn = 500
        shape_gcn = (256,)
        
        state_data_dict = {
            'low_fc': low_fc,
            'high_fc': high_fc,
            'shape_fc': shape_fc,
            
            'low_cnn': low_cnn,
            'high_cnn': high_cnn,
            'shape_cnn': shape_cnn,
            
            'low_gcn': low_gcn,
            'high_gcn': high_gcn,
            'shape_gcn': shape_gcn,
            
            'len_state_vec_for_walls': len_state_vec_for_walls,
            'len_state_vec_for_rooms': len_state_vec_for_rooms,
            'len_state_vec_for_edges': len_state_vec_for_edges,
            
            'len_state_vec_for_walls_rooms': len_state_vec_for_walls + len_state_vec_for_rooms,
            # 'len_state_vec_for_walls_rooms_edges': len_state_vec_for_walls + len_state_vec_for_rooms + len_state_vec_for_edges,
            'len_state_vec': len_state_vec,
            }
        
        return state_data_dict
    
        
    def _distance(self, coord_s, coord_e):
        return np.linalg.norm(np.array(coord_s)-np.array(coord_e))
        
    
    def wall_data_extractor_for_single_agent(self, plan_data_dict, active_wall_name=None):
        wall_important_points_dict = self._add_wall_important_points(plan_data_dict)
        if self.fenv_config['mask_flag']:
            if self.fenv_config['net_arch'] not in ['Fc', 'MetaFc']:
                wall_names = list(wall_important_points_dict.keys())
                wall_names = wall_names[plan_data_dict['mask_numbers']:]
                wall_important_points_dict = {wall_name: wall_important_points_dict[wall_name] for wall_name in wall_names}       
        
        walls_to_corners_distance_dict = self._add_walls_to_corners_distance(plan_data_dict, wall_important_points_dict)
        
        if self.fenv_config['net_arch'] == 'CnnGcn':
            if active_wall_name is not None:
                active_wall_data_dict = wall_important_points_dict[active_wall_name]
                active_wall_data_dict.update({'distances': walls_to_corners_distance_dict[active_wall_name]})
            else:
                active_wall_data_dict = None # meaning that there is no wall; meaning that we are just starting the episode; meaning that we are reseting  he episode.
                
            return None, active_wall_data_dict
            
        
        else:
            walls_to_walls_distance_dict = self._add_walls_to_walls_distance(plan_data_dict, wall_important_points_dict)
            
            
            wall_important_coords = [coord for point in wall_important_points_dict.values() for coord in point.values()]
            walls_to_corners_distance = [ds for ds in walls_to_corners_distance_dict.values()]
            walls_to_walls_distance = [ds for ds in walls_to_walls_distance_dict.values()]
            
            wall_important_coords_vec = list(itertools.chain.from_iterable(wall_important_coords))
            walls_to_corners_distance_vec = list(itertools.chain.from_iterable(walls_to_corners_distance))
            walls_to_walls_distance_vec = list(itertools.chain.from_iterable(walls_to_walls_distance))
            
            state_vec = wall_important_coords_vec + walls_to_corners_distance_vec + walls_to_walls_distance_vec
        
        
            return np.array(state_vec), None
        
    
    def _add_wall_important_points(self, plan_data_dict):
        important_points = ['start_of_wall', 'before_anchor', 'anchor', 'after_anchor', 'end_of_wall']
        n_walls_plus1 = plan_data_dict['number_of_total_walls'] + 1
        
        wall_coords_template = {k: [-1, -1] for k in important_points}
        wall_important_points_dict = {f'wall_{i}': copy.deepcopy(wall_coords_template) for i in range(1, n_walls_plus1+1)} # {wall_name: copy.deepcopy(wall_coords_template) for wall_name in plan_data_dict['walls_coords'].keys()} # 
        
        for wall_name, wall_data in plan_data_dict['walls_coords'].items():
            try:
                back_segment = wall_data['back_segment']
                front_segment = wall_data['front_segment']
                wall_important_points_dict[wall_name].update({ 'start_of_wall': list(back_segment['reflection_coord']),
                                                               'before_anchor': list(back_segment['end_coord']),
                                                               'anchor':        list(back_segment['start_coord']),
                                                               'after_anchor':  list(front_segment['end_coord']),
                                                               'end_of_wall':   list(front_segment['reflection_coord']) })
            except:
                print("wait in _wall_data_extractor_for_single_agent of observation")
                raise ValueError('Probably number of walls does not match with the number of rooms')
        
        if 'wall_1' not in plan_data_dict['walls_coords'].keys():
            for key in wall_important_points_dict['wall_1'].keys():
                wall_important_points_dict['wall_1'][key] = [0, 0]
            
        if 'wall_2' not in plan_data_dict['walls_coords'].keys():
            for key in wall_important_points_dict['wall_2'].keys():
                wall_important_points_dict['wall_2'][key] = [self.fenv_config['max_x'], 0]
            
        if 'wall_3' not in plan_data_dict['walls_coords'].keys():
            for key in wall_important_points_dict['wall_3'].keys():
                wall_important_points_dict['wall_3'][key] = [0, self.fenv_config['max_y']]
            
        if 'wall_4' not in plan_data_dict['walls_coords'].keys():
            for key in wall_important_points_dict['wall_4'].keys():
                wall_important_points_dict['wall_4'][key] = [self.fenv_config['max_x'], self.fenv_config['max_x']]
        
        return wall_important_points_dict


    def _add_walls_to_corners_distance(self, plan_data_dict, wall_important_points_dict):
        if self.fenv_config['net_arch'] == 'CnnGcn':
            important_points = ['anchor']
        else:
            important_points = ['start_of_wall', 'before_anchor', 'anchor', 'after_anchor', 'end_of_wall']
        # 5 wall's important points to 4 corners
        walls_to_corners_distance_dict = {wall_name: list(-1*np.ones((5*4),dtype=float)) for wall_name in wall_important_points_dict.keys()}
        
        for wall_name in walls_to_corners_distance_dict.keys(): # 3
            wall_to_corner_dist_vec = []
            for point_name in important_points: #5
                for corner_name, corner_coord in self.fenv_config['corners'].items(): # 4
                    wall_coord = wall_important_points_dict[wall_name][point_name]
                    if -1 in wall_coord:
                        d = -1
                    else:
                        d = self._distance(wall_coord, corner_coord)
                    wall_to_corner_dist_vec.append(d)
        
            walls_to_corners_distance_dict[wall_name] = copy.deepcopy(wall_to_corner_dist_vec)
        return walls_to_corners_distance_dict
        
    
    def _add_walls_to_walls_distance(self, plan_data_dict, wall_important_points_dict):
        important_points = ['start_of_wall', 'before_anchor', 'anchor', 'after_anchor', 'end_of_wall']
        walls_to_walls_distance_dict = {}
        for wall_name_s in wall_important_points_dict.keys():
            wall_s_i = int(wall_name_s.split('_')[-1])
            for wall_name_e in wall_important_points_dict.keys():
                wall_e_i = int(wall_name_e.split('_')[-1])
                if wall_e_i > wall_s_i:
                    wall_to_wall_dist_vec = []
                    for point_name_s in important_points:
                        for point_name_e in important_points:
                            s_coord = wall_important_points_dict[wall_name_s][point_name_s]
                            e_coord = wall_important_points_dict[wall_name_e][point_name_e]
                            if (-1 in s_coord) or (-1 in e_coord):
                                d = -1
                            else:
                                d = self._distance(s_coord, e_coord)
                            wall_to_wall_dist_vec.append(d)      
                    walls_to_walls_distance_dict[f"{wall_name_s}_to_{wall_name_e}"] = copy.deepcopy(wall_to_wall_dist_vec)
        return walls_to_walls_distance_dict
        
        
    def _distance_calculator(self, wall_state_dict):
        wall_to_wall_dist_vec = []
        wall_to_wall_dist_dict = {}
        walls_name = list(wall_state_dict.keys())
        points_name = list(wall_state_dict[walls_name[0]].keys())
        n_points = len(points_name)
        for i in range(len(walls_name)):
            wall_name_s = walls_name[i]
            for j in range(i+1, len(walls_name)):
                wall_name_e = walls_name[j]
                for point_name_s in points_name:
                    for point_name_e in points_name:
                        d = self._distance(wall_state_dict[wall_name_s][point_name_s],
                                           wall_state_dict[wall_name_e][point_name_e])
                        wall_to_wall_dist_dict[f"{wall_name_s}_{wall_name_e}_{point_name_s}_{point_name_e}"] = d
                        wall_to_wall_dist_vec.append(d)       
        
        wall_to_corner_dist_vec = []
        wall_to_corner_dist_dict = {}
        for wall_name in walls_name:
            for point_name in points_name:
                for corner_name, corner_coord in self.fenv_config['corners'].items():
                    d = self._distance(wall_state_dict[wall_name][point_name], corner_coord)
                    wall_to_corner_dist_dict[f"{wall_name}_{point_name}_{corner_name}"] = d
                    wall_to_corner_dist_vec.append(d)
        
        all_dist_values = wall_to_wall_dist_vec + wall_to_corner_dist_vec
        return all_dist_values
    
    
    def _wall_data_extractor(self, plan_data_dict):
        walls_state_vec = []
        wall_state_dict = {}
        for i in range(1, plan_data_dict['number_of_total_walls']+1):
            wall_name = f"wall_{i}"
            back_segment = plan_data_dict['walls_coords'][wall_name]['back_segment']
            front_segment = plan_data_dict['walls_coords'][wall_name]['front_segment']
            this_wall_state = {'start_of_wall': list(back_segment['reflection_coord']),
                               'before_anchor': list(back_segment['end_coord']),
                               'anchor':        list(back_segment['start_coord']),
                               'after_anchor':  list(front_segment['end_coord']),
                               'end_of_wall':   list(front_segment['reflection_coord'])}
            
            walls_state_vec.extend(list(this_wall_state.values()))
            wall_state_dict[wall_name] = this_wall_state
        
        walls_state_vec = list(itertools.chain.from_iterable(walls_state_vec))
        
        all_dist_values = self._distance_calculator(wall_state_dict)
        
        state_vec = walls_state_vec + all_dist_values
        
        return np.array(state_vec)
    
    
    def _get_obs_arr_for_conv(self, observation_matrix):
        obs_arr_conv = np.zeros((self.fenv_config['n_rows']*self.fenv_config['scaling_factor'], 
                                 self.fenv_config['n_cols']*self.fenv_config['scaling_factor'], 
                                 self.fenv_config['n_channels']), dtype=np.uint8)
        obs_arr_for_conv = np.zeros_like(obs_arr_conv) 
        obs_mat_scaled = np.kron(observation_matrix,
                                  np.ones((self.fenv_config['scaling_factor'], self.fenv_config['scaling_factor']), 
                                          dtype=observation_matrix.dtype))
        for r in range(self.fenv_config['n_rows']*self.fenv_config['scaling_factor']):
            for c in range(self.fenv_config['n_cols']*self.fenv_config['scaling_factor']):
                for v in list(self.fenv_config['color_map'].keys()): #[:self.fenv_config['n_rooms']+2]:
                    if obs_mat_scaled[r,c] == v:
                        obs_arr_for_conv[r,c,:] = list(  name_to_rgb(self.fenv_config['color_map'][v])  )
        return obs_arr_for_conv
    
    
    def update_block_cells(self, plan_data_dict):
        walls_coords = plan_data_dict['walls_coords']
        extended_walls_positions = []
        for wall_name, wall_val in walls_coords.items():
            front_segment_direction = wall_val['front_segment']['direction']
            back_segment_direction = wall_val['back_segment']['direction']
            wall_positions = np.array(wall_val['wall_positions'])
            anchor_coord = wall_val['anchor_coord']
            anchor_position = self._cartesian2image_coord(anchor_coord[0], anchor_coord[1], self.fenv_config['max_y'])
            
            extended_walls_positions += [list(wpos) for wpos in wall_positions]
            
            for segment_direction in [front_segment_direction, back_segment_direction]:
                if  segment_direction in ['east', 'west']:
                    horizental_wall_positions = np.squeeze(wall_positions[np.squeeze(np.argwhere(wall_positions[:,0] == anchor_position[0]))])
                    upper_row = [[wpos[0]+1, wpos[1]] for wpos in horizental_wall_positions]
                    lower_row = [[wpos[0]-1, wpos[1]] for wpos in horizental_wall_positions]
                    extended_walls_positions += upper_row
                    extended_walls_positions += lower_row
                    
                else:
                    vertical_wall_positions = np.squeeze(wall_positions[np.squeeze(np.argwhere(wall_positions[:,1] == anchor_position[1]))])
                    righert_col = [[wpos[0], wpos[1]+1] for wpos in vertical_wall_positions]
                    lefter_col = [[wpos[0], wpos[1]-1] for wpos in vertical_wall_positions]
                    extended_walls_positions += righert_col
                    extended_walls_positions += lefter_col
            
            
        obs_blocked_cells = copy.deepcopy(plan_data_dict['obs_blocked_cells'])
        for pos in extended_walls_positions:
            obs_blocked_cells[pos[0]][[pos[1]]] = 1
        obs_blocked_cells = np.clip(obs_blocked_cells + plan_data_dict['moving_ones'], 0, 1)
        plan_data_dict['obs_blocked_cells'] = copy.deepcopy(obs_blocked_cells)
        return plan_data_dict
    
    
    def get_embeded_observation(self, context):
        pd_graph, pc_graph_old, fc_graph_old = self._convert_to_graph(context['graph_data_numpy_old'])
        
        pd_graph, pc_graph, fc_graph = self._convert_to_graph(context['graph_data_numpy'])
        
        image_old = context['plan_canvas_arr_old']
        image = context['plan_canvas_arr']
        image = np.concatenate((image_old, image), axis=2)
        image = np.kron(image, np.ones((4, 4, 1)))
        image = image.astype(np.uint8)
        image = transform(image)
        image = torch.unsqueeze(image, dim=0)
        
        image = image.to(self.device)
        pd_graph = pd_graph.to(self.device)
        pc_graph_old = pc_graph_old.to(self.device)
        fc_graph_old = fc_graph_old.to(self.device)
        pc_graph = pc_graph.to(self.device)
        fc_graph = fc_graph.to(self.device)
        
        inputs = {'image': image.float(), 
                  'pd_graph': pd_graph, 
                  'pc_graph_old': pc_graph_old, 
                  'fc_graph_old': fc_graph_old,
                  'pc_graph': pc_graph, 
                  'fc_graph': fc_graph}
        
        emb = self._get_embeding(inputs)
        return emb
        
    
    def _get_embeding(self, inputs):
        self.model.eval()
        with torch.no_grad():
            return_layers = {
                '_reward_head.0': 'emb',
            }
            mid_getter = MidGetter(self.model, return_layers=return_layers, keep_output=True)
            mid_outputs, model_output = mid_getter(inputs)
        emb = mid_outputs['emb'].detach().cpu().numpy().flatten()
        return emb
    
    
    def _convert_to_graph(self, graph_data_numpy):
        def __shift_edge_by_one(e):
            e = np.array(e, dtype=np.int32)-1
            return e
        
        from torch_geometric.data import Data as gData
        
        graph_features_numpy = graph_data_numpy['graph_features_numpy']
        graph_edge_list_numpy = graph_data_numpy['graph_edge_list_numpy']

        partially_desired_graph_features_dict_numpy = graph_features_numpy['partially_desired_graph_features_dict_numpy']
        partially_current_graph_features_dict_numpy = graph_features_numpy['partially_current_graph_features_dict_numpy']
        fully_current_graph_features_dict_numpy = graph_features_numpy['fully_current_graph_features_dict_numpy']

        pd_edge_list = __shift_edge_by_one(graph_edge_list_numpy['partially_desired_graph_edge_list_numpy'])
        pc_edge_list = __shift_edge_by_one(graph_edge_list_numpy['partially_current_graph_edge_list_numpy'])
        fc_edge_list = __shift_edge_by_one(graph_edge_list_numpy['fully_current_graph_edge_list_numpy'])
        
        if len(pc_edge_list) == 0:
            pc_edge_list = [[0, 0]]
        if len(fc_edge_list) == 0:
            fc_edge_list = [[0, 0]]
                
        room_names = partially_desired_graph_features_dict_numpy.keys()
        dim_pd_features = dim_pc_features = 4
        dim_fc_features = 18
        n_nodes = n_rooms = len(room_names)
        pd_features = np.zeros((n_nodes, dim_pd_features))
        pc_features = np.zeros((n_nodes, dim_pc_features))
        fc_features = np.zeros((n_nodes, dim_fc_features))
        for i, room_name in enumerate(room_names):
            pd_features[i, :] = list(itertools.chain(list(partially_desired_graph_features_dict_numpy[room_name].values())))
            pc_features[i, :] = list(itertools.chain(list(partially_current_graph_features_dict_numpy[room_name].values())))
            fc_feats = fully_current_graph_features_dict_numpy[room_name].values()
            fc_feat_vec = []
            for fs in fc_feats:
                if isinstance(fs, list):
                    for f in fs:
                        fc_feat_vec.append(f)
                else:
                    fc_feat_vec.append(fs)
            fc_features[i, :] = fc_feat_vec
            
        pd_edge_index = torch.tensor(pd_edge_list, dtype=torch.long)
        pc_edge_index = torch.tensor(pc_edge_list, dtype=torch.long)
        fc_edge_index = torch.tensor(fc_edge_list, dtype=torch.long)
    
        pd_edge_index = pd_edge_index.t().contiguous()
        pc_edge_index = pc_edge_index.t().contiguous()
        fc_edge_index = fc_edge_index.t().contiguous()
    
        xpd = torch.tensor(pd_features, dtype=torch.float)
        xpc = torch.tensor(pc_features, dtype=torch.float)
        xfc = torch.tensor(fc_features, dtype=torch.float)
    
        ypd = ypc = yfc = torch.tensor([], dtype=torch.float) 
    
        
        pd_data = gData(edge_index=pd_edge_index, x=xpd, y=ypd)
        pc_data = gData(edge_index=pc_edge_index, x=xpc, y=ypc)
        fc_data = gData(edge_index=fc_edge_index, x=xfc, y=yfc)
    
        assert pd_data.x.shape == pc_data.x.shape == (n_nodes, dim_pd_features)
        assert fc_data.x.shape == (n_nodes, dim_fc_features)
        
        return pd_data, pc_data, fc_data
    
    
    def graph_normalization(self, graph_):
        graph = copy.deepcopy(graph_)
        for feat_cat_name, feat_cat_val in graph['graph_features_numpy'].items():
            for room_name, room_feat in feat_cat_val.items():
                for akey in ['desired_area', 'current_area', 'delta_area']:
                    graph['graph_features_numpy'][feat_cat_name][room_name][akey] /= 400
                    
                if feat_cat_name == 'fully_current_graph_features_dict_numpy':
                    for ckey in ['start_of_wall', 'before_anchor', 'anchor', 'after_anchor', 'end_of_wall']:
                        coord = graph['graph_features_numpy'][feat_cat_name][room_name][ckey]
                        if -1 not in coord:
                            graph['graph_features_numpy'][feat_cat_name][room_name][ckey] = list(np.array(coord)/20)
                        else:
                            break
                    
                    distances = graph['graph_features_numpy'][feat_cat_name][room_name]['distances']
                    if -1 not in distances:
                        graph['graph_features_numpy'][feat_cat_name][room_name]['distances'] = list(np.array(distances)/30)
                    
        return graph   
    
    
    def image_normalization(self, image_):
        image = copy.deepcopy(image_)
        image[:,:,0] /= 25
        return image
    
    
    @staticmethod
    def _cartesian2image_coord(x, y, max_y):
        return max_y-y, x