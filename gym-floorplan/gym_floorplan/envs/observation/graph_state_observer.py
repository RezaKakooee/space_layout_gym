#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 14:01:29 2023

@author: Reza Kakooee
"""

import copy
import itertools
import numpy as np
from collections import defaultdict

import torch
from torchvision.transforms import transforms

# from torch_intermediate_layer_getter import IntermediateLayerGetter as MidGetter

from gym_floorplan.envs.observation.layout_graph import LayoutGraph



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
class GraphStateObserver:
    def __init__(self, fenv_config):
        self.fenv_config = fenv_config
        
    
    
    def get_observation(self, plan_data_dict, active_wall_name):
        plan_data_dict = self._get_graph_data_numpy(plan_data_dict, active_wall_name)
        plan_data_dict = self._get_gcn_observation(plan_data_dict)
        plan_data_dict = self._get_gnn_observation(plan_data_dict)    
        
        plan_data_dict.update({
            'observation_gnn': {
                'gnn_nodes': np.array(list(plan_data_dict['gnn_nodes'].values())),
                'gnn_edges': plan_data_dict['gcn_data_dict']['fc_edge_list'],
                }
            })
        
        return plan_data_dict
    
    
    
    def _get_gnn_observation(self, plan_data_dict):
        warooge_data = plan_data_dict['warooge_data']
        node_features = {f"wall_{i}": [] for i in range(1, self.fenv_config['cell_id_max'])}
        
        ww_dist_dict = defaultdict(dict)#
        
        for ww, v in warooge_data['walls_to_walls_distance_dict'].items():
            fw, sw = ww.split('_to_')[0], ww.split('_to_')[-1]
            ww_dist_dict[fw][sw] = v
            ww_dist_dict[sw][fw] = v
            
        w_dist_dict = {f"wall_{i}": [] for i in range(2, self.fenv_config['cell_id_max'])}
        wall_names = list(w_dist_dict.keys())
        for wall_out in wall_names:
           for wall_in in wall_names:
               if wall_out != wall_in:
                   w_dist_dict[wall_out] += ww_dist_dict[wall_out][wall_in]
        
        
        w_pos_dict = defaultdict(list)
        for w_name in wall_names:
            w_pos_dict[w_name] = [p for pos in list(warooge_data['wall_important_points_dict'][w_name].values()) for p in pos]
        
        
        node_features = defaultdict(dict)
        
        for i in range(2, self.fenv_config['cell_id_max']):
            wall_name = f"wall_{i}"
            room_name = f"room_{i}"
            status = 1 if warooge_data['areas_acheived'][room_name] > 0 else 0
            
            node_features[room_name] = [i, status]
            
            node_features[room_name] += [warooge_data['areas_desired'][room_name]]
            node_features[room_name] += [warooge_data['areas_acheived'][room_name]]
            node_features[room_name] += [warooge_data['areas_delta'][room_name]]
            
            
            node_features[room_name] += w_pos_dict[wall_name]
            # node_features[room_name] += warooge_data['walls_to_corners_distance_dict'][wall_name]
            node_features[room_name] += w_dist_dict[wall_name]
            
        
        edge_dict = self._extract_current_edge_list(plan_data_dict)
        gcc_edge = ( (self._shift_edge_by_one(edge_dict['edge_list_room_achieved'])).tolist() + 
                     (self._shift_edge_by_one(edge_dict['edge_list_facade_achieved'])).tolist() + 
                     (self._shift_edge_by_one(edge_dict['edge_list_entrance_achieved'])).tolist() )
        
        plan_data_dict.update({
            'gnn_nodes': node_features,
            'gcc_edge': gcc_edge,
            })
        
        return plan_data_dict
        
        
    
    def _get_gcn_observation(self, plan_data_dict):
        normalized_graph_data_numpy_old = self._graph_normalization(plan_data_dict['graph_data_numpy_old'])
        normalized_graph_data_numpy = self._graph_normalization(plan_data_dict['graph_data_numpy'])
        # normalized_plan_canvas_arr_old = self.image_normalization(plan_data_dict['plan_canvas_arr_old'])
        # normalized_plan_canvas_arr = self.image_normalization(plan_data_dict['plan_canvas_arr'])
        
        context = {'graph_data_numpy_old': normalized_graph_data_numpy_old,
                   'graph_data_numpy': normalized_graph_data_numpy,
                   # 'plan_canvas_arr_old': normalized_plan_canvas_arr_old,
                   # 'plan_canvas_arr': normalized_plan_canvas_arr
                   }
        gcn_data_dict_old = self._convert_to_readable_dict_for_graph(context['graph_data_numpy_old'])
        gcn_data_dict = self._convert_to_readable_dict_for_graph(context['graph_data_numpy'])
        plan_data_dict.update({
            'gcn_data_dict_old': gcn_data_dict_old,
            'gcn_data_dict': gcn_data_dict
            })
        
        # if self.fenv_config['gnn_obs_method'] == 'embedded_image_graph':
        #     self.observation_gnn = self._get_gcn_obs(plan_data_dict)
            
        # if self.fenv_config['gnn_obs_method'] == 'embedded_image_graph':
        #     observation = copy.deepcopy(self.observation_gnn)
        # elif self.fenv_config['gnn_obs_method'] == 'image':
        #     observation = copy.deepcopy(self.observation_cnn)
        # elif self.fenv_config['gnn_obs_method'] == 'dummy_vector':
        #     if self.fenv_config['action_masking_flag']:
        #         observation = np.zeros(self._observation_space['real_obs'].shape)
        #     else:   
        #         observation = np.zeros(self._observation_space.shape)
        # else:
        #     raise ValueError("gnn_obs_method is unvalid or not-implemented yet!")
            
        return plan_data_dict
        
        
        
    def _get_graph_data_numpy(self, plan_data_dict, active_wall_name):
        if active_wall_name is None:
            plan_data_dict = self._create_initil_graph_data_numpy(plan_data_dict)
        else:
            active_room_name = f"room_{active_wall_name.split('_')[1]}"
            active_room_data_dict = self._get_active_room_data_dict(plan_data_dict, active_wall_name)
            
            plan_data_dict.update({'active_rooms_data_dict': {active_room_name: active_room_data_dict}})
            
            if plan_data_dict['last_room']['last_room_name'] is not None:
                last_room_name = plan_data_dict['last_room']['last_room_name']
                last_room_data_dict = copy.deepcopy(active_room_data_dict)
                last_room_data_dict['current_area'] = plan_data_dict['areas_achieved'][last_room_name]
                last_room_data_dict['delta_area'] = plan_data_dict['areas_delta'][last_room_name]
                plan_data_dict['active_rooms_data_dict'].update({last_room_name: last_room_data_dict})
                
                plan_data_dict = self._update_graph_data_numpy(plan_data_dict, active_room_name, last_room_name)
                
            else:
                plan_data_dict = self._update_graph_data_numpy(plan_data_dict, active_room_name)
            
        return plan_data_dict
    
    
    
    def _create_initil_graph_data_numpy(self, plan_data_dict):
        areas_desired = plan_data_dict['areas_desired'] # list(plan_data_dict['areas_desired'].values())
        
        partially_desired_graph_features_dict_numpy = {
            room_name: {
                'status': 1, 'desired_area': da, 'current_area': da, 'delta_area': 0
                }  for room_name, da in areas_desired.items()
            }
        
        partially_current_graph_features_dict_numpy = {
            room_name: {
                'status': 0, 'desired_area': da, 'current_area': 0, 'delta_area': da
                }  for room_name, da in areas_desired.items()
            }
        
        fully_current_graph_features_dict_numpy = {
            room_name: {
              'status': 0, 'desired_area': da, 'current_area': 0, 'delta_area': da,
              'start_of_wall': [-1, -1], 
              'before_anchor': [-1, -1], 
              'anchor': [-1, -1], 
              'after_anchor': [-1, -1], 
              'end_of_wall': [-1, -1], 
              # 'anchor_to_corner_00_distance': -1, 
              # 'anchor_to_corner_01_distance': -1, 
              # 'anchor_to_corner_10_distance': -1,
              # 'anchor_to_corner_11_distance': -1,
              'distances': [-1, -1, -1, -1],
              } for room_name, da in areas_desired.items()
            }
        
        graph_features_numpy = {
            'partially_desired_graph_features_dict_numpy': partially_desired_graph_features_dict_numpy,
            'partially_current_graph_features_dict_numpy': partially_current_graph_features_dict_numpy,
            'fully_current_graph_features_dict_numpy': fully_current_graph_features_dict_numpy,
                                }
        
        edge_list_room_desired = plan_data_dict['edge_list_room_desired']
        
        partially_desired_graph_edge_list_numpy = copy.deepcopy(edge_list_room_desired)
        partially_current_graph_edge_list_numpy = []
        fully_current_graph_edge_list_numpy = []
        
        graph_edge_list_numpy = {
            'partially_desired_graph_edge_list_numpy': partially_desired_graph_edge_list_numpy,
            'partially_current_graph_edge_list_numpy': partially_current_graph_edge_list_numpy,
            'fully_current_graph_edge_list_numpy': fully_current_graph_edge_list_numpy,
            }
        
        graph_data_numpy = {'graph_features_numpy': graph_features_numpy,
                            'graph_edge_list_numpy': graph_edge_list_numpy}
        plan_data_dict.update({'graph_data_numpy_old': graph_data_numpy,
                                   'graph_data_numpy': graph_data_numpy})
        plan_data_dict.update({'active_rooms_data_dict': {}})
        return plan_data_dict
    
            
    
    def _get_active_room_data_dict(self, plan_data_dict, active_wall_name):
        warooge_data_main = plan_data_dict['warooge_data_main']
        active_room_data_dict = warooge_data_main['wall_important_points_dict'][active_wall_name]
        if active_room_data_dict is not None:
            active_room_name = f"room_{active_wall_name.split('_')[1]}"
            active_room_data_dict['current_area'] = plan_data_dict['areas_achieved'][active_room_name]
            active_room_data_dict['delta_area'] = plan_data_dict['areas_delta'][active_room_name]
        return active_room_data_dict
    
    
    
    def _update_graph_data_numpy(self, plan_data_dict, active_room_name, last_room_name=None):
        graph_data_numpy = copy.deepcopy(plan_data_dict['graph_data_numpy'])
        
        ### features
        partially_current_graph_features_dict_numpy = copy.deepcopy(graph_data_numpy['graph_features_numpy']['partially_current_graph_features_dict_numpy'])
        fully_current_graph_features_dict_numpy = copy.deepcopy(graph_data_numpy['graph_features_numpy']['fully_current_graph_features_dict_numpy'])
        
        active_room_data_dict = plan_data_dict['active_rooms_data_dict'][active_room_name]
        
        partially_current_graph_features_dict_numpy[active_room_name]['status'] = 1
        fully_current_graph_features_dict_numpy[active_room_name]['status'] = 1
        
        for k, v in active_room_data_dict.items():
            if 'area' in k:
                partially_current_graph_features_dict_numpy[active_room_name][k] = v
            fully_current_graph_features_dict_numpy[active_room_name][k] = v
            
            
        if last_room_name is not None:
            last_room_data_dict = plan_data_dict['active_rooms_data_dict'][last_room_name]
        
            partially_current_graph_features_dict_numpy[last_room_name]['status'] = 1
            fully_current_graph_features_dict_numpy[last_room_name]['status'] = 1
            
            for k, v in last_room_data_dict.items():
                if 'area' in k:
                    partially_current_graph_features_dict_numpy[last_room_name][k] = v
                fully_current_graph_features_dict_numpy[last_room_name][k] = v
            
        graph_data_numpy['graph_features_numpy']['partially_current_graph_features_dict_numpy'] = copy.deepcopy(partially_current_graph_features_dict_numpy)
        graph_data_numpy['graph_features_numpy']['fully_current_graph_features_dict_numpy'] = copy.deepcopy(fully_current_graph_features_dict_numpy)
        
        ### edges
        edge_dict = self._extract_current_edge_list(plan_data_dict)
        
        current_edge_list = edge_dict['edge_list_room_achieved'] + edge_dict['edge_list_facade_achieved'] + edge_dict['edge_list_entrance_achieved']
        
        if current_edge_list:
            graph_data_numpy['graph_edge_list_numpy']['partially_current_graph_edge_list_numpy'] = current_edge_list
            graph_data_numpy['graph_edge_list_numpy']['fully_current_graph_edge_list_numpy'] = current_edge_list
        
        
        plan_data_dict['graph_data_numpy_old'] = copy.deepcopy(plan_data_dict['graph_data_numpy'])
        
        plan_data_dict['graph_data_numpy'] = copy.deepcopy(graph_data_numpy)
        
        plan_data_dict['edge_dict'] = edge_dict
        
        return plan_data_dict
    
    
    
    def _extract_current_edge_list(self, plan_data_dict):
        layout_graph = LayoutGraph(plan_data_dict, self.fenv_config)
        edge_dict = layout_graph.extract_graph_data()
        return edge_dict
    
    
    
    @staticmethod
    def _shift_edge_by_one(e):
        e = np.array(e, dtype=np.int32)-1
        return e
    
    
    
    def _convert_to_readable_dict_for_graph(self, graph_data_numpy):
        
        
        graph_features_numpy = graph_data_numpy['graph_features_numpy']
        graph_edge_list_numpy = graph_data_numpy['graph_edge_list_numpy']

        partially_desired_graph_features_dict_numpy = graph_features_numpy['partially_desired_graph_features_dict_numpy']
        partially_current_graph_features_dict_numpy = graph_features_numpy['partially_current_graph_features_dict_numpy']
        fully_current_graph_features_dict_numpy = graph_features_numpy['fully_current_graph_features_dict_numpy']

        pd_edge_list = self._shift_edge_by_one(graph_edge_list_numpy['partially_desired_graph_edge_list_numpy'])
        pc_edge_list = self._shift_edge_by_one(graph_edge_list_numpy['partially_current_graph_edge_list_numpy'])
        fc_edge_list = self._shift_edge_by_one(graph_edge_list_numpy['fully_current_graph_edge_list_numpy'])
        
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
        
        gcn_data_dict = {
            'dim_pd_features': dim_pd_features,
            'dim_fc_features': dim_fc_features,
            'n_nodes': n_nodes,
            'pd_edge_list': pd_edge_list,
            'pc_edge_list': pc_edge_list,
            'fc_edge_list': fc_edge_list,
            'pd_features': pd_features,
            'pc_features': pc_features,
            'fc_features': fc_features,
            }
        
        return gcn_data_dict
    
    
    
    def _graph_normalization(self, graph_):
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