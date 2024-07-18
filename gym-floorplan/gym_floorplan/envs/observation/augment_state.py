#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 22:50:48 2023

@author: rdbt
"""

#%%
import copy


class AugmentState:
    def __init__(self, fenv_config):
        self.fenv_config = fenv_config
        
    
    
    def augment(self, plan_data_dict, warooge_data_main, active_wall_name, state_data_dict):
        if len(plan_data_dict['wall_types']) == 0:
            shuffeled_warooge_data = copy.deepcopy(warooge_data_main) # no shuffeling
            observation_fc = self._get_fc_obs(plan_data_dict, shuffeled_warooge_data, active_wall_name, state_data_dict)
            
            augmented_observation_fc_tuple = tuple(observation_fc for _ in range(self.fenv_config['n_shuffles_for_data_augmentation']))
            warooge_data_tuple = tuple(shuffeled_warooge_data for _ in range(self.fenv_config['n_shuffles_for_data_augmentation']))
            
            plan_data_dict.update({
                        'warooge_data': shuffeled_warooge_data,
                        'observation': observation_fc,
                        'warooge_data_tuple': warooge_data_tuple,
                        'augmented_observation_fc_tuple': augmented_observation_fc_tuple,
                        })
        
        else:
            augmented_observation_fc_tuple = tuple()
            warooge_data_tuple = tuple()
            for n in range(self.fenv_config['n_shuffles_for_data_augmentation']):
                if n == 0:
                    shuffeled_warooge_data = copy.deepcopy(warooge_data_main) # no shuffeling
                    observation_fc = self._get_fc_obs(plan_data_dict, shuffeled_warooge_data, active_wall_name, state_data_dict)
                    
                    plan_data_dict.update({
                        'warooge_data': shuffeled_warooge_data,
                        'observation': observation_fc,
                        })       
                    
                else:
                    ref_tar_shuffeled_warooge_dict = self._map_ref_to_shuffeled_warooge_name(plan_data_dict['shuffeled_idxs'])
                    
                    shuffeled_warooge_data = self._modify_warooge_data_main(warooge_data_main, ref_tar_shuffeled_warooge_dict)
                    observation_fc = self._get_fc_obs(plan_data_dict, shuffeled_warooge_data, active_wall_name, state_data_dict)
                
                warooge_data_tuple += (shuffeled_warooge_data,)
                augmented_observation_fc_tuple += (observation_fc,)
            
            plan_data_dict.update({
                        'warooge_data_tuple': warooge_data_tuple,
                        'augmented_observation_fc_tuple': augmented_observation_fc_tuple,
                        })       
        
        return plan_data_dict
    
    
    
    def _map_ref_to_shuffeled_warooge_name(self, tar):
        ref = list(range(self.fenv_config['maximum_num_masked_rooms']+1, self.fenv_config['maximum_num_real_plus_masked_rooms']+1))
        ref_tar_shuffeled_warooge_dict = {i:i for i in range(0, self.fenv_config['maximum_num_masked_rooms']+1)}
        ref_tar_shuffeled_warooge_dict.update({r:t for r, t in zip(ref, tar)})
        return ref_tar_shuffeled_warooge_dict
    
    
    
    def _modify_warooge_data_main(self, warooge_data_main, ref_tar_shuffeled_warooge_dict):
        shuffeled_warooge_data = {
            'wall_important_points_dict': {
                f"wall_{ ref_tar_shuffeled_warooge_dict[ int( wall_name.split('_')[1] ) ] }": wall_val for wall_name, wall_val in warooge_data_main['wall_important_points_dict'].items() 
                },
            'obs_moving_labels': copy.deepcopy(warooge_data_main['obs_moving_labels'])
            }
        
        
        areas_desired_shuffeled = {}
        shuffeled_acheived_areas = {}
        areas_delta_shuffeled = {}
        for room_name in warooge_data_main['areas_desired']:
            new_room_name = f"room_{ref_tar_shuffeled_warooge_dict[ int( room_name.split('_')[1] ) ]}"
            areas_desired_shuffeled.update({new_room_name: warooge_data_main['areas_desired'][room_name]})
            shuffeled_acheived_areas.update({new_room_name: warooge_data_main['acheived_areas'][room_name]})
            areas_delta_shuffeled.update({new_room_name: warooge_data_main['areas_delta'][room_name]})
            
        shuffeled_warooge_data.update({
                'areas_desired': areas_desired_shuffeled,
                'acheived_areas': shuffeled_acheived_areas,
                'areas_delta': areas_delta_shuffeled
                })
        
        
        shuffeled_warooge_data.update({
            'edge_list_room_desired': [ [ref_tar_shuffeled_warooge_dict[i], ref_tar_shuffeled_warooge_dict[j]] for i, j in warooge_data_main['edge_list_room_desired']],
            'acheived_edge_list': [ [ref_tar_shuffeled_warooge_dict[i], ref_tar_shuffeled_warooge_dict[j]] for i, j in warooge_data_main['acheived_edge_list']],
            'edge_list_facade_desired': [ [i, ref_tar_shuffeled_warooge_dict[j]] for i, j in warooge_data_main['edge_list_facade_desired']],
            'acheived_facade_edge_list': [ [i, ref_tar_shuffeled_warooge_dict[j]] for i, j in warooge_data_main['acheived_facade_edge_list']],
            })
        
        move_label = self._change_obs_moving_labels_accordingly(warooge_data_main, ref_tar_shuffeled_warooge_dict)   
        shuffeled_warooge_data['obs_moving_labels'] = copy.deepcopy(move_label)
        
        return shuffeled_warooge_data
        


    def _change_obs_moving_labels_accordingly(self, warooge_data_main, ref_tar_shuffeled_warooge_dict):
        move_label = copy.deepcopy(warooge_data_main['obs_moving_labels'])
        for r in range(move_label.shape[0]):
            for c in range(move_label.shape[1]):
                move_label[r][c] = ref_tar_shuffeled_warooge_dict[move_label[r][c]]
        return move_label