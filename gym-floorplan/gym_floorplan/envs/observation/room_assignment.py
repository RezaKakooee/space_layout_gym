# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 15:38:31 2024

@author: Reza Kakooee
"""
import copy
import numpy as np

from gym_floorplan.envs.observation.room_assignment_helpers import (
    assign_room_ids,
    assign_rooms_initial_way,
    assign_room_ids_max_iou_stepwise,
    assign_room_ids_max_iou_min_mismatch_stepwise,
    assign_room_ids_max_iou_min_mismatch_ordered_permutaion,
    assign_room_ids_max_iou_min_mismatch_permutations,
    assign_room_ids_max_iou_by_base_wall,
    
    assign_room_ids_for_identity_less_walls,
    
    get_clean_matrices,
    get_geometrical_properties,
)




class RoomAssigment:
    def __init__(self, fenv_config: dict):
        self.fenv_config = fenv_config
        
        self.assign_rooms_method_name = 'assign_room_ids_max_iou_by_base_wall'
        
        if self.assign_rooms_method_name == 'assign_rooms_initial_way':
            self.assign_room_ids = assign_rooms_initial_way
            
        elif self.assign_rooms_method_name == 'assign_room_ids_max_iou_stepwise':
            self.assign_room_ids = assign_room_ids_max_iou_stepwise
            
        elif self.assign_rooms_method_name == 'assign_room_ids_max_iou_min_mismatch_stepwise':
            self.assign_room_ids = assign_room_ids_max_iou_min_mismatch_stepwise
            
        elif self.assign_rooms_method_name == 'assign_room_ids_max_iou_min_mismatch_ordered_permutaion':
            self.assign_room_ids = assign_room_ids_max_iou_min_mismatch_ordered_permutaion
            
        elif self.assign_rooms_method_name == 'assign_room_ids_max_iou_min_mismatch_permutations':
            self.assign_room_ids = assign_room_ids_max_iou_min_mismatch_permutations    
        
        elif self.assign_rooms_method_name == 'assign_room_ids_max_iou_by_base_wall':
            self.assign_room_ids = assign_room_ids_max_iou_by_base_wall    
            
        else:
            self.assign_room_ids = assign_room_ids
        

    
    def assign_rooms(self, plan_data_dict_backup, plan_data_dict):
        obs_moving_labels = copy.deepcopy(plan_data_dict_backup['obs_moving_labels'])
        obs_mat_w = copy.deepcopy(plan_data_dict['obs_mat_w'])
        obs_mat_base_w = copy.deepcopy(plan_data_dict['obs_mat_base_w'])
        agent_order = copy.deepcopy(plan_data_dict['shifted_agent_names_deque'])
        
        ## August 07, 2024
        obs_moving_labels = self._add_entrance_to_moving_labels(obs_moving_labels, plan_data_dict)
        obs_mat_w = self._add_entrance_to_moving_labels(obs_mat_w, plan_data_dict)
        
        obs_moving_labels, affected_rooms = self.assign_room_ids(obs_moving_labels, obs_mat_w, obs_mat_base_w, agent_order)
        
        wm = copy.deepcopy(plan_data_dict['wall_matrices']) # for identity full, we do not need wm, so we simply use its original values
        
        if self.fenv_config['wall_identity_mode'] == 'identity_less':
            obs_moving_labels_for_identity_less = copy.deepcopy(plan_data_dict_backup['obs_moving_labels'])
            obs_mat_w_identity_less = copy.deepcopy(plan_data_dict['obs_mat_w'])
            obs_moving_labels_for_identity_less = self._add_entrance_to_moving_labels(obs_moving_labels_for_identity_less, plan_data_dict)
            obs_mat_w_identity_less = self._add_entrance_to_moving_labels(obs_mat_w_identity_less, plan_data_dict)
            
            obs_moving_labels_for_identity_less, affected_rooms = assign_room_ids_for_identity_less_walls(obs_moving_labels_for_identity_less, obs_mat_w_identity_less, list(plan_data_dict['areas_desired'].values()))
        
            room_mapping = self.extract_room_mapping_from_full_to_less(obs_moving_labels, obs_moving_labels_for_identity_less)
            
            ### we need to modify wm for identity less
            # print(f"room_mapping: {room_mapping}")
            for wi in wm.keys():
                if -wi in room_mapping: # TOSO: Sep 23, 2024, why some room i do not exist?
                    wm[wi][wm[wi] == wi] = -room_mapping[-wi]
            wm = {-room_mapping.get(abs(k), abs(k)): v for k, v in wm.items()}
            plan_data_dict['wall_matrices'] = wm
            
            obs_mat_w = copy.deepcopy(obs_mat_w_identity_less)
            obs_moving_labels = copy.deepcopy(obs_moving_labels_for_identity_less)
        
        return obs_mat_w, obs_moving_labels, affected_rooms, wm
    
    
    
    def get_clean_matrices(self, obs_mat_w):
        return get_clean_matrices(obs_mat_w)
    
    
    
    def get_geometrical_properties(self, plan_data_dict):
        return get_geometrical_properties(plan_data_dict)
    
    
    
    def _add_entrance_to_moving_labels(self, mat, plan_data_dict):
        mat_ = copy.deepcopy(mat)
        for r, c in plan_data_dict['entrance_positions']:
            mat_[r][c] = self.fenv_config['entrance_cell_id']
        return mat_
    
    

    def extract_room_mapping_from_full_to_less(self, oml_full, oml_less):
        # Identify unique room IDs in both matrices (only for IDs >= 12)
        full_ids = set(id for id in np.unique(oml_full) if id >= 12)
        less_ids = set(id for id in np.unique(oml_less) if id >= 12)
        
        mapping = {}
        for full_id in full_ids:
            full_mask = oml_full == full_id
            max_overlap = 0
            max_overlap_id = None
            
            for less_id in less_ids:
                less_mask = oml_less == less_id
                overlap = np.sum(full_mask & less_mask)
                
                if overlap > max_overlap:
                    max_overlap = overlap
                    max_overlap_id = less_id
            
            if max_overlap_id is not None:
                mapping[full_id] = max_overlap_id
        
        return mapping

    
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 15:38:31 2024

@author: Reza Kakooee
"""
import copy
import numpy as np

from gym_floorplan.envs.observation.room_assignment_helpers import (
    assign_room_ids,
    assign_rooms_initial_way,
    assign_room_ids_max_iou_stepwise,
    assign_room_ids_max_iou_min_mismatch_stepwise,
    assign_room_ids_max_iou_min_mismatch_ordered_permutaion,
    assign_room_ids_max_iou_min_mismatch_permutations,
    assign_room_ids_max_iou_by_base_wall,
    
    assign_room_ids_for_identity_less_walls,
    
    get_clean_matrices,
    get_geometrical_properties,
)




class RoomAssigment:
    def __init__(self, fenv_config: dict):
        self.fenv_config = fenv_config
        
        self.assign_rooms_method_name = 'assign_room_ids_max_iou_by_base_wall'
        
        if self.assign_rooms_method_name == 'assign_rooms_initial_way':
            self.assign_room_ids = assign_rooms_initial_way
            
        elif self.assign_rooms_method_name == 'assign_room_ids_max_iou_stepwise':
            self.assign_room_ids = assign_room_ids_max_iou_stepwise
            
        elif self.assign_rooms_method_name == 'assign_room_ids_max_iou_min_mismatch_stepwise':
            self.assign_room_ids = assign_room_ids_max_iou_min_mismatch_stepwise
            
        elif self.assign_rooms_method_name == 'assign_room_ids_max_iou_min_mismatch_ordered_permutaion':
            self.assign_room_ids = assign_room_ids_max_iou_min_mismatch_ordered_permutaion
            
        elif self.assign_rooms_method_name == 'assign_room_ids_max_iou_min_mismatch_permutations':
            self.assign_room_ids = assign_room_ids_max_iou_min_mismatch_permutations    
        
        elif self.assign_rooms_method_name == 'assign_room_ids_max_iou_by_base_wall':
            self.assign_room_ids = assign_room_ids_max_iou_by_base_wall    
            
        else:
            self.assign_room_ids = assign_room_ids
        

    
    def assign_rooms(self, plan_data_dict_backup, plan_data_dict):
        obs_moving_labels = copy.deepcopy(plan_data_dict_backup['obs_moving_labels'])
        obs_mat_w = copy.deepcopy(plan_data_dict['obs_mat_w'])
        obs_mat_base_w = copy.deepcopy(plan_data_dict['obs_mat_base_w'])
        agent_order = copy.deepcopy(plan_data_dict['shifted_agent_names_deque'])
        
        ## August 07, 2024
        obs_moving_labels = self._add_entrance_to_moving_labels(obs_moving_labels, plan_data_dict)
        obs_mat_w = self._add_entrance_to_moving_labels(obs_mat_w, plan_data_dict)
        
        obs_moving_labels, affected_rooms = self.assign_room_ids(obs_moving_labels, obs_mat_w, obs_mat_base_w, agent_order)
        
        wm = copy.deepcopy(plan_data_dict['wall_matrices']) # for identity full, we do not need wm, so we simply use its original values
        
        if self.fenv_config['wall_identity_mode'] == 'identity_less':
            obs_moving_labels_for_identity_less = copy.deepcopy(plan_data_dict_backup['obs_moving_labels'])
            obs_mat_w_identity_less = copy.deepcopy(plan_data_dict['obs_mat_w'])
            obs_moving_labels_for_identity_less = self._add_entrance_to_moving_labels(obs_moving_labels_for_identity_less, plan_data_dict)
            obs_mat_w_identity_less = self._add_entrance_to_moving_labels(obs_mat_w_identity_less, plan_data_dict)
            
            obs_moving_labels_for_identity_less, affected_rooms = assign_room_ids_for_identity_less_walls(obs_moving_labels_for_identity_less, obs_mat_w_identity_less, list(plan_data_dict['areas_desired'].values()))
        
            room_mapping = self.extract_room_mapping_from_full_to_less(obs_moving_labels, obs_moving_labels_for_identity_less)
            
            ### we need to modify wm for identity less
            # print(f"room_mapping: {room_mapping}")
            for wi in wm.keys():
                if -wi in room_mapping: # TOSO: Sep 23, 2024, why some room i do not exist?
                    wm[wi][wm[wi] == wi] = -room_mapping[-wi]
            wm = {-room_mapping.get(abs(k), abs(k)): v for k, v in wm.items()}
            plan_data_dict['wall_matrices'] = wm
            
            obs_mat_w = copy.deepcopy(obs_mat_w_identity_less)
            obs_moving_labels = copy.deepcopy(obs_moving_labels_for_identity_less)
        
        return obs_mat_w, obs_moving_labels, affected_rooms, wm
    
    
    
    def get_clean_matrices(self, obs_mat_w):
        return get_clean_matrices(obs_mat_w)
    
    
    
    def get_geometrical_properties(self, plan_data_dict):
        return get_geometrical_properties(plan_data_dict)
    
    
    
    def _add_entrance_to_moving_labels(self, mat, plan_data_dict):
        mat_ = copy.deepcopy(mat)
        for r, c in plan_data_dict['entrance_positions']:
            mat_[r][c] = self.fenv_config['entrance_cell_id']
        return mat_
    
    

    def extract_room_mapping_from_full_to_less(self, oml_full, oml_less):
        # Identify unique room IDs in both matrices (only for IDs >= 12)
        full_ids = set(id for id in np.unique(oml_full) if id >= 12)
        less_ids = set(id for id in np.unique(oml_less) if id >= 12)
        
        mapping = {}
        for full_id in full_ids:
            full_mask = oml_full == full_id
            max_overlap = 0
            max_overlap_id = None
            
            for less_id in less_ids:
                less_mask = oml_less == less_id
                overlap = np.sum(full_mask & less_mask)
                
                if overlap > max_overlap:
                    max_overlap = overlap
                    max_overlap_id = less_id
            
            if max_overlap_id is not None:
                mapping[full_id] = max_overlap_id
        
        return mapping

    
