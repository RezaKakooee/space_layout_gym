# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 23:34:14 2021

@author: Reza Kakooee
"""
#%%

import os
import inspect
import ast
import copy
import numpy as np
import gymnasium as gym
from datetime import datetime
from collections import defaultdict, deque

# from gym_floorplan.base_env.observation.base_observation import BaseObservation

from gym_floorplan.envs.observation.sequential_painter import SequentialPainter
from gym_floorplan.envs.observation.room_extractor import RoomExtractor
from gym_floorplan.envs.observation.plan_constructor import PlanConstructor
from gym_floorplan.envs.observation.state_composer import StateComposer
from gym_floorplan.envs.observation.action_parser import ActionParser
from gym_floorplan.envs.observation.design_inspector import DesignInspector
from gym_floorplan.envs.observation.wall_transform import WallTransform
from gym_floorplan.envs.observation.wall_generator import WallGenerator
from gym_floorplan.envs.observation.room_assignment import RoomAssigment



# get root_dir from environment variable
HOUSING_DESIGN_ROOT_DIR = os.path.expandvars(str(os.getenv('HOUSING_DESIGN_ROOT_DIR')))
print(f"In observation.py HOUSING_DESIGN_ROOT_DIR: {HOUSING_DESIGN_ROOT_DIR}")


VERBOSE = 0

#%%
class Observation:
    def __init__(self, fenv_config:dict={}):
        # super().__init__()
        self.fenv_config = fenv_config
        
        self.plan_constructor = PlanConstructor(fenv_config=self.fenv_config)
        self.painter = SequentialPainter(fenv_config=self.fenv_config)
        self.rextractor = RoomExtractor(fenv_config=self.fenv_config)
        self.state_composer = StateComposer(fenv_config=self.fenv_config)
        self.action_parser = ActionParser(fenv_config=self.fenv_config)
        self.design_inspector = DesignInspector(fenv_config=self.fenv_config)
        self.room_assignment = RoomAssigment(fenv_config=self.fenv_config)
        
        self.observation_space = self._get_observation_space()

        self.time_dict_observation = defaultdict(dict)



    # @property
    def _get_observation_space(self): # def observation_space(self): 
        self.state_data_dict = self.state_composer.creat_observation_space_variables()
        
        _observation_space_fc = gym.spaces.Box(
            low=self.state_data_dict['low_fc'], 
            high=self.state_data_dict['high_fc'], 
            shape=self.state_data_dict['shape_fc'], 
            dtype=np.float32
        )

        _observation_space_cnn = gym.spaces.Box(
            low=self.state_data_dict['low_cnn'], 
            high=self.state_data_dict['high_cnn'], 
            shape=self.state_data_dict['shape_cnn'], 
            dtype=np.float64
        )
        
        _observation_space_meta = gym.spaces.Box(
            low=self.state_data_dict['low_meta'], 
            high=self.state_data_dict['high_meta'], 
            shape=self.state_data_dict['shape_meta'], 
            dtype=np.float64
        )

        if self.fenv_config['meta_observation_type'] == 'dict':
            _observation_space_metafc = gym.spaces.Dict(
                {'observation_fc': _observation_space_fc, 'observation_meta':  _observation_space_meta}
            )
            
            _observation_space_metacnn = gym.spaces.Dict(
                {'observation_cnn': _observation_space_cnn, 'observation_meta': _observation_space_meta,}
            )
        
        elif self.fenv_config['meta_observation_type'] == 'tuple':
            _observation_space_metafc = gym.spaces.Tuple(
                (_observation_space_fc, _observation_space_meta)
            )

            _observation_space_metacnn = gym.spaces.Tuple(
                (_observation_space_cnn, _observation_space_meta)
            )
        
        elif self.fenv_config['meta_observation_type'] == 'list':
            _observation_space_metacnn = gym.spaces.Box(
                    low=min(self.state_data_dict['low_cnn'], self.state_data_dict['low_meta']), 
                    high=max(self.state_data_dict['high_cnn'], self.state_data_dict['high_meta']), 
                    shape=np.prod(self.state_data_dict['shape_cnn']) + self.state_data_dict['shape_meta'], 
                    dtype=np.float32,
            )
            
            _observation_space_metafc = gym.spaces.Box(
                    low=min(self.state_data_dict['low_fc'], self.state_data_dict['low_meta']), 
                    high=max(self.state_data_dict['high_fc'], self.state_data_dict['high_meta']), 
                    shape=self.state_data_dict['shape_fc'] + self.state_data_dict['shape_meta'], 
                    dtype=np.float32,
            )
            
        _observation_space_gnn = gym.spaces.Dict({
            'gnn_nodes': gym.spaces.Box(low=self.state_data_dict['low_gnn'], 
                                    high=self.state_data_dict['high_gnn'], 
                                    shape=self.state_data_dict['shape_gnn'], 
                                    dtype=np.float16),
            'gnn_edge': gym.spaces.MultiDiscrete(self.fenv_config['num_nodes']*np.ones((1, 2)),
                                                 dtype=np.int16),
            })
        
        if self.fenv_config['action_masking_flag']:
            _observation_space_fc = gym.spaces.Dict({
                                'action_mask': gym.spaces.Box(low=0, high=1,  shape=(self.fenv_config['n_actions'],), dtype=np.float32),
                                'action_avail': gym.spaces.Box(low=0, high=1, shape=(self.fenv_config['n_actions'],), dtype=np.float32),
                                'real_obs': _observation_space_fc,
                                })
            _observation_space_cnn = gym.spaces.Dict({
                                'action_mask': gym.spaces.Box(low=0, high=1,  shape=(self.fenv_config['n_actions'],), dtype=np.float32),
                                'action_avail': gym.spaces.Box(low=0, high=1, shape=(self.fenv_config['n_actions'],), dtype=np.float32),
                                'real_obs': _observation_space_cnn,
                                })
            _observation_space_metafc = gym.spaces.Dict({
                                'action_mask': gym.spaces.Box(low=0, high=1,  shape=(self.fenv_config['n_actions'],), dtype=np.float32),
                                'action_avail': gym.spaces.Box(low=0, high=1, shape=(self.fenv_config['n_actions'],), dtype=np.float32),
                                'real_obs': _observation_space_metafc,
                                })
            _observation_space_metacnn = gym.spaces.Dict({
                                'action_mask': gym.spaces.Box(low=0, high=1,  shape=(self.fenv_config['n_actions'],), dtype=np.float32),
                                'action_avail': gym.spaces.Box(low=0, high=1, shape=(self.fenv_config['n_actions'],), dtype=np.float32),
                                'real_obs': _observation_space_metacnn,
                                })
            _observation_space_gnn = gym.spaces.Dict({
                                'action_mask': gym.spaces.Box(low=0, high=1, shape=(self.fenv_config['n_actions'],), dtype=np.float32),
                                'action_avail': gym.spaces.Box(low=0, high=1, shape=(self.fenv_config['n_actions'],), dtype=np.float32),
                                'real_obs': _observation_space_gnn,
                                })
        
        if self.fenv_config['net_arch'] == 'Fc':
            self._observation_space = _observation_space_fc
            
        elif self.fenv_config['net_arch'] == 'Cnn': 
            self._observation_space = _observation_space_cnn

        elif self.fenv_config['net_arch'] == 'MetaFc':
            self._observation_space = _observation_space_metafc

        elif self.fenv_config['net_arch'] == 'MetaCnn':
            self._observation_space = _observation_space_metacnn #gym.spaces.Tuple((_observation_space_cnn, _observation_space_meta))

        elif self.fenv_config['net_arch'] == 'Gnn':
            if self.fenv_config['gnn_obs_method'] in ['embedded_image_graph', 'dummy_vector']:
                self._observation_space = _observation_space_gnn
            elif self.fenv_config['gnn_obs_method'] == 'image':
                self._observation_space = _observation_space_cnn
            else:
                raise ValueError(f"Invalid gnn_obs_method! The current method is {self.fenv_config['gnn_obs_method']}")
                
        else:
            raise ValueError(f"{self.fenv_config['net_arch']} net_arch does not exist")
            
        if not self.fenv_config['load_from_inwalls_coords_fixed_for_debug']:
            self.inwalls_coords_fixed_for_debug = []
        else: 
            
            # TODO: remove the following line
            self.inwalls_coords_fixed_for_debug = [{'wall_12': {'anchor_coord': [18, 14], 'back_open_coord': [16, 14], 'front_open_coord': [18, 12], 'front_segment': {'orientation': 'axial', 'direction': 'south', 'location': 'in'}, 'back_segment': {'orientation': 'axial', 'direction': 'west', 'location': 'in'}, 'base_coords': [[16, 14], [18, 14], [18, 12], [18, 13], [17, 14]]}}, {'wall_13': {'anchor_coord': [8, 8], 'back_open_coord': [8, 6], 'front_open_coord': [6, 8], 'front_segment': {'orientation': 'axial', 'direction': 'west', 'location': 'in'}, 'back_segment': {'orientation': 'axial', 'direction': 'south', 'location': 'in'}, 'base_coords': [[8, 6], [8, 8], [6, 8], [7, 8], [8, 7]]}}, {'wall_14': {'anchor_coord': [16, 10], 'back_open_coord': [16, 12], 'front_open_coord': [16, 8], 'front_segment': {'orientation': 'axial', 'direction': 'south', 'location': 'in'}, 'back_segment': {'orientation': 'axial', 'direction': 'north', 'location': 'in'}, 'base_coords': [[16, 12], [16, 10], [16, 8], [16, 9], [16, 11]]}}, {'wall_15': {'anchor_coord': [14, 12], 'back_open_coord': [14, 14], 'front_open_coord': [14, 10], 'front_segment': {'orientation': 'axial', 'direction': 'south', 'location': 'in'}, 'back_segment': {'orientation': 'axial', 'direction': 'north', 'location': 'in'}, 'base_coords': [[14, 14], [14, 12], [14, 10], [14, 11], [14, 13]]}}, {'wall_16': {'anchor_coord': [8, 12], 'back_open_coord': [6, 12], 'front_open_coord': [8, 10], 'front_segment': {'orientation': 'axial', 'direction': 'south', 'location': 'in'}, 'back_segment': {'orientation': 'axial', 'direction': 'west', 'location': 'in'}, 'base_coords': [[6, 12], [8, 12], [8, 10], [8, 11], [7, 12]]}}, {'wall_17': {'anchor_coord': [12, 6], 'back_open_coord': [12, 4], 'front_open_coord': [12, 8], 'front_segment': {'orientation': 'axial', 'direction': 'north', 'location': 'in'}, 'back_segment': {'orientation': 'axial', 'direction': 'south', 'location': 'in'}, 'base_coords': [[12, 4], [12, 6], [12, 8], [12, 7], [12, 5]]}}]
        
        return self._observation_space
    
    
    
    def obs_reset(self, episode):
        self.n_actions_accepted = 0
        plan_data_dict = self.plan_constructor.get_plan_data_dict(episode=episode)
        plan_data_dict['plan_description'] = self.plan_constructor.get_plan_meta_data(plan_data_dict)

        if VERBOSE: print(f"obs_reset, line 189: plan_data_dict['areas_masked']: {plan_data_dict['areas_masked']}, oml areas maksed name: {[n for n in np.unique(plan_data_dict['obs_moving_labels']) if 2 <= n <= 5]}")
        
        
        self.active_wall_name = None  
        self.active_wall_status = None 
        plan_data_dict.update({'active_wall_name': self.active_wall_name,
                               'active_wall_status': self.active_wall_status})
        
        self.input_plan_data_dict = copy.deepcopy(plan_data_dict)
        
        if self.fenv_config['env_planning'] == 'One_Shot':
            plan_data_dict = self.state_composer.warooge_data_extractor(plan_data_dict)
            plan_data_dict = self.state_composer.refine_moving_labels(plan_data_dict)
            
            self.observation, plan_data_dict = self._make_observation(plan_data_dict)
                    
            plan_data_dict.update({'obs_arr_conv': self.observation})
            
            self.done = False 
            self.plan_data_dict = copy.deepcopy(plan_data_dict)
            self.plan_data_dict.update({'done': self.done,
                                        'state_data_dict': self.state_data_dict})
            return self.observation

        elif self.fenv_config['env_planning'] == 'Dynamic':
            plan_data_dict_pool_dir = os.path.join(HOUSING_DESIGN_ROOT_DIR, 'gym-floorplan/gym_floorplan/envs/observation/plan_data_dict_pool')
            os.makedirs(plan_data_dict_pool_dir, exist_ok=True)
            pdds = os.listdir(plan_data_dict_pool_dir)
            
            pdd_name = 'PDD__' + plan_data_dict['plan_id'] + '.npy'
            pdd_path = os.path.join(plan_data_dict_pool_dir, pdd_name)
            
            if self.fenv_config['load_pdd_flag'] and (pdd_name in pdds):
                plan_data_dict = np.load(pdd_path, allow_pickle=True).item()
                
            else:
                fec = copy.deepcopy(self.fenv_config)
                fec['n_walls'] = 1
                inwalls_coords = {}
                max_attempts = 100  # Maximum number of attempts to generate valid walls
        
                for i in range(plan_data_dict['n_walls']):
                    inwall_name = f"wall_{self.fenv_config['min_room_id']+i+1}"
                    
                    for attempt in range(max_attempts):
                        try:
                            plan_data_dict_old = copy.deepcopy(plan_data_dict)
                            valid_points_for_sampling = self.plan_constructor._get_valid_points_for_sampling(plan_data_dict)
                            inwall_coords = WallGenerator(fenv_config=fec).make_walls(valid_points_for_sampling)
                            inwall_coords = {inwall_name: inwall_coords['wall_11']}
                            self.inwalls_coords_fixed_for_debug.append(inwall_coords)
                            inwalls_coords.update(inwall_coords)
                            plan_data_dict.update({'inwalls_coords': inwalls_coords})
                            plan_data_dict = self.plan_constructor.update_plan_with_active_wall(plan_data_dict=plan_data_dict, walls_coords=plan_data_dict['inwalls_coords'], active_wall_name=inwall_name)
                            plan_data_dict = self.painter.update_obs_mat(plan_data_dict, inwall_name)
                            plan_data_dict = self.rextractor.update_room_dict(plan_data_dict, inwall_name)
                            if VERBOSE: print(f"obs_reset, line 189: plan_data_dict['areas_masked']: {plan_data_dict['areas_masked']}, oml areas maksed name: {[n for n in np.unique(plan_data_dict['obs_moving_labels']) if 2 <= n <= 5]}")
                            plan_data_dict = self.plan_constructor._update_block_cells(plan_data_dict)
                            active_wall_status = self.design_inspector.inspect_entrace_lvroom_relation_for_dyp(plan_data_dict, self.active_wall_name)
                            if 'rejected' in active_wall_status:
                                plan_data_dict = copy.deepcopy(plan_data_dict_old)
                                continue
                            plan_data_dict['wall_order'].update({i+1: inwall_name})
                            plan_data_dict['wall_types'].update({i+1: inwall_name})
                            break
                        except Exception as e:
                            if VERBOSE: print(f"Attempt {attempt + 1} failed: {str(e)}")
                            if attempt == max_attempts - 1:
                                raise ValueError(f"Failed to generate valid wall coordinates after {max_attempts} attempts")
                with open(pdd_path, "wb") as f:
                    np.save(f, plan_data_dict)
                
            # Rest of the method remains unchanged
            plan_data_dict = self.state_composer.warooge_data_extractor(plan_data_dict)
            plan_data_dict = self.state_composer.refine_moving_labels(plan_data_dict)
            
            n_agents = plan_data_dict['n_walls']
            agent_names_deque = deque([f"agent_{i+1+self.fenv_config['min_room_id']}" for i in range(n_agents)], maxlen=n_agents)
            self.observation, plan_data_dict = self._make_observation(plan_data_dict=plan_data_dict,
                                                                      active_wall_name=self.active_wall_name, # which is None
                                                                      active_wall_status=self.active_wall_status, # which is None
                                                                      agent_names_deque=agent_names_deque)
            plan_data_dict = self.design_inspector._extract_edge_list(plan_data_dict, end_of_episode=True) 
            
            self.done = False
            self.plan_data_dict = copy.deepcopy(plan_data_dict)
            self.plan_data_dict.update({'done': self.done,
                                        'state_data_dict': self.state_data_dict})
            self.active_wall_name_list = []
            return self.observation
        
    
    
    def update(self, episode, action, ep_time_step, agent_names_deque=None):
        self.episode = episode
        if ep_time_step > self.fenv_config['stop_ep_time_step']:
            print('wait in update of obervation')
            raise ValueError(f"ep_time_step went over than the limit! ep_time_step is {ep_time_step}, while the limit is self.fenv_config['stop_ep_time_step']")
        
        plan_data_dict = copy.deepcopy(self.plan_data_dict)
        if VERBOSE: print(f"obs_reset, line 189: plan_data_dict['areas_masked']: {plan_data_dict['areas_masked']}, oml areas maksed name: {[n for n in np.unique(plan_data_dict['obs_moving_labels']) if 2 <= n <= 5]}")

        if self.fenv_config['env_planning'] == 'One_Shot':
            if self.fenv_config['learn_room_size_category_order_flag']:
                self.decoded_action_dict = self.action_parser.decode_action(plan_data_dict, action)
            elif self.fenv_config['learn_room_order_directly']:
                self.decoded_action_dict = self.action_parser.decode_action_from_direct_order_learning(plan_data_dict, action)
            
            if self.decoded_action_dict['action_status'] is not None:
                self.active_wall_status, new_walls_coords = self.action_parser.select_wall(plan_data_dict, self.decoded_action_dict)
            else:
                self.active_wall_status = 'rejected_by_missing_room'
                # print("action_status is None")

            if self.active_wall_status == "check_room_area":
                self.active_wall_name = self.decoded_action_dict['active_wall_name']
                active_wall_i = self.decoded_action_dict['active_wall_i']
                
                try:
                    assert active_wall_i in self.fenv_config['real_room_id_range'], 'active_wall_i is bigger not in the range of valid real rooms' 
                except:
                    print('wait in update of observation')
                    raise ValueError('Probably sth need to be match with n_corners')
                
                plan_data_dict = self.plan_constructor.update_plan_with_active_wall(plan_data_dict, new_walls_coords, self.active_wall_name)

                plan_data_dict = self.painter.update_obs_mat(plan_data_dict, self.active_wall_name) # here we update plan_data_dict based on the wall order

                plan_data_dict = self.rextractor.update_room_dict(plan_data_dict, self.active_wall_name)

                plan_data_dict = self.plan_constructor._update_block_cells(plan_data_dict)

                # print(f"before: self.active_wall_status: {self.active_wall_status}")
                self.active_wall_status = self.design_inspector.inspect_constraints(plan_data_dict, self.active_wall_name)


                if self.active_wall_status == "accepted":
                    ### Note: active_wall_status will probably change in this section
                    self.n_actions_accepted += 1
                    plan_data_dict['wall_order'].update({self.n_actions_accepted: self.decoded_action_dict['active_wall_name']})
                    plan_data_dict['actions_accepted'].append(action)
                    plan_data_dict['wall_types'].update({self.decoded_action_dict['active_wall_name']: self.decoded_action_dict['wall_type']})
                    plan_data_dict['room_wall_occupied_positions'].extend(np.argwhere(plan_data_dict['obs_moving_ones']==1).tolist())
                    
                    if self.fenv_config['learn_room_size_category_order_flag']:
                        del plan_data_dict['room_i_per_size_category'][self.decoded_action_dict['room_size_cat_name']][0]
                        del plan_data_dict['room_area_per_size_category'][self.decoded_action_dict['room_size_cat_name']][0]   
                
                    self.done, self.active_wall_status = self._check_terminate(plan_data_dict, self.active_wall_name, self.active_wall_status, ep_time_step)
                    
                    plan_data_dict = self.state_composer.warooge_data_extractor(plan_data_dict)
                    plan_data_dict = self.state_composer.refine_moving_labels(plan_data_dict)
                    
                    if 'create' not in self.fenv_config['plan_config_source_name'] :
                        plan_data_dict = self.design_inspector._extract_edge_list(plan_data_dict, end_of_episode=False) 
                    
                    self.observation, plan_data_dict = self._make_observation(plan_data_dict, self.active_wall_name, self.active_wall_status)

                    if self.fenv_config['action_masking_flag']:
                        if self.done:
                            self.observation = {'action_mask': np.zeros(self.fenv_config['n_actions'], dtype=np.int16),
                                                'action_avail': np.ones(self.fenv_config['n_actions'], dtype=np.int16),
                                                'real_obs': self.observation}
                        else: # not self.done:
                            try:
                                self.observation = {'action_mask': self.action_parser.get_masked_actions(plan_data_dict),
                                                    'action_avail': np.ones(self.fenv_config['n_actions'], dtype=np.int16),
                                                    'real_obs': self.observation}
                            except:
                                self.observation = {'action_mask': np.zeros(self.fenv_config['n_actions'], dtype=np.int16),
                                                    'action_avail': np.ones(self.fenv_config['n_actions'], dtype=np.int16),
                                                    'real_obs': self.observation}
                                # print("Continue with no action left for this episode: badly_stopped!")
                                self.done = True
                                self.active_wall_status = "badly_stopped_"

                    self.plan_data_dict = copy.deepcopy(plan_data_dict) # only in this situation I change the self.plan_data_dict

                else: # self.active_wall_status != "accepted"
                    self.done, self.active_wall_status = self._is_time_over(self.active_wall_status, ep_time_step)
                
            else: # self.active_wall_status != "check_room_area"
                self.done, self.active_wall_status = self._is_time_over(self.active_wall_status, ep_time_step)
                
            
            if self.fenv_config['zero_constraint_flag']:
                if self.done and (self.active_wall_status not in ['check_room_area', 'accepted']):
                    self.plan_data_dict = self.design_inspector.inspect_objective(plan_data_dict)
            
            self.plan_data_dict.update({'done': self.done,
                                        'active_wall_status': self.active_wall_status,
                                        'obs_arr_conv': self.observation,
                                        'ep_time_step': ep_time_step})
        
        elif self.fenv_config['env_planning'] == 'Dynamic':
            self.plan_data_dict_backup = copy.deepcopy(self.plan_data_dict)
            self.shifted_agent_names_deque_backup = copy.deepcopy(agent_names_deque)
            self.observation_backup = copy.deepcopy(self.observation)
            self.done_backup = copy.deepcopy(self.done)
            
            self.decoded_action_dict = self.action_parser.decode_action_for_dynamic_agent(plan_data_dict, action)
            
            if self.decoded_action_dict['action_status'] != 'check':
                active_wall_status = 'rejected_by_not_available_room'
                self.plan_data_dict = copy.deepcopy(self.plan_data_dict_backup)
                self.shifted_agent_names_deque = self.shifted_agent_names_deque_backup
                self.observation = self.observation_backup
                self.done = self.done_backup
                self.active_wall_status = active_wall_status
                done, active_wall_status = self._is_time_over(active_wall_status, ep_time_step)
                if done:
                    self.done = done
                    self.active_wall_status = active_wall_status
            else:
                # TODO: assert self.decoded_action_dict['action_status'] == 'check', 'in this case action status here should be check'. no this is wrong. 
                active_agent_name = f"agent_{self.decoded_action_dict['active_wall_i']}"
                if active_agent_name is None:
                    raise ValueError(f"active_agent_name is None, while it should not be None")
                else:
                    shifted_agent_names_deque = copy.deepcopy(agent_names_deque)
                    shifted_agent_names_deque.remove(active_agent_name)
                    shifted_agent_names_deque.append(active_agent_name) # move the current agent to the last index. Because, we first need to draw the other agent, and then the current agent acts according to them
                    previous_wall_id = f"wall_{shifted_agent_names_deque[-2].split('_')[1]}"
                
                new_walls_coords = self._transform_walls(plan_data_dict, self.decoded_action_dict['active_wall_i'], self.decoded_action_dict['active_wall_transformation_i']) # for asp, actions only include the current agent. So, we only transform the corressponding walls
                plan_data_dict = copy.deepcopy(self.input_plan_data_dict)
                
                
                def off_light_plan_composer(plan_data_dict, shifted_agent_names_deque):
                    # continue based on obs_mat_base_w, keep the base of the moving wall. complete the plan by checking the contraints, and in the end compute the rooms
                    inwalls_coords = {}
                    ## TODO: when a wall moves off-light, do we place the new base wall into the plan, or the old one?
                    for i, ag_name in enumerate(shifted_agent_names_deque): # active_agent is already the very last agent
                        inwall_i = int(ag_name.split('_')[1])
                        inwall_name = f"wall_{inwall_i}"
                        inwall_coords = new_walls_coords[inwall_name]
                        inwalls_coords.update({inwall_name: inwall_coords})
                        plan_data_dict.update({'inwalls_coords': inwalls_coords})
                        plan_data_dict = self.plan_constructor.update_plan_with_active_wall(plan_data_dict=plan_data_dict, walls_coords=plan_data_dict['inwalls_coords'], active_wall_name=inwall_name)
                        plan_data_dict = self.painter.update_obs_mat(plan_data_dict, inwall_name)
                        active_wall_status = self.design_inspector.inspect_entrace_lvroom_relation_for_dyp(plan_data_dict, inwall_name)
                        if 'reject' in active_wall_status:
                            break
                        plan_data_dict['wall_order'].update({i+1: inwall_name})
                        plan_data_dict['wall_types'].update({i+1: inwall_name})
                    return plan_data_dict, active_wall_status
                        
                def on_light_plan_composer(plan_data_dict, shifted_agent_names_deque):
                    inwalls_coords = {}
                    for i, ag_name in enumerate(shifted_agent_names_deque): # active_agent is already the very last agent
                        inwall_i = int(ag_name.split('_')[1])
                        inwall_name = f"wall_{inwall_i}"
                        inwall_coords = new_walls_coords[inwall_name]
                        inwalls_coords.update({inwall_name: inwall_coords})
                        plan_data_dict.update({'inwalls_coords': inwalls_coords})
                        plan_data_dict = self.plan_constructor.update_plan_with_active_wall(plan_data_dict=plan_data_dict, walls_coords=plan_data_dict['inwalls_coords'], active_wall_name=inwall_name)
                        plan_data_dict['wall_order'].update({i+1: inwall_name})
                        plan_data_dict['wall_types'].update({i+1: inwall_name})
                    plan_data_dict = self.painter.update_obs_mat_for_dyp_step_size(self.input_plan_data_dict, plan_data_dict, shifted_agent_names_deque)
                    active_wall_status = self.design_inspector.inspect_entrace_lvroom_relation_for_dyp(plan_data_dict)
                    return plan_data_dict, active_wall_status
                
                
                if self.fenv_config['moving_laser_mode'] == 'off_light':
                    plan_data_dict['new_walls_coords_dyp'] = new_walls_coords
                    plan_data_dict, wall_status = off_light_plan_composer(plan_data_dict, shifted_agent_names_deque)
                elif self.fenv_config['moving_laser_mode'] == 'on_light':
                    plan_data_dict, wall_status = on_light_plan_composer(plan_data_dict, shifted_agent_names_deque)
                else: # flashing_light
                    if self.decoded_action_dict['active_wall_light_status_i'] == 0: # off-light
                        plan_data_dict, wall_status = off_light_plan_composer(plan_data_dict, shifted_agent_names_deque)
                    else: # on-light
                        plan_data_dict, wall_status = on_light_plan_composer(plan_data_dict, shifted_agent_names_deque)
                    
                if 'reject' not in wall_status:
                    if VERBOSE: print(f"obs_reset, line 189: plan_data_dict['areas_masked']: {plan_data_dict['areas_masked']}, oml areas maksed name: {[n for n in np.unique(plan_data_dict['obs_moving_labels']) if 2 <= n <= 5]}")
                    # here we use plan_data_dict_backup, bc plan_data_dict has been just updated with input_plan_data_dict, so it might not include enought data. but thsi is not correct. as we just updated plan_data_dict with painter
                    wall_repulsion_mat_without_active_wall = self.plan_constructor.get_wall_repulsion_cells(plan_data_dict, self.decoded_action_dict['active_wall_name'])
                    active_wall_status = self.inspect_new_wall_coords(plan_data_dict['obs_mat_base_w'], wall_repulsion_mat_without_active_wall, self.decoded_action_dict['active_wall_i'])
                else:
                    active_wall_status = 'reject' 
    
                if 'reject' in active_wall_status:
                    self.plan_data_dict = copy.deepcopy(self.plan_data_dict_backup)
                    self.shifted_agent_names_deque = self.shifted_agent_names_deque_backup
                    self.observation = self.observation_backup
                    self.done = self.done_backup
                    self.active_wall_status = active_wall_status
                    done, active_wall_status = self._is_time_over(active_wall_status, ep_time_step)
                    if done:
                        self.done = done
                        self.active_wall_status = active_wall_status
                                    
                else:
                    cleaned_obs_matrix, wall_matrices = self.room_assignment.get_clean_matrices(plan_data_dict['obs_mat_w'])
                    plan_data_dict.update({'shifted_agent_names_deque': shifted_agent_names_deque})
                    plan_data_dict.update({'cleaned_obs_matrix': cleaned_obs_matrix})
                    plan_data_dict.update({'wall_matrices': wall_matrices})
                    obs_mat_w, obs_moving_labels, affected_rooms, wm =  self.room_assignment.assign_rooms(self.plan_data_dict_backup, plan_data_dict)
                    plan_data_dict.update({'wall_matrices': wm}) # this line is correct, this does not do any thing for identity full, but does sth for identity less
                    plan_data_dict.update({'obs_moving_labels': obs_moving_labels, 'affected_rooms': affected_rooms})
                    # assert np.array_equal(obs_mat_w, plan_data_dict['obs_mat_w']), "obs_mat_w arrays are not exactly equal"
                    active_wall_status = self._check_if_any_room_disapread(plan_data_dict)
                    
                    if 'reject' in active_wall_status:
                        self.plan_data_dict = copy.deepcopy(self.plan_data_dict_backup)
                        self.shifted_agent_names_deque = self.shifted_agent_names_deque_backup
                        self.observation = self.observation_backup
                        self.done = self.done_backup
                        self.active_wall_status = active_wall_status
                        done, active_wall_status = self._is_time_over(active_wall_status, ep_time_step)
                        if done:
                            self.done = done
                            self.active_wall_status = active_wall_status
                        
                    else:
                        # plan_data_dict = self.plan_constructor._update_block_cells(plan_data_dict)
                        if VERBOSE: print(f"obs_reset, line 189: plan_data_dict['areas_masked']: {plan_data_dict['areas_masked']}, oml areas maksed name: {[n for n in np.unique(plan_data_dict['obs_moving_labels']) if 2 <= n <= 5]}")
                        plan_data_dict =  self.room_assignment.get_geometrical_properties(plan_data_dict) # obs_moving_labels_ : walled merged to it
                        plan_data_dict = self.state_composer.warooge_data_extractor(plan_data_dict)
                        plan_data_dict = self.state_composer.refine_moving_labels(plan_data_dict)
                        
                        plan_data_dict = self.design_inspector._extract_edge_list(plan_data_dict, end_of_episode=True) 
                        
                        observation, plan_data_dict = self._make_observation(plan_data_dict=plan_data_dict, 
                                                                             active_wall_name=self.decoded_action_dict['active_wall_name'], 
                                                                             active_wall_status=active_wall_status,
                                                                             agent_names_deque=agent_names_deque)
                        done, active_wall_status = self._is_time_over(active_wall_status, ep_time_step)
                        
                        self.plan_data_dict = copy.deepcopy(plan_data_dict)
                        self.shifted_agent_names_deque = shifted_agent_names_deque
                        self.observation = observation
                        self.done = done
                        self.active_wall_status = active_wall_status
                        
                        self.active_wall_name_list.append(self.decoded_action_dict['active_wall_name'])
                    
            # print(f"Action: {self.decoded_action_dict['action']:3d}, Room: {self.decoded_action_dict['active_room_i']:2d}, Transformation: {self.decoded_action_dict['active_wall_transformation_i']:2d}, Light: {self.decoded_action_dict['active_wall_light_status_i']:2d}, active_wall_status: {active_wall_status}")

        else:
            raise ValueError(f"Invalid env_type: {self.fenv_config['env_type']}")
            
        return self.observation
    
    
    
    def inspect_new_wall_coords(self, obs_mat_base_w, wall_repulsion_mat_without_active_wall, active_wall_i):
        potential_active_wall_positions = np.array(np.where(obs_mat_base_w == -active_wall_i)).flatten('F').reshape(-1, 2)
        active_wall_status = 'accepted'
        for r, c in potential_active_wall_positions:
            if wall_repulsion_mat_without_active_wall[r, c] != 0:
                active_wall_status = 'reject_by_repulsion'
                break
        return active_wall_status
           
    
    def _check_if_any_room_disapread(self, plan_data_dict):
        oml = copy.deepcopy(plan_data_dict['obs_moving_labels'])
        all_rooms_ids = np.unique(oml)
        real_room_ids = set([r for r in all_rooms_ids if r in self.fenv_config['real_room_id_range']])
        expected_room_ids = set(  np.array(range(plan_data_dict['n_rooms'])) + self.fenv_config['min_room_id']  )
        if len(expected_room_ids - real_room_ids) == 0:
            active_wall_status = 'accepted'
        else:
            active_wall_status = 'rejected_by_a_disapeared_room'
        return active_wall_status
        
        
        
    
    def place_base_walls(self, plan_data_dict:dict, new_walls_coords:dict, shifted_agent_names_deque):
        obs_mat_base = copy.deepcopy(plan_data_dict['obs_mat_base'])
        obs_mat_base_w = copy.deepcopy(plan_data_dict['obs_mat_base_w'])
        obs_mat_base_for_dot_prod = copy.deepcopy(plan_data_dict['obs_mat_base_for_dot_prod'])
        
        obs_mat = copy.deepcopy(plan_data_dict['obs_mat'])
        obs_mat_w = copy.deepcopy(plan_data_dict['obs_mat_w'])
        obs_mat_for_dot_prod = copy.deepcopy(plan_data_dict['obs_mat_for_dot_prod'])
        
        for i, ag_name in enumerate(shifted_agent_names_deque):
            inwall_i = int(ag_name.split('_')[1])
            inwall_name = f"wall_{inwall_i}"
            inwall_coords = new_walls_coords[inwall_name]
            
            for coord in inwall_coords['base_coords']:
                r, c = self._cartesian2image_coord(coord[0], coord[1], self.fenv_config['max_y'])
                
                obs_mat_base[r, c] = self.fenv_config['wall_pixel_value']
                obs_mat_base_w[r, c] = -inwall_i 
                obs_mat_base_for_dot_prod[r, c] = 0
                
                obs_mat[r, c] = self.fenv_config['wall_pixel_value']
                obs_mat_w[r, c] = -inwall_i 
                obs_mat_for_dot_prod[r, c] = 0
                
                

        plan_data_dict['obs_mat_base'] = copy.deepcopy(obs_mat_base)
        plan_data_dict['obs_mat_base_w'] = copy.deepcopy(obs_mat_base_w)
        plan_data_dict['obs_mat_base_for_dot_prod'] = copy.deepcopy(obs_mat_base_for_dot_prod)
        
        plan_data_dict['obs_mat'] = copy.deepcopy(obs_mat)
        plan_data_dict['obs_mat_w'] = copy.deepcopy(obs_mat_w)
        plan_data_dict['obs_mat_for_dot_prod'] = copy.deepcopy(obs_mat_for_dot_prod)
        return plan_data_dict
        
    
    
    def place_walls(self, plan_data_dict: dict, new_walls_coords: dict, shifted_agent_names_deque):
        obs_mat = copy.deepcopy(plan_data_dict['obs_mat'])
        obs_mat_w = copy.deepcopy(plan_data_dict['obs_mat_w'])
        obs_mat_for_dot_prod = copy.deepcopy(plan_data_dict['obs_mat_for_dot_prod'])
    
        for wall_name, wall_data in new_walls_coords.items():
            wall_i = int(wall_name.split('_')[1])
            
            for segment_name in ['front_segment', 'back_segment']:
                seg_val = wall_data[segment_name]
                
                start_coord = seg_val['start_coord']
                end_coord = seg_val['end_coord']
                direction = seg_val['direction']
                
                x, y = end_coord
                
                if direction == 'east':
                    while x <= self.fenv_config['max_x']:
                        r, c = self._cartesian2image_coord(x, y, self.fenv_config['max_y'])
                        if obs_mat[r, c] == 0:
                            obs_mat[r, c] = self.fenv_config['wall_pixel_value']
                            obs_mat_w[r, c] = -wall_i 
                            obs_mat_for_dot_prod[r, c] = 0
                        else:
                            break
                        x += 1
                
                elif direction == 'west':
                    while x >= 0:
                        r, c = self._cartesian2image_coord(x, y, self.fenv_config['max_y'])
                        if obs_mat[r, c] == 0:
                            obs_mat[r, c] = self.fenv_config['wall_pixel_value']
                            obs_mat_w[r, c] = -wall_i 
                            obs_mat_for_dot_prod[r, c] = 0
                        else:
                            break
                        x -= 1
                
                elif direction == 'north':
                    while y <= self.fenv_config['max_y']:
                        r, c = self._cartesian2image_coord(x, y, self.fenv_config['max_y'])
                        if obs_mat[r, c] == 0:
                            obs_mat[r, c] = self.fenv_config['wall_pixel_value']
                            obs_mat_w[r, c] = -wall_i 
                            obs_mat_for_dot_prod[r, c] = 0
                        else:
                            break
                        y += 1
                
                elif direction == 'south':
                    while y >= 0:
                        r, c = self._cartesian2image_coord(x, y, self.fenv_config['max_y'])
                        if obs_mat[r, c] == 0:
                            obs_mat[r, c] = self.fenv_config['wall_pixel_value']
                            obs_mat_w[r, c] = -wall_i 
                            obs_mat_for_dot_prod[r, c] = 0
                        else:
                            break
                        y -= 1
    
        plan_data_dict['obs_mat'] = obs_mat
        plan_data_dict['obs_mat_w'] = obs_mat_w
        plan_data_dict['obs_mat_for_dot_prod'] = obs_mat_for_dot_prod
        return plan_data_dict
        
    
    
    def _make_observation(self, plan_data_dict, active_wall_name=None, active_wall_status=None, agent_names_deque=None):
        if active_wall_name is not None:
            active_room_i = int(active_wall_name.split('_')[1])
            active_room_name = f"room_{active_room_i}"
        
        # plan_data_dict = self.state_composer.warooge_data_extractor(plan_data_dict)
        # plan_data_dict = self.state_composer.refine_moving_labels(plan_data_dict)
        
        plan_data_dict = self.state_composer.create_x_observation(plan_data_dict, active_wall_name) # i think active_wall_name does not affect dyp
        
        if self.fenv_config['net_arch'] == 'Fc':
            observation = copy.deepcopy(plan_data_dict['observation_fc'])
        
        elif self.fenv_config['net_arch'] == 'Cnn':
            observation = copy.deepcopy(plan_data_dict['observation_cnn'])
        
        elif self.fenv_config['net_arch'] == 'MetaFc':
            observation = copy.deepcopy(plan_data_dict['observation_metafc'])

        elif self.fenv_config['net_arch'] == 'MetaCnn':
            observation = copy.deepcopy(plan_data_dict['observation_metacnn'])
            
        elif self.fenv_config['net_arch'] == 'Gnn':
            observation = copy.deepcopy(plan_data_dict['observation_gnn'])

        else:
            raise ValueError(f"{self.fenv_config['net_arch']} net_arch does not exist")
            
        
        if self.fenv_config['dyp_is_multi_dim_observation']:
            observation = self.create_obs_for_dyp(plan_data_dict, agent_names_deque, observation)
            plan_data_dict.update({
                'observation': observation,
            })
            
        return observation, plan_data_dict
    
    
    def create_obs_for_dyp(self, plan_data_dict, agent_names_deque, observation):
        agent_names_deque_dict = self.rearrange_deque_to_dict(agent_names_deque)
        dummy_obs_mat_dict, dummy_obs_mat_w_dict = {}, {}
        obs_moving_ones = self.input_plan_data_dict['obs_moving_ones']
        for key, deq in agent_names_deque_dict.items():
            dummy_obs_mat = np.zeros_like(plan_data_dict['obs_mat']) + self.input_plan_data_dict['obs_mat'] + np.where(plan_data_dict['obs_mat_base_w'] <= -12, 10, 0)
            dummy_obs_mat_w = np.zeros_like(plan_data_dict['obs_mat_w']) + self.input_plan_data_dict['obs_mat_w'] + np.where(plan_data_dict['obs_mat_base_w'] <= -12, plan_data_dict['obs_mat_base_w'], 0)
            for i, ag_name in enumerate(deq[:-1]): # active_agent is already the very last agent
                inwall_i = int(ag_name.split('_')[1])
                inwall_name = f"wall_{inwall_i}"
                dummy_obs_mat, dummy_obs_mat_w = self.painter.update_dummy_obs_mat_for_dyp(dummy_obs_mat, dummy_obs_mat_w, obs_moving_ones, plan_data_dict['wall_inline_segments'], inwall_name)
            
            def _add_entrance_to_dummy_obs_mat_w(dummy_obs_mat_w):
                for r, c in plan_data_dict['entrance_positions']:
                    dummy_obs_mat_w[r][c] = -self.fenv_config['entrance_cell_id']
                return dummy_obs_mat_w
            dummy_obs_mat_w = _add_entrance_to_dummy_obs_mat_w(dummy_obs_mat_w)
            dummy_obs_mat_dict[key] = dummy_obs_mat
            dummy_obs_mat_w_dict[key] = dummy_obs_mat_w
        
        def stack_agent_arrays(agent_dict, start=12, end=19):
            # Determine the shape of the arrays
            sample_shape = next(iter(agent_dict.values())).shape
            # Create a list to hold the arrays
            stacked_arrays = []
            # Iterate through the agent numbers
            for i in range(start, end + 1):
                agent_key = f"agent_{i}"
                if agent_key in agent_dict:
                    stacked_arrays.append(agent_dict[agent_key])
                else:
                    # If the agent doesn't exist, add an array of zeros
                    stacked_arrays.append(np.zeros(sample_shape, dtype=np.int32))
            # Stack the arrays
            return np.stack(stacked_arrays, axis=0)
        arr = stack_agent_arrays(dummy_obs_mat_w_dict)
        arr = np.moveaxis(arr, 0, -1)
        arr = arr / -19.0
        if isinstance(observation, dict):
            observation_cnn = np.concatenate((arr, observation['observation_cnn']), axis=-1)
            observation['observation_cnn'] = observation_cnn
        else:
            observation= np.concatenate((arr, observation), axis=-1)
        # self.visualize_3d_array(observation_)
        return observation
        
    
    @staticmethod
    def visualize_3d_array(arr):
        import numpy as np
        import matplotlib.pyplot as plt
        # Ensure the input is a 3D numpy array
        assert isinstance(arr, np.ndarray) and arr.ndim == 3, "Input must be a 3D numpy array"
        
        # Get the number of slices (last dimension)
        num_slices = arr.shape[2]
        
        # Calculate the grid size
        grid_size = int(np.ceil(np.sqrt(num_slices)))
        
        # Create a figure and a grid of subplots
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(20, 20))
        
        # Flatten the axes array for easy indexing
        axes = axes.flatten()
        
        # Plot each slice
        for i in range(num_slices):
            im = axes[i].imshow(arr[:,:,i], cmap='viridis')
            axes[i].set_title(f'Channel {i+1}')
            axes[i].axis('off')
            fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        
        # Remove any unused subplots
        for i in range(num_slices, len(axes)):
            fig.delaxes(axes[i])
        
        # Adjust the layout and display the plot
        plt.tight_layout()
        plt.show()
        
        
    
    @staticmethod
    def rearrange_deque_to_dict(original_deque):
        result_dict = {}
        for i in range(len(original_deque)):
            # Create a new deque from the original
            new_deque = deque(original_deque)
            # Remove the i-th element and add it to the end
            element = new_deque[i]
            new_deque.remove(element)
            new_deque.append(element)
            # Add to the result dictionary
            result_dict[element] = list(new_deque)
        return result_dict
        

    
    def _check_terminate(self, plan_data_dict, active_wall_name, active_wall_status, ep_time_step):
        if len(plan_data_dict['areas_achieved']) == plan_data_dict['number_of_total_rooms']: # TODO
            done = True
            active_wall_status = 'well_finished'
        
        elif len(plan_data_dict['areas_achieved']) < plan_data_dict['number_of_total_walls']+1:
            done, active_wall_status = self._is_time_over(active_wall_status, ep_time_step)
        
        else:
            time = datetime.now().strftime("%Y%m%d_%H%M%S")
            # np.save(f"{self.fenv_config['root_dir']}/storage_nobackup/plan_data_dict_storage/plan_data_dict__{os.path.basename(__file__)}_{self.__class__.__name__}_{inspect.currentframe().f_code.co_name}_{time}.npy", self.plan_data_dict)
            message = f"""
            n_rooms cannot be bigger than num_desired_rooms. 
            The current one is {len(plan_data_dict['areas_achieved'])}, 
            while the limit is {plan_data_dict['number_of_total_rooms']}
            plan_id is: {plan_data_dict['plan_id']}
            
            """
            raise ValueError(message)
                
        return done, active_wall_status
    
    
    
    def _is_time_over(self, active_wall_status, ep_time_step):
        if self.fenv_config['env_planning'] == 'One_Shot':
            done = False
            if ep_time_step >= self.fenv_config['stop_ep_time_step']-1:
                done = True
                active_wall_status = 'badly_stopped'

        elif self.fenv_config['env_planning'] == 'Dynamic':
            done = False
            if ep_time_step >= self.fenv_config['stop_ep_time_step']-1:
                done = True
                active_wall_status = 'badly_stopped'
        else:
            raise ValueError(f"Invalid env_planning: {self.fenv_config['env_planning']}")
        return done, active_wall_status
    


    def _transform_walls(self, plan_data_dict, active_wall_i, active_wall_transformation_i): # in asp actions includes the current wall, so we only trasnform the current wall
        current_walls_coords = plan_data_dict['walls_coords'] # current_walls_coords means the walls before transformation
        new_walls_coords = copy.deepcopy(current_walls_coords)
        
        active_wall_name = f"wall_{active_wall_i}"
        active_wall_coords = current_walls_coords[active_wall_name]
        
        if self.fenv_config['action_dict'][active_wall_transformation_i] == 'no_action':
            pass
        else:
            n_wall = WallTransform(active_wall_name, active_wall_coords, active_wall_transformation_i, plan_data_dict, self.fenv_config).transform()
            new_walls_coords[active_wall_name] = list(n_wall.values())[0]
        self.__check_new_walls_coords(new_walls_coords)      
        # plan_data_dict = self._setup_plan(walls_coords=new_walls_coords) # every time we call _setup_plan, plan_data_dict updates. Meaning that plan_data_dict[0] also updates. Of course in asp we only update the new wall. and the rest of plan_data_dict will be the same as before. 
        return new_walls_coords #, plan_data_dict
    
    

    @staticmethod
    def __check_new_walls_coords(new_walls_coords):
        for wall_name, wall_data in new_walls_coords.items():
            if 'front_segment' not in wall_data.keys():
                # print(f"observation: __check_new_walls_coords: wall_name: {wall_name}, wall_data:{wall_data}")
                raise ValueError("observation: __check_new_walls_coords: why front_segment is missing?")
        # print("observation: __check_new_walls_coords: new_walls_coords is fine")
        
        
        
    @staticmethod
    def _cartesian2image_coord(x, y, max_y):
        return int(max_y-y), int(x)

        
    
#%% 
if __name__ == '__main__':
    from gym_floorplan.envs.fenv_config import LaserWallConfig
    fenv_config = LaserWallConfig().get_config()
    self = Observation(fenv_config)
    episode = None
    observation = self.obs_reset(episode)
    active_wall_name = 'wall_11'
    for ep_time_step, action in enumerate([992, 714, 874, 1456, 930, 134, 635]):
        self.update(episode, action, ep_time_step)
        
        if self.active_wall_status == "accepted":
            active_wall_name = f"wall_{int(self.active_wall_name.split('_')[1])+1}"
            
    plan_data_dict = self.plan_data_dict