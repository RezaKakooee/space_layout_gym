# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 23:34:14 2021

@author: Reza Kakooee
"""
#%%

import copy
import numpy as np
import gymnasium as gym
from collections import defaultdict

# from gym_floorplan.base_env.observation.base_observation import BaseObservation

from gym_floorplan.envs.observation.sequential_painter import SequentialPainter
from gym_floorplan.envs.observation.room_extractor import RoomExtractor
from gym_floorplan.envs.observation.plan_constructor import PlanConstructor
from gym_floorplan.envs.observation.state_composer import StateComposer
from gym_floorplan.envs.observation.action_parser import ActionParser
from gym_floorplan.envs.observation.design_inspector import DesignInspector



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
        
        self.observation_space = self._get_observation_space()

        self.time_dict_observation = defaultdict(dict)



    # @property
    def _get_observation_space(self): # def observation_space(self): 
        self.state_data_dict = self.state_composer.creat_observation_space_variables()
        
        _observation_space_fc = gym.spaces.Box(
            low=self.state_data_dict['low_fc'], 
            high=self.state_data_dict['high_fc'], 
            shape=self.state_data_dict['shape_fc'], 
            dtype=float
        )

        _observation_space_cnn = gym.spaces.Box(
            low=self.state_data_dict['low_cnn'], 
            high=self.state_data_dict['high_cnn'], 
            shape=self.state_data_dict['shape_cnn'], 
            dtype=np.uint8
        )
        
        _observation_space_meta = gym.spaces.Box(
            low=self.state_data_dict['low_meta'], 
            high=self.state_data_dict['high_meta'], 
            shape=self.state_data_dict['shape_meta'], 
            dtype=float
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
                    dtype=float,
            )
            
            _observation_space_metafc = gym.spaces.Box(
                    low=min(self.state_data_dict['low_fc'], self.state_data_dict['low_meta']), 
                    high=max(self.state_data_dict['high_fc'], self.state_data_dict['high_meta']), 
                    shape=self.state_data_dict['shape_fc'] + self.state_data_dict['shape_meta'], 
                    dtype=float,
            )
            
        # _observation_space_gnn = gym.spaces.Box(low=self.state_data_dict['low_gnn'], 
        #                                         high=self.state_data_dict['high_gnn'], 
        #                                         shape=self.state_data_dict['shape_gnn'], 
        #                                         dtype=float)
        
        # _observation_space_gnn = gym.spaces.Graph(node_space=gym.spaces.Box(low=self.state_data_dict['low_gnn'], 
        #                                                                     high=self.state_data_dict['high_gnn'], 
        #                                                                     shape=(18, 460),
        #                                                                     dtype=float), 
        #                                           edge_space=gym.spaces.Discrete(1),
        #                                           # num_nodes=self.fenv_config['max_room_id']-1,
        #                                           )
        
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
                                'action_mask': gym.spaces.Box(low=0, high=1,  shape=(self.fenv_config['n_actions'],), dtype=float),
                                'action_avail': gym.spaces.Box(low=0, high=1, shape=(self.fenv_config['n_actions'],), dtype=float),
                                'real_obs': _observation_space_fc,
                                })
        
            _observation_space_cnn = gym.spaces.Dict({
                                'action_mask': gym.spaces.Box(low=0, high=1,  shape=(self.fenv_config['n_actions'],), dtype=float),
                                'action_avail': gym.spaces.Box(low=0, high=1, shape=(self.fenv_config['n_actions'],), dtype=float),
                                'real_obs': _observation_space_cnn,
                                })
            
            _observation_space_metafc = gym.spaces.Dict({
                                'action_mask': gym.spaces.Box(low=0, high=1,  shape=(self.fenv_config['n_actions'],), dtype=float),
                                'action_avail': gym.spaces.Box(low=0, high=1, shape=(self.fenv_config['n_actions'],), dtype=float),
                                'real_obs': _observation_space_metafc,
                                })
            
            _observation_space_metacnn = gym.spaces.Dict({
                                'action_mask': gym.spaces.Box(low=0, high=1,  shape=(self.fenv_config['n_actions'],), dtype=float),
                                'action_avail': gym.spaces.Box(low=0, high=1, shape=(self.fenv_config['n_actions'],), dtype=float),
                                'real_obs': _observation_space_metacnn,
                                })
            
            _observation_space_gnn = gym.spaces.Dict({
                                'action_mask': gym.spaces.Box(low=0, high=1, shape=(self.fenv_config['n_actions'],), dtype=float),
                                'action_avail': gym.spaces.Box(low=0, high=1, shape=(self.fenv_config['n_actions'],), dtype=float),
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
            
        return self._observation_space
    
    
    
    def obs_reset(self, episode):
        self.n_actions_accepted = 0
        plan_data_dict = self.plan_constructor.get_plan_data_dict(episode=episode)
        plan_data_dict['plan_description'] = self.plan_constructor.get_plan_meta_data(plan_data_dict)
        self.input_plan_data_dict = copy.deepcopy(plan_data_dict)
        
        self.active_wall_name = None  
        self.active_wall_status = None 
        
        plan_data_dict.update({'active_wall_name': self.active_wall_name,
                               'active_wall_status': self.active_wall_status})
        
        plan_data_dict = self.state_composer.warooge_data_extractor(plan_data_dict)
        plan_data_dict = self.state_composer.refine_moving_labels(plan_data_dict)
        
        self.observation, plan_data_dict = self._make_observation(plan_data_dict)
        
        if self.fenv_config['action_masking_flag']:
            self.tmp_action_mask = self.action_parser.get_masked_actions(plan_data_dict)
            self.observation = {'action_mask': self.action_parser.get_masked_actions(plan_data_dict),
                                'action_avail': np.ones(self.fenv_config['n_actions'], dtype=np.int16),
                                'real_obs': self.observation}
        
        plan_data_dict.update({'obs_arr_conv': self.observation})
        
        self.done = False
        
        self.plan_data_dict = copy.deepcopy(plan_data_dict)
        
        shuffeled_idxs = np.random.choice(self.fenv_config['real_room_id_range'], self.fenv_config['maximum_num_real_rooms'], replace=False).tolist()
        
        self.plan_data_dict.update({'done': self.done,
                                    'shuffeled_idxs': shuffeled_idxs,
                                    'state_data_dict': self.state_data_dict})

        return self.observation
        
    
    
    def update(self, episode, action, ep_time_step):
        self.episode = episode
        if ep_time_step > self.fenv_config['stop_ep_time_step']:
            print('wait in update of obervation')
            raise ValueError(f"ep_time_step went over than the limit! ep_time_step is {ep_time_step}, while the limit is self.fenv_config['stop_ep_time_step']")
        
        plan_data_dict = copy.deepcopy(self.plan_data_dict)
        if self.fenv_config['learn_room_size_category_order_flag']:
            self.decoded_action_dict = self.action_parser.decode_action(plan_data_dict, action)
        elif self.fenv_config['learn_room_order_directly']:
            self.decoded_action_dict = self.action_parser.decode_action_from_direct_order_learning(plan_data_dict, action)
        
        if self.decoded_action_dict['action_status'] is not None:
            self.active_wall_status, new_walls_coords = self.action_parser.select_wall(plan_data_dict, self.decoded_action_dict)
        else:
            self.active_wall_status = 'rejected_by_missing_room'

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

                # TODO, uncomment following
                # try: 
                # if (self.active_wall_status in ['badly_stopped', 'well_finished'] or 
                #         self.fenv_config['phase'] in ['debug', 'test_'] ):
                #         plan_data_dict = self.design_inspector.inspect_objective(plan_data_dict)
                # else: # still we inspect to compute current adj
                #     plan_data_dict = self.design_inspector._extract_edge_list(plan_data_dict, end_of_episode=False)
                    

                # except:
                #     print(f"self.active_wall_status: {self.active_wall_status}")
                #     np.save("plan_data_dict__in_observation__update.npy", plan_data_dict)
                #     raise ValueError("design_inspector cannot be executed properly for some reason")
                    
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
                            print("Continue with no action left for this episode: badly_stopped!")
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
        
        return self.observation
    
    
    
    def _make_observation(self, plan_data_dict, active_wall_name=None, active_wall_status=None):
        if active_wall_name is not None:
            active_room_i = int(active_wall_name.split('_')[1])
            active_room_name = f"room_{active_room_i}"
        
        # plan_data_dict = self.state_composer.warooge_data_extractor(plan_data_dict)
        # plan_data_dict = self.state_composer.refine_moving_labels(plan_data_dict)
        
        plan_data_dict = self.state_composer.create_x_observation(plan_data_dict, active_wall_name)
        
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
            
        return observation, plan_data_dict
    
    
    
    def _check_terminate(self, plan_data_dict, active_wall_name, active_wall_status, ep_time_step):
        if len(plan_data_dict['areas_achieved']) == plan_data_dict['number_of_total_rooms']: # TODO
            done = True
            active_wall_status = 'well_finished'
        
        elif len(plan_data_dict['areas_achieved']) < plan_data_dict['number_of_total_walls']+1:
            done, active_wall_status = self._is_time_over(active_wall_status, ep_time_step)
        
        else:
            np.save("plan_data_dict__in__observation__check_terminate.npy", plan_data_dict)
            message = f"""
            n_rooms cannot be bigger than num_desired_rooms. 
            The current one is {len(plan_data_dict['areas_achieved'])}, 
            while the limit is {plan_data_dict['number_of_total_rooms']}
            plan_id is: {plan_data_dict['plan_id']}
            
            """
            raise ValueError(message)
                
        return done, active_wall_status
    
    
    
    def _is_time_over(self, active_wall_status, ep_time_step):
        done = False
        if ep_time_step >= self.fenv_config['stop_ep_time_step']-1:
            done = True
            active_wall_status = 'badly_stopped'
        return done, active_wall_status
    

        
    
#%% This is only for testing and debugging
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