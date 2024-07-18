#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 00:36:56 2023

@author: Reza Kakooee
"""


#%%
# import copy
import torch
import numpy as np
import pandas as pd
from collections import defaultdict
# from typing import Dict

from rl_logger import RlLogger



#%%
class PgRollout:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
            
        for k, v in agent.agent_config.items():
            setattr(self, k, v)
            
        self.sucess_counter = 0
        self.rollout_counter = 0
        
        self.plan_df = pd.DataFrame()
        self.res_df = pd.DataFrame()
        self.res_summary_df = pd.DataFrame()
        
        if agent.agent_config['phase'] == 'test':
            self.flogger = RlLogger(self.env.fenv_config, self.agent.agent_config)
        

        
    def _reset_observation_container(self):
        if self.net_arch == 'Fc':
            b_observations = np.zeros((self.n_interacts, self.n_envs, self.obs_dim), dtype=np.float32)
        elif self.net_arch == 'Cnn':
            b_observations = np.zeros((self.n_interacts, self.n_envs, *self.obs_dim), dtype=np.float32)
        elif self.net_arch == 'MetaFc':
            b_observations = {'observation_fc': np.zeros((self.n_interacts, self.n_envs, self.obs_fc_dim), dtype=np.float32),
                              'observation_meta': np.zeros((self.n_interacts, self.n_envs, self.obs_meta_dim), dtype=np.float32)}
        elif self.net_arch == 'MetaCnn':
            b_observations = {'observation_cnn': np.zeros((self.n_interacts, self.n_envs, *self.obs_cnn_dim), dtype=np.float32),
                              'observation_meta': np.zeros((self.n_interacts, self.n_envs, self.obs_meta_dim), dtype=np.float32)}
        else:
            raise NotImplementedError
            
        return b_observations
            
            
    
    def _append_to_observation_container(self, b_observations, observation, i):
        if self.net_arch == 'Fc':
            b_observations[i, :] = observation
        elif self.net_arch == 'Cnn':
            b_observations[i, :, :, :] = observation
        elif self.net_arch == 'MetaFc':
            b_observations['observation_fc'][i, :] = observation['observation_fc']
            b_observations['observation_meta'][i, :] = observation['observation_meta']
        elif self.net_arch == 'MetaCnn':
            b_observations['observation_cnn'][i, :, :, :] = observation['observation_cnn']
            b_observations['observation_meta'][i, :] = observation['observation_meta']
        else:
            raise NotImplementedError
            
        return b_observations
            
    

    def interact(self, n_interacts):
        self.n_interacts = n_interacts
        self.rollout_counter += 1
        
        b_observations = self._reset_observation_container()
        b_action_masks = np.zeros((self.n_interacts, self.n_envs, self.act_dim), dtype=np.int32)
        b_actions = np.zeros((self.n_interacts, self.n_envs), dtype=np.int32)
        b_log_pis = np.zeros((self.n_interacts, self.n_envs), dtype=np.float32)
        b_values = np.zeros((self.n_interacts+1, self.n_envs), dtype=np.float32)
        b_rewards = np.zeros((self.n_interacts, self.n_envs), dtype=np.float32)
        b_dones = np.zeros((self.n_interacts, self.n_envs), dtype=np.bool)
        
        b_infos = []
        infos = []
        self.info_dict = defaultdict(dict)
        self.res_dict = defaultdict(dict)

        observation, _ = self.env.reset()
        for i in range(self.n_interacts):
            b_observations = self._append_to_observation_container(b_observations, observation, i)
            
            with torch.no_grad():
                action, log_pi, value = self.agent.select_action(observation)
                
            state, reward, done, truncated, info = self.env.step(action)
            observation = state['real_obs'] if self.action_masking else state
            
            if self.action_masking: b_action_masks[i, :] = state['action_mask']
            b_actions[i, :] = action
            b_log_pis[i, :] = log_pi
            b_values[i, :] = value.squeeze()
            b_rewards[i, :] = reward
            b_dones[i, :] = done
            b_infos.append(info)
            
            try:
                if self.phase == 'train':
                    how_done = self._save_info_when_any_env_done(info, done, i)
            except:
                pass
                    
        
        with torch.no_grad():
            value = self.agent.get_value(observation)
            b_values[self.n_interacts, :] = value.squeeze()
            b_advantages = self.compute_gae(b_dones, b_rewards, b_values)
        
        
        batch = self._to_batch(b_observations, b_actions, b_log_pis, b_values, b_advantages)
        stats = self._to_stats(b_rewards, b_dones)
        
        
        try:
            if self.phase == 'train':
                self._save_on_interaction_end(how_done)
        except:
            pass
        
        return batch, stats
    
    
    
    def _to_batch(self, b_observations, b_actions, b_log_pis, b_values, b_advantages):
        if self.net_arch == 'Fc':
            bo_ = b_observations.reshape((-1,) + (self.obs_dim,))
        elif self.net_arch == 'Cnn':
            bo_ = b_observations.reshape((-1,) + self.obs_dim)
        elif self.net_arch == 'MetaFc':
            bo_ = {'observation_fc': b_observations['observation_fc'].reshape((-1,) + (self.obs_fc_dim,)),
                   'observation_meta': b_observations['observation_meta'].reshape((-1,) + (self.obs_meta_dim,))}
        elif self.net_arch == 'MetaCnn':
            bo_ = {'observation_cnn': b_observations['observation_cnn'].reshape((-1,) + self.obs_cnn_dim),
                   'observation_meta': b_observations['observation_meta'].reshape((-1,) + (self.obs_meta_dim,))}
        else:
            raise NotImplementedError
            
        batch = {
            'observations': bo_,
            # 'action_masks': b_action_masks.reshape(-1),
            'actions': b_actions.reshape(-1),
            'log_pis': b_log_pis.reshape(-1),
            'values': b_values[:-1, :].reshape(-1),
            # 'rewards': b_rewards.reshape(-1),
            # 'dones': b_dones.reshape(-1),
            # 'infos': b_infos.reshape(-1),
            'advantages': b_advantages.reshape(-1),
            }
        return batch
    
    
    
    def _to_stats(self, b_rewards, b_dones):
        stats = {
            'mean_b_rewards': np.mean(b_rewards),
            'max_b_rewards': np.max(b_rewards),
            'min_b_rewards': np.min(b_rewards),
            'sum_b_rewards': np.sum(b_rewards),
            'mean_ep_len': np.size(b_dones) / np.sum(b_dones),
            }
        return stats
    
    
    
    def _save_info_when_any_env_done(self, info, done, i):
        how_done = 'badly_finished'
        sum_done = sum(done) if isinstance(done, np.ndarray) else sum([done])
        if sum_done >= 1:
            done_idxs = np.array(np.where(done == True)).squeeze().tolist()
            # done_infos = np.array(info)[done_idxs]
            if not isinstance(done_idxs, list):
                done_idxs = [done_idxs]
            try:
                info = info['final_info']
                res_dict_keys = ['episode', 'ep_len', 'ep_mean_reward', 'ep_sum_reward', 'ep_max_reward', 'ep_last_reward']
                for di in done_idxs:
                    if 'env_data' in info[di]:
                        how_done = 'well_finished'
                        self.sucess_counter += 1
                        self.info_dict[self.sucess_counter] = info[di]['env_data']
                        self.res_dict[i] = {key: info[di]['env_data'][key] for key in res_dict_keys}
                    # else:
                    #     self.res_dict[i] = {key: info[di]['episode_end_data'][key] for key in res_dict_keys}
            except:
                print("wait in pg_rollout")
                raise ValueError('sth does not work correctly!')
                
        return how_done
    
                    
    
    def _save_on_interaction_end(self, how_done):
        if how_done == 'well_finished':
            info_df = pd.DataFrame.from_dict(self.info_dict).T
            self.plan_df = pd.concat([self.plan_df, info_df])# self.plan_df.append(info_df, ignore_index=True)
            
            res_df = pd.DataFrame.from_dict(self.res_dict).T
            self.res_df = pd.concat([self.res_df, res_df])#self.res_df.append(res_df, ignore_index=True)
            
            df_ = pd.DataFrame({
                'ep_len_avg': res_df['ep_len'].mean(),
                'ep_last_reward_avg': res_df['ep_last_reward'].mean(),
                }, index=[self.rollout_counter])
            self.res_summary_df = pd.concat([self.res_summary_df, df_])#self.res_summary_df.append(df_, ignore_index=True)
        
        
    
    def compute_gae(self, b_dones, b_rewards, b_values):
        b_advantages = np.zeros((self.n_interacts, self.n_envs), dtype=np.float32)
        last_advantage = 0
        last_value = b_values[-1, :]
        
        for t in reversed(range(self.n_interacts)):
            mask = 1.0 - b_dones[t, :]
            last_value = last_value * mask
            last_advantage = last_advantage * mask
            delta = b_rewards[t, :] + self.gamma * last_value - b_values[t, :]
            last_advantage = delta + self.gamma * self.gae_lambda * last_advantage
            b_advantages[t, :] = last_advantage
            last_value = b_values[t, :]

        return b_advantages
    
    
    
    def one_episode_rollout(self):
        this_episode_actions = []
        this_episode_rewards = []
        
        state, _ = self.env.reset()
        if self.net_arch == 'MetaCnn':
            # observation = state['real_obs'] if isinstance(state, dict) else state
            observation = {k: np.expand_dims(S, 0) for k, S in state.items()}
        else:
            observation = np.expand_dims(state, 0)
            
        done = False
        episode_i = 0
        selected_actions = []
        while not done:
            action, _, _ = self.agent.select_action(observation)
            selected_actions.append(action[0])
            if self.test_mode == 'offline':
                actions = self.env.obs.plan_constructor.adjustable_configs['potential_good_action_sequence']
                action = actions[episode_i]
            
            next_state, reward, done, truncated, _ = self.env.step(action)
            if self.net_arch == 'MetaCnn':
                # next_observation = next_state['real_obs'] if isinstance(state, dict) else next_state
                next_observation = {k: np.expand_dims(S, 0) for k, S in state.items()}
                
            else:
                next_observation = np.expand_dims(next_state, 0)
            
            this_episode_actions.append(action)
            this_episode_rewards.append(reward)
            
            observation = next_observation.copy()
            
            episode_i += 1
            
            if self.test_render_verbose >= 2:
                self.env.render()

        if done and self.env.obs.active_wall_status == "well_finished":        
            self.flogger.end_episode_logger(self.env.obs.plan_data_dict, reward, episode_i)
        
            self.env.render()
        else:
            if self.test_print_verbose >= 1: print("Episode badly finished!")
        
        episode_data = {
            'episode_actions': this_episode_actions,
            'episode_rewards': this_episode_rewards,
            'mean_episode_reward': np.mean(this_episode_rewards),
            'sum_episode_reward': np.sum(this_episode_rewards),
            'episode_len': episode_i,
            }

        return episode_data