#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 00:45:10 2023

@author: Reza Kakooee
"""

import numpy as np

import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

#%%
class DQNTrainer:
    def __init__(self, env, agent, dqn_config, writer):
        self.env = env
        self.agent = agent
        
        for k, v in dqn_config.items():
            setattr(self, k, v)
        
        self.writer = writer
        
        self.epsilon = self.max_epsilon
        
        self._initialize()
        


    def _initialize(self):
        self.epsilons = []
        self.losses = []
        self.mean_episode_rewards = []
        # self.episode_lens = []
        self.sum_episode_rewards = []
        self.update_counter = 0
        self.timestep = 0
        self.loss = 0
        
        
        
    def train(self):
        for i in range(self.n_episodes):
            
            try:
                self.dqn_rollout(i)
                if (i+1) % 10 == 0:
                    print(f"Episode: {i+1:05}, this_episode_reward: {self.this_episode_reward:05}, this_episode_len: {self.this_episode_len:05}")
                if (i+1) % self.checkpoint_freq == 0:
                    self.agent.save_model(self.model_path)
                    
                if self.timestep >= self.agent.n_iters:
                    break
                    
            except KeyboardInterrupt:
                print('Finishing ...')
                break
                
            
            
    def _update_agent(self, i):
        if len(self.agent.memory) >= self.batch_size:
            self.loss = self.agent.update_model()
            self.losses.append(self.loss)
            self.update_counter += 1
            self.agent.update_target_model(self.update_counter)
        
        if not self.use_noisy:
            self._update_epsilon()
            self.epsilons.append(self.epsilon)
            
        if self.use_per:
            self._update_beta(i, self.n_episodes)


        
    def dqn_rollout(self, i):
            self.this_episode_reward = 0
            self.this_episode_len = 0
            state, _ = self.env.reset()
            done = False
            while not done:
                action = self._select_action(state)
                next_state, reward, done, truncated, _ = self.env.step(action)
                
                transition = [state, action, next_state, reward, done]
                state = next_state
                
                if self.use_n_step:
                    self.one_step_transition = self.agent.memory_n.store(*transition)
                else:
                    self.one_step_transition = transition
                    
                if self.one_step_transition:
                    self.agent.memory.store(*self.one_step_transition)
                    
                self.timestep += 1
                self.this_episode_len += 1
                self.this_episode_reward += reward
                
                # if (self.timestep + 1) % 10 == 0:
                self._update_agent(i)
                
                if done:
                    self.mean_reward_of_this_episode = self.this_episode_reward / self.this_episode_len
                    self.mean_episode_rewards.append(self.mean_reward_of_this_episode)
                    self.sum_episode_rewards.append(self.this_episode_reward)
                    self.write_to_tb(i)
                    

                    
    def write_to_tb(self, episode_i):
        self.writer.add_scalar("charts/this_episode_reward", self.this_episode_reward, episode_i)
        self.writer.add_scalar("charts/mean_reward_of_this_episode", self.mean_reward_of_this_episode, episode_i)
        self.writer.add_scalar("charts/this_episode_len", self.this_episode_len, episode_i)
        self.writer.add_scalar("charts/epsilon", self.epsilon, episode_i)
        if len(self.agent.memory) >= self.batch_size: self.writer.add_scalar("losses/loss", self.loss, episode_i)
        
        if self.env.env_name == 'DOLW-v0':
            obs_moving_labels = self.env.obs.plan_data_dict['obs_moving_labels']
            obs_moving_labels = np.expand_dims(obs_moving_labels, 0)
            self.writer.add_image('images', obs_moving_labels, episode_i)
        # obs_cnn_arr = self.env.obs.plan_data_dict['obs_cnn_arr']
        # self.writer.add_image('images', np.array(obs_cnn_arr/255., float), i)
    
    
    
    def _select_action(self, state):
        if not self.use_noisy and (np.random.random() < self.epsilon):
            action = self.env.action_space.sample()
        else:
            state = torch.FloatTensor(state).to(self.device)
            
            if self.action_space_type == "continuous":
                action = self.agent.q_net(state).argmax() # selecting the greedy action 
                # dist = MultivariateNormal(logits, self.cov_mat)
            else:
                # logits = self.agent.q_net(state) # selecting the greedy action 
                # probs = F.softmax(logits.float(), dim=-1)
                # dist = Categorical(probs)
                # action = dist.sample()
                
                q_values = self.agent.q_net(state)
                actions = torch.argmax(q_values, dim=1).cpu().numpy()
            
            action = action.detach().cpu().numpy()
            
        return action



    def _update_epsilon(self):
        self.epsilon = max(
            self.min_epsilon,
            self.epsilon - (self.max_epsilon - self.min_epsilon) * self.epsilon_decay
            )
        
        
        
    def _update_beta(self, i, n_episodes):
        fraction = min(i / n_episodes, 1.0)
        self.beta = self.beta + fraction * (1 - self.beta)