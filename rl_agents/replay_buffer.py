#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 14:33:47 2023

@author: Reza Kakooee, Adapted from here: https://github.com/Curt-Park/rainbow-is-all-you-need
"""

import random
import numpy as np
from typing import List, Tuple, Dict, Deque
from collections import deque

from segment_tree import SumSegmentTree, MinSegmentTree



#%%
class ReplayBuffer_v0:
    def __init__(self, obs_dim: int, capacity: int, batch_size: int=32):
        self.capacity = capacity
        self.batch_size = batch_size
        
        self.obs_buf = np.zeros([capacity, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([capacity], dtype=np.float32)
        self.next_obs_buf = np.zeros([capacity, obs_dim], dtype=np.float32)
        self.rews_buf = np.zeros([capacity], dtype=np.float32)
        self.dones_buf = np.zeros([capacity], dtype=np.float32)
        
        self.ptr = 0
        self.curr_size = 0
        
    
    def store(self, obs: np.ndarray,
                    act: np.ndarray,
                    next_obs: np.ndarray,
                    rew: float,
                    done: bool,
                    ):
        
        self.obs_buf[self.ptr] = obs
        self.acts_buf[self.ptr] = act
        self.next_obs_buf[self.ptr] = next_obs
        self.rews_buf[self.ptr] = rew
        self.dones_buf[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.curr_size = min(self.curr_size + 1, self.capacity)
        
        
    def sample_batch(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.curr_size, size=self.batch_size, replace=False)
        
        batch = {
            'obs': self.obs_buf[idxs],
            'act': self.acts_buf[idxs],
            'next_obs': self.next_obs_buf[idxs],
            'rew': self.rews_buf[idxs],
            'done': self.dones_buf[idxs]
            }
        return batch
    
    
    def __len__(self):
        return self.curr_size
    
    

#%%
class ReplayBuffer:
    def __init__(self, obs_dim: int, 
                       capacity: int, 
                       batch_size: int=32,
                       n_step:int=1, 
                       gamma:float=0.99):
        
        self.capacity = capacity
        self.batch_size = batch_size
        self.n_step = n_step
        self.gamma = gamma
        
        self.obs_buf = np.zeros([capacity, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([capacity], dtype=np.float32)
        self.next_obs_buf = np.zeros([capacity, obs_dim], dtype=np.float32)
        self.rews_buf = np.zeros([capacity], dtype=np.float32)
        self.dones_buf = np.zeros([capacity], dtype=np.float32)
        
        self.n_step_buffer = deque(maxlen=n_step)
        
        self.ptr = 0
        self.curr_size = 0
    
    
    def store(self, obs: np.ndarray,
                    act: np.ndarray,
                    next_obs: np.ndarray,
                    rew: float,
                    done: bool
             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]:
        
        transition = (obs, act, next_obs, rew, done)
        self.n_step_buffer.append(transition)
        
        if self.n_step > 1:
            if len(self.n_step_buffer) < self.n_step:
                return ()
            
            obs, act = self.n_step_buffer[0][:2]
            next_obs, rew, done = self._get_n_step_info(self.n_step_buffer, self.gamma)
        
        self.obs_buf[self.ptr] = obs
        self.acts_buf[self.ptr] = act
        self.next_obs_buf[self.ptr] = next_obs
        self.rews_buf[self.ptr] = rew
        self.dones_buf[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.curr_size = min(self.curr_size + 1, self.capacity)
        
        return self.n_step_buffer[0] # why we return this? becasue if n_step = 1 then we only care about the first transition
    

    def sample_batch(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.curr_size, size=self.batch_size, replace=False)
        
        batch = {
            'obs': self.obs_buf[idxs],
            'act': self.acts_buf[idxs],
            'next_obs': self.next_obs_buf[idxs],
            'rew': self.rews_buf[idxs],
            'done': self.dones_buf[idxs],
            'idxs': idxs
            }
    
        return batch
    
    
    def sample_batch_from_idxs(self, idxs) -> Dict[str, np.ndarray]:
        batch = {
            'obs': self.obs_buf[idxs],
            'act': self.acts_buf[idxs],
            'next_obs': self.next_obs_buf[idxs],
            'rew': self.rews_buf[idxs],
            'done': self.dones_buf[idxs]
            }
    
        return batch
        
        
    def _get_n_step_info(self, n_step_buffer: Deque, gamma:float) -> Tuple[np.ndarray, np.int64, bool]:
        next_obs, rew, done = n_step_buffer[-1][-3:]
        # why we have [:-1] in the line below? bc we already pop this in the above line
        for transition in reversed(list(n_step_buffer)[:-1]): # reverse order means: r_{t+3}, r_{t+2}, r_{t+1}, r_{t}
            n_o, r, d = transition[-3:]
            rew = r + gamma * rew * (1 - d)
            next_obs, done = (n_o, d) if d else (next_obs, done) #?
            
        return next_obs, rew, done
        

    def __len__(self):
        return self.curr_size
    


#%%
class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, obs_dim: int, 
                       capacity:int, 
                       batch_size:int=32, 
                       alpha: float=0.6, 
                       beta: float=0.6,
                       n_step: int=1,
                       gamma: float=0.99):
        
        assert alpha >= 0
        
        super(PrioritizedReplayBuffer, self).__init__(obs_dim, capacity, batch_size)
        
        self.alpha = alpha
        self.beta = beta
        
        self.n_step = n_step
        self.gamma = gamma

        self.max_priority = 1.0
        self.tree_ptr = 0

        tree_capacity = 1
        while tree_capacity < self.capacity:
            tree_capacity *= 2
        
        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)


    def store(self, 
              obs: np.ndarray,
              act: int,
              next_obs: np.ndarray,
              rew: float,
              done: bool):

        transition = super().store(obs, act, next_obs, rew, done)
        
        if transition:
            self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
            self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
            self.tree_ptr = (self.tree_ptr + 1) % self.capacity
        
        return transition
    
    
    def sample_batch(self) -> Dict[str, np.ndarray]:
        assert len(self) >= self.batch_size
        assert self.beta > 0
        
        idxs = self._sample_proportional()

        batch = {
            'obs': self.obs_buf[idxs],
            'act': self.acts_buf[idxs],
            'next_obs': self.next_obs_buf[idxs],
            'rew': self.rews_buf[idxs],
            'done': self.dones_buf[idxs],
            'weights': np.array([self._calculate_weight(i, self.beta) for i in idxs]),
            'idxs': idxs,
        }
        
        return batch

    
    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)
          
            
    def _sample_proportional(self) -> List[int]:
        """Sample indices based on proportions."""
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / self.batch_size
        
        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)
            
        return indices
    
    
    def _calculate_weight(self, idx: int, beta: float):
        """Calculate the weight of the experience at idx."""
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)
        
        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight
        
        return weight