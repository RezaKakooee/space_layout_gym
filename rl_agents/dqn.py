#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 15:48:23 2023

@author: Reza Kakooee
"""
import os 
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import numpy as np
from typing import Dict



import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

# import torchvision

from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from network import DqNet, NoisyDqNet, C51Net, RainbowNet, TinyLinearNet


#%%
class DQN:
    def __init__(self, env, dqn_config):
        self.env = env
        
        for k, v in dqn_config.items():
            setattr(self, k, v)
        
        if self.use_per:
            self.memory = PrioritizedReplayBuffer(self.obs_dim, self.memory_capacity, self.batch_size, self.alpha, self.beta)
        else:
            self.memory = ReplayBuffer(self.obs_dim, self.memory_capacity, self.batch_size)
        
        if self.use_n_step:
            self.memory_n = ReplayBuffer(self.obs_dim, self.memory_capacity, self.batch_size, n_step=self.n_step, gamma=self.gamma)
        
        if self.network_name == 'TinyLinearNet':
            self.q_net = TinyLinearNet(self.obs_dim, self.act_dim).to(self.device)
            self.q_net_target = TinyLinearNet(self.obs_dim, self.act_dim).to(self.device)
            
        if self.network_name == 'NoisyDqNet':
            self.q_net = NoisyDqNet(self.obs_dim, self.act_dim).to(self.device)
            self.q_net_target = NoisyDqNet(self.obs_dim, self.act_dim).to(self.device)
            
        elif self.network_name == 'C51Net':
            self.support = torch.linspace(self.v_min, self.v_max, self.atom_size).to(self.device)
            self.q_net = C51Net(self.obs_dim, self.act_dim, self.hidden_dim, self.atom_size, self.support).to(self.device)
            self.q_net_target = C51Net(self.obs_dim, self.act_dim, self.hidden_dim, self.atom_size, self.support).to(self.device)
            
        elif self.network_name == "RainbowNet":
            self.support = torch.linspace(self.v_min, self.v_max, self.atom_size).to(self.device)
            self.q_net = RainbowNet(self.obs_dim, self.act_dim, self.hidden_dim, self.atom_size, self.support).to(self.device)
            self.q_net_target = RainbowNet(self.obs_dim, self.act_dim, self.hidden_dim, self.atom_size, self.support).to(self.device)
            
        else:
            self.q_net = DqNet(self.obs_dim, self.act_dim, self.hidden_dim).to(self.device)
            self.q_net_target = DqNet(self.obs_dim, self.act_dim, self.hidden_dim).to(self.device)
            
            
        self.q_net_target.load_state_dict(self.q_net.state_dict())
        self.q_net_target.eval()
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.q_net_lr)
        
        self.transition = []
        
        
        
    def update_model(self) -> torch.Tensor:
        batch = self.memory.sample_batch()  
        loss = self._compute_dqn_loss(batch, self.gamma)
        
        if self.use_n_step:
            idxs = batch['idxs']
            samples = self.memory_n.sample_batch_from_idxs(idxs)
            gamma = self.gamma ** self.n_step
            n_loss = self._compute_dqn_loss(samples, gamma)
            loss += n_loss
        
        if self.use_per:
            weights = torch.FloatTensor(batch['weights'].reshape(-1, 1)).to(self.device)
            loss_for_prior = loss.detach().cpu().numpy()
            new_priorities = loss_for_prior + self.prior_eps
            self.memory.update_priorities(idxs, new_priorities)
            loss = torch.mean(loss * weights)
            
        self.optimizer.zero_grad()
        loss.backward()
        if self.use_dueling:
            clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        if self.use_noisy:
            self.q_net.reset_noise()
            self.q_net_target.reset_noise()
        
        return loss.item()
    
    
    
    def _compute_dqn_loss(self, batch: Dict[str, np.ndarray], gamma: float) -> torch.Tensor:
        state = torch.FloatTensor(batch['obs']).to(self.device)
        action = torch.LongTensor(batch['act']).to(self.device)
        next_state = torch.FloatTensor(batch['next_obs']).to(self.device)
        reward = torch.FloatTensor(batch['rew'].reshape(-1, 1)).to(self.device)
        done = torch.FloatTensor(batch['done'].reshape(-1, 1)).to(self.device)
        
        if not self.use_categorical:
            cur_q_value = self.q_net(state).gather(1, action.reshape(-1,1))
            
            if self.use_double:
                action_prime = self.q_net(next_state).argmax(dim=1, keepdim=True).detach() # action selection with q_net
                next_q_value = self.q_net_target(next_state).gather(1, action_prime) # action evaluation with q_net_target
            else:
                next_q_value = self.q_net_target(next_state).max(dim=1, keepdim=True)[0].detach()
            
            mask = 1 - done
            target = (reward + self.gamma * next_q_value * mask).to(self.device)
            
            if self.use_per:
                loss  = F.smooth_l1_loss(cur_q_value, target, reduction="none")
            else:
                loss  = F.smooth_l1_loss(cur_q_value, target)
        
        else:
            delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

            with torch.no_grad():
                next_action = self.q_net_target(next_state).argmax(1)
                next_dist = self.q_net_target.dist(next_state)
                next_dist = next_dist[range(self.batch_size), next_action]
    
                # _compute_belman_update
                t_z = reward + (1 - done) * self.gamma * self.support
                t_z = t_z.clamp(min=self.v_min, max=self.v_max)
                
                # _align_bellman_update_with_atoms
                b = (t_z - self.v_min) / delta_z
                l = b.floor().long()
                u = b.ceil().long()
    
                offset = (torch.linspace(0, (self.batch_size - 1) * self.atom_size, self.batch_size).long()
                          .unsqueeze(1).expand(self.batch_size, self.atom_size).to(self.device))
    
                proj_dist = torch.zeros(next_dist.size(), device=self.device)
                proj_dist.view(-1).index_add_(0, 
                                              (l + offset).view(-1), 
                                              (next_dist * (u.float() - b)).view(-1))
                proj_dist.view(-1).index_add_(0,
                                              (u + offset).view(-1), 
                                              (next_dist * (b - l.float())).view(-1))
    
            dist = self.q_net.dist(state)
            log_p = torch.log(dist[range(self.batch_size), action])
    
            if self.use_per:
                loss = -(proj_dist * log_p).sum(1)
            else:
                loss = -(proj_dist * log_p).sum(1).mean()

        return loss
   
        
   
    def update_target_model(self, count):
        if self.target_update_type == 'hard':
            if count % self.hard_freq == 0:
                self.q_net_target.load_state_dict(self.q_net.state_dict())
        else:
            for param, target_param in zip(self.q_net.parameters(), self.q_net_target.parameters()):
                target_param.data.copy_(
                    (1.0 - self.soft_tau) * target_param + self.soft_tau * param
                    )
                
              
                
    def save_model(self, model_path='model/model_weights.pth'):
        torch.save(self.q_net.state_dict(), model_path)
        
        
        
    def load_model(self, model_path='model/model_weights.pth'):
        self.q_net.load_state_dict(torch.load(model_path))
        self.q_net_target.load_state_dict(self.dqn_agent.q_net.state_dict())    