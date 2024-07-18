#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 18:22:45 2023

@author: Reza Kakooee
"""

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions import MultivariateNormal

from network import TinyLinearNet, LinearNet, CnnNet, MetaFc, MetaFc, MetaCnn, MetaCnnResnet, MetaCnnResidual


#%% 
class PPO:
    def __init__(self, env, fenv_config, agent_config):
        self.env = env
        self.fenv_config = fenv_config
        self.agent_config = agent_config
        
        for k, v in agent_config.items():
            setattr(self, k, v)
        
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)
        
        if self.model_name == "TinyLinearNet":
            self.net = TinyLinearNet(self.obs_dim, self.act_dim, self.hidden_dim).to(self.device)
        elif self.model_name == "LinearNet":
            self.net = LinearNet(self.obs_dim, self.act_dim, self.hidden_dim).to(self.device)
        elif self.model_name == "CnnNet":
            self.net = CnnNet(self.obs_dim, self.act_dim, self.fenv_config).to(self.device)
        elif self.model_name == "MetaFc":
            self.net = MetaFc(self.obs_dim, self.act_dim, self.fenv_config).to(self.device)
        elif self.model_name == "MetaCnn":
            self.net = MetaCnn(self.obs_dim, self.act_dim, self.fenv_config).to(self.device)
        elif self.model_name == "MetaCnnResnet":
            self.net = MetaCnnResnet(self.obs_dim, self.act_dim, self.fenv_config).to(self.device)
        elif self.model_name == 'MetaCnnResidual':
            self.net = MetaCnnResidual(self.obs_dim, self.act_dim, self.fenv_config).to(self.device)
        else:
            print(f"self.model_name: {self.model_name}")
            raise NotImplementedError
        
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.net_lr)
        
    
    
    def update_model(self, batch, cur_lr) -> torch.Tensor:
        batch_len = len(batch['actions'])
        batch = self._batch_to_tensor(batch)
        
        for _ in range(self.n_updates_per_batch):
            indices = np.random.choice(batch_len, batch_len, replace=False)
            for start in range(0, self.batch_size, self.mini_batch_size):
                mb_observations, mb_actions, mb_log_pis, mb_values, mb_qvalues, mb_advantages_norm = self._get_mini_batch(batch, start, indices)

                new_mb_log_pis, new_mb_values, new_mb_entropies = self._policy_evaluation(mb_observations, mb_actions)
                
                actor_loss, critic_loss, entropy_loss, loss = self._compute_loss(mb_log_pis, mb_advantages_norm, new_mb_log_pis, 
                                                                                 new_mb_values, mb_qvalues, mb_values,
                                                                                 new_mb_entropies)  
                self._policy_improvment(loss, cur_lr)
        
        loss_dict = {
            'actor_loss': actor_loss.item(), 
            'critic_loss': critic_loss.item(), 
            'entropy_loss': entropy_loss.item(), 
            'loss': loss.item()
            }
        
        return loss_dict
        


    def _policy_evaluation(self, mb_observations, mb_actions):
        new_mb_logits, new_mb_values = self.net(mb_observations)
        new_mb_dist = MultivariateNormal(new_mb_logits, self.cov_mat) if self.action_space_type == "continuous" else Categorical(F.softmax(new_mb_logits, dim=-1))
        new_mb_log_pis = new_mb_dist.log_prob(mb_actions)
        new_mb_entropies = new_mb_dist.entropy()
        return new_mb_log_pis, new_mb_values, new_mb_entropies

        
        
    def _compute_loss(self, mb_log_pis, mb_advantages_norm, new_mb_log_pis, 
                            new_mb_values, mb_qvalues, mb_values,
                            new_mb_entropies) -> torch.Tensor:
        actor_loss = self._get_actor_loss(mb_log_pis, mb_advantages_norm, new_mb_log_pis)
        critic_loss = self._get_critic_loss(new_mb_values, mb_qvalues, mb_values)
        entropy_loss = self._get_entropy_loss(new_mb_entropies)
        loss = actor_loss + self.vf_coef * critic_loss - self.entropy_coef * entropy_loss
        return actor_loss, critic_loss, entropy_loss, loss
    
    
    
    def _get_actor_loss(self, mb_log_pis, mb_advantages_norm, new_mb_log_pis):
        log_ratios = new_mb_log_pis - mb_log_pis
        ratios = torch.exp(log_ratios)
        clipped_ratios = torch.clamp(ratios, 1 - self.clip_coef, 1 + self.clip_coef)
        surr1 = mb_advantages_norm * ratios
        surr2 = mb_advantages_norm * clipped_ratios
        actor_reward = torch.min(surr1, surr2)
        actor_loss = - actor_reward.mean()
        return actor_loss
        
        
        
    def _get_critic_loss(self, new_mb_values, mb_qvalues, mb_values):
        new_mb_values = new_mb_values.squeeze()
        
        if self.clip_critic_loss_flag:
            new_mb_values_clipped = mb_values + torch.clamp(new_mb_values - mb_values, -self.clip_coef, self.clip_coef)
            
            v_loss_unclipped = 0.5 * nn.MSELoss()(new_mb_values, mb_qvalues)
            v_loss_clipped   = 0.5 * nn.MSELoss()(new_mb_values_clipped, mb_qvalues)
            
            v_loss = torch.max(v_loss_unclipped, v_loss_clipped)
            return v_loss
        
        return 0.5 * nn.MSELoss()(new_mb_values, mb_qvalues)
        
    
        
    def _get_entropy_loss(self, entropies):
        return entropies.mean()
        
        
    
    def _get_kl(self, log_ratios, ratios):
        with torch.no_grad():
            # calculate approx_kl http://joschu.net/blog/kl-approx.html
            old_approx_kl = (-log_ratios).mean()
            approx_kl = ((ratios - 1) - log_ratios).mean()
            self.clipfracs += [((ratios - 1.0).abs() > self.clip_coef).float().mean().item()]
            
            #or
            # approx_kl_divergence = .5 * ((mbatch['log_pis'] - new_mb_quantities[log_pi]) ** 2).mean()
        
        
        
    def _policy_improvment(self, loss, cur_lr):
        self.optimizer.param_groups[0]["lr"] = cur_lr
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=0.5)
        self.optimizer.step()
        


    def select_action(self, observation):
        observation = self._observation_to_tensor(observation)
        logits, values = self.net(observation)
        
        if self.action_space_type == "continuous":
            dist = MultivariateNormal(logits, self.cov_mat)
        else:
            dist = Categorical(F.softmax(logits, dim=-1))
                
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.detach().cpu().numpy(), log_prob.detach().cpu().numpy(), values.detach().cpu().numpy()
    
    
    
    def get_value(self, observation):
        observation = self._observation_to_tensor(observation)
        _, values = self.net(observation)
        return values.detach().cpu().numpy()
    
    
    
    
    def _observation_to_tensor(self, observation):
        if self.net_arch == 'Fc':
            observation = torch.FloatTensor(observation).to(self.device)
        elif self.net_arch == 'Cnn':
            observation = torch.FloatTensor(observation).to(self.device)/255.0
        elif self.net_arch == 'MetaFc':
            observation = {'observation_fc': torch.FloatTensor(observation['observation_fc']).to(self.device),
                           'observation_meta': torch.FloatTensor(observation['observation_meta']).to(self.device)}
        elif self.net_arch == 'MetaCnn':
            observation = {'observation_cnn': torch.FloatTensor(observation['observation_cnn']).to(self.device)/255.0,
                           'observation_meta': torch.FloatTensor(observation['observation_meta']).to(self.device)}
        else:
            raise NotImplementedError
        return observation
    
    
    
    def _batch_to_tensor(self, batch):
        for k, v in batch.items():
            if k == 'observations':
                batch[k] = self._observation_to_tensor(batch[k])
            if k in ['log_pis', 'values', 'rewards', 'advantages']:
                batch[k] = torch.FloatTensor(v).to(self.device)
            if k in ['action_masks', 'actions']:
                batch[k] = torch.LongTensor(v).to(self.device)
            if k in ['dones']:
                batch[k] = torch.FloatTensor(v).to(self.device)
        return batch
    
    
    
    def observation_batch_to_mbatch(self, v, mb_indices):
        if self.net_arch == 'Fc':
            mb_o = v[mb_indices]
        elif self.net_arch == 'Cnn':
            mb_o = v[mb_indices]
        elif self.net_arch == 'MetaFc':
            mb_o = {'observation_fc': v['observation_fc'][mb_indices],
                    'observation_meta': v['observation_meta'][mb_indices]}
        elif self.net_arch == 'MetaCnn':
            mb_o = {'observation_cnn': v['observation_cnn'][mb_indices],
                    'observation_meta': v['observation_meta'][mb_indices]}
        else:
            raise NotImplementedError
    
        return mb_o



    def _get_mini_batch(self, batch, start, indices):
        end = start + self.mini_batch_size
        mb_indices = indices[start:end]
        mbatch = {}
        for k, v in batch.items():
            if k == 'observations':
                mbatch[k] = self.observation_batch_to_mbatch(v, mb_indices)
            else:
                mbatch[k] = v[mb_indices]
                
        mb_observations, mb_actions = mbatch['observations'], mbatch['actions']
        mb_log_pis, mb_advantages, mb_values = mbatch['log_pis'], mbatch['advantages'], mbatch['values']
        mb_advantages_norm = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
        mb_qvalues = mb_values + mb_advantages
        return mb_observations, mb_actions, mb_log_pis, mb_values, mb_qvalues, mb_advantages_norm