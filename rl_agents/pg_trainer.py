#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 16:03:10 2023

@author: Reza Kakooee
"""

import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
import numpy as np
from pg_rollout import PgRollout



#%%
class PgTrainer:
    def __init__(self, env, agent, writer):
        self.env = env
        self.agent = agent
        self.writer = writer
        
        for k, v in agent.agent_config.items():
            setattr(self, k, v)
        
        if self.load_model_flag:
            self.load_model(agent.agent_config['chkpt_path'])   
            print("The model loaded for further training.")
            
        self.pg_rollout = PgRollout(env, agent)



    def train(self):
        self.timestep = 0
        self.episode_counter = 0
        self.batch_couner = 0 
        goal_visit_counter = 0
        for update_i in range(1, self.agent.agent_config['n_updates']+1):
            try:
                batch, stats = self.pg_rollout.interact(self.n_interacts)
                self.timestep += len(batch['actions'])
                
                if self.anneal_lr:
                    frac = 1.0 - (update_i - 1.0) / self.agent.agent_config['n_updates']
                    cur_lr = frac * self.net_lr
                    
                loss_dict = self.agent.update_model(batch, cur_lr)
                
                if (self.batch_couner+1) % self.verbos_freq == 0:
                    print(f"-------------------- Batch: {self.batch_couner+1:06} --------------------")
                    print(f"Batch: {self.batch_couner+1}")
                    print(f"TimeStep: {self.timestep}")
                    
                    print(f"MeanBatchReward: {stats['mean_b_rewards']}")
                    print(f"MinBatchReward: {stats['min_b_rewards']}")
                    print(f"MaxBatchReward: {stats['max_b_rewards']}")
                    print(f"SumBatchReward: {stats['sum_b_rewards']}")
                    print(f"MeanEpisodeLen: {stats['mean_ep_len']}\n")
                    
                if (self.batch_couner+1) % self.checkpoint_freq == 0:
                    self.save_model(self.batch_couner+1)
                    
                if (self.batch_couner+1) % self.log_freq == 0:
                    self.write_to_tb(stats, loss_dict, self.batch_couner)
                    self.pg_rollout.plan_df.to_csv(self.agent.agent_config['plan_df_path'], index=False)
                    self.pg_rollout.res_df.to_csv(self.agent.agent_config['res_df_path'], index=False)
                    self.pg_rollout.res_summary_df.to_csv(self.agent.agent_config['res_summary_df_path'], index=False)
                
                if stats['mean_ep_len'] <= 10:
                        goal_visit_counter += 1
                        if goal_visit_counter == self.n_goal_visits:
                            self.save_model(0)
                            self.pg_rollout.plan_df.to_csv(self.agent.agent_config['plan_df_path'], index=False)
                            self.pg_rollout.res_df.to_csv(self.agent.agent_config['res_df_path'], index=False)
                            self.pg_rollout.res_summary_df.to_csv(self.agent.agent_config['res_summary_df_path'], index=False)
                            break
                        
            except KeyboardInterrupt:
                if self.save_model_flag:
                    print("The model saved.")
                    self.save_model(0)
                    self.pg_rollout.plan_df.to_csv(self.agent.agent_config['plan_df_path'], index=False)
                    self.pg_rollout.res_df.to_csv(self.agent.agent_config['res_df_path'], index=False)
                    self.pg_rollout.res_summary_df.to_csv(self.agent.agent_config['res_summary_df_path'], index=False)
                
                break        
            
            self.batch_couner += 1    
                
                
                
    def write_to_tb(self, stats, loss_dict, i):
            self.writer.add_scalar("charts/SumBatchReward", stats['sum_b_rewards'], i)
            self.writer.add_scalar("charts/MinBatchReward", stats['min_b_rewards'], i)
            self.writer.add_scalar("charts/MaxBatchReward", stats['max_b_rewards'], i)
            self.writer.add_scalar("charts/MeanBatchReward", stats['mean_b_rewards'], i)
            self.writer.add_scalar("charts/MeanEpisodeLen", stats['mean_ep_len'], i)
            
            self.writer.add_scalar("charts/actor_loss", loss_dict['actor_loss'], i)            
            self.writer.add_scalar("charts/critic_loss", loss_dict['critic_loss'], i)
            self.writer.add_scalar("charts/entropy_loss", loss_dict['entropy_loss'], i)
            self.writer.add_scalar("charts/loss", loss_dict['loss'], i)
           
            self.writer.add_scalar("charts/MeanEpisodeLen", stats['mean_ep_len'], i)
            if self.env_name == 'DOLW-v0_':
                obs_moving_labels = self.env.obs.plan_data_dict['obs_moving_labels']
                obs_moving_labels = np.expand_dims(obs_moving_labels, 0)
                self.writer.add_image('images', obs_moving_labels, i)



    def save_model(self, i):
        model_path = os.path.join(self.model_dir, f'model_{i:04}.pth')
        torch.save(self.agent.net.state_dict(), model_path)
        with open(self.chkpt_txt_fixed_path, "w") as f:
            f.write(str(model_path))
        
        
        
    def load_model(self, model_path):
        self.agent.net.load_state_dict(torch.load(model_path))
