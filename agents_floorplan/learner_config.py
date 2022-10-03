#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 17:57:00 2021

@author: RK

"""

#%%
import os
from ray import tune
from policy import MyPolicy
from ray.rllib.agents.callbacks import DefaultCallbacks
# from my_callbacks import MyCallbacks


#%%
def get_learner_config(env, env_name, agent_config, config):
    config.update({
                "framework": agent_config['framework'],
                "env": env_name,
                "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", agent_config['RLLIB_NUM_GPUS'])),
                "num_workers": agent_config['num_workers'],
                "callbacks": DefaultCallbacks,
                "simple_optimizer": True,
                'horizon': env.fenv_config['stop_time_step'],
            })
    
    
    if agent_config['some_agents'] == 'Single':
        config.update({"log_level": "WARN"})
    
        
    if agent_config['some_agents'] == 'Multi':
       my_policy = MyPolicy(env, agent_config['num_policies'])
       config.update({
                      "multiagent": {"policies": my_policy.define_policy(),
                                     "policy_mapping_fn": my_policy.policy_mapping_fn}
                      })
       
    
    if (agent_config['agent_first_name'] == 'dqn') and (agent_config['agent_last_name'] == 'Rainbow'):
        config.update({
                       "num_atoms": 51,
                       "noisy": True,
                       "double_q": True,
                       "dueling": True,
                       # "n_step": 5,
                       # "v_min": -100.0, 
                       # "v_max": 100.0,
                       "log_level": "ERROR",
                       
                       'num_atoms': 51,
                       'noisy': True,
                       'gamma': 0.99,
                       'lr': .0001,
                       'hiddens': [512],
                       'learning_starts': 10000,
                       'buffer_size': 50000,
                       'rollout_fragment_length': 4,
                       'train_batch_size': 32,
                       'exploration_config': {
                                  'epsilon_timesteps': 2,
                                  'final_epsilon': 0.0
                                  },
                       'target_network_update_freq': 500,
                       'prioritized_replay': True,
                       'prioritized_replay_alpha': 0.5,
                       'final_prioritized_replay_beta': 1.0,
                       'prioritized_replay_beta_annealing_timesteps': 400000,
                       'n_step': 3,
                       'gpu': True,
                       'model': {
                            'grayscale': True,
                            'zero_mean': False,
                            'dim': 42
                            },
                       })
        
    
    if env_name == 'master_env-v0':
        if agent_config['model_source'] == 'RllibCustomConfig':
            if agent_config['net_arch'] == 'Cnn':
                if env.fenv_config['resolution'] == 'Low':
                    config.update({
                        "model": {
                            "dim": 23,
                            "conv_filters": [
                              [16, [3, 3], 1], # [Channel, [Kernel, Kernel], Stride]]
                              [32, [5, 5], 2],
                              [512, [11, 11], 1]
                                  ],  
                            },
                        })
                else:
                    config.update({
                        "model": {
                            "dim": 41,
                            "conv_filters": [
                              [16, [3, 3], 1],
                              [32, [5, 5], 2],
                              [512, [11, 11], 1],
                                  ],  
                            },
                        })
            
            elif agent_config['net_arch'] == 'CnnGcn':
                config.update({
                    "model": {
                'fcnet_hiddens': [256, 256, 256, 256, 256],
                "use_lstm": True,
                "lstm_cell_size": 256,
                },
                    })
            
        
        elif agent_config['model_source'] == 'RllibCustomModel':
              if agent_config['net_arch'] in ['Cnn', 'FcCnn']:
                  if env.fenv_config['resolution'] == 'Low':
                      config.update({
                            "model": {
                                "dim": 21,
                                "conv_filters": [
                                  [16, [3, 3], 1],
                                  [32, [5, 5], 2],
                                  [512, [11, 11], 1]
                                      ],
                                "custom_model": agent_config['model_name'],
                                "vf_share_layers": True,
                                "custom_model_config": {
                                    "conv_filters": [21, 21, 1]
                                    },
                                },
                            })
                    
                  else:
                      raise NotImplementedError('Needs to be implemented for high resolution')
                
        elif agent_config['model_source'] == 'MyCustomModel':
            if agent_config['agent_first_name'] == 'ppo':
                config.update({
                    "model": {
                        "custom_model": agent_config['model_name'],
                        "custom_model_config": {},
                        },
                    "entropy_coeff": 0.01,
                    })
            elif agent_config['agent_first_name'] == 'dqn':
                config.update({
                    "model": {
                        "custom_model": agent_config['model_name'],
                        # "custom_model_config": {},
                        },
                    })
            else:
                raise NotImplementedError('For now only ppo and dqn are supported.')
        
        elif agent_config['model_source'] == 'RllibModel':
            pass
        else:
            raise ValueError('model_source or model_source-net_arch combination is unknown!')

    elif env.env_name == "pistonball_v4":
            config.update({"model": {"custom_model": "CNNModelV2"}})
            config.update({
                "log_level": "ERROR",
                "framework": "torch",
                "num_envs_per_worker": 1,
                "compress_observations": False,
                "batch_mode": 'truncate_episodes',

                # 'use_critic': True,
                'use_gae': True,
                "lambda": 0.9,
                "gamma": .99,
                # "kl_coeff": 0.001,
                # "kl_target": 1000.,
                "clip_param": 0.4,
                'grad_clip': None,
                "entropy_coeff": 0.1,
                'vf_loss_coeff': 0.25,
                "sgd_minibatch_size": 64,
                "num_sgd_iter": 10,  # epoc
                'rollout_fragment_length': 512,
                "train_batch_size": 512 * 4,
                'lr': 2e-05,
                "clip_actions": True})

    else:
        raise ValueError('A wrogn env name')
        
        
    ### exploration config
    # config.update({"exploration_config": {# The Exploration class to use.
    #                                       "type": "EpsilonGreedy",
    #                                       # Config for the Exploration class' constructor:
    #                                       "initial_epsilon": 1.0,
    #                                       "final_epsilon": 0.02,
    #                                       "epsilon_timesteps": 10000000,  # Timesteps over which to anneal epsilon.

    #                                        # For soft_q, use:
    #                                        # "exploration_config" = {
    #                                        #   "type": "SoftQ"
    #                                        #   "temperature": [float, e.g. 1.0]
    #                                        # }
    #                                      }})
    
    
    ### env data 
    if agent_config['save_env_data_flag']:
        config.update({"output": agent_config['env_data_dir'], "output_max_file_size": 5000000})
        
        
    ### Hyper-parameter tunning
    if agent_config['hyper_tune_flag']:
        if agent_config['agent_first_name'] == 'ppo':
            # config.update({"gamma": ray.tune.grid_search([0.8, 0.9, 0.99])})
            # config.update({"lambda": ray.tune.grid_search([0.9, 0.95, 1.0])})
            # config.update({"lr": ray.tune.grid_search([0.01, 0.001, 0.0001])})
            # config.update({"train_batch_size": ray.tune.grid_search([3000, 4000, 5000])})
            # config.update({"sgd_minibatch_size": ray.tune.grid_search([64, 128, 256])})
            
            # config.update({"vf_loss_coeff": ray.tune.grid_search([0.5, 0.75, 1.0])})
            # config.update({"clip_param": ray.tune.grid_search([0.1, 0.2, 0.3])})
            # config.update({"kl_target": ray.tune.grid_search([0.01, 0.02, 0.03])})
            # config.update({"entropy_coeff": ray.tune.grid_search([0.0, 0.01])})
            
            
            
            ### copy
            # config.update({"gamma": ray.tune.grid_search([0.8, 0.99])})
            # config.update({"lambda": ray.tune.grid_search([0.9, 1.0])})
            # config.update({"lr": ray.tune.grid_search([0.01, 0.0001])})
            # # config.update({"train_batch_size": ray.tune.grid_search([3000, 5000])})
            # config.update({"sgd_minibatch_size": ray.tune.grid_search([64, 256])})
            
            # config.update({"vf_loss_coeff": ray.tune.grid_search([0.5, 1.0])})
            # config.update({"clip_param": ray.tune.grid_search([0.1, 0.3])})
            # config.update({"kl_target": ray.tune.grid_search([0.01, 0.03])})
            # config.update({"entropy_coeff": ray.tune.grid_search([0.0, 0.01])})
            
            
            ### copy choice
            config.update({"gamma": tune.uniform(0.8, 0.99)})
            # config.update({"lambda": tune.uniform(0.9, 1.0)})
            # config.update({"lr": ray.tune.choice([0.01, 0.0001])})
            # config.update({"train_batch_size": ray.tune.grid_search([3000, 5000])})
            # config.update({"sgd_minibatch_size": ray.tune.choice([64, 256])})
            
            # config.update({"vf_loss_coeff": ray.tune.grid_search([0.5, 1.0])})
            # config.update({"clip_param": ray.tune.grid_search([0.1, 0.3])})
            # config.update({"kl_target": ray.tune.grid_search([0.01, 0.03])})
            # config.update({"entropy_coeff": ray.tune.grid_search([0.0, 0.01])})
            
    return config
