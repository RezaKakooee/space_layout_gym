#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 13:15:44 2021

@author: RK
"""

# %% 
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env


# %% Env
env_name = 'master_env' # master_env, simple_fc_env simple_cnn_env simple_fccnn_env
 
if 'simple' in env_name:
    from simple_env import SimpleEnv
    
    env_config = {'env_name': env_name}
    env = SimpleEnv(env_config)
    
elif env_name == 'master_env':
    from gym_floorplan.envs.fenv_config import LaserWallConfig
    from gym_floorplan.envs.master_env import MasterEnv

    fenv_config = LaserWallConfig().get_config()
    env = MasterEnv(fenv_config)
    
check_env(env, warn=True)

# env = make_vec_env(SingleDOLWEnv, n_envs=4)

# %% Agent
simple_storage_dir = "storage/simples/simple"
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=simple_storage_dir)
# model = PPO.load("storage/saved_models/ppo_scenario21_80-90-100", env=env)
model.learn(total_timesteps=5000)
model.save(simple_storage_dir)

# %%
obs = env.reset()
n_episodes = 2
for e in range(n_episodes):
    time_step = 0
    while True:
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        # env.render()
        time_step+=1        
        if done:
          # env.render()
          obs = env.reset()
          print(f"Episode: {e}, Total TimeStep: {time_step}")
          break