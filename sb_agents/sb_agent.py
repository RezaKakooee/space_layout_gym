#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 13:15:44 2021

@author: Reza Kakooee
"""

# %% 
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env

import gym_floorplan
from gym_floorplan.envs.fenv_config import LaserWallConfig
from gym_floorplan.envs.single_dolw_env import SingleDOLWEnv


# %%

fenv_config = LaserWallConfig().get_config()
env = SingleDOLWEnv(fenv_config)
check_env(env, warn=True)

# env = make_vec_env(SingleDOLWEnv, n_envs=4)

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log='/storage/sb_tb_log_dir')
# model = PPO.load("storage/saved_models/ppo_first_scenario", env=env)
model.learn(total_timesteps=500000)
model.save("storage/saved_models/ppo_first_scenario")

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
          env.render()
          obs = env.reset()
          print(f"Episode: {e}, Total TimeStep: {time_step}")
          break

# env._illustrate()
