# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 23:44:54 2024

@author: Reza Kakooee
"""


from gym_floorplan.envs.fenv_config import LaserWallConfig
from gym_floorplan.envs.master_env import SpaceLayoutGym

fenv_config = LaserWallConfig().get_config()
env = SpaceLayoutGym(fenv_config)

obs = env.reset()

done = False
while not done:
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    env.render()
env.close()