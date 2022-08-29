#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 13:32:45 2021

@author: RK
"""
import os

import ray
from ray import tune
from ray.rllib.agents import ppo
from ray.tune.logger import pretty_print

import gym_floorplan
from gym_floorplan.envs.fenv_config import LaserWallConfig
from gym_floorplan.envs.single_dolw_env import SingleDOLWEnv

fenv_config = LaserWallConfig().get_config()
env = SingleDOLWEnv(fenv_config)
env_name = "single_dolw_env-v0"
ray.tune.register_env(env_name, lambda config: SingleDOLWEnv(fenv_config))

if __name__ == "__main__":
    ray.init(local_mode=True)
    config = {
        "env": env_name,  # or "corridor" if registered above
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "1")),
        "num_workers": 1,  # parallelism
        "framework": 'torch',
    }

    stop = {
        "training_iteration": 50,
        "timesteps_total": 100000,
        "episode_reward_mean": 6,
    }
    learner_name = 'tunner'
    if learner_name == 'trainer':
        ppo_config = ppo.DEFAULT_CONFIG.copy()
        ppo_config.update(config)
        ppo_config["lr"] = 1e-3
        trainer = ppo.PPOTrainer(config=ppo_config, env=env_name)
        for _ in range(stop['training_iteration']):
            result = trainer.train()
            print(pretty_print(result))
            # stop training of the target train steps or reward are reached
            if result["timesteps_total"] >= stop['timesteps_total'] or \
                    result["episode_reward_mean"] >= stop['episode_reward_mean']:
                break
    else:
        chckp_path = "./storage/PPO/PPO_single_dolw_env-v0_0e511_00000_0_2021-12-23_15-03-08/checkpoint_000025/checkpoint-25"
        # automated run with Tune and grid search and TensorBoard
        print("Training automatically with Ray Tune")
        results = tune.run("PPO", config=config, stop=stop,
                           local_dir='storage',
                           checkpoint_freq=10,
                           checkpoint_at_end=True,
                           restore=chckp_path)

    ray.shutdown()