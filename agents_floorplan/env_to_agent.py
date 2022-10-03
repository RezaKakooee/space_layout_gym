# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 10:44:10 2021

@author: RK
"""
#%%
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# %%
import ray
import gym

# %% ENV # if custom
from gym_floorplan.envs.master_env import MasterEnv


# %%
# from pettingzoo.butterfly import pistonball_v4
# from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
# import supersuit as ss

# %% AGENT
from tunner import MyTunner
from trainner import MyTrainer
# from callbacks import MyCallbacks

from ray.rllib.models import ModelCatalog
from model import (SimpleFc, 
                   SimpleCnn, 
                   SimpleFcCnn, 
                   SimpleActionMaskFc, 
                
                   MySimpleFc, 
                   MySimpleCnnGcn, # this is a FC model
                   MySimpleCnn,
                   MySimpleConv, # for testing high resolution
                   MySimpleFcCnn,
                   MySimpleGnnCnn,
                   MySimpleActionMaskFc,
                   MySimpleActionMaskCnnGcn,
                   
                   MySimpleMetaFc,
                   MySimpleActionMaskMetaFc,
                )


# %%
class EnvToAgent:
    def __init__(self, fenv_config=None, agent_config=None):
        if fenv_config['env_name'] == 'CartPole-v0': # gym env
            self.env_name = fenv_config['env_name']
            self.env = gym.make(self.env_name)
            self.env.env_name = self.env_name

        else: # Custom env
            self.env_name = fenv_config['env_name']
            
            ### for tunner
            self._register_my_env(env_name=self.env_name, env_config=fenv_config)

            ### for trainer
            self.env = self._env_creator(env_config=fenv_config) #self._env_creator(env_config={'env_name': self.env_name})
            
            if agent_config['model_source'] in ['RllibCustomModel', 'MyCustomModel']:
                self._register_my_model()
                
                
        self.agent_config = agent_config
        self.learner_name = self.agent_config['learner_name']
        
        
    def _register_my_env(self, env_name: str = 'master_env-v0', env_config={}):
        if env_name == "master_env-v0":
            ray.tune.register_env(env_name, lambda config: MasterEnv(env_config))
        # elif env_name == "pistonball_v4":
        #     ray.tune.register_env(env_name, lambda config: ParallelPettingZooEnv(self._pt_env_creator(config)))


    def _env_creator(self, env_config):
        if self.env_name == 'master_env-v0':
            return MasterEnv(env_config)
        
    # def _pt_env_creator(self, env_config):
    #     env = pistonball_v4.parallel_env(n_pistons=20, time_penalty=-0.1, continuous=True, random_drop=True, random_rotate=True, ball_mass=0.75, ball_friction=0.3, ball_elasticity=1.5, max_cycles=125)
    #     env = ss.color_reduction_v0(env, mode='B')
    #     env = ss.dtype_v0(env, 'float32')
    #     env = ss.resize_v0(env, x_size=84, y_size=84)
    #     env = ss.frame_stack_v1(env, 3)
    #     env = ss.normalize_obs_v0(env, env_min=0, env_max=1)
    #     #env = ss.flatten_v0(env)
    #     self.env = env
    #     self.env.env_name = "pistonball_v4"
    #     return env

    def _register_my_model(self):
        ModelCatalog.register_custom_model("SimpleFc", SimpleFc)
        ModelCatalog.register_custom_model('SimpleCnn', SimpleCnn)
        ModelCatalog.register_custom_model("SimpleFcCnn", SimpleFcCnn)
        
        ModelCatalog.register_custom_model("MySimpleMetaFc", MySimpleMetaFc)
        ModelCatalog.register_custom_model("SimpleActionMaskFc", SimpleActionMaskFc)
        
        ModelCatalog.register_custom_model('MySimpleFc', MySimpleFc)
        ModelCatalog.register_custom_model('MySimpleCnnGcn', MySimpleCnnGcn)
        ModelCatalog.register_custom_model('MySimpleCnn', MySimpleCnn)
        ModelCatalog.register_custom_model('MySimpleConv', MySimpleConv)
        ModelCatalog.register_custom_model('MySimpleFcCnn', MySimpleFcCnn)
        ModelCatalog.register_custom_model('MySimpleGnnCnn', MySimpleGnnCnn)
        ModelCatalog.register_custom_model('MySimpleActionMaskFc', MySimpleActionMaskFc)
        ModelCatalog.register_custom_model('MySimpleActionMaskCnnGcn', MySimpleActionMaskCnnGcn)
        ModelCatalog.register_custom_model('MySimpleActionMaskMetaFc', MySimpleActionMaskMetaFc)
    
    
    def _run_the_tunner(self):
        self.my_tunner = MyTunner(self.env, self.agent_config)
        self.my_tunner._the_tunner()
        
        
    def _run_the_trainer(self):
        self.my_trainer = MyTrainer(self.env, self.agent_config)
        self.my_trainer._the_trainer()


    # %%
    def learn(self):
        if self.learner_name == "tunner":
            self._run_the_tunner()
        elif self.learner_name == 'trainer':
            self._run_the_trainer()        
        else:
            raise ValueError("env_to_agent: Unvalid learner name")
