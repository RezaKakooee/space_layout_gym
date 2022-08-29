# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 00:05:42 2021

@author: RK
"""

# %%
import os
from pathlib import Path

# %%
root_dir = os.path.realpath(Path(os.getcwd()).parents[0])


# %%
class AgentConfig:
    def __init__(self):
        ####### Parameters and Hyper-parameters
        self.num_iters = 1
        self.num_episodes = 1

        self.learner_name = 'trainer'  # 'tunner' # 'random' # 'trainer'
        self.RLLIB_NUM_GPUS = 0
        self.num_workers = 1

        self.agent_first_name = 'ppo'
        self.agent_last_name = '_Rainbow'

        self.load_agent_flag = False
        self.save_agent_flag = True
        self.checkpoint_freq = 1

        self.hyper_tune_flag = False

        ## Agents
        self.some_agents = 'Single'

        self.num_policies = 1

        self.default_metric = 'episode_reward_mean'
        self.default_mode = 'max'

        self.custom_model_flag = True

        self.framework = 'torch'

        self.save_env_data_flag = False

        ####### Directories and Pathsl
        self.housing_design_dir = root_dir
        self.agents_floorplan_dir = f"{self.housing_design_dir}/agents_floorplan"

        # self.wandb_api_key_file_path = f"{self.agents_floorplan_dir}/wandb_api_key_file.txt"

        ##### Storage
        self.storage = f"{self.agents_floorplan_dir}/storage"

        ### chkpt_dir and path
        self.chkpt_dir = f"{self.storage}/chkpt_dir"
        if not os.path.exists(self.chkpt_dir):
            os.makedirs(self.chkpt_dir)

        self.chkpt_txt_fixed_path = f"{self.chkpt_dir}/chkpt_path.txt"
        if os.path.exists(self.chkpt_txt_fixed_path):
            with open(self.chkpt_txt_fixed_path) as f:
                self.chkpt_path = f.readline()
        else:
            with open(self.chkpt_txt_fixed_path, 'w') as f:
                f.write('')

        self.chkpt_config_npy_fixed_path = f"{self.chkpt_dir}/chkpt_config.npy"
        

    def get_config(self):
        return self.__dict__


# %%
if __name__ == '__main__':
    self = AgentConfig()
    agent_config = self.get_config()
