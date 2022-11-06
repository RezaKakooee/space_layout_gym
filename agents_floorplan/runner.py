# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 00:09:58 2021
@author: RK
"""

# %%
import argparse
parser = argparse.ArgumentParser(description='Process some parameters.')
parser.add_argument('--env_name', type=str, default='master_env-v0', help='Custom env flag') # pistonball_v4

parser.add_argument('--num_iters',       type=int, default=1, help='Num iterations for tunner')
parser.add_argument('--num_episodes',    type=int, default=1, help='Num episodes for trainer')
parser.add_argument('--RLLIB_NUM_GPUS',  type=int, default=0, help='Number of GPUs')
parser.add_argument('--num_workers',     type=int, default=1, help='Number of Workers')
parser.add_argument('--local_mode_flag', type=int, default=1, help='set the local mode')

parser.add_argument('--learner_name',     type=str, default='trainer',   help='Learner name, like tunner')
parser.add_argument('--agent_first_name', type=str, default='ppo',      help='Learner name, like dqn')
parser.add_argument('--agent_last_name',  type=str, default='_Rainbow', help='Learner name, like Rainbow')

parser.add_argument('--load_agent_flag', type=int, default=0, help='Restore trained agent')
parser.add_argument('--save_agent_flag', type=int, default=0, help='Store checkpoint')
parser.add_argument('--checkpoint_freq', type=int, default=1, help='Checkpoint frequency')

parser.add_argument('--load_agent_for_longer_training_flag', type=int, default=0, help='Restore trained agent for longer training')

parser.add_argument('--num_policies', type=int, default=1, help='Number of policies')

parser.add_argument('--hyper_tune_flag', type=int, default=0, help='Hyper-parameters tuning')

args = parser.parse_args()

from pprint import pprint 
print("- - - - - - - - - - args:")
pprint(args.__dict__)

# %%
import os
import ray
import time
import json
import shutil
import numpy as np
from pathlib import Path
from datetime import datetime

from gym_floorplan.envs.fenv_config import LaserWallConfig
from agent_config import AgentConfig
from env_to_agent import EnvToAgent

from json_reader import json_reader

# %%
class Runner:
    def __init__(self):
        self.start_time = time.time()
        self.latest_chkpt_number = 0 # for run=1
        self.num_iterations_in_a_run = args.num_iters
        
        self._update_fenv_config()
        self._update_agent_config()
        
        if args.load_agent_flag:
            chkpt_path = self._get_latest_chkpt()
        
    
        if args.load_agent_for_longer_training_flag:
            assert self.fenv_config['plan_config_source_name'] == 'longer_training_config', "load_agent_for_longer_training_flag only works when you have a single set of"
            self.fenv_config.update({
                'plan_config_source_name': 'longer_training_config',
                'learner_name': args.learner_name,
                'agent_first_name': args.agent_first_name,
                'chkpt_path': chkpt_path,                
                }) 
    
    def _update_fenv_config(self):
        if args.env_name == 'master_env-v0':
            self.fenv_config = LaserWallConfig().get_config()
        elif args.env_name == 'CartPole-v0':
            self.fenv_config = {'env_name': 'CartPole-v0'}
        elif args.env_name == 'pistonball_v4':
            self.fenv_config = {'env_name': 'pistonball_v4'}
        else:
            raise ValueError('Not a valid env_name')
        
        self.fenv_config.update({'env_name': args.env_name})
        
    
    def _update_agent_config(self):
        self.agent_config = AgentConfig().get_config()
        if args.env_name == 'master_env-v0':
            self.agent_config.update({'some_agents': self.fenv_config['env_type']})
        else:
            raise ValueError('Not a valid env_name')
        
        self.agent_config.update({
            'stop_tunner_iteration': self.latest_chkpt_number + self.num_iterations_in_a_run,
            'num_episodes': args.num_episodes,
            'RLLIB_NUM_GPUS': args.RLLIB_NUM_GPUS,
            'num_workers': args.num_workers,
            'learner_name': args.learner_name,
            'agent_first_name': args.agent_first_name,
            'agent_last_name': args.agent_last_name,
            'model_source': self.fenv_config['model_source'],
            'model_name': self.fenv_config['model_name'],
            'net_arch': self.fenv_config['net_arch'],
            'use_lstm': self.fenv_config['use_lstm'],
            'load_agent_flag': bool(args.load_agent_flag),
            'save_agent_flag': bool(args.save_agent_flag),
            'checkpoint_freq': args.checkpoint_freq,
            'num_policies': args.num_policies,
            'hyper_tune_flag': bool(args.hyper_tune_flag),
            })
        
        ### Local_dir
        local_dir = f"{self.agent_config['storage']}/{args.learner_name}/local_dir"
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)
        self.agent_config.update({'local_dir': local_dir})
        
        ### only for trainer
        if args.learner_name == 'trainer':
            trainer_dir_name = f"{args.agent_first_name.upper()}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"  
            trainer_chkpt_dir = f"{local_dir}/{trainer_dir_name}"
            if not os.path.exists(trainer_chkpt_dir):
                os.makedirs(trainer_chkpt_dir)
            self.agent_config.update({'trainer_chkpt_dir': trainer_chkpt_dir})
        
        ### env env_data_dir
        if args.learner_name == 'trainer':
            self.scenario_dir = f"{local_dir}/{self.fenv_config['scenario_name']}"
            env_data_dir = f"{self.scenario_dir}/env_data"
        else:
            self.scenario_dir = f"{local_dir}/{self.fenv_config['scenario_name']}"
            env_data_dir = f"{self.scenario_dir}/env_data"

        if not os.path.exists(env_data_dir):
            os.makedirs(env_data_dir)
        self.agent_config.update({'env_data_dir': env_data_dir})
        
        
    def _get_latest_chkpt(self):
        with open(self.agent_config['chkpt_txt_fixed_path']) as f:
            chkpt_path = f.readline()
        
        if len(chkpt_path) == 0:
            raise ValueError("load_agent_flag is True, but chkpt_path is empty. Set loat_agent_flat to False, as seems it is the first time you are running the code!")
        
        self.latest_chkpt_number = int(chkpt_path.split("/")[-1].split('-')[-1])
        self.agent_config['stop_tunner_iteration'] = self.latest_chkpt_number + self.num_iterations_in_a_run
        
        return chkpt_path
        
    
    def start(self, n_runs=1):
        for n in range(n_runs):
            print(f"====================================== Run number is: {n}")
            print(f"- - - - - - Start iteration: {self.latest_chkpt_number}")
            print(f"- - - - - - Expected end iteration: {self.agent_config['stop_tunner_iteration']}") 
            print(f"- - - - - - Env name: {self.fenv_config['env_name']}")
            print(f"- - - - - - Scenario name: {self.fenv_config['scenario_name']}")

            info = ray.init(
                            ignore_reinit_error=True,
                            log_to_driver=False,
                            local_mode=args.local_mode_flag,
                            object_store_memory=10**8,
                            )
            
            self.env_to_agent = EnvToAgent(fenv_config=self.fenv_config, agent_config=self.agent_config)
            self.env_to_agent.learn()
            
            ray.shutdown()
        
        self._terminate()
        
        
    def _terminate(self):
        print("\n\n ----- Terminating .... ==================================")
        end_time = time.time()
        elapsed_time = end_time - self.start_time
        
        if args.save_agent_flag:
            if args.learner_name == 'trainer':
                self.chkpt_path = self.env_to_agent.my_trainer.chkpt_path
                self.chkpt_dir = self.chkpt_path.parents[1]
                
                shutil.move(os.path.normpath(self.chkpt_dir), self.scenario_dir)
            
                self.chkpt_path_ = os.path.join(self.chkpt_path.parents[2], 
                                           self.fenv_config['scenario_name'], 
                                           '/'.join(str(self.chkpt_path).split('/')[-3:]))
                
                env_data_dir = f"{self.scenario_dir}/env_data"
                json_file_pathes = [f"{env_data_dir}/{json_name}" for json_name in os.listdir(env_data_dir)]
                plans_df = json_reader(json_file_pathes)
                
                plans_df.to_csv(f"{self.scenario_dir}/plans_df.csv", index=False)
            
            elif args.learner_name == 'tunner':
                self.chkpt_path = Path(self.env_to_agent.my_tunner.chkpt_path)
                self.chkpt_dir = self.chkpt_path.parents[2]
                

                env_data_dir = f"{self.chkpt_dir}/env_data"
                json_file_pathes = [f"{env_data_dir}/{json_name}" for json_name in os.listdir(env_data_dir)]
                plans_df = json_reader(json_file_pathes)
                
                plans_df.to_csv(f"{self.scenario_dir}/plans_df.csv", index=False)

                shutil.move(os.path.normpath(self.chkpt_dir), self.scenario_dir)
            
                self.chkpt_path_ = os.path.join(self.chkpt_path.parents[3], 
                                           self.fenv_config['scenario_name'], 
                                           '/'.join(str(self.chkpt_path).split('/')[-4:]))
                
            
            with open(self.agent_config['chkpt_txt_fixed_path'], "w") as f:
                f.write(str(self.chkpt_path_))
                
            chkpt_json = {
                    'scenario_name': self.fenv_config['scenario_name'],
                    "chkpt_path": str(self.chkpt_path_),
                    }
                    
            # scenario_dir = f"{self.agent_config['local_dir']}/{self.fenv_config['scenario_name']}"
            chkpt_json_path = os.path.join(self.scenario_dir, "chkpt_json.json")
            with open(chkpt_json_path, 'w') as f:
                json.dump(chkpt_json, f, indent=4)
        
        
        run_info_json = {
                'elapsed_time (minutes)': elapsed_time/60,
                'RLLIB_NUM_GPUS': args.RLLIB_NUM_GPUS,
                'num_iters': args.num_iters,
                'num_workers': args.num_workers,
        }
        
        
        ## Store the current version env for inferencing in the future
        root_dir = os.path.normpath('/home/rdbt/ETHZ/dbt_python/housing_design')
        source_env_dir = os.path.join(root_dir, 'gym-floorplan')
        destination_env_dir = os.path.join(self.scenario_dir, 'gym-floorplan')
        shutil.copytree(source_env_dir, destination_env_dir)
        
        
        run_info_json_path = os.path.join(self.scenario_dir, 'run_info_json.json')
        with open(run_info_json_path, 'w') as f:
            json.dump(run_info_json, f, indent=4)
            
        print(f"- - - - - - Scenario name: {self.fenv_config['scenario_name']}")
        print(f"- - - - Total time for {self.num_iterations_in_a_run} iterations was: {elapsed_time} (s)")   
        print(" = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =")
        
        
        
        
# %%
def main():
    runner = Runner()
    runner.start()
    return runner
    
    
# %%
if __name__ == '__main__':
    runner = main()