#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 17:01:46 2021

@author: RK
"""
import os
import argparse

parser = argparse.ArgumentParser(description="Process some parameters.")
parser.add_argument("--phase", type=str, default="second_train", help="Custom env flag") # "first_dev" # first_dev, dev, first_train, train
args = parser.parse_args()

if args.phase == "first_dev":
    os.system("python runner.py")

elif args.phase == "dev":
    os.system("python runner.py --load_agent_flag 0")
    
elif args.phase == "first_train":
    os.system("python runner.py --num_iters 1 \
                                   --checkpoint_freq 1\
                                   --RLLIB_NUM_GPUS 1 \
                                   --num_workers 10 \
                                   --local_mode_flag 0 \
                                   --learner_name trainer \
                                   --agent_first_name ppo \
                                   --agent_last_name _Rainbow \
                                   --load_agent_flag 0\
                                   --save_agent_flag 1 \
                                   --num_policies 1")

elif args.phase == "second_train":
    os.system("python runner.py --num_iters 1 \
                                   --checkpoint_freq 1\
                                   --RLLIB_NUM_GPUS 1 \
                                   --num_workers 10 \
                                   --local_mode_flag 0 \
                                   --learner_name tunner \
                                   --agent_first_name ppo \
                                   --agent_last_name _Rainbow \
                                   --load_agent_flag 1 \
                                   --save_agent_flag 1 \
                                   --num_policies 1")
                                   
elif args.phase == "very_short_train":
    os.system("python runner.py --num_iters 20 \
                                   --checkpoint_freq 10\
                                   --RLLIB_NUM_GPUS 1 \
                                   --num_workers 10 \
                                   --local_mode_flag 0 \
                                   --learner_name tunner \
                                   --agent_first_name ppo \
                                   --agent_last_name _Rainbow \
                                   --load_agent_flag 1 \
                                   --save_agent_flag 1 \
                                   --num_policies 1")

elif args.phase == "short_train":
    os.system("python runner.py --num_iters 200 \
                                   --checkpoint_freq 25\
                                   --RLLIB_NUM_GPUS 1 \
                                   --num_workers 10 \
                                   --local_mode_flag 0 \
                                   --learner_name tunner \
                                   --agent_first_name ppo \
                                   --agent_last_name _Rainbow \
                                   --load_agent_flag 1 \
                                   --save_agent_flag 1 \
                                   --num_policies 1")


elif args.phase == "longer_train":
    os.system("python runner.py --num_iters 200 \
                                   --checkpoint_freq 20\
                                   --RLLIB_NUM_GPUS 1 \
                                   --num_workers 10 \
                                   --local_mode_flag 0 \
                                   --learner_name tunner \
                                   --agent_first_name ppo \
                                   --agent_last_name _Rainbow \
                                   --load_agent_for_longer_training_flag 1\
                                   --load_agent_flag 1 \
                                   --save_agent_flag 1 \
                                   --num_policies 1")
                                   
                                   
elif args.phase == "medium_train":
    os.system("python runner.py --num_iters 1000 \
                                   --checkpoint_freq 50\
                                   --RLLIB_NUM_GPUS 1 \
                                   --num_workers 10 \
                                   --local_mode_flag 0 \
                                   --learner_name tunner \
                                   --agent_first_name ppo \
                                   --agent_last_name _Rainbow \
                                   --load_agent_flag 1 \
                                   --save_agent_flag 1 \
                                   --num_policies 1")
                                   
elif args.phase == "long_train":
    os.system("python runner.py --num_iters 5000 \
                                   --checkpoint_freq 50\
                                   --RLLIB_NUM_GPUS 1 \
                                   --num_workers 10 \
                                   --local_mode_flag 0 \
                                   --learner_name tunner \
                                   --agent_first_name ppo \
                                   --agent_last_name _Rainbow \
                                   --load_agent_flag 1 \
                                   --save_agent_flag 1 \
                                   --num_policies 1")

elif args.phase == "very_long_train":
    os.system("python runner.py --num_iters 30000 \
                                   --checkpoint_freq 200\
                                   --RLLIB_NUM_GPUS 1 \
                                   --num_workers 10 \
                                   --local_mode_flag 0 \
                                   --learner_name tunner \
                                   --agent_first_name ppo \
                                   --agent_last_name _Rainbow \
                                   --load_agent_flag 1 \
                                   --save_agent_flag 1 \
                                   --num_policies 1")