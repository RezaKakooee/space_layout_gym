# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 09:47:49 2021

@author: RK
"""

# from ray.rllib.agents.ppo import PPOTrainer, PPOTFPolicy, PPOTorchPolicy

import random

# %%
class MyPolicy:
    def __init__(self, env, num_policies):
        self.env = env
        self.num_policies = num_policies
    
    def define_policy(self):
        policies = {"policy_{}".format(i+1): (None, self.env.observation_space, self.env.action_space, {})
                        for i in range(self.num_policies)}
        self.policy_ids = list(policies.keys())
        return policies

    # def _policy_mapping_fn(self, agent_id, episode, **kwargs):
    #     return (lambda agent_id: random.choice(self.policy_ids))
    
    def policy_mapping_fn(self, agent_id, episode, **kwargs):
        if self.num_policies == 1:
            pol_id = random.choice(self.policy_ids)
        else:
            agent_id = int(agent_id.split('_')[-1])
            pol_id = self.policy_ids[agent_id-1]
        # print(f"agent_id: {agent_id}, pol_id: {pol_id}")
        return pol_id

