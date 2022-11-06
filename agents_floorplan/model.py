#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 17:42:15 2021

@author: RK
"""

#%%
import gym
from abc import ABC

import numpy as np
import torch
from torch import nn

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models.torch.visionnet import VisionNetwork as TorchVis
from ray.rllib.utils.torch_utils import FLOAT_MIN

from gym_floorplan.envs.fenv_config import LaserWallConfig
fenv_config = LaserWallConfig().get_config()


#%%
class SimpleFc(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs,
                     model_config, name):

        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)
        self.torch_sub_model = TorchFC(obs_space, action_space, num_outputs,
                                        model_config, name)


    def forward(self, input_dict, state, seq_lens):
        input_dict["obs"] = input_dict["obs"].float()
        fc_out, _ = self.torch_sub_model(input_dict, state, seq_lens)
        return fc_out, []


    def value_function(self):
        return self.torch_sub_model.value_function()
    
    
    
#%%
class SimpleCnn(TorchModelV2, nn.Module, ABC):
    def __init__(self, obs_space, action_space, num_outputs,
                 model_config, name):

        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        self.torch_vis_model = TorchVis(obs_space, action_space, num_outputs,
                                        model_config, name)


    def forward(self, input_dict, state, seq_lens):
        input_dict["obs"] = input_dict["obs"].float()
        fc_out, _ = self.torch_vis_model(input_dict, state, seq_lens)
        return fc_out, []


    def value_function(self):
        return self.torch_sub_model.value_function()
        
    
    
#%%
class SimpleFcCnn(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs,
                     model_config, name):

        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)
        self.my_obs_space = obs_space
        self.my_action_space = action_space
        self.my_num_outputs = num_outputs
        self.my_model_config = model_config
        self.my_name = name
        self.torch_sub_fc_model = TorchFC(obs_space.original_space[0], action_space, num_outputs,
                                    model_config, name)

        self.torch_sub_vis_model = TorchVis(obs_space.original_space[1], action_space, num_outputs,
                                        model_config, name)

        self.value_fn = nn.Linear(2, 1)
        self.head = nn.Linear(2*num_outputs, num_outputs) # nn.Linear(2*action_space.n, num_outputs)
        # print(f"num_outputs: {num_outputs}")
        # print(f"action_space.n: {action_space.n}")
        # assert num_outputs == action_space.n


    def forward(self, input_dict, state, seq_lens):
        # print(f"input_dict['obs'][0].shape: {input_dict['obs'][0].shape}")
        # print(f"input_dict['obs'][1].shape: {input_dict['obs'][1].shape}")
        # print(f"self.my_obs_space: {self.my_obs_space.original_space}")
        # print(f"self.my_obs_space[0].shape: {self.my_obs_space.original_space[0].shape}")
        # print(f"self.my_obs_space[1].shape: {self.my_obs_space.original_space[1].shape}")
        assert input_dict["obs"][0].shape[1:] == self.my_obs_space.original_space[0].shape
        assert input_dict["obs"][1].shape[1:] == self.my_obs_space.original_space[1].shape
        fc_out, _ = self.torch_sub_fc_model({"obs": input_dict["obs"][0]}, state, seq_lens)
        cnn_out, _ = self.torch_sub_vis_model({"obs": input_dict["obs"][1]}, state, seq_lens)

        x = torch.cat((fc_out, cnn_out), -1)
        # print(f"x.shape[1]: {x.shape[1]}")
        # print(f"2*self.my_action_space.n: {2*self.my_action_space.n}")
        # assert x.shape[1] == 2*self.my_action_space.n
        out = self.head(x)

        return out, []


    def value_function(self):
        vf_fc = self.torch_sub_fc_model.value_function()
        vf_cnn = self.torch_sub_vis_model.value_function()
        vf_combined = torch.stack([vf_fc, vf_cnn], -1)

        return self.value_fn(vf_combined).squeeze(-1)



#%%
class SimpleActionMaskFc(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs,
                     model_config, name):
        
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)
        
        self.obs_space = obs_space
        self.action_space = action_space
        self.num_outputs = num_outputs
        self.model_config = model_config
        self.name = name
        
        orig_space = getattr(obs_space, "original_space", obs_space)
        assert (
            isinstance(orig_space, gym.spaces.Dict)
            and "action_mask" in orig_space.spaces
            and "real_obs" in orig_space.spaces
        )
        
        self.action_embed_model = TorchFC(orig_space['real_obs'], action_space, num_outputs,
                                          model_config, name+'_action_embed_model')
        
        # disable action masking --> will likely lead to invalid actions
        self.no_masking = False
        if "no_masking" in model_config["custom_model_config"]:
            self.no_masking = model_config["custom_model_config"]["no_masking"]


    def forward(self, input_dict, state, seq_lens):
        # Compute the unmasked logits.
        logits, _ = self.action_embed_model({"obs": input_dict["obs"]["real_obs"]})

        # If action masking is disabled, directly return unmasked logits
        if self.no_masking:
            return logits, state

        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]
        
        # Convert action_mask into a [0.0 || -inf]-type mask.
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        masked_logits = logits + inf_mask

        # Return masked logits.
        return masked_logits, state


    def value_function(self):
        return self.action_embed_model.value_function()
        
  
    
#%%
class MySimpleFc(TorchModelV2, nn.Module, ABC):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        self.obs_space = obs_space
        self.action_space = action_space
        self.num_outputs = num_outputs
        self.model_config = model_config
        self.name = name
        
        self.network = nn.Sequential(
            nn.Linear(in_features=self.obs_space.shape[0], out_features=256),
            nn.Tanh(),
            nn.Linear(in_features=256, out_features=256),
            )
        
        self._policy_fn = nn.Sequential(
            nn.Linear(in_features=256, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=num_outputs))
        
        self._value_fn = nn.Sequential(
            nn.Linear(in_features=256, out_features=1))
        
        
    def forward(self, input_dict, state, seq_lens):
        network_out = self.network(input_dict["obs"])
        value = self._value_fn(network_out)
        self._value = value.reshape(-1)
        logits = self._policy_fn(network_out)
        return logits, state


    def value_function(self):
        return self._value#.flatten # maybe i need to remove flatten
    


#%%
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN

class MySimpleCnnGcn(TorchRNN, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, 
                fc_size=64, lstm_state_size=256):

        nn.Module.__init__(self)
        
        super().__init__(obs_space, action_space, num_outputs, model_config,
                         name)

        self.obs_size = get_preprocessor(self.obs_space)(self.obs_space).size
        self.fc_size = fc_size
        self.lstm_state_size = lstm_state_size

        self.obs_space = obs_space
        self.action_space = action_space
        self.num_outputs = num_outputs
        self.model_config = model_config
        self.name = name
        
        self.num_frames = 20
         
        self.fc_size = self.num_frames * self.lstm_state_size
        
        self.linear = nn.Linear(in_features=self.obs_space.shape[0], out_features=256)

        self.lstm = nn.LSTM(self.obs_size, self.lstm_state_size, batch_first=True)
        
        self.network = nn.Sequential(
            # nn.Linear(in_features=self.obs_space.shape[0], out_features=256),
            # nn.Tanh(),
            nn.Linear(in_features=256, out_features=256),
            nn.Tanh(),
            nn.Linear(in_features=256, out_features=256),
            nn.Tanh(),
            nn.Linear(in_features=256, out_features=256),
            )
        
        self._policy_fn = nn.Sequential(
            nn.Linear(in_features=256, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=num_outputs))
        
        self._value_fn = nn.Sequential(
            nn.Linear(in_features=256, out_features=1))
        
        
    def forward(self, input_dict, states, seq_lens):
        s1 = torch.unsqueeze(states[0], 0)
        s2 = torch.unsqueeze(states[1], 0)
        features, [h, c] = self.lstm(input_dict["obs"], [s1, s2])
        features = torch.reshape(features, [-1, features.shape[1] * features.shape[2]])
        network_out = self.network(features)
        value = self._value_fn(network_out)
        self._value = value.reshape(-1)
        logits = self._policy_fn(network_out)
        return logits, states


    def value_function(self):
        return self._value#.flatten # maybe i need to remove flatten
    
    
#%%

class MySimpleCnnGcn(TorchRNN, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, 
                fc_size=64, lstm_state_size=256):

        nn.Module.__init__(self)
        
        super().__init__(obs_space, action_space, num_outputs, model_config,
                         name)

        self.obs_size = get_preprocessor(self.obs_space)(self.obs_space).size
        self.fc_size = fc_size
        self.lstm_state_size = lstm_state_size

        self.obs_space = obs_space
        self.action_space = action_space
        self.num_outputs = num_outputs
        self.model_config = model_config
        self.name = name
        
        self.num_frames = 20
         
        self.fc_size = self.num_frames * self.lstm_state_size
        
        self.linear = nn.Linear(in_features=self.obs_space.shape[0], out_features=256)

        self.lstm = nn.LSTM(self.obs_size, self.lstm_state_size, batch_first=True)
        
        self.network = nn.Sequential(
            # nn.Linear(in_features=self.obs_space.shape[0], out_features=256),
            # nn.Tanh(),
            nn.Linear(in_features=256, out_features=256),
            nn.Tanh(),
            nn.Linear(in_features=256, out_features=256),
            nn.Tanh(),
            nn.Linear(in_features=256, out_features=256),
            )
        
        self._policy_fn = nn.Sequential(
            nn.Linear(in_features=256, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=num_outputs))
        
        self._value_fn = nn.Sequential(
            nn.Linear(in_features=256, out_features=1))
        
        
    def forward(self, input_dict, states, seq_lens):
        # s1 = torch.unsqueeze(states[0], 0)
        # s2 = torch.unsqueeze(states[1], 0)
        # features, [h, c] = self.lstm(input_dict["obs"], [s1, s2])
        # features = torch.reshape(features, [-1, features.shape[1] * features.shape[2]])
        
        network_out = self.network(input_dict["obs"])
        value = self._value_fn(network_out)
        self._value = value.reshape(-1)
        logits = self._policy_fn(network_out)
        return logits, states


    def value_function(self):
        return self._value#.flatten # maybe i need to remove flatten
    
    
    
#%%
class MySimpleCnn(TorchModelV2, nn.Module, ABC):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        self.obs_space = obs_space
        self.action_space = action_space
        self.num_outputs = num_outputs
        self.model_config = model_config
        self.name = name
        
        self._in_channels = obs_space.shape[2]

        self.network = nn.Sequential(
            nn.Conv2d(self._in_channels, 16, 3),#, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 5),#, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 256, 11),#, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256*5*5, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU())

        self._policy_fn = nn.Sequential(
            nn.Linear(in_features=256, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=num_outputs))

        self._value_fn = nn.Sequential(
            nn.Linear(in_features=256, out_features=1))


    def forward(self, input_dict, state, seq_lens):
        obs_transformed = input_dict['obs'].permute(0, 3, 1, 2)
        network_output = self.network(obs_transformed.float())
        value = self._value_fn(network_output)
        self._value = value.reshape(-1)
        logits = self._policy_fn(network_output)
        return logits, state


    def value_function(self):
        return self._value
    
    

#%%
class MySimpleConv_(TorchModelV2, nn.Module, ABC):
    def __init__(self, obs_space, act_space, num_outputs, *args, **kwargs):
        TorchModelV2.__init__(self, obs_space, act_space, num_outputs, *args, **kwargs)
        nn.Module.__init__(self)
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, [8, 8], stride=(4, 4)),
            nn.ReLU(),
            nn.Conv2d(32, 64, [4, 4], stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(64, 64, [3, 3], stride=(1, 1)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
        )
        self.policy_fn = nn.Linear(512, num_outputs)
        self.value_fn = nn.Linear(512, 1)


    def forward(self, input_dict, state, seq_lens):
        model_out = self.model(input_dict["obs"].permute(0, 3, 1, 2))
        self._value_out = self.value_fn(model_out)
        return self.policy_fn(model_out), state


    def value_function(self):
        return self._value_out.flatten()



#%%
class MySimpleConv(TorchModelV2, nn.Module, ABC):
    def __init__(self, obs_space, act_space, num_outputs, *args, **kwargs):
        TorchModelV2.__init__(self, obs_space, act_space, num_outputs, *args, **kwargs)
        nn.Module.__init__(self)
        
        self._in_channels = obs_space.shape[2]
        self._num_actions = num_outputs#action_space.n
        
        self.conv1 = nn.Conv2d(self._in_channels, 16, [3, 3], stride=(1, 1))
        self.conv2 = nn.Conv2d(16, 32, [5, 5], stride=(2, 2))
        self.conv3 = nn.Conv2d(32, 512, [11, 11], stride=(1, 1))
        self.lin1 = nn.Linear(32768, 512)
        self.policy_fn = nn.Linear(512, num_outputs)
        self.value_fn = nn.Linear(512, 1)


    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"].permute(0, 3, 1, 2)
        x = self.conv1(x.float())
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.conv3(x)
        x = nn.ReLU()(x)
        x = nn.Flatten()(x)
        x = self.lin1(x)
        x = nn.ReLU()(x)
        
        self._value_out = self.value_fn(x)
        return self.policy_fn(x), state


    def value_function(self):
        return self._value_out.flatten()
    
    

#%%
class MySimpleFcCnn(TorchModelV2, nn.Module, ABC):
    def __init__(self, obs_space, action_space, num_outputs, 
                 model_config, name):
        
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, 
                              model_config, name)
        nn.Module.__init__(self)
        
        self.obs_space = obs_space
        self.action_space = action_space
        self.num_outputs = num_outputs
        self.model_config = model_config
        self.name = name
        
        self.orig_space = getattr(obs_space, "original_space", obs_space)
        
        
        self._in_channels_fc = self.orig_space[0].shape[0]
        self._in_channels_cnn = self.orig_space[1].shape[2]
        
        
        self.network_fc = nn.Sequential(
            nn.Linear(in_features=self._in_channels_fc, out_features=256),
            nn.Tanh(),
            nn.Linear(in_features=256, out_features=256))
        
        
        self.network_cnn = nn.Sequential(
            nn.Conv2d(self._in_channels_cnn, 16, [3, 3], stride=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(16, 32, [5, 5], stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 512, [11, 11], stride=(1, 1)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            )
        
        self._policy_fn = nn.Sequential(
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=num_outputs))
        
        self._value_fn = nn.Sequential(
            nn.Linear(in_features=512, out_features=1))
        
        
    def forward(self, input_dict, state, seq_lens):
        
        fc_out = self.network_fc(input_dict["obs"][0])
        cnn_out = self.network_cnn(input_dict["obs"][1].permute(0, 3, 1, 2))

        x = torch.cat((fc_out, cnn_out), -1)
        # print(f"x.shape[1]: {x.shape[1]}")
        # print(f"2*self.my_action_space.n: {2*self.my_action_space.n}")
        # assert x.shape[1] == 2*self.my_action_space.n
        out = self.head(x)
        
        logits = self._policy_fn(out)
        
        value = self._value_fn(out)
        self._value = value.reshape(-1)

        return logits, []
        
    
    def value_function(self):
        return self._value#.flatten
   


#%%
class MySimpleActionMaskFc(TorchModelV2, nn.Module, ABC):
    def __init__(self, obs_space, action_space, num_outputs, 
                 model_config, name):
        
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, 
                              model_config, name)
        nn.Module.__init__(self)
        
        self.obs_space = obs_space
        self.action_space = action_space
        self.num_outputs = num_outputs
        self.model_config = model_config
        self.name = name
        
        self.orig_space = getattr(obs_space, "original_space", obs_space)
        
        self.network = nn.Sequential(
            nn.Linear(in_features=self.orig_space['real_obs'].shape[0], out_features=256),
            nn.Tanh(),
            # nn.Linear(in_features=256, out_features=256),
            # nn.Tanh(),
            # nn.Linear(in_features=256, out_features=256),
            # nn.Tanh(),
            # nn.Linear(in_features=256, out_features=256),
            # nn.Tanh(),
            # nn.Linear(in_features=256, out_features=256),
            # nn.Tanh(),
            # nn.Linear(in_features=256, out_features=256),
            # nn.Tanh(),
            # nn.Linear(in_features=256, out_features=256),
            # nn.Tanh(),
            nn.Linear(in_features=256, out_features=256),
            )
        
        self._actor_head = nn.Sequential(
            nn.Linear(in_features=256, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=self.action_space.n)) ## ???? num_outputs?   action_space.n
        
        self._critic_head = nn.Sequential(
            nn.Linear(in_features=256, out_features=1))
        
        
    def forward(self, input_dict, state, seq_lens):
        network_out = self.network(input_dict["obs"]["real_obs"])
        logits = self._actor_head(network_out)
        
        action_mask = input_dict["obs"]["action_mask"]
        
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        logits += inf_mask
        
        
        value = self._critic_head(network_out)
        self._value = value.reshape(-1)
        
        return logits, state


    def value_function(self):
        return self._value#.flatten
   


#%%
class MySimpleMetaFc(TorchModelV2, nn.Module, ABC):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        self.obs_space = obs_space
        self.action_space = action_space
        self.num_outputs = num_outputs
        self.model_config = model_config
        self.name = name
        
        self.network = nn.Sequential(
            nn.Linear(in_features=self.obs_space.shape[0], out_features=256),
            nn.Tanh(),
            nn.Linear(in_features=256, out_features=256),
            nn.Tanh(),
            nn.Linear(in_features=256, out_features=256),
            nn.Tanh(),
            nn.Linear(in_features=256, out_features=256),
            nn.Tanh(),
            nn.Linear(in_features=256, out_features=256),
            nn.Tanh(),
            nn.Linear(in_features=256, out_features=256),
            nn.Tanh(),
            nn.Linear(in_features=256, out_features=256),
            nn.Tanh(),
            nn.Linear(in_features=256, out_features=96),
            )
        
        self.desired_area_layer = nn.Sequential(
            nn.Linear(in_features=9, out_features=16),
            # nn.Tanh(),
            )
        
        self.acheived_area_layer = nn.Sequential(
            nn.Linear(in_features=9, out_features=16),
            # nn.Tanh(),
            )
        
        self.desired_adj_layer = nn.Sequential(
            nn.Linear(in_features=36, out_features=16),
            # nn.Tanh(),
            )
        
        self.acheived_adj_layer = nn.Sequential(
            nn.Linear(in_features=36, out_features=16),
            # nn.Tanh(),
            )
        
        self.meta_linear_layer = nn.Sequential(
            nn.Linear(in_features=32, out_features=32),
            nn.Tanh(),
            )
        
        
        self.two_last_linear_layewrs = nn.Sequential(
            nn.Linear(in_features=128, out_features=256),
            nn.Tanh(),
            nn.Linear(in_features=256, out_features=256),
            )
        
        self._actor_head = nn.Sequential(
            nn.Linear(in_features=256, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=num_outputs)) 
        
        self._critic_head = nn.Sequential(
            nn.Linear(in_features=256, out_features=1))
        
        
    def forward(self, input_dict, state, seq_lens):
        real_obs = input_dict["obs"]
        network_out = self.network(real_obs)
        
        area_state_vec = real_obs[:, fenv_config['len_state_vec_for_walls']:fenv_config['len_state_vec_for_walls_rooms']]
        half_len_of_area_state_vec = int(area_state_vec.shape[1]/2)
        
        desired_area_layer_out = self.desired_area_layer(area_state_vec[:, :half_len_of_area_state_vec])
        acheived_area_layer_out = self.acheived_area_layer(area_state_vec[:, half_len_of_area_state_vec:])
        
        self.area_layer_out = desired_area_layer_out - acheived_area_layer_out
        # print(f"area_layer_out dist to origin: {torch .linalg.norm(area_layer_out, dim=1).mean()}")
        
        adj_state_vec = real_obs[:, fenv_config['len_state_vec_for_walls_rooms']:] # 2358
        half_len_of_adj_state_vec = int(adj_state_vec.shape[1]/2)
        
        desired_adj_layer_out = self.desired_adj_layer(adj_state_vec[:, :half_len_of_adj_state_vec])
        
        acheived_adj_layer_out = self.acheived_adj_layer(adj_state_vec[:, half_len_of_adj_state_vec:])
        
        self.adj_layer_out = desired_adj_layer_out - acheived_adj_layer_out
        # print(f"adj_layer_out dist to origin: {torch .linalg.norm(adj_layer_out, dim=1).mean()}")
        
        meta_out = torch.cat((self.area_layer_out, self.adj_layer_out), 1)
        
        self.meta_out = self.meta_linear_layer(meta_out)
        
        out = torch.cat((network_out, self.meta_out), 1)
        
        out = self.two_last_linear_layewrs(out)
        
        logits = self._actor_head(out)
        
        value = self._critic_head(out)
        self._value = value.reshape(-1)
        
        return logits, state


    def value_function(self):
        return self._value#.flatten
    
    

#%%
class MySimpleActionMaskMetaFc(TorchModelV2, nn.Module, ABC):
    def __init__(self, obs_space, action_space, num_outputs, 
                 model_config, name):
        
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, 
                              model_config, name)
        nn.Module.__init__(self)
        
        self.obs_space = obs_space
        self.action_space = action_space
        self.num_outputs = num_outputs
        self.model_config = model_config
        self.name = name
        
        self.orig_space = getattr(obs_space, "original_space", obs_space)
        
        self.network = nn.Sequential(
            nn.Linear(in_features=self.orig_space['real_obs'].shape[0], out_features=256),
            nn.Tanh(),
            nn.Linear(in_features=256, out_features=256),
            nn.Tanh(),
            nn.Linear(in_features=256, out_features=256),
            nn.Tanh(),
            nn.Linear(in_features=256, out_features=256),
            nn.Tanh(),
            nn.Linear(in_features=256, out_features=256),
            nn.Tanh(),
            nn.Linear(in_features=256, out_features=256),
            nn.Tanh(),
            nn.Linear(in_features=256, out_features=256),
            nn.Tanh(),
            nn.Linear(in_features=256, out_features=96),
            )
        
        self.desired_area_layer = nn.Sequential(
            nn.Linear(in_features=9, out_features=16),
            # nn.Tanh(),
            )
        
        self.acheived_area_layer = nn.Sequential(
            nn.Linear(in_features=9, out_features=16),
            # nn.Tanh(),
            )
        
        self.desired_adj_layer = nn.Sequential(
            nn.Linear(in_features=36, out_features=16),
            # nn.Tanh(),
            )
        
        self.acheived_adj_layer = nn.Sequential(
            nn.Linear(in_features=36, out_features=16),
            # nn.Tanh(),
            )
        
        self.meta_linear_layer = nn.Sequential(
            nn.Linear(in_features=32, out_features=32),
            nn.Tanh(),
            )
        
        
        self.two_last_linear_layewrs = nn.Sequential(
            nn.Linear(in_features=128, out_features=256),
            nn.Tanh(),
            nn.Linear(in_features=256, out_features=256),
            )
        
        self._actor_head = nn.Sequential(
            nn.Linear(in_features=256, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=self.action_space.n)) 
        
        self._critic_head = nn.Sequential(
            nn.Linear(in_features=256, out_features=1))
        
        
    def forward(self, input_dict, state, seq_lens):
        real_obs = input_dict["obs"]["real_obs"]
        network_out = self.network(real_obs)
        
        area_state_vec = real_obs[:, fenv_config['len_state_vec_for_walls']:fenv_config['len_state_vec_for_walls_rooms']]
        half_len_of_area_state_vec = int(area_state_vec.shape[1]/2)
        
        desired_area_layer_out = self.desired_area_layer(area_state_vec[:, :half_len_of_area_state_vec])
        acheived_area_layer_out = self.acheived_area_layer(area_state_vec[:, half_len_of_area_state_vec:])
        
        area_layer_out = desired_area_layer_out - acheived_area_layer_out
        
        adj_state_vec = real_obs[:, fenv_config['len_state_vec_for_walls_rooms']:] # 2358
        half_len_of_adj_state_vec = int(adj_state_vec.shape[1]/2)
        
        desired_adj_layer_out = self.desired_adj_layer(adj_state_vec[:, :half_len_of_adj_state_vec])
        
        acheived_adj_layer_out = self.acheived_adj_layer(adj_state_vec[:, half_len_of_adj_state_vec:])
        
        adj_layer_out = desired_adj_layer_out - acheived_adj_layer_out
        
        meta_out = torch.cat((area_layer_out, adj_layer_out), 1)
        
        meta_out = self.meta_linear_layer(meta_out)
        
        out = torch.cat((network_out, meta_out), 1)
        
        out = self.two_last_linear_layewrs(out)
        
        logits = self._actor_head(out)
        
        action_mask = input_dict["obs"]["action_mask"]
        
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        logits += inf_mask
        
        
        value = self._critic_head(out)
        self._value = value.reshape(-1)
        
        return logits, state


    def value_function(self):
        return self._value#.flatten
    
    
    
#%%
class MySimpleActionMaskCnnGcn(TorchModelV2, nn.Module, ABC):
    def __init__(self, obs_space, action_space, num_outputs, 
                 model_config, name):
        
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, 
                              model_config, name)
        nn.Module.__init__(self)
        
        self.obs_space = obs_space
        self.action_space = action_space
        self.num_outputs = num_outputs
        self.model_config = model_config
        self.name = name
        
        self.orig_space = getattr(obs_space, "original_space", obs_space)
        
        self.network = nn.Sequential(
            nn.Linear(in_features=self.orig_space['real_obs'].shape[0], out_features=256),
            nn.Tanh(),
            nn.Linear(in_features=256, out_features=256),
            nn.Tanh(),
            nn.Linear(in_features=256, out_features=256),
            nn.Tanh(),
            nn.Linear(in_features=256, out_features=256),
            nn.Tanh(),
            nn.Linear(in_features=256, out_features=256),
            nn.Tanh(),
            nn.Linear(in_features=256, out_features=256),
            )
        
        self._actor_head = nn.Sequential(
            nn.Linear(in_features=256, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=self.action_space.n)) ## ???? num_outputs?   action_space.n
        
        self._critic_head = nn.Sequential(
            nn.Linear(in_features=256, out_features=1))
        
        
    def forward(self, input_dict, state, seq_lens):
        network_out = self.network(input_dict["obs"]["real_obs"])
        logits = self._actor_head(network_out)
        
        action_mask = input_dict["obs"]["action_mask"]
        
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        logits += inf_mask
        
        
        value = self._critic_head(network_out)
        self._value = value.reshape(-1)
        
        return logits, state


    def value_function(self):
        return self._value#.flatten
    
    

#%%
# class GCN(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         torch.manual_seed(1234)
#         self.conv1 = GCNConv(dataset.num_features, 4)
#         self.conv2 = GCNConv(4, 4)
#         self.conv3 = GCNConv(4, 2)
#         self.classifier = Linear(2, dataset.num_classes)

#     def forward(self, x, edge_index):
#         h = self.conv1(x, edge_index)
#         h = h.tanh()
#         h = self.conv2(h, edge_index)
#         h = h.tanh()
#         h = self.conv3(h, edge_index)
#         h = h.tanh()  # Final GNN embedding space.
        
#         # Apply a final (linear) classifier.
#         out = self.classifier(h)

#         return out, h     
  
     
  
    
#%%
class MySimpleGnnCnn(TorchModelV2, nn.Module, ABC):
    def __init__(self, obs_space, action_space, num_outputs, 
                 model_config, name):
        
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, 
                              model_config, name)
        nn.Module.__init__(self)
        
        self.obs_space = obs_space
        self.action_space = action_space
        self.num_outputs = num_outputs
        self.model_config = model_config
        self.name = name
        
        
        
#%%
if __name__ == '__main__':
    img = np.random.normal(0, 1, (1, 41, 41, 3))
    img = torch.from_numpy(img).float()
    net = MySimpleConv()
    net.forward(img)
    