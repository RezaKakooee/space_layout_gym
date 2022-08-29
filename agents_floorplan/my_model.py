#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 17:42:15 2021

@author: RK
"""

# %%
from abc import ABC

import torch
from torch import nn

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models.torch.visionnet import VisionNetwork as TorchVis

# %%
from gym_floorplan.envs.fenv_config import LaserWallConfig

fevn_config = LaserWallConfig().get_config()
complex_obs_space = False
if fevn_config['net_arch'] in ['cnnfc', 'fccnn']:
    complex_obs_space = True


# %%
class CnnFcModel(TorchModelV2, nn.Module, ABC):
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

        if complex_obs_space:
            self.torch_sub_vis_model = TorchVis(obs_space.original_space[0], action_space, num_outputs,
                                                model_config, name)

            self.torch_sub_fc_model = TorchFC(obs_space.original_space[1], action_space, num_outputs,
                                              model_config, name)

            self.value_f = nn.Linear(2, 1)
            self.head = nn.Linear(2 * num_outputs, action_space.n)
            self.head_ = nn.Linear(2 * action_space.n, num_outputs)
        else:
            self.torch_sub_model = TorchFC(obs_space, action_space, num_outputs,
                                           model_config, name)

    def forward(self, input_dict, state, seq_lens):
        if complex_obs_space:
            cnn_out, _ = self.torch_sub_vis_model({"obs": input_dict["obs"][0]}, state, seq_lens)
            fc_out, _ = self.torch_sub_fc_model({"obs": input_dict["obs"][1]}, state, seq_lens)

            x = torch.cat((cnn_out, fc_out), -1)
            out = self.head(x)
            out_ = self.head_(x)

            return out, []
        else:
            input_dict["obs"] = input_dict["obs"].float()
            fc_out, _ = self.torch_sub_model(input_dict, state, seq_lens)
            return fc_out, []

    def value_function(self):
        if complex_obs_space:
            vf_cnn = self.torch_sub_vis_model.value_function()
            vf_fc = self.torch_sub_fc_model.value_function()
            vf_combined = torch.stack([vf_cnn, vf_fc], -1)

            return self.value_f(vf_combined).squeeze(-1)
        else:
            return self.torch_sub_model.value_function()


# %%
class FcCnnModel(TorchModelV2, nn.Module):
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
        if complex_obs_space:
            self.torch_sub_fc_model = TorchFC(obs_space.original_space[0], action_space, num_outputs,
                                        model_config, name)

            self.torch_sub_vis_model = TorchVis(obs_space.original_space[1], action_space, num_outputs,
                                            model_config, name)

            self.value_f = nn.Linear(2, 1)
            self.head = nn.Linear(2*action_space.n, num_outputs)
            # print(f"num_outputs: {num_outputs}")
            # print(f"action_space.n: {action_space.n}")
            assert num_outputs == action_space.n
        else:
            self.torch_sub_model = TorchFC(obs_space, action_space, num_outputs,
                                        model_config, name)


    def forward(self, input_dict, state, seq_lens):
        if complex_obs_space:
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
            assert x.shape[1] == 2*self.my_action_space.n
            out = self.head(x)

            return out, []
        else:
            input_dict["obs"] = input_dict["obs"].float()
            fc_out, _ = self.torch_sub_model(input_dict, state, seq_lens)
            return fc_out, []


    def value_function(self):
        if complex_obs_space:
            vf_fc = self.torch_sub_fc_model.value_function()
            vf_cnn = self.torch_sub_vis_model.value_function()
            vf_combined = torch.stack([vf_fc, vf_cnn], -1)

            return self.value_f(vf_combined).squeeze(-1)
        else:
            return self.torch_sub_model.value_function()


# %%
class CnnModel(TorchModelV2, nn.Module, ABC):
    def __init__(self, obs_space, action_space, num_outputs,
                 model_config, name):

        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        self.torch_vis_model = TorchVis(obs_space.original_space, action_space, num_outputs,
                                        model_config, name)

    def forward(self, input_dict, state, seq_lens):
        input_dict["obs"] = input_dict["obs"].float()
        fc_out, _ = self.torch_sub_model(input_dict, state, seq_lens)
        return fc_out, []

    def value_function(self):
        if complex_obs_space:
            vf_cnn = self.torch_sub_vis_model.value_function()
            vf_fc = self.torch_sub_fc_model.value_function()
            vf_combined = torch.stack([vf_cnn, vf_fc], -1)

            return self.value_f(vf_combined).squeeze(-1)
        else:
            return self.torch_sub_model.value_function()


# %%
class CNNModelV2(TorchModelV2, nn.Module, ABC):
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
            (nn.Linear(3136, 512)),
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
