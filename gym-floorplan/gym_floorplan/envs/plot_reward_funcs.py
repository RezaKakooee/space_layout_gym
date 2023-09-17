# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 00:16:37 2023

@author: team
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


from gym_floorplan.envs.fenv_config import LaserWallConfig

fenv_config = LaserWallConfig().get_config()


#%%
n_bins = 21 #reward_measurements_dict['n_desired_adjacencies'] + 1
edge_range = np.linspace(fenv_config['edge_diff_min'], fenv_config['edge_diff_max'], n_bins)

rew_lin = fenv_config['positive_final_reward'] + 1 - fenv_config['linear_reward_coeff'] * edge_range

rew_exp = np.exp( (fenv_config['positive_final_reward'] - fenv_config['exp_reward_coeff']*edge_range ) / fenv_config['exp_reward_temperature']  ).astype(int)

rew_qua = fenv_config['positive_final_reward'] - fenv_config['quad_reward_coeff'] * edge_range**2 

df = pd.DataFrame(
    {'edge_rage': edge_range,
     'Linear Reward': rew_lin, 
     'Exponential Reward': rew_exp, 
     'Quadratic Reward': rew_qua, 
     }
    )

fig = go.Figure()
fig.add_trace(go.Scatter(x=edge_range, y=rew_lin,
                    mode='lines',
                    name='Linear'))
fig.add_trace(go.Scatter(x=edge_range, y=rew_exp,
                    mode='lines',
                    name='Exponential'))
fig.add_trace(go.Scatter(x=edge_range, y=rew_qua,
                    mode='lines', 
                    name='Quadratic'))

fig.update_layout(title='Types of Reward Function',
                   xaxis_title='Delta Adjacency',
                   yaxis_title='Reward')

fig.write_image("Types of Reward Function.png", scale=3)

fig.show()






