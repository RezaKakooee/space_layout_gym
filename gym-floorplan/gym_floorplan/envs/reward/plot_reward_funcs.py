# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 00:16:37 2023

@author: team
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go


from gym_floorplan.envs.fenv_config import LaserWallConfig

fenv_config = LaserWallConfig().get_config()


#%%
n_bins = 21 #reward_measurements_dict['n_desired_adjacencies'] + 1
edge_range = np.linspace(fenv_config['edge_diff_min'], fenv_config['edge_diff_max'], n_bins)

rew_lin = fenv_config['positive_final_reward'] + 1 - fenv_config['linear_reward_coeff'] * edge_range

rew_exp = np.exp( (fenv_config['positive_final_reward'] - fenv_config['exp_reward_coeff']*edge_range ) / fenv_config['exp_reward_temperature']  ).astype(int)

rew_qua = fenv_config['positive_final_reward'] - fenv_config['quad_reward_coeff'] * edge_range**2 


log_reward_coeff = 1000 
log_reward_base = 15 

rew_log = -log_reward_coeff * np.log(1 + edge_range) / np.log(log_reward_base) + log_reward_coeff

# df = pd.DataFrame(
#     {'edge_rage': edge_range,
#      'Linear Reward': rew_lin, 
#      'Exponential Reward': rew_exp, 
#      'Quadratic Reward': rew_qua, 
#      }
#     )


#%%
# fig = go.Figure()
# fig.add_trace(go.Scatter(x=edge_range, y=rew_lin, mode='lines', name='Linear'))
# fig.add_trace(go.Scatter(x=edge_range, y=rew_exp, mode='lines', name='Exponential'))
# fig.add_trace(go.Scatter(x=edge_range, y=rew_qua, mode='lines', name='Quadratic'))
# # fig.add_trace(go.Scatter(x=edge_range, y=rew_log, mode='lines', name='Logarithmic'))

# fig.update_layout(title='Types of Reward Function', xaxis_title='Delta Adjacency', yaxis_title='Reward')
# fig.write_image("Types of Reward Function.png", scale=3)
# fig.show()



#%%
import numpy as np
import matplotlib.pyplot as plt

# Function parameters
x_start = 0
x_end = 31
y_start = 1
y_end = 1000

# Define the functions as lambda functions
linear = lambda x: y_end - ((y_end-y_start)/(x_end-x_start))*(x - x_start)
quadratic = lambda x: y_end - ((y_end-y_start)/((x_end-x_start)**2)) * ((x-x_start)**2)
logarithmic = lambda x: y_end - ((y_end-y_start) / np.log(x_end - x_start + 1)) * np.log(x - x_start + 1)
exponential = lambda x: y_end * ((y_start/y_end)**((x - x_start)/(x_end - x_start)))

x = np.linspace(x_start, x_end, 500)

# Create the plot
plt.figure(figsize=(8,6), dpi=300)

plt.plot(x, linear(x), label='Linear', linewidth=4)
plt.plot(x, quadratic(x), label='Quadratic', linewidth=4)
plt.plot(x, logarithmic(x), label='Logarithmic', linewidth=4)
# plt.plot(x, exponential(x), label='Exponential')

plt.title('Different Types of Reward Functions', fontsize=20)
plt.xlabel('Delta Adjacency', fontsize=20)
plt.ylabel('Reward', fontsize=20)
plt.legend(fontsize=18)
# plt.grid(True)
# plt.xticks([])
# plt.yticks([])
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
# plt.tight_layout()

plt.savefig('Reward_Funcs.png', dpi=300, transparent=True)
plt.show()