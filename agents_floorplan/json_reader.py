# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 13:54:11 2021

@author: RK
"""
# %%
import os
import ast
import copy
import json
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
#pio.renderers.default = 'svg'
pio.renderers.default = 'browser'

from ray.rllib.offline.json_reader import JsonReader

from gif_maker import make_gif
from gym_floorplan.envs.fenv_config import LaserWallConfig

# %%
def json_reader(json_file_pathes):
    exp = defaultdict()
    for i, json_path in enumerate(json_file_pathes):
        df = pd.read_json(json_path, lines=True)
        cols = df.columns
        for index, row in df.iterrows():
            for col in cols:
                val = row[col]
                if len(val) > 0:
                    exp[f"worker_{i}_{val['time_step']}"] = val['env_data']
    
    exp_dict = {}
    for key,value in exp.items():
        if value not in exp_dict.values():
            exp_dict[key] = value
    
    exp_df = pd.DataFrame.from_dict(exp_dict).T
    
    return exp_df
                
    
def get_data_from_json(json_path):
    c = 0
    infos_list = []
    dones_list = []
    reader = JsonReader(json_path)
    for batch in reader.read_all_files():
        c += 1
        batch_list = batch.split_by_episode()
        dones = []
        infos = []
        for eps_batch in batch_list:
            infos.append(eps_batch["infos"])
            dones.append(eps_batch["dones"])
        
        infos_list.extend(infos[0])
        dones_list.extend(dones)
    print(f"_ - _ - _ Number of acts: {c}")
    return infos_list, dones_list


def scatter_room_area(areas_df, save_dir):
    area_names = list(areas_df.columns)
    room_names = [f"Room {i+1}" for i in range(len(area_names))]
    colors = ['r', 'g', 'b', 'm']
    xx = list(range(len(areas_df)))
    areas_df.insert(len(room_names), 'time', xx)
    fig = px.line(areas_df, x="time", y=area_names, title="Rooms areaa")
    fig.show()

    # fig, axs = plt.subplots(4, 1)
    # xx = list(range(len(areas_df)))
    # for i, area_name in enumerate(area_names):
    #     axs[i].plot(xx, areas_df[area_name], colors[i])
    #     axs[i].set_ylabel(f"{room_names[i]}'s area")
    #     axs[i].get_xaxis().set_visible(False)
    # fig.savefig(f"{save_dir}/room_areas.png")
    
    
def barplot_room_area(areas, i, save_dir):
    room_areas = list(areas.values())
    fig = plt.figure()
    room_names = [f"Room_{i+1}" for i in range(len(areas))]
    
    barlist = plt.bar(room_names, room_areas)
    barlist[0].set_color('r')
    barlist[1].set_color('g')
    barlist[2].set_color('b')
    barlist[3].set_color('m')
    plt.ylim([0, 250])
    plt.ylabel("Room Areas")
    plt.savefig(f"{save_dir}/room_areas_{i:05}.png")
    plt.show()
    

# %%
if __name__ == "__main__":
    fenv_config = LaserWallConfig().get_config()
    storage_dir = fenv_config['storage_dir']
    scenario_name = 'Scenario__DOLW__Masked_FTC__09_Rooms__LRres__Area__Adj__METAFC__2022_09_27_2325'
    scenario_dir = f"{storage_dir}/tunner/local_dir/{scenario_name}"
    agent_folder = [ name for name in os.listdir(scenario_dir) if os.path.isdir(os.path.join(scenario_dir, name)) ][0]
    env_data_dir = f"{scenario_dir}/{agent_folder}/env_data" 
    json_file_pathes = [f"{env_data_dir}/{json_name}" for json_name in os.listdir(env_data_dir)]
    
    
    exp_df = json_reader(json_file_pathes)
    
    optim_exp_df_path = f"{scenario_dir}/optim_exp_df_{scenario_name}.csv"
    exp_df.to_csv(optim_exp_df_path, index=False)
    
    
    selected_cols = ['n_rooms', 'mask_numbers', 'masked_corners',
       'mask_lengths', 'mask_widths', 'desired_areas', 'desired_edge_list']
    
    df = copy.deepcopy(exp_df)
    df = df[selected_cols]
    
    for col in df.columns:
        df[col] = df[col].astype(str)#.drop_duplicates().to_frame()
        
    df_ = df.drop_duplicates()

    # areas_df = pd.DataFrame(infos_list)
    # scatter_room_area(areas_df, save_dir=fenv_config['area_plots_dir'])
    
    # for i, areas in enumerate(infos_list):
    #     barplot_room_area(areas, i, save_dir=fenv_config['area_plots_dir'])
    
    # make_gif(input_dir=fenv_config['area_plots_dir'], output_dir=fenv_config['area_gif_dir'], gif_name="room_areas")
