#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 12:42:20 2023

@author: Reza Kakooee
"""
import os
import pandas as pd
#import seaborn as sns
# import matplotlib.pylab as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


#%%
class Visualizer:
    def __init__(self, agent_config):
        self.agent_config = agent_config
    
    
    
    def plot_tb(self):
        df_path = os.path.join(self.agent_config['results_dir'], 'res_summary_df.csv')
        self.df = pd.read_csv(df_path)
        self.df = self.df.reset_index()
        self.df = self.df.rename(columns={'index': 'Batch'})
        self._plot_tb_from_res_summary(self.df, y_name='ep_len_avg')
        self._plot_tb_from_res_summary(self.df, y_name='ep_last_reward_avg')
    
    
    
    def _plot_tb_from_res_summary(self, df, y_name):
        if isinstance(y_name, list):
            fig = px.line(df, x='Batch', y=y_name, #["episode_reward_mean", "episode_len_mean"],
                           labels={"timesteps_total": "Time Step", y_name: y_name.replace("_", " ").title()})
            fig.update_yaxes(title="Episode Mean Rewartd & Length", title_font=dict(size=20))
        else:
            fig = px.line(df, x='Batch', y=y_name, #"episode_reward_mean",
                          labels={"timesteps_total": "Time Step", y_name: y_name.replace("_", " ").title()})
        
        
        fig.update_traces(line=dict(width=3))
        fig.update_layout(height=600, width=800, hovermode='x')
        fig.update_layout({
                        # 'plot_bgcolor': 'rgba(0, 255, 255, 0.2)', 
                        'paper_bgcolor': 'rgba(0, 0, 0, 0)'})
        fsize = 30
        fig.update_xaxes(title_font=dict(size=fsize, family='Courier', color='black'), tickangle = 270)
        fig.update_yaxes(title_font=dict(size=25, family='Courier', color='black'))
        fig.update_xaxes(tickfont=dict(family='Rockwell', color='black', size=fsize), tickangle = 270)
        fig.update_yaxes(tickfont=dict(family='Rockwell', color='black', size=fsize))
        # fig.update_xaxes(range=[1_600_000, 2_200_000])
            
        
        # fig.update_yaxes(range=[ymax, ymin])
        fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99))
        fig.update_layout(legend={'title_text':''})
        fig.update_layout(legend_font_size=30)
        # fig.update_layout(template="presentation")

        fig.show()
        
        agent_performance_metrics_dir = os.path.join(self.agent_config['results_dir'])
        fig.write_image(f"{agent_performance_metrics_dir}/tb_plots_{y_name}.png", scale=3)