# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 14:27:30 2021

@author: Reza Kakooee
"""

import numpy as np
from PIL import Image

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_agg import FigureCanvasAgg

#%%
colors = {'left_segment': 'red',
          'down_segment': 'green',
          'right_segment': 'blue',
          'up_segment': 'yellow',
          'wall_0_back_segment': 'purple',
          'wall_0_front_segment': 'purple',
          'wall_1_back_segment': 'turquoise',
          'wall_1_front_segment': 'turquoise',
          'wall_2_back_segment': 'coral',
          'wall_2_front_segment': 'coral',
          'wall_3_back_segment': 'lightblue',
          'wall_3_front_segment': 'lightblue',
          'wall_4_back_segment': 'pink' ,
          'wall_4_front_segment': 'pink',
          
          'wall_5_back_segment': 'silver',
          'wall_5_front_segment': 'silver',
          'wall_6_back_segment': 'beige',
          'wall_6_front_segment': 'beige',
          'wall_7_back_segment': 'gold',
          'wall_7_front_segment': 'gold',
          'wall_8_back_segment': 'orange',
          'wall_8_front_segment': 'orange',
          'wall_9_back_segment': 'aqua' ,
          'wall_9_front_segment': 'aqua',
          'wall_10_back_segment': 'darkorchid' ,
          'wall_10_front_segment': 'darkorchid',
          'wall_11_back_segment': 'deeppink' ,
          'wall_11_front_segment': 'deeppink',
          }

    
#%%
def show_plan(plan_data_dict, 
              episode, time_step,
              fenv_config=None):
    
    colors = {'left_segment': 'black',
              'down_segment': 'black',
              'right_segment': 'black',
              'up_segment': 'black',
              
              'wall_1_back_segment': 'blue',
              'wall_1_front_segment': 'blue',
              'wall_2_back_segment': 'red',
              'wall_2_front_segment': 'red',
              'wall_3_back_segment': 'green',
              'wall_3_front_segment': 'green',
              'wall_4_back_segment': 'magenta',
              'wall_4_front_segment': 'magenta',
              'wall_5_back_segment': 'darkblue' ,
              'wall_5_front_segment': 'darkblue',
              'wall_6_back_segment': 'darkred',
              'wall_6_front_segment': 'darkred',
              'wall_7_back_segment': 'darkgreen',
              'wall_7_front_segment': 'darkgreen',
              'wall_8_back_segment': 'mediumvioletred',
              'wall_8_front_segment': 'mediumvioletred',
              'wall_9_back_segment': 'orange',
              'wall_9_front_segment': 'orange',
              'wall_10_back_segment': 'aqua',
              'wall_10_front_segment': 'aqua',
              'wall_11_back_segment': 'darkorchid',
              'wall_11_front_segment': 'darkorchid',
              'wall_12_back_segment': 'deeppink',
              'wall_12_front_segment': 'deeppink',
              'wall_13_back_segment': 'pink',
              'wall_13_front_segment': 'pink',
              'wall_14_back_segment': 'gold',
              'wall_14_front_segment': 'gold',
              'wall_15_back_segment': 'beige',
              'wall_15_front_segment': 'beige',
              }
    
    mask_numbers = plan_data_dict['mask_numbers']
    for i in range(mask_numbers):
        colors[f"wall_{i+1}_back_segment"] = 'black'
        colors[f"wall_{i+1}_front_segment"] = 'black'
    
    
    time_step += 1 # for fig title
    
    outline_segments = plan_data_dict['outline_segments']
    inline_segments = plan_data_dict['inline_segments']
    obs_mat = plan_data_dict['obs_mat']
    
    method = fenv_config['wall_correction_method']
    
    marker_dict = {'north': '^', 'south': 'v', 'east': '>', 'west': '<'}
    
    if fenv_config['show_render_flag']:
        dpi = 100
        figsize = (8,8)
    else:
        dpi = 100
        figsize = (10,10)
        plt.ioff()
        
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, frameon=False)
    fig.tight_layout()
    fig.patch.set_visible(False)
    wall_linewidth = 4
    basewall_linewidth = 4
    masked_walls_linewidth = 5
    markersize = 15
    for key, segment in outline_segments.items():
        x = [segment['start_coord'][0] , segment['end_coord'][0]]
        y = [segment['start_coord'][1] , segment['end_coord'][1]]
        ax.plot(x,y, 
                color=colors[key],#colors[key], 
                linewidth=basewall_linewidth)
        
    for key, seg_data in inline_segments.items():
        x = [seg_data['start_coord'][0], seg_data['end_coord'][0]]
        y = [seg_data['start_coord'][1], seg_data['end_coord'][1]]
        
        ax.plot(x,y, 
            color=colors[key],#colors[key[:6]], 
            label=key, 
            linewidth=basewall_linewidth,
            ) 
        
        # if 'front' in key:
        #     ax.plot([seg_data['end_coord'][0]], [seg_data['end_coord'][1]], 
        #             marker=marker_dict[seg_data['direction']],
        #             markersize=markersize,
        #             color=colors[key])
      
    if method is not None:    
        for key, seg_data in inline_segments.items():
            if method == 'concurrent':
                x = [seg_data['start_coord'][0], seg_data['extended_end_coord'][0]]
                y = [seg_data['start_coord'][1], seg_data['extended_end_coord'][1]]
            elif method == 'sequential':
                x = [seg_data['start_coord'][0], seg_data['reflection_coord'][0]]
                y = [seg_data['start_coord'][1], seg_data['reflection_coord'][1]]
            
            ax.plot(x,y, 
                color=colors[key],#colors[key[:6]], 
                # label=key, 
                linewidth=wall_linewidth,
                ) 
    
    if fenv_config['mask_flag']:
        for i in range(plan_data_dict['mask_numbers']):
            ax.add_patch(Rectangle(
                         plan_data_dict['rectangles_vertex'][i],
                         plan_data_dict['mask_lengths'][i], 
                         plan_data_dict['mask_widths'][i],
                         fc='white', ec='none', lw=10))
     
        
        
    # plot outline segments again
    for key, segment in outline_segments.items():
        x = [segment['start_coord'][0] , segment['end_coord'][0]]
        y = [segment['start_coord'][1] , segment['end_coord'][1]]
        ax.plot(x,y, 
                color=colors[key],#colors[key], 
                linewidth=basewall_linewidth)
        
        
    if fenv_config['mask_flag']:
        for i, (key, seg_data) in enumerate(plan_data_dict['walls_segments'].items()):
            if i <= fenv_config['mask_numbers']-1:
                if fenv_config['masked_corners'][i] == 'corner_00':
                    corner_xy = [fenv_config['min_x'], fenv_config['min_y']]
                elif fenv_config['masked_corners'][i] == 'corner_01':
                    corner_xy = [fenv_config['max_x'], fenv_config['min_y']]
                elif fenv_config['masked_corners'][i] == 'corner_10':
                    corner_xy = [fenv_config['min_x'], fenv_config['max_y']]
                elif fenv_config['masked_corners'][i] == 'corner_11':
                    corner_xy = [fenv_config['max_x'], fenv_config['max_y']]
                else:
                    raise ValueError('Wrong corner to display')
            
                x = [corner_xy[0] , seg_data['back_segment']['reflection_coord'][0]]
                y = [corner_xy[1] , seg_data['back_segment']['reflection_coord'][1]]
                    
                ax.plot(x,y, 
                    color='white',#colors[key[:6]], 
                    # label=key, 
                    linewidth=masked_walls_linewidth,
                    ) 
                
                x = [corner_xy[0] , seg_data['front_segment']['reflection_coord'][0]]
                y = [corner_xy[1] , seg_data['front_segment']['reflection_coord'][1]]
                    
                ax.plot(x,y, 
                    color='white',#colors[key[:6]], 
                    # label=key, 
                    linewidth=masked_walls_linewidth,
                    ) 
    
    
    # ax.set_axis_off()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    # fig.savefig("test.png")
    # ax.grid(True)

    
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    # ... convert to a NumPy array ...
    X = np.asarray(buf)
    # ... and pass it to PIL.
    
    im = Image.fromarray(X)
    
    obs_mat = edit_obs_mat(X[:,:,0])
    
    if fenv_config['show_render_flag']:
        # plt.title(method)
        # lgd = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.045), fancybox=True, shadow=False, ncol=4)
        # ax.legend(loc='upper center', bbox_to_anchor=(0.95, 1), fancybox=True, shadow=False, ncol=4)
        # ax.set_title(f"time_step: {time_step}")
        plt.show(block=False)
        
        if fenv_config['save_render_flag']:
            generated_plans_dir = fenv_config['generated_plans_dir']
            plan_fig_path = f"{generated_plans_dir}/trial_{fenv_config['trial']:02}_layout_episode_{episode:02}_timestep_{time_step:03}.png"
            # fig.savefig(plan_fig_path, bbox_extra_artists=(lgd,), bbox_inches='tight')
            fig.savefig(plan_fig_path, bbox_inches='tight')
        
    return obs_mat


def crop_center(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]


def edit_obs_mat(obs_mat):
    # mask = np.where(obs_mat==255)
    # obs_mat[mask] = 2
    
    obs_mat = crop_center(obs_mat,800,800)
    # mask10 = np.where(obs_mat<10)
    # obs_mat[mask10] = 300
    # mask255 = np.where(obs_mat<255)
    # obs_mat[mask255] = 100
    obs_mat[0:10,:] = 20
    obs_mat[-10:,:] = 20
    obs_mat[:,0:10] = 20
    obs_mat[:,-10:] = 20
    # mask300 = np.where(obs_mat==300)
    # obs_mat[mask300] = 0
    
    return obs_mat
    
        
def show_env(obs_mat, labels, episode, time_step, fenv_config, save_fig=True):
    time_step += 1
    import matplotlib as mpl

    mpl.rcParams['axes.linewidth'] = 0.1 #set the value globallys
    from skimage import color
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    # ax.matshow(1-obs_mat, cmap='gray')
    plt.imshow(color.label2rgb(labels, image=obs_mat))
    
    # row, col = self.env_data_model.state_coords_for_obs_dict[f'S{start_state}']
    # plt.plot(col, row, '*')
    plt.title('Segmentation Map')
    plt.axis('off')
    # plt.pause(0.000000001)
    
    # if not done:
    #     plt.close('all')
    if save_fig:
        generated_plans_dir = fenv_config['generated_plans_dir']
        fig.savefig(f"{generated_plans_dir}/trial_{fenv_config['trial']:02}_rooms_episode_{episode:02}_timestep_{time_step:03}.png")



def plot_contour(contours):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(5,5))
    X = contours[0][:,0]
    Y = contours[0][:,1]
    plt.plot(X, Y)
    plt.show()
    
    
def plot_obs_mat_for_conv(obs_mat_for_conv):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(5,5))
    plt.imshow(obs_mat_for_conv)
    plt.show()
    

def show_obs_mat(obs_mat):
    fig = plt.figure(figsize=(5,5))
    plt.imshow(obs_mat)
    plt.title('obs_mat')
    plt.axis('off')
    
def show_obs_arr_conv(obs_arr_conv, episode, time_step, fenv_config, save_fig=True):
    time_step += 1
    fig = plt.figure(figsize=(5,5))
    plt.imshow(obs_arr_conv)
    plt.title('obs_arr_conv')
    plt.axis('off')
    if save_fig:
        generated_plans_dir = fenv_config['generated_plans_dir']
        fig.savefig(f"{generated_plans_dir}/trial_{fenv_config['trial']:02}_segmentation_episode_{episode:02}_timestep_{time_step:03}.png")