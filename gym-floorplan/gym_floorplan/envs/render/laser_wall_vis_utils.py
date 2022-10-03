# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 14:27:30 2021

@author: RK
"""
import os
import copy
import numpy as np
from PIL import Image

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import matplotlib.patheffects as path_effects
from matplotlib.backends.backend_agg import FigureCanvasAgg

#%%
colors = {'yellow', 'purple', 'turquoise', 'coral', 'lightblue', 'silver'}

#%%
def render_plan(plan_data_dict, 
              episode, time_step,
              fenv_config=None,
              ):
    
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
    
    room_colors = {
        'room_1': 'blue',
        'room_2': 'red',
        'room_3': 'green',
        'room_4': 'magenta',
        'room_5': 'darkblue',
        'room_6': 'darkred',
        'room_7': 'darkgreen',
        'room_8': 'mediumvioletred',
        'room_9': 'orange',
        'room_10': 'aqua',
        'room_11': 'darkorchid',
        'room_12': 'deeppink',
        'room_13': 'pink',
        'room_14': 'gold',
        'room_15': 'beige',
        }
    
    outline_segments = plan_data_dict['outline_segments']
    inline_segments = plan_data_dict['inline_segments']
    obs_mat = plan_data_dict['obs_mat']
    # obs_m = copy.deepcopy(obs_mat)/10
    
    moving_labels = plan_data_dict['moving_labels']
    mask_numbers = plan_data_dict['mask_numbers']
    n_corners = 4
    
    number_of_total_walls = plan_data_dict['number_of_total_walls'] # add +1 to add when putting in range in the line below
    for i in range(n_corners):
        colors[f"wall_{i+1}_back_segment"] = 'black'
        colors[f"wall_{i+1}_front_segment"] = 'black'
    
    time_step += 1 # for fig title
    
    
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
    
    if fenv_config['so_thick_flag']:
        if fenv_config['max_x'] == 20:
            wall_linewidth = 24
            basewall_linewidth = 4
            outline_linewidth = 24
            masked_walls_linewidth = 25
        elif fenv_config['max_x'] == 40:
            wall_linewidth = 12
            basewall_linewidth = 4
            outline_linewidth = 13
            masked_walls_linewidth = 13
            
    else:
        wall_linewidth = 4
        basewall_linewidth = 4
        outline_linewidth = 4
        masked_walls_linewidth = 5
        
    if fenv_config['so_thick_flag']:
        outline_control_points = [
            [[-1, -1], [-1, 21]],
            [[-1, 21], [-1, -1]],
            [[21, 21], [-1, 21]],
            [[-1, 21], [21, 21]],
            ]
        for conrol_points in outline_control_points:
            ax.plot(conrol_points[0], conrol_points[1], 
                    color='black',#colors[key], 
                    linewidth=basewall_linewidth)
    else:
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
            # label=key, 
            linewidth=basewall_linewidth,
            ) 
        
        # if 'front' in key:
        #     ax.plot([seg_data['end_coord'][0]], [seg_data['end_coord'][1]], 
        #             marker=marker_dict[seg_data['direction']],
        #             markersize=markersize,
        #             color=colors[key])
      
    for key, seg_data in inline_segments.items():
        x = [seg_data['start_coord'][0], seg_data['reflection_coord'][0]]
        y = [seg_data['start_coord'][1], seg_data['reflection_coord'][1]]
        
        ax.plot(x,y, 
            color=colors[key],#colors[key[:6]], 
            # label=key, 
            linewidth=wall_linewidth,
            ) 
    
    if fenv_config['mask_flag']:
        for i in range(mask_numbers):
            ax.add_patch(Rectangle(
                         plan_data_dict['rectangles_vertex'][i],
                         plan_data_dict['mask_lengths'][i], 
                         plan_data_dict['mask_widths'][i],
                         fc='white', ec='none', lw=10))
     
        
    # plot outline segments again
    if fenv_config['so_thick_flag']:
        outline_control_points = [
            [[-1, -1], [-1, 21]],
            [[-1, 21], [-1, -1]],
            [[21, 21], [-1, 21]],
            [[-1, 21], [21, 21]],
            ]
        for conrol_points in outline_control_points:
            ax.plot(conrol_points[0], conrol_points[1], 
                    color='black',#colors[key], 
                    linewidth=outline_linewidth)
            
    else:
        for key, segment in outline_segments.items():
            x = [segment['start_coord'][0] , segment['end_coord'][0]]
            y = [segment['start_coord'][1] , segment['end_coord'][1]]
            ax.plot(x,y, 
                    color=colors[key],#colors[key], 
                    linewidth=outline_linewidth)
        
    
    delta_l = 1 if fenv_config['so_thick_flag'] else 0 # delta length
    
    if fenv_config['so_thick_flag']:
        if fenv_config['mask_flag']:
            for i, (key, seg_data) in enumerate(plan_data_dict['walls_segments'].items()):
                
                if i <= mask_numbers-1:
                    if plan_data_dict['masked_corners'][i] == 'corner_00':
                        corner_xy = [fenv_config['min_x'], fenv_config['min_y']]
                        corner_xy = [corner_xy[0]-1, corner_xy[1]-1]
                        for k, sd in seg_data.items():
                            if sd['direction'] == 'south':
                                x1 = [corner_xy[0] , sd['reflection_coord'][0]-delta_l]
                                y1 = [corner_xy[1] , sd['reflection_coord'][1] - 1]
                            elif sd['direction'] == 'west':
                                x2 = [corner_xy[0] , sd['reflection_coord'][0]-1]
                                y2 = [corner_xy[1] , sd['reflection_coord'][1]-delta_l ]
                                
                    elif plan_data_dict['masked_corners'][i] == 'corner_01':
                        corner_xy = [fenv_config['min_x'], fenv_config['max_y']]
                        corner_xy = [corner_xy[0]-1, corner_xy[1]+1]
                        for k, sd in seg_data.items():
                            if sd['direction'] == 'north':
                                x1 = [corner_xy[0] , sd['reflection_coord'][0]-delta_l]
                                y1 = [corner_xy[1] , sd['reflection_coord'][1] + 1]
                            elif sd['direction'] == 'west':
                                x2 = [corner_xy[0] , sd['reflection_coord'][0] - 1]
                                y2 = [corner_xy[1] , sd['reflection_coord'][1]+delta_l]
                        
                    elif plan_data_dict['masked_corners'][i] == 'corner_10':
                        corner_xy = [fenv_config['max_x'], fenv_config['min_y']]
                        corner_xy = [corner_xy[0]+1, corner_xy[1]-1]
                        for k, sd in seg_data.items():
                            if sd['direction'] == 'south':
                                x1 = [corner_xy[0] , sd['reflection_coord'][0]+delta_l]
                                y1 = [corner_xy[1] , sd['reflection_coord'][1] - 1]
                            elif sd['direction'] == 'east':
                                x2 = [corner_xy[0] , sd['reflection_coord'][0] + 1]
                                y2 = [corner_xy[1] , sd['reflection_coord'][1]-delta_l]
                        
                    elif plan_data_dict['masked_corners'][i] == 'corner_11':
                        corner_xy = [fenv_config['max_x'], fenv_config['max_y']]
                        corner_xy = [corner_xy[0]+1, corner_xy[1]+1]
                        for k, sd in seg_data.items():
                            if sd['direction'] == 'north':
                                x1 = [corner_xy[0] , sd['reflection_coord'][0]+delta_l]
                                y1 = [corner_xy[1] , sd['reflection_coord'][1] + 1]
                            elif sd['direction'] == 'east':
                                x2 = [corner_xy[0] , sd['reflection_coord'][0] + 1]
                                y2 = [corner_xy[1]+1 , sd['reflection_coord'][1]+delta_l]
                                
                    else:
                        raise ValueError('Wrong corner to display')
                    fenv_config['show_render_flag']    
                        
                        
                    ax.plot(x1,y1, 
                        color='white',#colors[key[:6]], 
                        # label=key, 
                        linewidth=masked_walls_linewidth,
                        ) 
                    
                    ax.plot(x2,y2, 
                        color='white',#colors[key[:6]], 
                        # label=key, 
                        linewidth=masked_walls_linewidth,
                        ) 
        
    else:
        if fenv_config['mask_flag']:
            for i, (key, seg_data) in enumerate(plan_data_dict['walls_segments'].items()):
                
                if i <= mask_numbers-1:
                    if plan_data_dict['masked_corners'][i] == 'corner_00':
                        corner_xy = [fenv_config['min_x'], fenv_config['min_y']]
                        for k, sd in seg_data.items():
                            if sd['direction'] == 'south':
                                x1 = [corner_xy[0] , sd['reflection_coord'][0]-delta_l]
                                y1 = [corner_xy[1] , sd['reflection_coord'][1]]
                            elif sd['direction'] == 'west':
                                x2 = [corner_xy[0] , sd['reflection_coord'][0]]
                                y2 = [corner_xy[1] , sd['reflection_coord'][1]-delta_l]
                        
                    elif plan_data_dict['masked_corners'][i] == 'corner_01':
                        corner_xy = [fenv_config['min_x'], fenv_config['max_y']]
                        for k, sd in seg_data.items():
                            if sd['direction'] == 'north':
                                x1 = [corner_xy[0] , sd['reflection_coord'][0]-delta_l]
                                y1 = [corner_xy[1] , sd['reflection_coord'][1]]
                            elif sd['direction'] == 'west':
                                x2 = [corner_xy[0] , sd['reflection_coord'][0]]
                                y2 = [corner_xy[1] , sd['reflection_coord'][1]+delta_l]
                        
                    elif plan_data_dict['masked_corners'][i] == 'corner_10':
                        corner_xy = [fenv_config['max_x'], fenv_config['min_y']]
                        for k, sd in seg_data.items():
                            if sd['direction'] == 'south':
                                x1 = [corner_xy[0] , sd['reflection_coord'][0]+delta_l]
                                y1 = [corner_xy[1] , sd['reflection_coord'][1]]
                            elif sd['direction'] == 'east':
                                x2 = [corner_xy[0] , sd['reflection_coord'][0]]
                                y2 = [corner_xy[1] , sd['reflection_coord'][1]-delta_l]
                        
                    elif plan_data_dict['masked_corners'][i] == 'corner_11':
                        corner_xy = [fenv_config['max_x'], fenv_config['max_y']]
                        for k, sd in seg_data.items():
                            if sd['direction'] == 'north':
                                x1 = [corner_xy[0] , sd['reflection_coord'][0]+delta_l]
                                y1 = [corner_xy[1] , sd['reflection_coord'][1]]
                            elif sd['direction'] == 'east':
                                x2 = [corner_xy[0] , sd['reflection_coord'][0]]
                                y2 = [corner_xy[1] , sd['reflection_coord'][1]+delta_l]
                                
                    else:
                        raise ValueError('Wrong corner to display')
                    fenv_config['show_render_flag']    
                        
                        
                    ax.plot(x1,y1, 
                        color='white',#colors[key[:6]], 
                        # label=key, 
                        linewidth=masked_walls_linewidth,
                        ) 
                    
                    ax.plot(x2,y2, 
                        color='white',#colors[key[:6]], 
                        # label=key, 
                        linewidth=masked_walls_linewidth,
                        ) 
                    
    
    if fenv_config['show_room_dots_flag']:
        for i, (room_name, room_data) in enumerate(plan_data_dict['rooms_dict'].items()):
            if i+1 > mask_numbers:
                room_i = int(room_name.split('_')[1])
                room_positions = np.argwhere(moving_labels == room_i).tolist()
                room_coords = [
                        _image_coords2cartesian(p[0], p[1], fenv_config['n_rows']) for p in room_positions
                        ]
                for rc in room_coords:
                    ax.plot(rc[0]+1, rc[1]+1, 
                            color=room_colors[room_name], 
                            marker='.',
                            markersize=15,
                            ) 
            
    
    if fenv_config['show_graph_on_plan_flag']:
        rooms_gravity_coord_dict = plan_data_dict['rooms_gravity_coord_dict']
        for room_name, room_gravity in rooms_gravity_coord_dict.items():
            ax.plot([room_gravity[0]], [room_gravity[1]], 
                    color=room_colors[room_name], 
                    label=room_name, 
                    # linewidth=masked_walls_linewidth,
                    marker='*',
                    markersize=20
                    ) 
        
        desired_edge_list = (np.array(plan_data_dict['desired_edge_list'], dtype=int)+n_corners).tolist()
        # desired_edge_list = desired_edge_list[:int(0.6*len(desired_edge_list))]
        edge_list = (np.array(plan_data_dict['edge_list'], dtype=int)+n_corners).tolist()
        # desired_edge_list = _filter_edge_list(desired_edge_list, n_corners)
        # edge_list = _filter_edge_list(edge_list, n_corners)
        all_edges = edge_list + [edge for edge in desired_edge_list if edge not in edge_list]
        for edge in all_edges:
            
            if (edge in desired_edge_list) and (edge in edge_list):
                color = 'green'
            elif (edge in desired_edge_list) and (edge not in edge_list):
                color = 'red'
            elif (edge not in desired_edge_list) and (edge in edge_list):
                color = 'blue'
            else:
                raise ValueError('Not possible! Edge must be at least in edge_list or desierd_edge_list')
            
            # color = 'green' if edge in desired_edge_list else 'red'
            from_room = f"room_{edge[0]}"
            to_room = f"room_{edge[1]}"
            coord_1 = rooms_gravity_coord_dict[from_room]
            coord_2 = rooms_gravity_coord_dict[to_room]
            # coord_1_adj, sig_adj = _conn_sig(coord_1, coord_2, beta=1).
            # xx, yy = _hanging_line(coord_1, coord_2)
            # ax.plot(xx, yy, color=color)
            
            xx = [coord_1[0], coord_2[0]]
            yy = [coord_1[1], coord_2[1]]
            ax.plot(xx, yy, 
                    linewidth=5, color=color,
             path_effects=[path_effects.SimpleLineShadow(shadow_color='white', alpha=0.7, rho=0.5),
                           path_effects.Normal()])
    
    
    # ax.set_axis_off()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    # fig.savefig("test.png")
    ax.grid(True)

    
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    # ... convert to a NumPy array ...
    X = np.asarray(buf)
    # ... and pass it to PIL.
    
    im = Image.fromarray(X)
    
    obs_mat = _edit_obs_mat(X[:,:,0])
    
    
    if fenv_config['show_render_flag']:
        # plt.title(f"mask_numbers: {mask_numbers}")
        # lgd = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.015), fancybox=True, shadow=False, ncol=4)
        # ax.legend(loc='upper center', bbox_to_anchor=(0.95, 1), fancybox=True, shadow=False, ncol=4)
        # ax.set_title(f"time_step: {time_step}")
        plt.show(block=False)
        
    if fenv_config['save_render_flag']:
        scenario_dir = fenv_config['the_scenario_dir']
        designed_layouts_dir = os.path.join(scenario_dir, 'inference_files/designed_layouts/')
        if not os.path.exists(designed_layouts_dir):
            os.makedirs(designed_layouts_dir)
        plan_fig_path = os.path.join(designed_layouts_dir, f"trial_{fenv_config['trial']:02}_layout_episode_{episode:04}_timestep_{time_step:04}.png")
        # fig.savefig(plan_fig_path, bbox_extra_artists=(lgd,), bbox_inches='tight')
        fig.savefig(plan_fig_path, bbox_inches='tight')
        
        
        
    return obs_mat


def __crop_center(img, cropx, cropy):
    y,x = img.shape
    startx = x//2 - (cropx//2)
    starty = y//2 - (cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]


def _edit_obs_mat(obs_mat):
    obs_mat = __crop_center(obs_mat, 800, 800)
    obs_mat[0:10, :] = 20
    obs_mat[-10:, :] = 20
    obs_mat[:, 0:10] = 20
    obs_mat[:, -10:] = 20
    return obs_mat


def _get_obs_grayscale(plan_data_dict):
    obs_m = plan_data_dict['obs_mat']/10
    n_rows, n_cols = obs_m.shape
    obs_mat_w = plan_data_dict['obs_mat_w']
    moving_labels = plan_data_dict['moving_labels']
    obs_grayscale = copy.deepcopy(obs_m)
    obs_grayscale[0, :] = 1
    obs_grayscale[-1, :] = 1
    obs_grayscale[:, 0] = 1
    obs_grayscale[:, -1] = 1
    for r in range(n_rows):
        for c in range(n_cols):
            if (moving_labels[r][c] > 0) and  (moving_labels[r][c] <= 2):
                obs_grayscale[r][c] = 0
    return obs_grayscale


def _image_coords2cartesian(r, c, n_rows):
    return c-1, n_rows-2-r
    

def _cartesian2image_coord(x, y, max_y):
        return max_y-y, x
    

def _conn_sig(coord_1, coord_2, beta=1):
    coord_1_ = np.linspace(-5, 5)
    coord_1_adj = np.linspace(*coord_1)
    sig = 1/(1 + np.exp(-coord_1_*beta))
    sig = (sig - sig.min()) / sig.max()
    sig_adj = sig * (coord_2[1] - coord_2[0]) + coord_2[0]
    return coord_1_adj, sig_adj
    


def conn_sig(coord_1, y, beta=1):
    coord_1_ = np.linspace(-5, 5)
    coord_1_adj = np.linspace(*coord_1)
    sig = 1/(1+np.exp(-coord_1_*beta))
    sig = (sig-sig.min())/sig.max()
    sig_adj = sig *(y[1]-y[0]) + y[0]


def _hanging_line(coord_1, coord_2):
    a = (coord_2[1] - coord_1[1])/(np.cosh(coord_2[0]) - np.cosh(coord_1[0]))# +np.finfo(float).eps)
    b = coord_1[1] - a*np.cosh(coord_1[0])
    xx = np.linspace(coord_1[0], coord_2[0], 100)
    yy = a*np.cosh(xx) + b
    return xx, yy
    
    
def _filter_edge_list(edge_list, n_corners):
    edge_list_ = [
        [ [edge[0], edge[1]] for edge in edge_list if ( (edge[0]>n_corners) and (edge[1]>n_corners) ) ]
        ]
    return edge_list_[0]
    
    
# %%        
def display_env(obs_mat, labels, episode, time_step, fenv_config, save_fig=True):
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
    if fenv_config['save_render_flag']:
        scenario_dir = fenv_config['scenario_dir']
        fig.savefig(f"{scenario_dir}/trial_{fenv_config['trial']:02}_rooms_episode_{episode:04}_timestep_{time_step:04}.png")


# %%
def illustrate_obs_arr_conv(obs_arr_conv, episode, time_step, fenv_config, save_fig=True):
    time_step += 1
    fig = plt.figure(figsize=(5,5))
    plt.imshow(obs_arr_conv)
    plt.title('obs_arr_conv')
    plt.axis('off')
    if fenv_config['save_render_flag']:
        scenario_dir = fenv_config['scenario_dir']
        fig.savefig(f"{scenario_dir}/trial_{fenv_config['trial']:02}_segmentation_episode_{episode:04}_timestep_{time_step:04}.png")
       
        
# %%   
cmaps = ['CMRmap', 'CMRmap_r', 'Paired', 'Paired_r', 
         'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 
         'Spectral', 'Spectral_r', 'binary', 'binary_r', 'bone', 'bone_r', 
         'brg', 'brg_r', 'bwr', 'bwr_r', 
         'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r',  
         'hsv_r', 'jet', 'jet_r', 'nipy_spectral', 'plasma']

def demonestrate_rooms_with_walls(plan_data_dict, episode, time_step, fenv_config, save_fig=True):
    rw = plan_data_dict['moving_labels'] - plan_data_dict['obs_mat_w']
    rw = rw - (plan_data_dict['mask_numbers']-1)
    rw[rw <= 0] = 1 
    # rw_ = rw ** 2
    fig = plt.figure(figsize=(5,5))
    plt.imshow(rw, cmap='CMRmap_r')
    # plt.title(cmap)
    plt.axis('off')
    
    if fenv_config['save_render_flag']:
        scenario_dir = fenv_config['scenario_dir']
        designed_rooms_with_walls_dir = os.path.join(scenario_dir, 'inference_files/designed_layouts/')
        
        if not os.path.exists(designed_rooms_with_walls_dir):
            os.makedirs(designed_rooms_with_walls_dir)
                
        plan_fig_path = os.path.join(designed_rooms_with_walls_dir, f"trial_{fenv_config['trial']:02}_rooms_with_walls_episode_{episode:04}_timestep_{time_step:04}.png")
        fig.savefig(plan_fig_path, bbox_inches='tight')
       
        
    


# %%

# def plot_contour(contours):
#     import matplotlib.pyplot as plt
#     fig = plt.figure(figsize=(5,5))
#     X = contours[0][:,0]
#     Y = contours[0][:,1]
#     plt.plot(X, Y)
#     plt.show()
    
    
# def plot_obs_mat_for_conv(obs_mat_for_conv):
#     import matplotlib.pyplot as plt
#     fig = plt.figure(figsize=(5,5))
#     plt.imshow(obs_mat_for_conv)
#     plt.show()
    

# def show_obs_mat(obs_mat):
#     fig = plt.figure(figsize=(5,5))
#     plt.imshow(obs_mat)
#     plt.title('obs_mat')
#     plt.axis('off')