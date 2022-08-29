# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 23:00:12 2021

@author: RK
"""

# %%
import os
import imageio
import numpy as np


# %%
def make_gif(input_dir, output_dir, gif_name):
    images = []
    filenames = os.listdir(input_dir)
    filenames = np.sort(filenames)
    for file_name in filenames:
        if file_name.endswith('.png'):
            file_path = os.path.join(input_dir, file_name)
            images.append(imageio.imread(file_path))
    imageio.mimsave(f'{output_dir}/{gif_name}.gif', images)
