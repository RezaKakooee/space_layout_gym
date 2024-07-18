#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 20:19:30 2023

@author: Reza Kakooee
"""

import numpy as np



def linear(x, x_start, x_end, y_start, y_end, k=1):
    return y_end - ((y_end-y_start)/(x_end-x_start))*(x - x_start)



def quadratic(x, x_start, x_end, y_start, y_end, k=1):
    return y_end - ((y_end-y_start)/((x_end-x_start)**2)) * ((x-x_start)**2)



def logarithmic(x, x_start, x_end, y_start, y_end, k=1):
    return y_end - ((y_end-y_start) / np.log(k * (x_end - x_start) + 1)) * np.log(k * (x - x_start) + 1)



def exponential(x, x_start, x_end, y_start, y_end, k=1):
    return y_end * ((y_start/y_end)**((x - x_start)/(x_end - x_start)))



def direct_linear(x, den):
    return - x / den

