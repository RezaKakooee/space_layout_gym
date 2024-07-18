#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 16:00:35 2023

@author: Reza Kakooee
"""

import numpy as np
from typing import List
import matplotlib.pyplot as plt

class Visualize:
    def __init__(self):
        pass 
    
    
    def plot(self, frame_idx: int, 
                   scores: List[float], 
                   losses: List[float], 
                   ):
        plt.figure(figsize=(20, 5))
        plt.subplot(121)
        plt.title('frame %s. score: %s' % (frame_idx, np.mean(scores[-10:])))
        plt.plot(scores)
        plt.subplot(122)
        plt.title('loss')
        plt.plot(losses)
