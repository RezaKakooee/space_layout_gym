# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 21:01:47 2021

@author: Reza Kakooee
"""

from abc import ABC, abstractmethod


class BaseReward(ABC):
    def __init__(self):
        self._reward = None

    @property
    def reward(self):
        return self._reward