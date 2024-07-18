# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 00:40:28 2021

@author: Reza Kakooee
"""

from abc import ABC, abstractmethod


class BaseObservation(ABC):
    def __init__(self):
        self._observation_space = None

    @property
    def observation_space(self):
        return self._observation_space