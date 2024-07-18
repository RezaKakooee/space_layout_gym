# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 22:20:17 2021

@author: Reza
"""

from abc import ABC, abstractmethod


class BaseAction(ABC):
    def __init__(self):
        self._action_space = None

    @property
    def action_space(self):
        return self._action_space