# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 00:24:19 2021

@author: Reza
"""
from abc import ABC, abstractmethod

class BaseRender(ABC):
    def __init__(self):
        pass

    @property
    def show(self):
        raise NotImplementedError