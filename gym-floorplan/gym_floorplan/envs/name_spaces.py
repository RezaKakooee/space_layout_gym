# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 16:46:24 2024

@author: Reza Kakooee
"""

net_archs = {
        # model_last_name -> model_arch_name
        'TinyCnnEncoder': 'Cnn',
        'MetaCnnEncoder': 'MetaCnn',

        
        'TinyFcNet':'Fc', 
        'FcNet':'Fc', 
        'MetaFcNet': 'MetaFc',
        
        'TinyCnnNet':'Cnn', 
        'CnnNet':'Cnn', 
        'MetaCnnNet':'MetaCnn',
        'MetaCnnResNet':'MetaCnn',
        'MetaCnnResidualNet':'MetaCnn',
        
        'TinyFcNetActionMasking': 'Fc', 
        'FcNetActionMasking': 'Fc',
        'MetaFcNetActionMasking': 'MetaFc',
        
        'TinyCnnNetActionMasking': 'Cnn',
        'CnnNetActionMasking': 'Cnn',
        'MetaCnnNetActionMasking': 'MetaCnn',
        'MetaCnnResNetActionMasking': 'MetaCnn',
        'MetaCnnResidualNetActionMasking':'MetaCnn',
        'MetaCnnResidualNetActionMaskingQ':'MetaCnn',
        'MetaCnnResidualNetActionMaskingP':'MetaCnn',
        
        'TinyGnnNet': 'Gnn',
        'GnnNet': 'Gnn',
        
        'SimpleFc': 'Fc', 
        'SimpleCnn': 'Cnn',
        
        'DQNTinyFcNet': 'Fc',
        'DQNTinyFcNetActionMasking': 'Fc',

        'DQNCnnNetActionMasking': 'Cnn',

        'BaseCnnNet': 'Cnn',
        'CnnMiniResNet': 'Cnn',
        
        'MetaCnnNetFineTune': 'MetaCnn',
        'MetaCnnNetFromPreTrain': 'MetaCnn',
        
        'MetaCnnNetwork': 'MetaCnn',
        'MetaCnnNetFromPretrainedNet': 'MetaCnn',
        'MetaCnnNetPreTrainedEncoder': 'MetaCnn',

        }


### CNN obs
color_map = {
            0: 'cyan', #'blue', 
            
            1: 'dimgray', #'red', 
            2: 'gray', #'magenta', 
            3: 'gainsboro', #'darkblue', 
            4: 'silver', #'darkred', 
            
            5: 'darkblue', 
            6: 'darkgreen', # 'mediumvioletred', 
            7: 'darkred', # 'orange', 
            8: 'pink', # 'magenta', #'aqua', 
            9: 'darkseagreen', # 'darkorchid', 
            10: 'darkorchid', # 'skyblue', 
            11: 'blue', # 'pink', 
            12: 'darkviolet', # 'gold', 
            13: 'darkmagenta', # 'beige', 
            
            14: 'lime', #'silver', # white
            
            15: 'antiquewhite', # 'yellow',
            16: 'antiquewhite', # 'purple',  navajowhite
            17: 'antiquewhite', # 'teal',  blanchedalmond
            18: 'antiquewhite', # 'violet',  papayawhip
            19: 'moccasin', # chocolate
            
            20: 'hotpink',
            }