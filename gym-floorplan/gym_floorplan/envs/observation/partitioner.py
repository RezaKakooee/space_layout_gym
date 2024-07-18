# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 14:45:18 2021

@author: Reza Kakooee
"""

import numpy as np


#%%
class Partitioner:
    def __init__(self, obs_mat):
        self.obs_mat = obs_mat

    
    def get_rectangules(self):
        rects = self._parse_rows(self.obs_mat) 
        
        all_rects = {}
        rect_counter = 0
        for i in range(1, len(rects)+1):
            rows = rects[i]['rows']
            cols = rects[i]['cols']
            if self._strictly_insrementally_increasing(cols[0]): # cols[0] -> we send only one row
                rect_counter += 1
                all_rects.update({rect_counter: rects[i]})
                
            else:
                sub_rects = self._parse_cols(cols[0])
                for ind, col_set in sub_rects.items():
                    rect_counter += 1
                    
                    all_rects.update({ rect_counter: {'rows': rows, 'cols': np.vstack([col_set]*len(rows)).tolist() }  })
        
        return all_rects
        
    
    def _parse_rows(self, obs_mat):
        n_rows, n_cols = np.shape(obs_mat)
        cols_list = []
        for r in range(n_rows):
            cols_list.append( list(np.where(obs_mat[r]!=0)[0]) )
        cols_list.append([])
        r = 0
        count = 0
        rects = {}
        row_addition_flag = True
        while row_addition_flag:
            if r <= n_rows - 1:
                if len(cols_list[r]) > 0:
                    count += 1
                    rects.update({ count: {'rows': [r], 'cols':[cols_list[r]]} })
                    this_rect_flag = True
                    while this_rect_flag:
                        if len(cols_list[r]) == len(cols_list[r+1]):
                            if np.all(cols_list[r] == cols_list[r+1]):
                                rects[count]['rows'].append(r+1)
                                rects[count]['cols'].append(cols_list[r+1])
                                r += 1
                                if r >= n_rows:
                                    row_addition_flag = False
                            else:
                                this_rect_flag = False
                        else:
                            this_rect_flag = False
            r += 1
            if r >= n_rows:
                row_addition_flag = False
        
        return rects
    
    
    def _parse_cols(self, cols0):
        c = 0
        count = 0
        col_addition_flag = True
        len_cols = len(cols0)
        clusters = {}
        cols0.append(-1)
        while col_addition_flag:
            count += 1
            if c <= len_cols - 1:
                clusters.update({ count: [cols0[c]] })
                this_rect_flag = True
                while this_rect_flag:
                    if (cols0[c] == cols0[c+1]-1):
                        clusters[count].append(cols0[c+1])
                        c += 1
                        if c >= len_cols:
                            col_addition_flag = False
                    else:
                        this_rect_flag = False
            c += 1
            if c >= len_cols:
                col_addition_flag = False
        
        return clusters
    
    
    @staticmethod
    def _strictly_insrementally_increasing(L):
        return all(x==y-1 for x, y in zip(L, L[1:]))
    

#%%
if __name__ == '__main__':
    obs_mat = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 1., 1., 0., 0., 1., 1., 0., 0.],
                        [0., 0., 1., 1., 0., 0., 1., 1., 0., 0.],
                        [0., 0., 1., 1., 1., 1., 0., 0., 0., 0.],
                        [0., 0., 1., 1., 1., 1., 0., 0., 0., 0.],
                        [0., 0., 0., 1., 1., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 1., 1., 0., 0., 0., 0., 0.],
                        [0., 1., 1., 0., 1., 1., 0., 0., 0., 0.],
                        [0., 1., 1., 0., 1., 1., 0., 0., 0., 0.]])
    
    
    obs_mat = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]])
    
    self = Partitioner(obs_mat)
    all_rects = self.get_rectangules()
    print(all_rects)