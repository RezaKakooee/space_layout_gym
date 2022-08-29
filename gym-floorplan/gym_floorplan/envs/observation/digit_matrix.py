# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 10:41:26 2022

@author: RK
"""
import numpy as np
import matplotlib.pyplot as plt

# %%
class DigitMatrix:
    def __init__(self, size=(7, 5)):
        self.size = size
        self.empty = np.zeros(self.size)
        
        
    def digit_zero(self):
        zero = np.zeros(self.size)
        zero[1:6, 1] = 1
        zero[1:6, 3] = 1
        zero[1:6, 3] = 1 
        zero[1, 2] = 1
        zero[5, 2] = 1
        return zero


    def digit_one(self):
        one = np.zeros(self.size)
        one[1:-1, 2] = 1
        return one
    
    
    def digit_two(self):
        two = np.zeros(self.size)
        two[1, 1:4] = 1
        two[3, 1:4] = 1
        two[5, 1:4] = 1 
        two[2, 3] = 1
        two[4, 1] = 1
        return two
    
    
    def digit_three(self):
        three = np.zeros(self.size)
        three[1, 1:4] = 1
        three[3, 1:4] = 1
        three[5, 1:4] = 1 
        three[2, 3] = 1
        three[4, 3] = 1
        return three
    
    
    def digit_four(self):
        four = np.zeros(self.size)
        four[1:4, 1] = 1
        four[1:6, 3] = 1
        four[3, 2] = 1
        return four
    
    
    def digit_five(self):
        five = np.zeros(self.size)
        five[1, 1:4] = 1
        five[3, 1:4] = 1
        five[5, 1:4] = 1 
        five[2, 1] = 1
        five[4, 3] = 1
        return five
    
    
    def digit_six(self):
        six = np.zeros(self.size)
        six[1, 1:4] = 1
        six[3, 1:4] = 1
        six[5, 1:4] = 1 
        six[2, 1] = 1
        six[4, 1] = 1
        six[4, 3] = 1
        return six
    
    
    def digit_seven(self):
        seven= np.zeros(self.size)
        seven[1, 1:4] = 1
        seven[1:-1, 3] = 1
        return seven
    
    
    def digit_eight(self):
        eight= np.zeros(self.size)
        eight[1:-1, 1] = 1
        eight[1:-1, 3] = 1
        eight[1, 2] = 1
        eight[3, 2] = 1
        eight[5, 2] = 1
        return eight
    
    
    def digit_nine(self):
        nine= np.zeros(self.size)
        nine[1:-1, 1] = 1
        nine[1:-1, 3] = 1
        nine[1, 2] = 1
        nine[3, 2] = 1
        nine[5, 2] = 1
        nine[4, 1] = 0
        return nine
    
    
    def print_digit(self, digit_name):
        print('Digit name: ', digit_name)
        if digit_name == 'zero':
            print(self.digit_zero())
        elif digit_name == 'one':
            print(self.digit_one())
        elif digit_name == 'two':
            print(self.digit_two())    
        elif digit_name == 'three':
            print(self.digit_three())
        elif digit_name == 'four':
            print(self.digit_four())
        elif digit_name == 'five':
            print(self.digit_five())
        elif digit_name == 'six':
            print(self.digit_six())
        elif digit_name == 'seven':
            print(self.digit_seven())
        elif digit_name == 'eight':
            print(self.digit_eight())
        elif digit_name == 'nine':
            print(self.digit_nine())
    
    
    @staticmethod
    def _show_digit(digit, digit_name):
        fig = plt.figure()
        plt.imshow(digit)
        plt.title(digit_name)
        plt.show()
        
        
    def show_digit(self, digit_name):
        if digit_name == 'zero':
            digit = self.digit_zero()
        elif digit_name == 'one':
            digit = self.digit_one()
        elif digit_name == 'two':
            digit = self.digit_two()   
        elif digit_name == 'three':
            digit = self.digit_three()
        elif digit_name == 'four':
            digit = self.digit_four()
        elif digit_name == 'five':
            digit = self.digit_five()
        elif digit_name == 'six':
            digit = self.digit_six()
        elif digit_name == 'seven':
            digit = self.digit_seven()
        elif digit_name == 'eight':
            digit = self.digit_eight()
        elif digit_name == 'nine':
            digit = self.digit_nine()
            
        self._show_digit(digit, digit_name)
        
        
        
# %%
if __name__ == '__main__':
    self = DigitMatrix()
    digit_names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    for digit_name in digit_names:
        self.print_digit(digit_name)
        self.show_digit(digit_name)