#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 01:45:08 2022

@author: RK
"""

#%%
import numpy as np
from PIL import Image, ImageOps
from skimage import io, transform

import torch
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader

# from torchvision.io import read_image


#%%
class CreateDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        image = np.load(self.image_paths[idx])
        image = np.kron(image, np.ones((4, 4)))
        
        # image = Image.open(self.image_paths[idx]).convert('RGB')
        # image = image.resize((100, 100))
        # image = ImageOps.grayscale(image)
        # image = np.array(image)
        # image = image.astype(float)
        # image = image/255.0
        
        image = np.expand_dims(image, axis=2)
        
        label = self.labels[idx]
        
        sample = {'image': image, 'label': label}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
        # if self.transform:
        #     image = self.transform(image)
        #     image = image.cuda()
        
        # return (image, label)
    
#%%
class ToTensor(object):
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = image.transpose((2, 0, 1))
        transformed_sample = {'image': torch.from_numpy(image),
                              'label': torch.tensor(label)}
        return transformed_sample
    
    
class ToTensor_(object):
    def __call__(self, image):
        image = image.transpose((2, 0, 1))
        transformed_image = torch.from_numpy(image)
        return transformed_image
    
    
    
