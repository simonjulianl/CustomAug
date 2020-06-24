#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 20:12:47 2020

@author: simon
"""

import torch
import math

class hide_and_seek(torch.nn.Module):
    '''hide and seek data augmentation based on https://arxiv.org/abs/1704.04232
    S X S : total number of patches, 
    p : probability of hiding
    mean : value of pixel that is masked ( default is normalized, hence mean = 0 )'''
    def __init__(self, S = 4, p =0.5, mean = 0):
        super().__init__()
        self.S = S
        self.p = p
        self.mean = mean # mean of all RGB Vector of the dataset
    
    def forward(self,img):
        return self.patch_image(img)
        
    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
    
    def patch_image(self,img):
        if len(img.shape) == 3 : #RGB for C X H X W ( torch format )
            H,W = img.shape[1:]
        else : # BLACK_WHITE
            H,W = img.shape
            
        H_patch = math.floor(H/self.S)
        W_patch = math.floor(W/self.S)
        
        H_patch_list = torch.arange(0, H, H_patch)
        W_patch_list = torch.arange(0, W, W_patch)
        
        assert len(H_patch_list) == len(W_patch_list) # S X S
        # cartesian product of all possible lists coordinates
        for i in range(self.S): # loop over H
            for j in range(self.S): # loop over W
                cover = torch.rand(1).item() <= self.p # less than 0.5 chance
        
                if cover : 
                    if len(img.shape) == 3 : #RGB for C X H X W ( torch format )
                        img[:,
                            i * H_patch : (None if (i + 1) == self.S else (i+1) * H_patch),
                            j * W_patch : (None if (j + 1) == self.S else (j+1) * W_patch)] = self.mean
                    else : # BLACK_WHITE
                        img[i * H_patch : (None if (i + 1) == self.S else (i+1) * H_patch),
                            j * W_patch : (None if (j + 1) == self.S else (j+1) * W_patch)] = self.mean
                        
        return img
