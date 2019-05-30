#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper functions 

"""
import numpy as np

#  normalize RGB before segmentation
def RGB_normalizer(img):
    '''
    Normalizes RGB color channels to 0-1 range.
    Takes as input an RGB image and normalizes each channel to range 0-1.
    Returns stacked array with three arrays of normalized R, G, and B
    '''
    Red = img[:,:,0].astype('float')
    Green = img[:,:,1].astype('float')
    Blue = img[:,:,2].astype('float')
    # normalize to range 0-1
    Rnorm = (Red - Red.min())/(Red.max() - Red.min())
    Gnorm = (Green - Green.min())/(Green.max() - Green.min())
    Bnorm = (Blue - Blue.min())/(Blue.max() - Blue.min())
    
    img_out = np.dstack((Rnorm, Gnorm, Bnorm))
    
    return(img_out)



# compute chromatic coordinates
def chromatic_coords(img, normalized=True):
    '''
    Converts RGB to chromatic coordinates r, g, and b.
    Takes as input either raw RGB or normalized RGB image 
    and boolean normalization status.
    Returns 3d stacked array of r, g, and b.
    '''
    if normalized == False:
        img_norm = RGB_normalizer(img)
    elif normalized == True:
        img_norm = img.copy()
    # to avoid zero division
    img_norm = np.where(img_norm==0, 0.000001, img_norm)
    # separate the channels
    Rnorm = img_norm[:,:,0]
    Gnorm = img_norm[:,:,1]
    Bnorm = img_norm[:,:,2]
    # compute chromatic coords
    r =  Rnorm / (Rnorm + Gnorm + Bnorm)
    g = Gnorm / (Rnorm + Gnorm + Bnorm)
    b = Bnorm / (Rnorm + Gnorm + Bnorm)
    
    img_out = np.dstack((r, g, b))
    return(img_out)



# compute vegetation index
def veg_ind(img, ind_type, colorsch):
    '''
    Computes selected vegetation index (pixel-wise).
    Takes as input an image matrix, 
    the type of vegetation index to be computed ('cive', 'wi', 'exg', 'exgr'), 
    and the color scheme (options are 'rgb' and 'chromatic').
    Returns the 2d vegetation index matrix. 
    '''
    if colorsch == 'rgb':
        img_chrom = chromatic_coords(img, normalized=False)
    elif colorsch == 'chromatic':
        img_chrom = img.copy()
    # separate chromatic channels:
    r = img_chrom[:,:,0]
    g = img_chrom[:,:,1]
    b = img_chrom[:,:,2]
    
    # compute index:
    if ind_type == 'cive':
        ## CIVE (Color Index of Vegetation)
        # CIVE =  0.441r - 0.881g + 0.385b + 18.78745
        img_out = 0.441*r - 0.881*g + 0.385*b + 18.78745
    
    elif ind_type == 'wi':
      ### Woebbecke index
      # WI = (g - b) / |r - g|
      # note: values -0.1<x<0.1 are set to 0.1
      denom = abs(r - g)
      denom = np.where(denom<0.1, 0.1, denom)
      img_out = (g - b) / denom    
    
    elif ind_type == 'exg':
        ### ExG (Excess Green)
        # ExG = 2g - r - b
        img_out = 2*g - r - b  
    
    elif ind_type == 'wi':
        ### ExG - ExR (Excess Green minus Excess Red)
        # ExGR - ExG - 1.4r - g
        img_out = (2*g - r - b) - (1.4*r - g)
    
    return(img_out)
    


# simple segmentation threshold finder for 2 classes:
def threshold_finder(img_in, T0, delta_T):
    '''
    Takes as input the image to be segmented, an initial threshold,
    and minimum difference cutoff value.
    Returns a new threshold.
    '''
    T_diff = delta_T+1
    while T_diff > delta_T:
        i1 = np.where(img_in<T0,
                      1,np.nan)*img_in
        i2 = np.where(img_in>T0,
                      1,np.nan)*img_in
        m1 = np.nanmean(i1)
        m2 = np.nanmean(i2)
        T = (1/2) * (m1 + m2)
        T_diff = abs(T0-T)
        T0=T
        
    return T0    

