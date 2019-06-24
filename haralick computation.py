#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image processing project
Extracting Forest Areas from Google Maps Aerial Images

Haralick Features Computation

"""

from os import listdir
from mahotas.features import haralick
from helper_functions import attributes_lum

    

# sliding window to compute haralick descriptors per pixel
# note: uses haralick descriptors from mahotas library for efficiency
def haralick_window_slider(gray, window_step=3):    
    '''
    Computes average haralick features over a window centered around each 
    pixel of an image; default window size is set to 7x7.
    Takes as input a gray-level image and step size.
    Returns array 5 times the size of the input image 
    (1 dimension per haralick feature).
    '''
    # note: default sliding window (7x7) (Giannini, Merola, and Allegrini, 2012)
    # padding the array by reflection of neighbor pixels
    gray_pad = np.pad(gray, pad_width=window_step, mode='reflect').astype(int)
    # output matrix
    haralick_pixels = np.zeros((gray.shape[0],gray.shape[1],5))
    # move window and compute haralick descriptors
    for x in range(gray.shape[0]):
        for y in range(gray.shape[1]):
            window = gray_pad[ x:(x+2*window_step+1), y:(y+2*window_step+1) ]
            haralick_window = haralick(window)
            haralick_pixels[x,y,:] = np.sum(haralick_window[:,[0,1,2,4,8]], 
                           axis = 0) / 4
    
    return(haralick_pixels)
    


images = listdir()

for i in images:
    # read image
    img = imageio.imread(i)
    # cut logo
    img = img[0:490,:]
    # convert to gray-level using luminance
    gray = attributes_lum(img).reshape((img.shape[0],img.shape[1]))
    # compute haralick features
    haralick_img = haralick_window_slider(gray)
    # write to file
    np.save(i[0:-4]+'_haralick.npy', haralick_img)



