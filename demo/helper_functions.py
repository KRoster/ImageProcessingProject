#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper functions 

"""
import numpy as np
import random
from mahotas.features import haralick
from scipy import ndimage
from matplotlib.colors import rgb_to_hsv
from sklearn.metrics.pairwise import cosine_similarity



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
def veg_ind(img, colorsch):
    '''
    Computes four vegetation indices (pixel-wise).
    Takes as input an image matrix and the color scheme 
    (options are 'rgb' and 'chromatic').
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
    
    img_out = np.zeros((img.shape[0],img.shape[1],4))
    # compute index:
    ## CIVE (Color Index of Vegetation)
    # CIVE =  0.441r - 0.881g + 0.385b + 18.78745
    img_out[:,:,0] = 0.441*r - 0.881*g + 0.385*b + 18.78745
    
    ### Woebbecke index
    # WI = (g - b) / |r - g|
    # note: values -0.1<x<0.1 are set to 0.1
    denom = abs(r - g)
    denom = np.where(denom<0.1, 0.1, denom)
    img_out[:,:,1] = (g - b) / denom    
    
    ### ExG (Excess Green)
    # ExG = 2g - r - b
    img_out[:,:,2] = 2*g - r - b  

    ### ExG - ExR (Excess Green minus Excess Red)
    # ExGR - ExG - 1.4r - g
    img_out[:,:,3] = (2*g - r - b) - (1.4*r - g)
    
    return(img_out)

    
def RGB_seg(img, normalized = True):
    '''
    Segments image by RGB thresholds.
    Takes as input RGB image and boolean indicating if RGB values are raw 
    or normalized.
    Returns segmented image, where forest areas = 1, other areas = 0.
    '''
    # normalize RGB channels if necessary:
    if normalized==False:
        img_norm = RGB_normalizer(img)
    elif normalized==True:
        img_norm = img.copy()
    # apply threshold:
    img_rgb_trees = np.where(((img_norm[:,:,0]<0.16) & (img_norm[:,:,0]>0.03) &
                         (img_norm[:,:,1]>0.11) & (img_norm[:,:,1]<0.22)
                         & (img_norm[:,:,2]>0.13) & (img_norm[:,:,2]<0.24)),
                         1,0)
    return(img_rgb_trees)

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



# kmeans clustering algorithm
def kmeans(img, k, S, n_iterations):
    '''
    Assigns each pixel to one of k clusters.
    Takes as input the image as a pixel-feature matrix plus parameters 
    k,S,n_iterations.
    Returns vector of cluster allocations.
    '''
    i=1
    # initialize the k cluster centroids:
    random.seed(S)
    ids = np.sort(random.sample(range(0, img.shape[0]), k))
    # get starting centroid values
    cent = np.take(img, ids, axis=0)
    while i<=n_iterations:
        # compute distances
        dist = np.sqrt(((img - cent[:,np.newaxis])**2).sum(axis=2))
        clusters = np.argmin(dist, axis=0)
        # recompute the centroids
        for j in np.unique(clusters):
            cent[j,:] = np.mean(np.take(img, np.where(clusters==j), axis=0), axis=1)
        i+=1
    clusters=clusters+1
    return(clusters)
    

# reshape image
def attributes_reshape(img):
    '''
    Flattens image into long format, i.e. pixel list with features as columns.
    '''
    m,n = img.shape[0], img.shape[1]
    img = img.astype('float')
    # flatten image into pixel vector:
    img_out = img.reshape((m*n,img.shape[2]))
    
    return(img_out)


# luminance
def attributes_lum(img):
    '''
    Calculates luminance feature.
    Takes as input RGB image.
    Returns vector of luminance for each pixel.
    '''
    m,n = img.shape[0], img.shape[1]
    img = img.astype('float')
    img_out = 0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]
    img_out = img_out.reshape((m*n, 1))
    
    return(img_out)


# calculate glcm 
def glcm_calculation(gray, neigh_x, neigh_y):
    '''
    Calculates Gray Level Co-occurance Matrix (GLCM), where the x-direction 
    represents the reference pixel's grey-level value and y-direction 
    represents the neighbor pixel's grey level.
    Takes as input gray-level input image, and the x- and y-direction position 
    of the neighbor pixel.
    Returns the GLCM.
    '''
    assert neigh_x>=0 and neigh_y>=0
    
    # create discrete grey-level bins by rounding input to 0 decimals:
    gray = np.around(gray).astype(int)
    # compute GLCM:
    glcm = np.zeros((255,255))
    
    for x in range(0, gray.shape[0]-neigh_x):
        for y in range(0, gray.shape[1]-neigh_y):
            glcm_x = gray[x,y]
            glcm_y = gray[x+neigh_x, y+neigh_y]
            glcm[glcm_x,glcm_y] += 1.0
    
    return(glcm)


# compute the six independent haralick descriptors
def haralick_descriptors(glcm):
    '''
    Computes six haralick descriptors over the input probability matrix:
    maximum probability, correlation, contrast, energy, homogeneity, entropy.
    Takes as input a probability matrix with values ranging from 0 to 1.
    Returns a list of the six descriptor values.
    '''
    prob = glcm/np.sum(glcm)
    # row and column means and variances    
    mr = 0
    for i in range(255):
        pij=np.sum(prob[i,:])
        mr += i*pij  
    
    mc = 0
    for j in range(255):
        pij=np.sum(prob[:,j])
        mc += j*pij 

    vr = 0
    for i in range(255):
        pij = np.sum(prob[i,:])
        vr += pij * (i - mr)**2
    vr = vr**(1/2) 
    
    vc = 0
    for j in range(255):
        pij = np.sum(prob[:,j])
        vc += pij * (j - mc)**2
    vc = vc**(1/2)
        
        
    # Descriptor 1: maximum probability
    max_prob = prob.max()
    
    # Descriptor 2: correlation    
    if vr!=0 and vc!=0:
        corr_h=0
        for i in range(255):
            for j in range(255):
                corr_h += ((i-mr)*(j-mc)*prob[i,j]) / (vr*vc)
    else: 
        corr_h = np.nan
       
    # Descriptor 3: contrast
    contrast_h = 0
    for i in range(255):
        for j in range(255):
            contrast_h += prob[i,j]*((i-j)**2)
    
    # Descriptor 4: energy (aka uniformity)
    energy_h = np.nansum(prob**2)
    
    # Descriptor 5: homogeneity
    L_x = np.array([np.arange(1, 256)] * 255)
    L_y = np.array([np.arange(1, 256)] * 255).transpose()
    denom = np.abs(L_x-L_y)
    homogeneity_h = np.sum(prob/(1+denom))
    
    # Descriptor 6: entropy
    entropy_h = 0
    for i in range(255):
        for j in range(255):
            if prob[i,j]!=0:
                entropy_h += prob[i,j]*np.log2(prob[i,j])
    entropy_h = -entropy_h
    
    
    return(np.array([max_prob, corr_h, contrast_h, energy_h, 
            homogeneity_h, entropy_h]))
    

# sliding window to compute haralick descriptors per pixel
# note: uses haralick descriptors from mahotas library (for efficiency)
def haralick_window_slider(gray, window_step=3):    
    '''
    Computes average haralick features over a 7x7 window centered around each 
    pixel of an image.
    Takes as input a gray-level image.
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
            haralick_pixels[x,y,:] = np.sum(haralick_window[:,[0,1,2,4,8]], axis = 0) / 4
    
    return(haralick_pixels)


def outline_overlay(seg, img):
    '''
    Cleans segments and overlays segment outline, performing the following
    steps: hole filling, closing, opening of closing, outline using erosion.
    Takes as input the segments and original image.
    Returns original image with outline overlayed.
    '''
    # zero-padding
    in_image = np.pad(seg, mode='reflect', pad_width=((7,7),(7,7)))
    
    # structuring elements:
    # 5x5 cross
    struc_element = np.zeros((3,3))
    struc_element[:,1]=1
    struc_element[1,:]=1
    # 7x7 cross
    struc_element7 = np.zeros((7,7))
    struc_element7[:,3]=1
    struc_element7[3,:]=1

    # fill holes
    filled = ndimage.binary_fill_holes(input = in_image, structure = struc_element)

    # opening/closing of double filled
    closing_filled = ndimage.binary_closing(input = filled, structure = struc_element)
    open_of_close_filled = ndimage.binary_opening(input = closing_filled, structure = struc_element7)

    # outline using a 7x7 cross structuring element
    outline_opcl_filled = open_of_close_filled.astype('int') - ndimage.binary_erosion(input = open_of_close_filled, structure = struc_element7).astype('int')
    # remove padding
    outline_opcl_filled = outline_opcl_filled[7:-7,7:-7]
    ## input image overlayed with outline
    out_image = img.copy()
    out_image[np.where(outline_opcl_filled==1)]=[255, 30, 30]

    return(out_image)


def generate_features(img, haralick_file):
    # Prepare data with all features
    img_features = np.zeros((img.shape[0]*img.shape[1], 4+6+5))
    # vegetation features
    img_features[:,0:4] = attributes_reshape(veg_ind(img, colorsch='rgb'))
    # color features
    img_features[:,4:7] = attributes_reshape(RGB_normalizer(img))  # RGB
    img_features[:,7:10] = attributes_reshape(rgb_to_hsv((RGB_normalizer(img))))
    # haralick descriptors
    haralick1 = np.load(haralick_file)
    haralick1_flat = np.reshape(haralick1, 
                                (haralick1.shape[0]*haralick1.shape[1], 
                                 haralick1.shape[2]))
    img_features[:,10:15] = haralick1_flat

    return(img_features)
    

def cosine_classifier(features, clusters, comparison_vector, cosine_threshold):
    # prepare final segmentation output matrix:
    final_segment = np.zeros(features.shape[0])
    for c in np.unique(clusters):
        cluster_features = features[(clusters==c).flatten(),:]
        seg_avg = np.median(cluster_features,axis=0)
        # cosine similarity between cluster and comparison vector (forest)
        similarity = cosine_similarity([seg_avg], [comparison_vector])
        # classify
        if similarity>cosine_threshold:
            final_segment[(clusters==c).flatten()] = 1

    return(final_segment.reshape(clusters.shape))
    
def RGB_classifier(features, clusters):
    # prepare final segmentation output matrix:
    final_segment = np.zeros(features.shape[0])
    for c in np.unique(clusters):
        cluster_features = features[(clusters==c).flatten(),4:7]
        seg_med = np.median(cluster_features,axis=0)
        # rgb thresholds
        if ((seg_med[0]>0.03) & (seg_med[0]<0.4) &
            (seg_med[1]>0.11) & (seg_med[1]<0.28) &
            (seg_med[2]>0.13) & (seg_med[2]<0.31)):
            final_segment[(clusters==c).flatten()] = 1

    return(final_segment.reshape(clusters.shape))