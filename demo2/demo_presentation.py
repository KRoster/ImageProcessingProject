#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image processing project
Demo


"""
import numpy as np
import imageio
import matplotlib.pyplot as plt
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


def full_pipeline_rgb(img_name, haralick_name):
    '''
    Applies feature engineering, segmentation, classification and outline
    methods to extract forest areas.
    Takes as input an image and its haralick features (which are computed 
    separately).
    Returns a list of 4 arrays: the clusters and three different 
    classification results of the clusters (std, raw, rgb).
    '''
    # read image
    img = imageio.imread(img_name)
    # remove google logo
    img = img[0:490,:,0:3]
    # generate features
    img_features = generate_features(img, haralick_name)
    # standardize features
    std_img_features = (img_features - np.mean(img_features, axis=0))/np.std(img_features, axis=0)
    # clustering
    clusters = kmeans(std_img_features, k=4, S=12, n_iterations=100).reshape((img.shape[0], img.shape[1]))
    # forest prototype
    forest_avg_std = np.array([-0.61, 0.24, 0.61, 0.62,
                           -0.87, -0.9, -0.86, 0.62, 0.52, -0.93, 
                           0.35, -0.11, 0.05, 0.4, -0.47])
    forest_avg_raw = np.array([18.74, -0.06, 0.06, 0.04, 
                               0.17, 0.21, 0.23, 0.5, 0.41, 0.25, 
                               0.05, 75.95, 0.43, 0.33, 4.81])
    # classify
    #final_seg_raw = cosine_classifier(img_features, clusters, forest_avg_raw, cosine_threshold=0.994)
    #final_seg_std = cosine_classifier(std_img_features, clusters, forest_avg_std, cosine_threshold=0.32)
    final_seg_rgb = RGB_classifier(img_features, clusters)
    # cleaning and outline
    #final_raw = outline_overlay(final_seg_raw, img)
    #final_std = outline_overlay(final_seg_std, img)
    final_rgb = outline_overlay(final_seg_rgb, img)

    return([final_rgb, clusters])


# name of example input images
#img1_example = 'image1.png'
#haralick1_example = 'haralick1.npy'
img2_example = 'image2.png'
haralick2_example = 'haralick2.npy'
img3_example = 'image3.png'
haralick3_example = 'haralick3.npy'
#img4_example = 'image4.png'
#haralick4_example = 'haralick4.npy'


# apply forest area extraction
#out1 = full_pipeline_rgb(img1_example, haralick1_example)
out2 = full_pipeline_rgb(img2_example, haralick2_example)
out3 = full_pipeline_rgb(img3_example, haralick3_example)
#out4 = full_pipeline_rgb(img4_example, haralick4_example)


# Plot results
plt.figure(figsize=(10,5))
plt.axis('off')
plt.subplot(121); plt.imshow(out2[0]); plt.title('Image 2')
plt.subplot(122); plt.imshow(out3[0]); plt.title('Image 3')
plt.show()


    


