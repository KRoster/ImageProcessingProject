#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image processing project

Full Pipeline
"""

import numpy as np
import imageio
import matplotlib.pyplot as plt

# color processing fcts:
from helper_functions import generate_features, kmeans
from helper_functions import cosine_classifier, RGB_classifier
from helper_functions import outline_overlay

'''
The following function implements the forest area extraction pipeline.
All the steps are described in the Final Report notebook.
'''
def full_pipeline(img_name, haralick_name):
    '''
    Applies feature engineering, segmentation, classification and outline
    methods to extract forest areas.
    Takes as input an image and its haralick features (which are computed
    separately).
    Returns a list of 4 arrays: the clusters and three different
    classification results of the clusters (std, raw, rgb).
    '''
    # read image
    img = imageio.imread('images/'+img_name)
    # remove google logo
    img = img[0:490,:,0:3]
    # generate features
    img_features = generate_features(img, haralick_name)
    # standardize features
    std_img_features = (img_features - np.mean(img_features, axis=0))/np.std(img_features, axis=0)
    # clustering
    clusters = kmeans(std_img_features, k=4, S=12, n_iterations=100).reshape((img1.shape[0], img1.shape[1]))
    # classify
    forest_avg_std = np.array([-0.61, 0.24, 0.61, 0.62, 
                           -0.87, -0.9, -0.86, 0.62, 0.52, -0.93, 
                           0.35, -0.11, 0.05, 0.4, -0.47])
    forest_avg_raw = np.array([18.74, -0.06, 0.06, 0.04, 
                               0.17, 0.21, 0.23, 0.5, 0.41, 0.25, 
                               0.05, 75.95, 0.43, 0.33, 4.81])
    final_seg_raw = cosine_classifier(img_features, clusters, forest_avg_raw, cosine_threshold=0.994)
    final_seg_std = cosine_classifier(std_img_features, clusters, forest_avg_std, cosine_threshold=0.32)
    final_seg_rgb = RGB_classifier(img_features, clusters)
    # cleaning and outline
    final_raw = outline_overlay(final_seg_raw, img)
    final_std = outline_overlay(final_seg_std, img)
    final_rgb = outline_overlay(final_seg_rgb, img)

    return([final_raw, final_std, final_rgb, clusters])
    
# apply the pipeline steps to the dataset of input images
# available images and their pre-computed haralick features:
haralick_names = listdir('haralick')
h=haralick_names[1]
for h in haralick_names[1:57]:
    i = h[0:-13]+'.png'
    h = 'haralick/'+h
    # extraction pipeline
    result = full_pipeline(i, h)
    # select individual results
    result_std = result[0]
    result_raw = result[1]
    result_rgb = result[2]
    clusters = result[3]
    clusters = (255*(clusters - clusters.min())/(clusters.max() - clusters.min())).astype('uint8')
    # save results
    imageio.imwrite('results/'+i[0:-4]+'_result_clusters.png', clusters)
    imageio.imwrite('results/'+i[0:-4]+'_result_std.png', result_std)
    imageio.imwrite('results/'+i[0:-4]+'_result_raw.png', result_raw)
    imageio.imwrite('results/'+i[0:-4]+'_result_rgb.png', result_rgb)
    plt.imshow(clusters)
    plt.savefig('results/'+i[0:-4]+'_result_clusters_color.png')

    
    
