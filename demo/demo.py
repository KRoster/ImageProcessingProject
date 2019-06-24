#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image processing project
Demo


"""
import numpy as np
import imageio
import matplotlib.pyplot as plt

# color processing fcts:
from helper_functions import generate_features, kmeans
from helper_functions import cosine_classifier, RGB_classifier
from helper_functions import outline_overlay


def full_pipeline_all(img_name, haralick_name):
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
    
# name of example input image
img_example = 'lat-22.200495034324955_long -48.69176482972345_map.png'
haralick_example = 'haralick3.npy'

# read image
img = imageio.imread(img_example)


# apply forest area extraction
out = full_pipeline_all(img_example, haralick_example)
out_example_raw = out[0]
out_example_std = out[1]
out_example_rgb = out[2]
out_example_clusters = out[3]

# Plot results
plt.figure(figsize=(15,10))
plt.subplot(231); plt.imshow(img); plt.title('Image')
plt.subplot(232); plt.imshow(out_example_clusters); plt.title('Clusters')
plt.subplot(234); plt.imshow(out_example_std); plt.title('Std')
plt.subplot(235); plt.imshow(out_example_raw); plt.title('Raw')
plt.subplot(236); plt.imshow(out_example_rgb); plt.title('RGB')

    


