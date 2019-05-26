# -*- coding: utf-8 -*-
"""
Extracting forest areas from Google Maps aerial images

Final Project
Course: SCC5830 - Image Processing
2019/01

Collect sample images using the Google Maps API

"""
import numpy as np
import pandas as pd
import imageio
import matplotlib.pyplot as plt
import requests



api_key = open('APIkey.txt').readline()

url = "https://maps.googleapis.com/maps/api/staticmap?"

maptype = 'satellite'
zoom = 14  # zoom level (0-21+)
size = '512x512' 

# define an (arbitrary) square inside Sao Paulo state 
topleft = [-20.8166393,-49.4523966] #(Sao Jose do Rio Preto)
topright = [-20.5933745,-47.564257] #(Franca)
bottomleft = [-23.0999162,-48.9578333] #(Avare)

# generate random lat/long coordinates within square
np.random.seed(12)
lats = np.random.uniform(topleft[0], bottomleft[0], size=10)
longs = np.random.uniform(topleft[1], topright[1], size=10)
locations = zip(lats, longs)

# collect and save the images
for i in locations:
    # api request
    img = requests.get(url + "center=" + str(i[0]) + ", " + str(i[1]) + 
                       "&zoom=" + str(zoom) + "&maptype" + maptype + 
                       "&size=" size +"&key=" + str(api_key)) 
    
    # write to file 
    f = open("lat" + str(i[0]) + "_long " + str(i[1])+'_map.png', 'wb') 
    f.write(img.content) 
    f.close()



    