
# Extracting Forest Areas in Google Maps Aerial Images
**Kirstin Roster - NUSP: 11375819**


This project aims to identify forest areas in Google Maps aerial images, by (i) segmenting the images into separate regions and (ii) classifying tree-covered areas. The output consists of the original Google maps images overlayed with outlines of forest areas. I use features related to color, texture, and vegetation of pixels and their neighbors, including the Color Index of Vegetation (CIVE), the Woebbecke index (WI), the Excess Green index (ExG) and the Excess Green minus Excess Red index (ExGR) as well as Haralick descriptors. K-means clustering successfully separates forest areas. Cosine similarity and RGB classification are employed to classify clusters consisting mostly of forest areas. Finally, mathematical morphology techniques help clean the segmentation and generate an outline. The results can be used for additional analysis such as measuring the share of forest coverage.  

The images are sourced from the Google Maps static maps API, centered at randomly sampled locations within Sao Paulo state in Brazil. Collected images have size 512x512 pixels and zoom level 14 (see sample [images](https://github.com/KRoster/ImageProcessingProject/tree/master/Sample%20images) and [script](https://github.com/KRoster/ImageProcessingProject/blob/master/image%20collection.py)).  Details on the Google Maps API are available [here](https://developers.google.com/maps/documentation/maps-static/intro).


**Limitations:**
* The approach does not distinguish between different types of tree-covered areas. For example, primary forest and soybean plantations differ not only by their tree density and tree diversity, but also by their ecological and economic impacts on our planet and society. Distinction between these types of forest areas is beyond the scope of this project. [This study](https://www.nature.com/articles/nature14967) serves as an example of how such a differentiation may be achieved.
* Meant for illustrative purposes, this project makes use of current satellite imagery, not historical data, which restricts monitoring of forest area to present-day stocks and does not allow following flows over time.  


**Context:**

Global forest coverage has declined continuously [since at least 1992](https://data.worldbank.org/indicator/ag.lnd.frst.zs) due to, for example, economic exploration of the rainforest. Reducing deforestation is an important contributor to climate change mitigation, the first step of which is effective monitoring of forest areas. The [value of monitoring forest areas](https://globalforestatlas.yale.edu/conservation/forest-monitoring) has been demonstrated in many contexts and various organizations invest in research projects to better map tree cover. For example, [Global Forest Watch](https://www.globalforestwatch.org/map) publicly share their global map of forest cover including tree gain and loss over time. Analysis based on this map can be used to detect and respond to illegal mining and deforestation activities, report on forest fires, or assess environmental risks of businesses.

This image processing project aims to exemplify how forest areas can be monitored using publicly available satellite data. Later implementations can be expanded globally and historically to monitor changes over time. 


