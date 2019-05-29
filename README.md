
# Extracting Forest Areas in Google Maps Aerial Images
Kirstin Roster


This project aims to identify forest areas in Google Maps aerial images. After preliminary noise reduction, this will be achieved through (i) segmenting the images into separate regions and (ii) classifying tree-covered areas. The output will be the satellite images overlayed with labels of forest areas. In addition to color and texture analysis, I will compute several vegetation indices to help identify areas with forest cover, including the Color Index of Vegetation (CIVE), the Woebbecke index (WI), the Excess Green index (ExG) and the Excess Green minus Excess Red index (ExGR). Different methods of segmentation will be tested, including thresholding and clustering. After extracting forest areas, additional analysis may include measuring forest coverage from the pixel share of identified forest areas or overlaying the forest maps with geo-coded economic data, such as industry locations.

The images are sourced from the Google Maps static maps API, centered at randomly sampled locations within Sao Paulo state in Brazil. Collected images have size 512x512 pixels and zoom level 14 (see sample [images](./images) and [script](./image collection.py).  Details on the Google Maps API are available [here](https://developers.google.com/maps/documentation/maps-static/intro).


**Limitations:**
* The approach does not distinguish between different types of tree-covered areas. For example, primary forest and soybean plantations differ not only by their tree density and tree diversity, but also by their ecological and economic impacts on our planet and society. Distinction between these types of forest areas is beyond the scope of this project. The following study serves as an example of how such a differentiation may be achieved:
T.W. Crowther et al. (2015). Mapping Tree Density at a Global Scale. *Nature* (525), 201–205
* Meant for illustrative purposes, this project makes use of current satellite imagery, not historical data, which restricts monitoring of forest area to present-day stocks and does not allow following flows over time.  


**Context:**
Global forest coverage has declined continuously [since at least 1992](https://data.worldbank.org/indicator/ag.lnd.frst.zs) due to, for example, economic exploration of the rainforest. Reducing deforestation is an important contributor to climate change mitigation, the first step of which is effective monitoring of forest areas. The [value of monitoring forest areas](https://globalforestatlas.yale.edu/conservation/forest-monitoring) has been demonstrated in many contexts and various organizations invest in research projects to better map tree cover. For example, [Global Forest Watch](https://www.globalforestwatch.org/map) publicly share their global map of forest cover including tree gain and loss over time. Analysis based on this map can be used to detect and respond to illegal mining and deforestation activities, report on forest fires, or assess environmental risks of businesses.
This project aims to contribute to monitoring efforts by demonstrating how forest areas can be identified using satellite imagery in an example region. Later implementations can be expanded globally and historically to monitor changes over time. 
