# Estimation of Above Ground Biomass (AGB) of Winter Wheat 

## Field trip to Demmin
To monitor and analyse crop over time, each year, some students from the EAGLE program go to the experimental fields in Demmin for a field trip as part of the MB3 course “From field measurements to Geoinformation”. This year’s field trip took place from 29.05.2023 to 03.06.2023. The students took many different measurements and samples, including hyperspectral measurements as well as plant height and density, chlorophyll content of the leaves and soil moisture. For each site (SSU), a few (5) plants were then harvested and their biomass and leaf area index measured in the laboratory. Beside the vegetation and soil measurements, they also took drone images using different sensors.

## Aim
This project aims to estimate the above ground biomass in two winter wheat fields using the in-situ data to train a model and additional Sentinel-2 and drone imagery. For both estimations, two models are created and trained (linear model and random forest).  The results of the estimation will be compared based on their model’s performance. The analysis is done in R with preprocessing in QGIS.
You can find the commented scripts in this repository.

## Study Area
![StudyArea](https://github.com/IsasGithub/MB3_Fieldwork_AGB_Estimation/blob/main/figs/StudyArea.PNG)
The observed fields belong to Demmin, a town in northern Germany (Western-Pomerania), where they grow winter wheat at that time of year. Each field consists of 6 ESUs with each 9 SSUs. The arrangement of these is analogous to the Sentinel and Landsat grid (10m und 30m). For this project, however, we will only use the ESUs 1 & 2 of each field.



***Table 1** Sentinel-2A scenes downloaded from the ESA Copernicus Open Access Hub and used for the land cover change analysis.*
| Image Nr. | Acquisition Date | Processing Level | Image ID                                                     |
| --------- | ---------------- | ---------------- | ------------------------------------------------------------ |
| 1         | 2019-10-27       | MSIL2A           | S2A_MSIL2A_20191027T141051_N0213_R110_T20KQB_20191027T163011 |
| 2         | 2022-12-20       | MSIL2A           | S2A_MSIL2A_20221210T140711_N0509_R110_T20KQB_20221210T200157 |

## R Scripts
### SentinelDataDownload
Script to enable the direct download of Sentinel-2 scenes for December 2022 from the ESA Copernicus Open Access Hub. The data for October 2019 is no longer available for direct download from Copernicus and therefore provided in the folder 'SentinelData'. The images for December 2022 are saved in this folder as well, and are advised to be used for the further analysis to ensure a complication-free run of the script 'SupervisedClassification.R'.

### SupervisedClassification
This script was used to perform a supervised classification of the two Sentinel-2 scenes using a trained random forest model as well as a change detection. The satellite imagery used for the classification is located in the folder 'SentinelData/' and can be loaded directly into the script. Moreover, it was not possible to used the packages 'raster' and 'RStoolbox' to create training areas directly within RStudio. The reasons for this is, that 'RStoolbox' relies on some functions of the 'raster' package which are depracated and do not work properly with the newest version of R. Therefore, the training samples were previously created in QGIS, stored in the folder 'TrainingData' and can be loaded directly into the script. There are only two classes, namely agriculture and natural vegetation, because the main objective was to map the expansion of cropland in the study region.

### DeforestationAnalysis
Besides the actual analysis, I found an interesting data set containing various information about the percentage of global forest area and net forest conversion, as well as causes of deforestation in Brazil. I used it to practice my data visualization skills using ggplot2 and included it as a nice addition of information. Coincidentally, the data set also stored infromation about the global use of soybeans, which was filtered for Paraguay and visualizes the earlier mentioned sharp increase in soybean plantations.

## Results
First, a false color images was used for a quick visualization of the land cover change. The methodical approach is a combination of Image Differencing and Multi Temporal Stacking using NDVI. This way the change is coded by color and the color depends on NDVI intensity in both time stamps (Figure 1).

| ![falseColorNDVI](https://user-images.githubusercontent.com/116877154/232325367-0825ddb0-d269-4f53-b3b8-cc790ff1133e.png) |
|:--:|
| ***Figure1** False color image used for a first visualization of the land cover change in the study region. Reddish colors indicate vegetation loss, yellowish colors vegetation gain. Blue refers to no change.* 

Next, a trained random forest model was used to classify agriculture as well as natural vegetation in the two Sentinel-2 scenes. Performance analysis for both classifications yielded and overall accuracy of 100%. The two classification results can be seen in the Figure below:

| ![LC_19-22](https://user-images.githubusercontent.com/116877154/232326113-5b348110-c2b3-4257-a798-b8e3ac44290f.png) |
|:--:|
| ***Figure 2** Land cover classification results for the study region in Paraguay for October 2019 and December 2022.* |

For further visualization the area occupied by each land cover class was plotted for the two years using a pie chart:

| ![pieChart](https://user-images.githubusercontent.com/116877154/232326333-eb74b313-3dbb-46d0-b622-46d624d14f40.png) |
|:--:|
| ***Figure 3** Area of each land cover class in ha for the years 2019 and 2022.* |

The last step was to map and quatify the land cover change in the study region between the two time stamps. The results of the analysis suggest, that the study region experienced significant expansion of cropland and therefore a loss of natural vegetation (Figure 4). Based on the change detection, the amount of natural vegetation area lost to agriculture was calculated to be around 115 km2.

| ![lcChange](https://user-images.githubusercontent.com/116877154/232326656-653f3e0c-21dc-4143-9e3b-8ce9574f1757.png) |
|:--:|
| ***Figure 4** Map of the land cover change in the study region, which directly refers to increases in cultivated land (red).* |
