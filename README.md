# Estimation of Above Ground Biomass (AGB) of Winter Wheat 

## Field trip to Demmin
To monitor and analyse crop over time, each year, some students from the EAGLE program go to the experimental fields in Demmin for a field trip as part of the MB3 course “From field measurements to Geoinformation”. This year’s field trip took place from 29.05.2023 to 03.06.2023. The students took many different measurements and samples, including hyperspectral measurements as well as plant height and density, chlorophyll content of the leaves and soil moisture. For each site (SSU), a few (5) plants were then harvested and their biomass and leaf area index measured in the laboratory. Beside the vegetation and soil measurements, they also took drone images using different sensors.

## Aim
This project aims to estimate the above ground biomass in two winter wheat fields using the in-situ data to train a model and additional Sentinel-2 and drone imagery. For both estimations, two models are created and trained (linear model and random forest).  The results of the estimation will be compared based on their model’s performance. The analysis is done in R with preprocessing in QGIS.

## Study Area
![StudyArea](https://github.com/IsasGithub/MB3_Fieldwork_AGB_Estimation/blob/main/figs/StudyArea.PNG)
***Figure 1:** Study area with fields and SSUs.* 

The observed fields belong to Demmin, a town in northern Germany (Western-Pomerania), where they grow winter wheat at that time of year. Each field consists of 6 ESUs with each 9 SSUs. The arrangement of these is analogous to the Sentinel and Landsat grid (10m und 30m). For this project, however, we will only use the ESUs 1 & 2 of each field.

## Data
* In-situ measurements:
  * 30.05.2023 - 01.06.2023
* Sentinel-2 image:
  * 02.06.2023 
* Drone imagery:
  * 30.05.2023 - 01.06.2023
  * M600 Mica Sense Dual System
  * Since the drone images are too large, please download them from my google drive folder and save them in your data folder: https://drive.google.com/drive/folders/1fvr-hXuy83VxH4pg9DGirB_3QvS9eb1D?usp=sharing

## Methodology 
![Workflow](https://github.com/IsasGithub/MB3_Fieldwork_AGB_Estimation/blob/main/figs/workflow.png)
***Figure 2:** Simplified workflow.* 

### Data preprocessing in QGIS
* Create area of interest which contains both fields that we want to observe based on the satellite image of that point of time.
* Create a new csv-table with only the relevant in-situ measurements (only the weights of the sample-plants from the laboratory) and add this ground truth information to the shapefile with the point location and names, the SSUs. 
* Since we only weighted the bowl, the wet and later, after 24 hours in the dry oven, the dry plants in the bowl, we need to subtract the weight of the bowl off the weight of the wet weights in order to get the AGB information. This is done in the field calculator by creating a new column “AGB”. 
* Drone preprocessing: reprojecting the orthomosaiks.
* Those can then be imported to R studio.

### Preprocessing in R
The following is performed for both satellite and drone imagery. You can find the commented scripts in this repository.

#### Load data
In the beginning, the required data (aoi, samples with ground truth information and raster image) are imported into the script, and the raster is masked to the area of interest, our fields.

#### Calculate NDVI and create buffer
A common practice in remote sensing to estimate the AGB is using a Sentinel-2 (or drone raster) derived vegetation index, the NDVI, as a predictor. Therefore, we need to calculate this index and convert it to a raster layer. Then, a 10 m buffer around the sample points will be created in order to extract later the mean values of the pixels.

#### Zonal statistics
For this buffered points we extract the mean NDVI from the newly created ndvi_raster, which we combine through a adding it as a new column with our ground truth AGB values from the `samples` data frame. Now we can plot the linear relationship in a scatterplot style between the ndvi values and AGB values. It can be seen, that there is a (slight) relationship between them. It is especially visible in the Sentinel plot, because it has more samples and therefore, it is easier to identify a relation.

| <img src="https://github.com/IsasGithub/MB3_Fieldwork_AGB_Estimation/blob/main/figs/scatterplot_sat.jpeg" alt="scatterplot_sat" width="300"/> | <img src="https://github.com/IsasGithub/MB3_Fieldwork_AGB_Estimation/blob/main/figs/scatterplot_dr1.jpeg" alt="scatterplot_dr1" width="300"/> |<img src="https://github.com/IsasGithub/MB3_Fieldwork_AGB_Estimation/blob/main/figs/scatterplot_dr2.jpeg" alt="scatterplot_dr2" width="300"/> |
| -- | --- | --- |
| Linear Relationship Sentinel | Linear Relationship Drone 1 | Linear Relationship Drone 2 |

***Figure 3:** Linear relationship between NDVI and AGB values for Sentinel and drone images.* 

### AGB estimatiom
#### Dividing data set in training & testing
Splitting the dataset into training and testing subsets helps ensure the model's performance on new data. The training set is used to train the model. The model learns the relationships between the predictor (NDVI) and the target (AGB) variables. The testing set is used to evaluate the model's performance on new, unseen data. This helps assess how well the model generalises to data it hasn't seen during training. 70% of the SSU points (with ground truth and NDVI values) are randomly selected for training, and the remaining 30% are assigned to testing.


In this exercise we will investigate the relation of vegetation indices to above-ground biomass values over study area. We will perform multi-linear and random forest machine learning regression models in order to predict the AGB values based on vegetation indices as predictors.


## Results

## Conclusion
The field trip gave a good insight into common agricultural field work. We learned about the "Demmin" project in general and about the measurements needed for it. Since, at least most of us, had very little experience with fieldwork, it was very exciting for us to explore how all these measurements work and to get outside, away from the computer, to get the data we need for our analysis ourselves. The analysis of this project has demonstrated the building of two different models to estimate AGB of winter wheat using additional Sentinel-2 and drone data, which can now be applied to similar locations or different dates.
