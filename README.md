# Estimation of Above Ground Biomass (AGB) of Winter Wheat 

## Field trip to Demmin
To monitor and analyse crop over time, each year, some students from the EAGLE program go to the experimental fields in Demmin for a field trip as part of the MB3 course “From field measurements to Geoinformation”. This year’s field trip took place from 29.05.2023 to 03.06.2023. The students took many different measurements and samples, including hyperspectral measurements as well as plant height and density, chlorophyll content of the leaves and soil moisture. For each site (SSU), a few (5) plants were then harvested and their biomass and leaf area index was measured in the laboratory. Beside the vegetation and soil measurements, they also took drone images using different sensors.

## Aim
This project aims to estimate the above ground biomass in two winter wheat fields using the in-situ data to train a model and additional Sentinel-2 and drone imagery. For both estimations, two models are created and trained (linear model and random forest).  The results of the estimation will be compared based on their model’s performance. The analysis is done in R with preprocessing in QGIS.

## Study Area
![StudyArea](https://github.com/IsasGithub/MB3_Fieldwork_AGB_Estimation/blob/main/figs/StudyArea.PNG)
***Figure 1:** Study area with fields and SSUs.* 

The observed fields belong to Demmin, a town in northern Germany (Western-Pomerania), where they grow winter wheat at that time of year. Each field consists of 6 ESUs with each 9 SSUs. The arrangement of these is analogous to the Sentinel and Landsat grid (10m and 30m). For this project, however, we will only use the ESUs 1 & 2 of each field.

## Data
* In-situ measurements:
  * 30.05.2023 - 01.06.2023
* Sentinel-2 image:
  * 02.06.2023
  * https://scihub.copernicus.eu/dhus/#/home
* Drone imagery:
  * 30.05.2023 - 01.06.2023
  * Only available for ESU 2 in Field 2
  * M600 Mica Sense Dual System
  * Since the drone images are too large, please download them from my google drive folder and save them in your data folder: https://drive.google.com/drive/folders/1fvr-hXuy83VxH4pg9DGirB_3QvS9eb1D?usp=sharing

## Methodology & Results
![Workflow](https://github.com/IsasGithub/MB3_Fieldwork_AGB_Estimation/blob/main/figs/workflow.png)
***Figure 2:** Simplified workflow.* 

### Data preprocessing in QGIS
* Create area of interest which contains both fields that we want to observe based on the satellite/drone image of that point of time.
* Create a new csv-table with only the relevant in-situ measurements (only the weights of the sample-plants from the laboratory) and add this ground truth information to the shapefile with the point location and names, the SSUs. 
* Since we only weighted the bowl, the wet and later, after 24 hours in the dry oven, the dry plants in the bowl, we need to subtract the weight of the bowl off the weight of the wet weights in order to get the AGB information. This is done in the field calculator by creating a new column “AGB”. 
* Drone preprocessing: reprojecting the orthomosaiks.
* Those can then be imported into R studio.

### Preprocessing in R
The following is performed for both satellite and drone imagery. To keep the page clearer, no code snippets are included in the text. Nevertheless, the documented scripts can of course also be sourced from this repository.

#### Load data
In the beginning, the required data (aoi, samples with ground truth information and raster image) are imported into the script, and the raster is masked to the area of interest, our fields.

#### Calculate NDVI and create buffer
A common practice in remote sensing to estimate the AGB is using a Sentinel-2 (or drone raster) derived vegetation index, the **NDVI**, as a predictor. Therefore, we need to calculate this index and convert it to a raster layer. Then, a 10 m buffer around the sample points will be created in order to extract later the mean values of the pixels.

#### Zonal statistics
In this part, we investigate the relation of the vegetation index to above-ground biomass values over the study area. For these **buffered points** we extract the mean NDVI from the newly created **ndvi_raster**, which we combine through adding it as a new column with our ground truth AGB values from the **samples** data frame. Now we can plot the linear relationship in a scatterplot style between the ndvi values and AGB values. It can be seen, that there is a (slight) relationship between them. It is especially visible in the Sentinel plot, because it has more samples and therefore, it is easier to identify a relation.

| <img src="https://github.com/IsasGithub/MB3_Fieldwork_AGB_Estimation/blob/main/figs/scatterplot_sat.jpeg" alt="scatterplot_sat" width="300"/> | <img src="https://github.com/IsasGithub/MB3_Fieldwork_AGB_Estimation/blob/main/figs/scatterplot_dr1.jpeg" alt="scatterplot_dr1" width="300"/> |<img src="https://github.com/IsasGithub/MB3_Fieldwork_AGB_Estimation/blob/main/figs/scatterplot_dr2.jpeg" alt="scatterplot_dr2" width="300"/> |
| -- | --- | --- |
| Linear Relationship Sentinel | Linear Relationship Drone 1 | Linear Relationship Drone 2 |

***Figure 3:** Linear relationship between NDVI and AGB values for Sentinel and drone images.* 

### AGB estimation
#### Dividing data set into training & testing
Splitting the data set into training and testing subsets helps ensure the model's performance on new data. The training set is used to train the model. The model learns the relationships between the predictor (NDVI) and the target (AGB) variables. The testing set is used to evaluate the model's performance on new, unseen data. This helps assess how well the model generalises to data it hasn't seen during training. 70% of the SSU points (with ground truth and NDVI values) are randomly selected for training, and the remaining 30% are assigned to testing.

We will now perform linear and random forest machine learning regression models in order to predict the AGB values based on the NDVI as a predictor.

#### Linear model

The linear model is created using the `lm()` function. It models the relationship between the target variable AGB and the predictor NDVI_mean using the training data. Predictions are made by using the `predict()` function on the model, providing the testing data as newdata. The results of the prediction are the estimated AGB values based on the linear model. We will then convert it into a data frame containing the columns: **true_AGB** (actual AGB values from the testing data), and **estim_AGB** (predicted AGB values from the linear regression model). 

In order to visualise the predictions properly, we reshape the predicitons data frame from a wide format to a long format using the `pivot_longer()` function.

#### Random forest

The random forest model is created using the `randomForest()` function with 700 trees to grow. We do the same steps as for the prediction with the linear model.


### Visualisation
The two data frames from both predictions will be combined into one. We generate a `geom_bar` plot to visualise the comparison between true and the two estimated AGB values.
The resulting plot shows a comparison between them for different samples - represented by short names - in alphabetical order. **Figure 4** and **figure 5** show that for both, Sentinel and drone image, the values for the predicted AGB with the linear model seem to be closer to the true AGB values. 

![agb_combined_sat](https://github.com/IsasGithub/MB3_Fieldwork_AGB_Estimation/blob/main/figs/agb_combined_sat.jpeg)

***Figure 4:** Visualisation of true and estimated AGB (Sentinel).* 


![agb_combined_drones](https://github.com/IsasGithub/MB3_Fieldwork_AGB_Estimation/blob/main/figs/agb_combined_drones.jpeg)

***Figure 5:** Visualisation of true and estimated AGB (drone).* 

### Error metrics
This assumption should be substantiated by calculating different error metrics to assess the model's accuracy. There are three, which are commonly used:
* The Mean Absolute Error (MAE): measures the average absolute difference between the predicted values and the true values and gives an idea of the average magnitude of the difference.
* Mean Squared Error (MSE): calculates the average of the squared differences between the predicted values and the true values.
* Root Mean Squared Error (RMSE): is the square root of the MSE and provides a measure of the average magnitude of the errors.

The following applies to all errors: Lower values indicate a better model performance. The lower the values, the more accurate the model's predictions are on average. The benchmark to consider a result good can be based on the range of the actual AGB values. 


**Table 1:** Error metrics for linear and random forest model for Sentinel and drone images.

| Error metric | Sentinel       | Drone 1          | Drone 2           |
| ------------ | -------------- | ---------------- | ----------------- |
| MAE_lm       | 6.918877       | 5.145503         | 7.559921          |
| MSE_lm       | 72.53601       | 31.30135         | 82.64145          |
| RMSE_lm      | 8.516807       | 5.594761         | 9.090734          |
| MAE_rf       | 6.571995       | 7.097156         | 7.495691          |
| MSE_rf       | 70.91848       | 62.1553          | 81.02835          |
| RMSE_rf      | 8.421311       | 7.883863         | 9.001575          |


The results show, that for the Sentinel image, both the linear model and the random forest model are performing fairly similarly. The models seem to have moderate accuracy. Comparing these results to the results of the first drone image, the linear model's performance on drone 1 data is better. The random forest model, however, seems to perform worse. Similar to the first drone image, the linear model's performance on the second drone image is better than that of the random forest model. Compared to the Sentinel image, both models seem to have slightly higher errors. Nonetheless, we should keep in mind while comparing them, that the spatial extends of the images are not similar, which makes it difficult to compare.
Based on these results, the linear model consistently performs better compared to the random forest model, which indicates that the linear model might be more robust across different imaging sources. However, the accuracies could be better, although it seemed from the plot, that the models predicted the AGB quite well. The accuracies could be improved by e.g. generating additional samples. 

### Visualisation of AGB estimation for entire study area
The AGB can also be estimated for the entire study area, our two fields. It will be calculated and shown exemplarily for the Sentinel-2 image using the linear model. For this we will extract the NDVI values for the entire **aoi**, not just for the sample points, and make new predictions with these values as newdata. We can then convert the previously created **ndvi_raster** to a data frame and add the predictions as a new column, that we can then plot (see **figure 6**).

![agb_aoi_sat](https://github.com/IsasGithub/MB3_Fieldwork_AGB_Estimation/blob/main/figs/agb_aoi_sat.jpeg)

***Figure 6:** Estimated AGB for entire study area.* 


## Conclusion
The field trip gave a good insight into common agricultural field work. We learned about the "Demmin" project in general and about the measurements needed for it. Since, at least most of us, had very little experience with fieldwork, it was very exciting for us to explore how all these measurements work and to get outside, away from the computer, to get the data we need for our analysis ourselves. The analysis of this project has demonstrated the building of two different models to estimate AGB of winter wheat using additional Sentinel-2 and drone data, which can now be applied to similar locations or different dates.
