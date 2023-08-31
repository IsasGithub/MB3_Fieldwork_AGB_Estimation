### MB3 Isabella - AGB estimation Sentinel image

# load libraries

library(raster)
library(terra)
library(sf)
library(mapview)
library(exactextractr)
library(corrplot)
library(randomForest)
library(Metrics)
library(ggplot2)
library(ggdark)
library(tidyr)
library(dplyr)

# --> please change the path to your working directory
setwd("D:/DATEN ZWEI/Wue/SS_23/MB3_FieldCourse_Demmin/data")

# load study area and sample data with locations
aoi <- st_read("D:/DATEN ZWEI/Wue/SS_23/MB3_FieldCourse_Demmin/pythonProject/fieldaoi.shp")

samples <- st_read("D:/DATEN ZWEI/Wue/SS_23/MB3_FieldCourse_Demmin/pythonProject/SSUs_GT.shp")
# rename column names 
colnames(samples)[7] ="bowl_weight"
colnames(samples)[8] ="wet_weight"
colnames(samples)[9] ="dry_weight"


mapview(aoi) + 
  mapview(samples, legend = TRUE)


# import sentinel data and mask it to the study area
sentinel <- rast("D:/DATEN ZWEI/Wue/SS_23/MB3_FieldCourse_Demmin/data/sentinel_image.tif")
sentinel2_fields <- mask(sentinel, aoi)
plot(sentinel2_fields)


## AGB Estimation sentinel image

# calculate vegetation index
red <- sentinel2_fields[[3]]
nir <- sentinel2_fields[[4]]

# NDVI
ndvi <- (nir - red) / (nir + red)


# Convert numeric vectors to Rasterlayer
ndvi_raster <- raster(ndvi)



# create buffer around sample points
buffer_distance <- 10
buffers <- st_buffer(samples, dist = buffer_distance)


## extract zonal statistics from satellite imagery for buffered regions

# Extract mean NDVI from NDVI raster for buffered regions
ndvi_mean <- raster::extract(ndvi_raster, buffers, fun = mean, na.rm = TRUE)

# Add NDVI mean values to the samples dataframe
samples$NDVI_mean <- ndvi_mean


# Create linear relation scatter plot between NDVI and AGB
ggplot(data = samples, aes(x = NDVI_mean, y = AGB)) +
  ggtitle("Linear Relation: AGB - NDVI") + 
  geom_point(color = "olivedrab3", size = 3) +
  labs(x = "NDVI",
       y = "AGB") +
  dark_theme_gray()



# create a model to estimate agb values 

## preprocessing

# divide data into training and testing
# also put column short (names) in data frame 
samples_df <- samples[c(6, 10, 13)]

set.seed(132) # For reproducibility
div <- sample(1:nrow(samples_df), 0.7 * nrow(samples_df)) # 70% for training, 30% for testing
training_df <- samples_df[div, ]
testing_df <- samples_df[-div, ]



#### Linear Model

# Create linear model using NDVI_mean from the training data set
model_lm <- lm(AGB ~ as.vector(NDVI_mean), data = training_df)

# Make predictions: predict agb values for sample points from the testing data set
predictions_lm <- predict(model_lm, newdata = testing_df)

## visualisation

# Create a data frame containing the predicted agb values
pred_lm_df <- data.frame(
  short = testing_df$short,
  estim_AGB = predictions_lm 
)

# Create a data frame containing true AGB values for testing data
true_agb_lm_df <- data.frame(
  short = testing_df$short,
  true_AGB = testing_df$AGB
)


# Combine the data frames 
predictions_lm_df <- bind_rows(true_agb_lm_df, pred_lm_df)

# Convert the data frame to long format for plotting
predictions_lm_df_long <- pivot_longer(predictions_lm_df, cols = c(true_AGB, estim_AGB))

# Create a ggplot object
plot_lm <- ggplot(predictions_lm_df_long, aes(x = short, y = value, fill = name)) +
  ggtitle("True & Estimated AGB (linear model)") + 
  geom_bar(stat = "identity", position = "dodge", width = 1.3) +
  xlab("SSUs") +
  ylab("AGB [g/5 stems]") +
  scale_fill_manual(
    values = c("indianred3", "olivedrab3"),
    labels = c("True AGB", "Estim. AGB"),
    guide_colorbar(title = "")
  ) +
  dark_theme_gray() +  
  theme(
    axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1)  
  ) 

# Order x-axis (= SSU names) alphabetically
plot_lm + scale_x_discrete(limits = predictions_lm_df$short[order(predictions_lm_df$short)])



##### Random Forest

# Create random forest model using NDVI_mean
model_rf <- randomForest(AGB ~ NDVI_mean, data = training_df, ntree = 700)

# Make predictions using the random forest model
predictions_rf <- predict(model_rf, newdata = testing_df)


# Create a data frame containing the predicted agb values
pred_rf_df <- data.frame(
  short = testing_df$short,
  estim_AGB = predictions_rf 
)

# Create a data frame containing true AGB values for testing data
true_agb_rf_df <- data.frame(
  short = testing_df$short,
  true_AGB = testing_df$AGB
)


# Combine the data frames 
predictions_rf_df <- bind_rows(true_agb_rf_df, pred_rf_df)

# Convert the data frame to long format for plotting
predictions_rf_df_long <- pivot_longer(predictions_rf_df, cols = c(true_AGB, estim_AGB))


# Create a ggplot object
plot_rf <- ggplot(predictions_rf_df_long, aes(x = short, y = value, fill = name)) +
  ggtitle("True & Estimated AGB (random forest)") + 
  geom_bar(stat = "identity", position = "dodge", width = 1.3) +
  xlab("SSUs") +
  ylab("AGB [g/5 stems]") +
  scale_fill_manual(
    values = c("indianred3", "forestgreen"),
    labels = c("True AGB", "Estim. AGB"),
    guide_colorbar(title = "")
  ) +
  dark_theme_gray() +  
  theme(
    axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1)  
  ) 
  
  
# Order x-axis (= SSU names) alphabetically
plot_rf + scale_x_discrete(limits = predictions_rf_df$short[order(predictions_rf_df$short)])




### visualize both in one

# Combine the data frames and select relevant columns
combined_predictions_df <- data.frame(
  short = testing_df$short,
  estim_AGB_lm = predictions_lm,
  true_AGB = testing_df$AGB,
  estim_AGB_rf = predictions_rf
)

# Create a long format data frame for plotting
combined_predictions_df_long <- pivot_longer(
  combined_predictions_df,
  cols = c(estim_AGB_lm, true_AGB, estim_AGB_rf),
  names_to = "Variable",
  values_to = "Value"
)

# Create the plot
combined_plot <- ggplot(combined_predictions_df_long, aes(x = short, y = Value, fill = Variable)) +
  ggtitle("True & Estimated AGB") +
  geom_bar(stat = "identity", position = "dodge", width = 0.7) +
  xlab("SSUs") +
  ylab("AGB [g/5 stems]") +
  scale_fill_manual(
    values = c("olivedrab3", "indianred3", "forestgreen"),
    labels = c("Estim. AGB (linear model)", "True AGB", "Estim. AGB (random forest)"),
    guide_colorbar(title = "")
  ) +
  dark_theme_gray() +
  theme(
    axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
    legend.position = "bottom"
  )

# Order x-axis (= SSU names) alphabetically
combined_plot + scale_x_discrete(limits = combined_predictions_df$short[order(combined_predictions_df$short)])




### calculate errors for accuracy assessment

# Calculate MAE, MSE, and RMSE for linear model
mae_lm <- mean(abs(predictions_lm - testing_df$AGB))
mse_lm <- mean((predictions_lm - testing_df$AGB)^2)
rmse_lm <- sqrt(mse_lm)

# Calculate MAE, MSE, and RMSE for random forest model
mae_rf <- mean(abs(predictions_rf - testing_df$AGB))
mse_rf <- mean((predictions_rf - testing_df$AGB)^2)
rmse_rf <- sqrt(mse_rf)

# Print the evaluation metrics for both models
cat("Linear Model:\n")
cat("MAE:", mae_lm, "\n")
cat("MSE:", mse_lm, "\n")
cat("RMSE:", rmse_lm, "\n")

cat("\nRandom Forest Model:\n")
cat("MAE:", mae_rf, "\n")
cat("MSE:", mse_rf, "\n")
cat("RMSE:", rmse_rf, "\n")





## Visualisation of estimated agb for entire study area exemplaryly using the linear model

# extract values based on the positions of the pixel in the raster
aoi_rast_pixels <- raster::extract(ndvi_raster, seq_along(0:length(values(ndvi_raster))))

# Add column NDVI_mean with calculated ndvi
aoi_rast_pixels$NDVI_mean <- ((nir - red) / (nir + red))

# Make prediction
predictions_aoi <- predict(model_lm, newdata = aoi_rast_pixels)

# Set negative values to 0
predictions_aoi <- pmax(predictions_aoi, 0)


# Convert the raster to a data frame
ndvi_df <- as.data.frame(ndvi_raster, xy = TRUE)

# Add the predicted AGB values as a new column
ndvi_df$AGB_estimated <- predictions_aoi

# Plot the data
ggplot() +
  ggtitle("Estimated AGB [g/5 stems]") +
  geom_raster(data = ndvi_df, aes(x = x, y = y, fill = AGB_estimated)) +
  scale_fill_gradientn(
    colors = c("gray20", "olivedrab3"),
    na.value = "transparent",
    guide_colorbar(title = "AGB"),
  ) +
  dark_theme_gray()
