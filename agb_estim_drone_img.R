### MB3 Isabella - AGB estimation drone images

## this code is basically the same like the code for the sentinel image ("agb_estim_sat_img.R),
## therefore, the amount of comments is reduced

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

# there are two different drone orhtophotos, that is why there is 1 and 2
aoi <- st_read("./aoiF1drones.shp")
aoi2 <- st_read("./aoiF2drones.shp")

samples <- st_read("./SSUs_GT_f1d.shp")
colnames(samples)[7] ="bowl_weight"
colnames(samples)[8] ="wet_weight"
colnames(samples)[9] ="dry_weight"


mapview(aoi) + 
  mapview(samples, legend = TRUE)


samples2 <- st_read("./SSUs_GT_f2d.shp")
colnames(samples2)[7] ="bowl_weight"
colnames(samples2)[8] ="wet_weight"
colnames(samples2)[9] ="dry_weight"


mapview(aoi2) + 
  mapview(samples2, legend = TRUE)


# import drone data
drone1 <- rast("./F1drones.tif")
dr1 <- mask(drone1, aoi)
plot(dr1)

drone2 <- rast("./F2drones.tif")
dr2 <- mask(drone2, aoi2)
plot(dr2)



## AGB Estimation drone image 1

red <- dr1[[3]]
nir <- dr1[[4]]
ndvi <- (nir - red) / (nir + red)

ndvi_raster <- raster(ndvi)

buffer_distance <- 10
buffers <- st_buffer(samples, dist = buffer_distance)

ndvi_mean <- raster::extract(ndvi_raster, buffers, fun = mean, na.rm = TRUE)

samples$NDVI_mean <- ndvi_mean

ggplot(data = samples, aes(x = NDVI_mean, y = AGB)) +
  ggtitle("Linear Relation: AGB - NDVI") + 
  geom_point(color = "olivedrab3", size = 3) +
  labs(x = "NDVI",
       y = "AGB") +
  dark_theme_gray()



# create a model to estimate agb values 

## preprocessing

samples_df <- samples[c(6, 10, 13)]

set.seed(132) 
div <- sample(1:nrow(samples_df), 0.7 * nrow(samples_df)) 
training_df <- samples_df[div, ]
testing_df <- samples_df[-div, ]


#### Linear Model

model_lm <- lm(AGB ~ as.vector(NDVI_mean), data = training_df)
predictions_lm <- predict(model_lm, newdata = testing_df)


pred_lm_df <- data.frame(
  short = testing_df$short,
  estim_AGB = predictions_lm 
)


true_agb_lm_df <- data.frame(
  short = testing_df$short,
  true_AGB = testing_df$AGB
)

predictions_lm_df <- bind_rows(true_agb_lm_df, pred_lm_df)



##### Random Forest

model_rf <- randomForest(AGB ~ NDVI_mean, data = training_df, ntree = 700)

predictions_rf <- predict(model_rf, newdata = testing_df)

pred_rf_df <- data.frame(
  short = testing_df$short,
  estim_AGB = predictions_rf 
)

true_agb_rf_df <- data.frame(
  short = testing_df$short,
  true_AGB = testing_df$AGB
)

predictions_rf_df <- bind_rows(true_agb_rf_df, pred_rf_df)


# visualisation

combined_predictions_df <- data.frame(
  short = testing_df$short,
  estim_AGB_lm = predictions_lm,
  true_AGB = testing_df$AGB,
  estim_AGB_rf = predictions_rf
)

combined_predictions_df_long <- pivot_longer(
  combined_predictions_df,
  cols = c(estim_AGB_lm, true_AGB, estim_AGB_rf),
  names_to = "Variable",
  values_to = "Value"
)

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

combined_plot + scale_x_discrete(limits = combined_predictions_df$short[order(combined_predictions_df$short)])




### calculate errors for accuracy assessment

mae_lm <- mean(abs(predictions_lm - testing_df$AGB))
mse_lm <- mean((predictions_lm - testing_df$AGB)^2)
rmse_lm <- sqrt(mse_lm)

mae_rf <- mean(abs(predictions_rf - testing_df$AGB))
mse_rf <- mean((predictions_rf - testing_df$AGB)^2)
rmse_rf <- sqrt(mse_rf)

cat("Linear Model:\n")
cat("MAE:", mae_lm, "\n")
cat("MSE:", mse_lm, "\n")
cat("RMSE:", rmse_lm, "\n")

cat("\nRandom Forest Model:\n")
cat("MAE:", mae_rf, "\n")
cat("MSE:", mse_rf, "\n")
cat("RMSE:", rmse_rf, "\n")


## AGB Estimation drone image 2

red2 <- dr2[[3]]
nir2 <- dr2[[4]]

ndvi_2 <- (nir2 - red2) / (nir2 + red2)

ndvi_raster_2 <- raster(ndvi_2)


buffer_distance <- 10
buffers_2 <- st_buffer(samples2, dist = buffer_distance)


ndvi_mean_2 <- raster::extract(ndvi_raster_2, buffers_2, fun = mean, na.rm = TRUE)

samples2$NDVI_mean <- ndvi_mean_2

ggplot(data = samples2, aes(x = NDVI_mean, y = AGB)) +
  ggtitle("Linear Relation: AGB - NDVI") + 
  geom_point(color = "olivedrab3", size = 3) +
  labs(x = "NDVI",
       y = "AGB") +
  dark_theme_gray()



# create a model to estimate agb values 

## preprocessing

samples2_df <- samples2[c(6, 10, 13)]

set.seed(132)
div <- sample(1:nrow(samples2_df), 0.7 * nrow(samples2_df)) 
training2_df <- samples2_df[div, ]
testing2_df <- samples2_df[-div, ]


#### Linear Model

model_lm_2 <- lm(AGB ~ as.vector(NDVI_mean), data = training2_df)

predictions_lm_2 <- predict(model_lm_2, newdata = testing2_df)


pred_lm2_df <- data.frame(
  short = testing2_df$short,
  estim_AGB = predictions_lm_2
)

true_agb_lm2_df <- data.frame(
  short = testing2_df$short,
  true_AGB = testing2_df$AGB
)

predictions_lm2_df <- bind_rows(true_agb_lm2_df, pred_lm2_df)




##### Random Forest

model_rf_2 <- randomForest(AGB ~ NDVI_mean, data = training2_df, ntree = 700)

predictions_rf_2 <- predict(model_rf_2, newdata = testing2_df)

pred_rf2_df <- data.frame(
  short = testing2_df$short,
  estim_AGB = predictions_rf_2
)

true_agb_rf2_df <- data.frame(
  short = testing2_df$short,
  true_AGB = testing2_df$AGB
)

predictions_rf2_df <- bind_rows(true_agb_rf2_df, pred_rf2_df)


# Visualisation

combined_predictions2_df <- data.frame(
  short = testing2_df$short,
  estim_AGB_lm = predictions_lm_2,
  true_AGB = testing2_df$AGB,
  estim_AGB_rf = predictions_rf_2
)

combined_predictions2_df_long <- pivot_longer(
  combined_predictions2_df,
  cols = c(estim_AGB_lm, true_AGB, estim_AGB_rf),
  names_to = "Variable",
  values_to = "Value"
)

combined_plot_2 <- ggplot(combined_predictions2_df_long, aes(x = short, y = Value, fill = Variable)) +
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

combined_plot_2 + scale_x_discrete(limits = combined_predictions2_df$short[order(combined_predictions2_df$short)])


### calculate errors for accuracy assessment

mae_lm2 <- mean(abs(predictions_lm_2 - testing2_df$AGB))
mse_lm2 <- mean((predictions_lm_2 - testing2_df$AGB)^2)
rmse_lm2 <- sqrt(mse_lm2)

mae_rf2 <- mean(abs(predictions_rf_2 - testing2_df$AGB))
mse_rf2 <- mean((predictions_rf_2 - testing2_df$AGB)^2)
rmse_rf2 <- sqrt(mse_rf2)

cat("Linear Model:\n")
cat("MAE:", mae_lm2, "\n")
cat("MSE:", mse_lm2, "\n")
cat("RMSE:", rmse_lm2, "\n")

cat("\nRandom Forest Model:\n")
cat("MAE:", mae_rf2, "\n")
cat("MSE:", mse_rf2, "\n")
cat("RMSE:", rmse_rf2, "\n")





### show both drone images in one graph

# Create a data frame containing the predicted agb values
dr2_df <- data.frame(
  short = testing2_df$short,
  estim_AGB_lm = predictions_lm_2,
  true_AGB = testing2_df$AGB,
  estim_AGB_rf = predictions_rf_2
)

dr1_df <- data.frame(
  short = testing_df$short,
  estim_AGB_lm = predictions_lm,
  true_AGB = testing_df$AGB,
  estim_AGB_rf = predictions_rf
)

# Combine the data frames 
dr12_df <- bind_rows(dr1_df, dr2_df)

dr12_df_long <- pivot_longer(dr12_df, cols = c(estim_AGB_lm, true_AGB, estim_AGB_rf))


plot_dr12 <- ggplot(dr12_df_long, aes(x = short, y = value, fill = name)) +
  ggtitle("True & Estimated AGB") + 
  geom_bar(stat = "identity", position = "dodge", width = 0.6) +
  xlab("SSUs") +
  ylab("AGB [g/5 stems]") +
  scale_fill_manual(
    values = c("olivedrab3", "indianred3", "forestgreen"),
    labels = c("Estim. AGB (lm)", "True AGB", "Estim. AGB (rf)"),
    guide_colorbar(title = "")
  ) +
  dark_theme_gray() +  
  theme(
    axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1)  
  ) 

plot_dr12 + scale_x_discrete(limits = dr12_df$short[order(dr12_df$short)])

