# ======================================================================
# Random Forest prediction of years-since-disturbance (ysd_bin3)
# using a pre-trained probabilistic model
#
# Author: Lisa Mandl
# Project: EO4Backcasting
#
# Purpose:
#   - Load a pre-trained RF classifier (3 ysd classes)
#   - Predict per-pixel class probabilities for a raster tile
#   - Derive (i) maximum probability and (ii) hard class assignment
#
# Assumptions:
#   - Model was trained with probability = TRUE
#   - Raster bands match the training predictors exactly
#   - Class order is preserved in the trained model
# ======================================================================


# ======================================================================
# 0. PACKAGES
# ======================================================================
suppressPackageStartupMessages({
  library(terra)
  library(ranger)
})


# ======================================================================
# 1. LOAD PRE-TRAINED MODEL
# ======================================================================
model_file <- "/mnt/eo/EO4Backcasting/_intermediates/rf_ysd_bin3_prob.rds"
rf_model <- readRDS(model_file)

# extract class labels from model
ysd_levels <- rf_model$forest$levels
stopifnot(length(ysd_levels) == 3)


# ======================================================================
# 2. LOAD PREDICTION RASTER
# ======================================================================
# SpatRaster must contain the same spectral bands as used for training
# (blue, green, red, nir, swir1, swir2)

bap_med <- rast("/mnt/eo/EO4Backcasting/_tiles/bap_med_tile.tif")

expected_bands <- c("blue", "green", "red", "nir", "swir1", "swir2")
stopifnot(all(expected_bands %in% names(bap_med)))

# enforce band order (defensive programming)
bap_med <- bap_med[[expected_bands]]


# ======================================================================
# 3. PREDICTION WRAPPER FOR terra::predict()
# ======================================================================
# This function:
#   - receives raster blocks as data.frame
#   - predicts class probabilities
#   - returns a numeric matrix [n_pixels × 3]

rf_fun_probs <- function(model, x, ...) {
  
  x_df <- as.data.frame(x)
  n    <- nrow(x_df)
  
  # pre-allocate output
  out <- matrix(NA_real_, nrow = n, ncol = length(ysd_levels))
  if (n == 0) return(out)
  
  # predict only where predictors are complete
  idx <- stats::complete.cases(x_df)
  
  if (any(idx)) {
    p <- predict(
      model,
      data = x_df[idx, , drop = FALSE]
    )$predictions
    
    out[idx, ] <- as.matrix(p)
  }
  
  out
}


# ======================================================================
# 4. RUN TILE-WISE PROBABILITY PREDICTION
# ======================================================================
out_prob_file <- "/mnt/eo/EO4Backcasting/_predictions/ysd_probs_tile.tif"

prob_ras <- predict(
  bap_med,
  rf_model,
  rf_fun_probs,
  filename  = out_prob_file,
  overwrite = TRUE
)

# name probability layers explicitly
names(prob_ras) <- paste0("prob_", ysd_levels)


# ======================================================================
# 5. DERIVED PRODUCTS
# ======================================================================

# ---- 5.1 Maximum class probability (prediction confidence) ----
p_max <- app(prob_ras, fun = max, na.rm = TRUE)
names(p_max) <- "p_max"

writeRaster(
  p_max,
  "/mnt/eo/EO4Backcasting/_predictions/ysd_pmax_tile.tif",
  overwrite = TRUE
)


# ---- 5.2 Hard class assignment (argmax) ----
hard_class <- app(prob_ras, fun = function(v) {
  if (all(is.na(v))) return(NA_real_)
  which.max(v)
})

names(hard_class) <- "ysd_class_id"

writeRaster(
  hard_class,
  "/mnt/eo/EO4Backcasting/_predictions/ysd_class_tile.tif",
  overwrite = TRUE
)


# ======================================================================
# OUTPUT SUMMARY
# ======================================================================
# Generated files:
#   - ysd_probs_tile.tif  : class probability cube (3 bands)
#   - ysd_pmax_tile.tif   : maximum probability per pixel
#   - ysd_class_tile.tif  : hard class (1 = ysd1_5, 2 = ysd6_10, 3 = ysd>10)
#
# Notes:
#   - NA values propagate consistently from input rasters
#   - Class numbering follows model training order
# ======================================================================
