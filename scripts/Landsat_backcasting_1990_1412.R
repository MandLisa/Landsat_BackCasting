# ======================================================================
# Random Forest classification of years-since-disturbance (ysd_bin3)
# trained on single-year spectral observations and applied to BAP 1990
#
# Author: Lisa Mandl
# Project: EO4Backcasting
#
# Purpose:
#   - Train a 3-class RF classifier (ysd1–5, ysd6–10, ysd>10)
#   - Training data: single-year Landsat observations (no composites)
#   - Prediction data: annual Best Available Pixel (BAP) composite (1990)
#   - Produce probabilistic and hard class predictions
#
# Methodological note:
#   - The training dataset is reused unchanged, as both training and
#     prediction rely on single-epoch spectral observations.
#   - BAP acts as a quality-based selection operator within a year and
#     does not introduce temporal aggregation.
# ======================================================================


# ======================================================================
# 0. PACKAGES
# ======================================================================
suppressPackageStartupMessages({
  library(data.table)
  library(terra)
  library(ranger)
})


# ======================================================================
# 1. TRAINING DATA PREPARATION
# ======================================================================

# ---- 1.1 Read training data (single-year observations) ----
train_csv <- "/mnt/eo/EO4Backcasting/_intermediates/training_healthy_disturbed_2711_final.csv"
dt <- fread(train_csv)

# required columns
req_cols <- c(
  "ID", "ysd", "state",
  "blue", "green", "red", "nir", "swir1", "swir2"
)
stopifnot(all(req_cols %in% names(dt)))

# keep only disturbed pixels
dt <- dt[state == "disturbed"]


# ---- 1.2 Create 3-class years-since-disturbance bins ----
dt[, ysd_bin3 := NA_character_]

dt[ysd >=  1 & ysd <=  5, ysd_bin3 := "ysd1_5"]
dt[ysd >=  6 & ysd <= 10, ysd_bin3 := "ysd6_10"]
dt[ysd >  10,              ysd_bin3 := "ysd>10"]

dt <- dt[!is.na(ysd_bin3)]

dt[, ysd_bin3 := factor(
  ysd_bin3,
  levels = c("ysd1_5", "ysd6_10", "ysd>10")
)]


# ---- 1.3 Predictor selection (raw spectral bands only) ----
base_pred <- c("blue", "green", "red", "nir", "swir1", "swir2")

# restrict to complete cases
dt_base <- dt[complete.cases(dt[, ..base_pred])]

# ensure reproducibility
setorder(dt_base, ID)


# ======================================================================
# 2. TRAIN / TEST SPLIT (BY PIXEL ID)
# ======================================================================
set.seed(42)

ids_all   <- unique(dt_base$ID)
n_ids     <- length(ids_all)
train_ids <- sample(ids_all, size = floor(0.7 * n_ids))
test_ids  <- setdiff(ids_all, train_ids)

train_base <- dt_base[ID %in% train_ids]
test_base  <- dt_base[ID %in% test_ids]

cat("Number of IDs  – train:", length(train_ids),
    " test:", length(test_ids), "\n")
cat("Number of rows – train:", nrow(train_base),
    " test:", nrow(test_base), "\n")


# ======================================================================
# 3. CLASS WEIGHTS (INVERSE FREQUENCY)
# ======================================================================
freq <- train_base[, .N, by = ysd_bin3][order(ysd_bin3)]

w_vec <- freq$N
names(w_vec) <- as.character(freq$ysd_bin3)

# inverse-frequency weighting
class_weights <- max(w_vec) / w_vec

print(freq)
print(class_weights)


# ======================================================================
# 4. RANDOM FOREST TRAINING
# ======================================================================
rf_formula <- as.formula(
  paste("ysd_bin3 ~", paste(base_pred, collapse = " + "))
)

rf_model <- ranger(
  formula        = rf_formula,
  data           = train_base[, c(base_pred, "ysd_bin3"), with = FALSE],
  num.trees      = 500,
  mtry           = 3,
  min.node.size  = 5,
  importance     = "impurity",
  probability    = TRUE,
  classification = TRUE,
  class.weights  = class_weights,
  num.threads    = 30
)

# persist model
saveRDS(
  rf_model,
  "/mnt/eo/EO4Backcasting/_models/rf_ysd_bin3_prob_1990.rds"
)

ysd_levels <- levels(train_base$ysd_bin3)

# OOB error rate
oob_error <- rf_model$prediction.error
oob_accuracy <- 1 - oob_error

cat("OOB accuracy:", round(oob_accuracy, 3), "\n")


# ======================================================================
# 5. PREDICTION HELPER FOR terra::predict()
# ======================================================================
# Returns a matrix [n_pixels × 3] of class probabilities
rf_fun_probs <- function(model, x, ...) {
  x_df <- as.data.frame(x)
  n    <- nrow(x_df)
  
  out <- matrix(NA_real_, nrow = n, ncol = length(ysd_levels))
  if (n == 0) return(out)
  
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
# 6. TILE-WISE PREDICTION (BAP 1990)
# ======================================================================

# BAP composite for 1990 (single-epoch spectral representation)
bap_1990 <- rast("/mnt/dss_europe/level3_interpolated/X0016_Y0020/19900801_LEVEL3_LNDLG_IBAP.tif")

names(bap_1990) <- c(
  "blue",
  "green",
  "red",
  "nir",
  "swir1",
  "swir2"
)

# ---- 6.2 Load forest mask (1 = forest, NA/0 = non-forest) ----
forest_mask <- rast(
  "/mnt/eo/EO4Backcasting/_data/forest_mask_eroded_2px.tif"
)

forest_mask_crop <- crop(
  forest_mask,
  ext(bap_1990)
)

# ---- 6.3 Ensure spatial alignment ----
# (fail loudly if something is wrong)
stopifnot(
  compareGeom(bap_1990, forest_mask_crop, stopOnError = FALSE)
)

# ---- 6.4 Mask BAP to forest pixels only ----
# keep pixels where forest_mask == 1
bap_1990_forest <- mask(
  bap_1990,
  forest_mask_crop
)

# optional: free memory
rm(bap_1990)
gc()


prob_file <- "/mnt/eo/EO4Backcasting/_predictions/ysd_probs_BAP1990.tif"

prob_ras <- predict(
  bap_1990_forest,
  rf_model,
  rf_fun_probs,
  filename  = prob_file,
  overwrite = TRUE
)

names(prob_ras) <- paste0("prob_", ysd_levels)
prob_ras


# ======================================================================
# 7. DERIVED PRODUCTS
# ======================================================================

# maximum class probability
p_max <- app(prob_ras, fun = max, na.rm = TRUE)

# confidence mask
conf_mask <- p_max >= 0.5

# connected components on confident pixels
clumps <- patches(
  conf_mask,
  directions = 8
)

# pixel count per clump
clump_freq <- freq(clumps)

# clumps to keep (MMU >= 6)
keep_ids <- clump_freq$value[clump_freq$count >= 6]

# final spatial mask
mmu_mask <- clumps %in% keep_ids


# initialise output raster
binary_ras <- rast(prob_ras)
values(binary_ras) <- NA
names(binary_ras) <- paste0("bin_", ysd_levels)

# loop over classes
for (i in seq_len(nlyr(prob_ras))) {
  binary_ras[[i]] <- ifel(
    mmu_mask & (prob_ras[[i]] >= 0.5),
    1,
    NA
  )
}


# ensure no pixel is assigned to more than one class
overlap_check <- app(binary_ras, fun = function(v) sum(v == 1, na.rm = TRUE))
global(overlap_check > 1, "sum")


writeRaster(
  binary_ras,
  "/mnt/eo/EO4Backcasting/_predictions/ysd_bins_BAP1990_forest_p05_mmu6.tif",
  datatype  = "INT1U",
  gdal      = c("COMPRESS=LZW", "TILED=YES"),
  overwrite = TRUE
)
