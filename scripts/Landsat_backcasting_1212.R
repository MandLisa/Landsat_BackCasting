# ======================================================================
# Random Forest classification of years-since-disturbance (ysd_bin3)
# using spectral bands only, with probabilistic output
#
# Author: Lisa Mandl
# Project: EO4Backcasting
# Purpose:
#   - Train a 3-class RF classifier (ysd1–5, ysd6–10, ysd>10)
#   - Predict class probabilities per pixel
#   - Derive hard class and maximum class probability
#
# Notes:
#   - Only disturbed pixels are used for training
#   - Predictors are raw spectral bands (no indices)
#   - Class imbalance is handled via inverse-frequency weights
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

# ---- 1.1 Read training data ----
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


# ---- 1.2 Create 3-class ysd bins ----
dt[, ysd_bin3 := NA_character_]

dt[ysd >=  1 & ysd <=  5, ysd_bin3 := "ysd1_5"]
dt[ysd >=  6 & ysd <= 10, ysd_bin3 := "ysd6_10"]
dt[ysd >  10,              ysd_bin3 := "ysd>10"]

# drop undefined bins
dt <- dt[!is.na(ysd_bin3)]

# enforce ordered factor
dt[, ysd_bin3 := factor(
  ysd_bin3,
  levels = c("ysd1_5", "ysd6_10", "ysd>10")
)]


# ---- 1.3 Predictor selection (spectral bands only) ----
base_pred <- c("blue", "green", "red", "nir", "swir1", "swir2")

# restrict to complete cases
dt_base <- dt[complete.cases(dt[, ..base_pred])]

# sort for reproducibility
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

# inverse frequency weighting
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
  probability    = TRUE,    # probabilistic output
  classification = TRUE,
  class.weights  = class_weights,
  num.threads    = 30
)

# optional persistence
saveRDS(
  rf_model,
  "/mnt/eo/EO4Backcasting/_intermediates/rf_ysd_bin3_prob.rds"
)

ysd_levels <- levels(train_base$ysd_bin3)


# ======================================================================
# 5. PREDICTION HELPER FOR terra::predict()
# ======================================================================
# Returns a matrix [n_pixels × 3] of class probabilities
rf_fun_probs <- function(model, x, ...) {
  x_df <- as.data.frame(x)
  n    <- nrow(x_df)
  
  # pre-allocate output
  out <- matrix(NA_real_, nrow = n, ncol = 3)
  if (n == 0) return(out)
  
  # predict only on complete cases
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
# 6. TILE-WISE PREDICTION (PROBABILITY CUBE)
# ======================================================================
prob_file <- "/mnt/eo/EO4Backcasting/_predictions/ysd_probs_tile.tif"

prob_ras <- predict(
  bap_med,        # SpatRaster with spectral bands
  rf_model,       # ranger model
  rf_fun_probs,   # custom probability wrapper
  filename  = prob_file,
  overwrite = TRUE
)

# name probability layers
names(prob_ras) <- paste0("prob_", ysd_levels)
prob_ras


# ======================================================================
# 7. DERIVED PRODUCTS
# ======================================================================

# maximum class probability
p_max <- app(prob_ras, fun = max, na.rm = TRUE)
names(p_max) <- "p_max"

# hard class label (1, 2, 3)
hard_class <- app(prob_ras, fun = function(v) {
  if (all(is.na(v))) return(NA_real_)
  which.max(v)
})
names(hard_class) <- "ysd_class_id"

# outputs:
#   - prob_ras   : probability cube
#   - p_max      : prediction confidence
#   - hard_class : final class assignment
