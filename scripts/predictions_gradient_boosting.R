# ======================================================================
# Gradient-Boosted Decision Trees (XGBoost) classification of YSD bins
# trained on single-year spectral observations and applied to BAP 1990
#
# Author: Lisa Mandl
# Project: EO4Backcasting
#
# Purpose:
#   - Train a 3-class GBDT classifier (ysd1–5, ysd6–10, ysd>10)
#   - Training data: single-year Landsat observations (no composites)
#   - Prediction data: annual Best Available Pixel (BAP) composite (1990)
#   - Produce probabilistic and hard class predictions
#
# Methodological note:
#   - Same as RF pipeline: backcasting uses single-epoch spectral state.
#   - BAP is treated as a within-year quality selector, not temporal aggregation.
# ======================================================================

# ======================================================================
# 0. PACKAGES
# ======================================================================
suppressPackageStartupMessages({
  library(data.table)
  library(terra)
  library(xgboost)
})

# ======================================================================
# 1. TRAINING DATA PREPARATION
# ======================================================================

# ---- 1.1 Read training data (single-year observations) ----
train_csv <- "/mnt/eo/EO4Backcasting/_intermediates/training_healthy_disturbed_2711_final.csv"
dt <- fread(train_csv)

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

ysd_levels <- levels(dt$ysd_bin3)
K <- length(ysd_levels)

# ---- 1.3 Predictor selection (raw spectral bands only) ----
base_pred <- c("blue", "green", "red", "nir", "swir1", "swir2")

# restrict to complete cases
dt_base <- dt[complete.cases(dt[, ..base_pred])]

# ensure reproducibility & stable ordering
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
# 3. CLASS WEIGHTS (INVERSE FREQUENCY) -> ROW WEIGHTS FOR XGBOOST
# ======================================================================
freq <- train_base[, .N, by = ysd_bin3][order(ysd_bin3)]

w_vec <- freq$N
names(w_vec) <- as.character(freq$ysd_bin3)

# inverse-frequency weighting
class_weights <- max(w_vec) / w_vec

print(freq)
print(class_weights)

# row-wise weights
w_train <- class_weights[as.character(train_base$ysd_bin3)]
w_test  <- class_weights[as.character(test_base$ysd_bin3)]  # optional, for eval symmetry

# ======================================================================
# 4. XGBOOST TRAINING (MULTICLASS PROBABILITIES)
# ======================================================================

# labels must be 0..K-1 for xgboost multiclass
y_train <- as.integer(train_base$ysd_bin3) - 1L
y_test  <- as.integer(test_base$ysd_bin3) - 1L

X_train <- as.matrix(train_base[, ..base_pred])
X_test  <- as.matrix(test_base[, ..base_pred])

dtrain <- xgb.DMatrix(data = X_train, label = y_train, weight = w_train)
dtest  <- xgb.DMatrix(data = X_test,  label = y_test,  weight = w_test)

watchlist <- list(train = dtrain, eval = dtest)

# Reasonable default params for EO tabular spectra:
# - shallow trees + shrinkage + subsampling for robustness to label noise
params <- list(
  booster = "gbtree",
  objective = "multi:softprob",
  num_class = K,
  eval_metric = "mlogloss",
  eta = 0.05,
  max_depth = 6,
  min_child_weight = 5,
  subsample = 0.8,
  colsample_bytree = 0.8,
  gamma = 0,
  lambda = 1,
  alpha = 0,
  tree_method = "hist",   # usually fast on CPU
  nthread = 30
)

set.seed(42)
xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 3000,
  watchlist = watchlist,
  early_stopping_rounds = 50,
  verbose = 1
)

# persist model
saveRDS(
  xgb_model,
  "/mnt/eo/EO4Backcasting/_models/xgb_ysd_bin3_prob_1990.rds"
)

cat("Best iteration:", xgb_model$best_iteration, "\n")
cat("Best eval mlogloss:", xgb_model$best_score, "\n")

# quick accuracy on held-out IDs (hard labels from max prob)
p_test <- predict(xgb_model, dtest)  # vector length = nrow(test)*K
p_test <- matrix(p_test, ncol = K, byrow = TRUE)
pred_test <- max.col(p_test) - 1L
acc_test <- mean(pred_test == y_test)

cat("Test accuracy (ID-split):", round(acc_test, 3), "\n")

# ======================================================================
# 8. PER-CLASS ACCURACY (TEST SET, ID-SPLIT)
# ======================================================================

# y_test: true labels (0..K-1)
# pred_test: predicted labels (0..K-1)

# confusion matrix
cm <- table(
  truth = factor(y_test, levels = 0:(K-1), labels = ysd_levels),
  pred  = factor(pred_test, levels = 0:(K-1), labels = ysd_levels)
)

print(cm)

# per-class (producer's) accuracy
class_acc <- diag(prop.table(cm, margin = 1))

cat("\nPer-class accuracy:\n")
print(round(class_acc, 3))

# overall accuracy (for reference)
overall_acc <- sum(diag(cm)) / sum(cm)
cat("\nOverall accuracy:", round(overall_acc, 3), "\n")




# ======================================================================
# 5. PREDICTION HELPER FOR terra::predict()
# ======================================================================
# Returns a matrix [n_pixels × K] of class probabilities
xgb_fun_probs <- function(model, x, ...) {
  x_df <- as.data.frame(x)
  n    <- nrow(x_df)
  
  out <- matrix(NA_real_, nrow = n, ncol = K)
  if (n == 0) return(out)
  
  idx <- stats::complete.cases(x_df)
  
  if (any(idx)) {
    X <- as.matrix(x_df[idx, , drop = FALSE])
    d <- xgb.DMatrix(X)
    
    p <- predict(model, d)  # vector length = sum(idx)*K
    p <- matrix(p, ncol = K, byrow = TRUE)
    
    out[idx, ] <- p
  }
  
  out
}

# ======================================================================
# 6. TILE-WISE PREDICTION (BAP 1990)
# ======================================================================

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
forest_mask <- rast("/mnt/eo/EO4Backcasting/_data/forest_mask_eroded_2px.tif")
forest_mask_crop <- crop(forest_mask, ext(bap_1990))

# ---- 6.3 Ensure spatial alignment ----
stopifnot(compareGeom(bap_1990, forest_mask_crop, stopOnError = FALSE))

# ---- 6.4 Mask BAP to forest pixels only ----
bap_1990_forest <- mask(bap_1990, forest_mask_crop)

rm(bap_1990)
gc()

prob_file <- "/mnt/eo/EO4Backcasting/_predictions/ysd_probs_BAP1990_xgb.tif"

prob_ras <- predict(
  bap_1990_forest,
  xgb_model,
  xgb_fun_probs,
  filename  = prob_file,
  overwrite = TRUE
)

names(prob_ras) <- paste0("prob_", ysd_levels)
prob_ras

# ======================================================================
# 7. DERIVED PRODUCTS (same logic as RF pipeline)
# ======================================================================

# maximum class probability
p_max <- app(prob_ras, fun = max, na.rm = TRUE)

# confidence mask
conf_mask <- p_max >= 0.5

# connected components on confident pixels
clumps <- patches(conf_mask, directions = 8)

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
  "/mnt/eo/EO4Backcasting/_predictions/ysd_bins_BAP1990_forest_p05_mmu6_xgb.tif",
  datatype  = "INT1U",
  gdal      = c("COMPRESS=LZW", "TILED=YES"),
  overwrite = TRUE
)
