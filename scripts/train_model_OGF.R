# ======================================================================
# Consolidate per-tile training tables + train a global XGBoost model
# (tile-based split to avoid spatial leakage)
#
# Assumes you have per-tile files like:
#   /mnt/eo/EO4Backcasting/training_data/training_<TILE>_features.csv
#
# The features CSV should contain at least:
#   point_id, x, y, tile, label_undisturbed_20y, ibap_*, nbr_*
# ======================================================================

suppressPackageStartupMessages({
  library(data.table)
  library(xgboost)
})

# ---------------------------- SETTINGS ---------------------------------

in_dir  <- "/mnt/eo/EO4Backcasting/training_data"
out_dir <- "/mnt/eo/EO4Backcasting/model_xgb"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

pattern_features <- "^training_.*_features\\.csv$"

seed <- 42

# Tile split fractions
p_train <- 0.70
p_val   <- 0.15
p_test  <- 0.15

# If you want to FORCE specific tiles into test (e.g., Central Europe tiles),
# list them here; otherwise leave empty.
force_test_tiles <- character(0)

# Optional: downsample for quick prototyping (set to NA to disable)
max_rows_total <- NA_integer_  # e.g., 2e6

# -------------------------- HELPERS ------------------------------------

read_one_features <- function(f) {
  dt <- fread(f)
  # basic sanity
  stopifnot("label_undisturbed_20y" %in% names(dt))
  stopifnot("tile" %in% names(dt))
  dt
}

drop_non_predictors <- function(dt) {
  # Keep tile for splitting but drop from X matrix later
  drop_cols <- intersect(names(dt), c("point_id", "x", "y"))
  if (length(drop_cols) > 0) dt[, (drop_cols) := NULL]
  dt
}

get_feature_cols <- function(dt) {
  # Predictor columns: everything except label and tile
  setdiff(names(dt), c("label_undisturbed_20y", "tile"))
}

rm_constant_or_all_na <- function(dt, cols) {
  keep <- logical(length(cols))
  for (i in seq_along(cols)) {
    v <- dt[[cols[i]]]
    if (all(is.na(v))) { keep[i] <- FALSE; next }
    # constant if only one unique non-NA value
    u <- unique(v[!is.na(v)])
    keep[i] <- length(u) > 1
  }
  cols[keep]
}

# ---------------------- 1) CONSOLIDATION -------------------------------

files <- list.files(in_dir, pattern = pattern_features, full.names = TRUE)
if (length(files) == 0) stop("No feature CSVs found in: ", in_dir)

# Read and row-bind (tile-wise files are typically manageable)
dt_list <- vector("list", length(files))
for (i in seq_along(files)) {
  dt_list[[i]] <- drop_non_predictors(read_one_features(files[i]))
}
dt_all <- rbindlist(dt_list, use.names = TRUE, fill = TRUE)
rm(dt_list); gc()

# Optional: limit total rows for a fast prototype run
if (!is.na(max_rows_total) && nrow(dt_all) > max_rows_total) {
  set.seed(seed)
  dt_all <- dt_all[sample.int(nrow(dt_all), max_rows_total)]
}

# Basic checks
stopifnot(all(dt_all$label_undisturbed_20y %in% c(0, 1)))
dt_all[, tile := as.character(tile)]

# Identify predictor columns and remove degenerate ones
feature_cols <- get_feature_cols(dt_all)
feature_cols <- rm_constant_or_all_na(dt_all, feature_cols)

# Save the FINAL feature list (critical for inference: same order!)
fwrite(data.table(feature = feature_cols),
       file.path(out_dir, "feature_list.csv"))

# ---------------------- 2) TILE-BASED SPLIT ----------------------------

tiles <- sort(unique(dt_all$tile))

set.seed(seed)
tiles_shuffled <- sample(tiles)

# Force some tiles into test if requested
if (length(force_test_tiles) > 0) {
  force_test_tiles <- intersect(force_test_tiles, tiles_shuffled)
  tiles_shuffled <- setdiff(tiles_shuffled, force_test_tiles)
} else {
  force_test_tiles <- character(0)
}

n <- length(tiles_shuffled)
n_train <- floor(p_train * n)
n_val   <- floor(p_val   * n)
# remainder goes to test (plus forced)
train_tiles <- tiles_shuffled[seq_len(n_train)]
val_tiles   <- tiles_shuffled[(n_train + 1):(n_train + n_val)]
test_tiles  <- tiles_shuffled[(n_train + n_val + 1):n]
test_tiles  <- union(test_tiles, force_test_tiles)

# Save splits
splits <- rbindlist(list(
  data.table(tile = train_tiles, split = "train"),
  data.table(tile = val_tiles,   split = "val"),
  data.table(tile = test_tiles,  split = "test")
))
fwrite(splits, file.path(out_dir, "tile_splits.csv"))

# Subset
dt_train <- dt_all[tile %in% train_tiles]
dt_val   <- dt_all[tile %in% val_tiles]
dt_test  <- dt_all[tile %in% test_tiles]

# ---------------------- 3) BUILD DMATRICES -----------------------------

# Important: tile is NOT a predictor, remove it from X matrices
make_matrix <- function(dt, feature_cols) {
  X <- as.matrix(dt[, ..feature_cols])
  # ensure numeric
  storage.mode(X) <- "double"
  X
}

X_train <- make_matrix(dt_train, feature_cols)
y_train <- dt_train$label_undisturbed_20y

X_val <- make_matrix(dt_val, feature_cols)
y_val <- dt_val$label_undisturbed_20y

X_test <- make_matrix(dt_test, feature_cols)
y_test <- dt_test$label_undisturbed_20y

dtrain <- xgb.DMatrix(data = X_train, label = y_train)
dval   <- xgb.DMatrix(data = X_val,   label = y_val)
dtest  <- xgb.DMatrix(data = X_test,  label = y_test)

rm(X_train, X_val, X_test); gc()

# ---------------------- 4) TRAIN GLOBAL XGBOOST ------------------------

params <- list(
  booster = "gbtree",
  objective = "binary:logistic",
  eval_metric = c("auc", "aucpr"),
  eta = 0.05,
  max_depth = 6,
  min_child_weight = 5,
  subsample = 0.8,
  colsample_bytree = 0.8,
  lambda = 1.0,
  alpha = 0.0
)

set.seed(seed)
watchlist <- list(train = dtrain, val = dval)

model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 5000,
  watchlist = watchlist,
  early_stopping_rounds = 50,
  verbose = 1
)

# Save model
xgb.save(model, file.path(out_dir, "xgb_model.json"))

# ---------------------- 5) EVALUATION (TEST) ---------------------------

pred_test <- predict(model, dtest)

# Simple threshold-free metrics from xgboost are already in training logs (AUC, AUCPR).
# For a concrete threshold, you choose based on your use-case.
# Example: choose threshold that maximizes F1 on validation set.
pred_val <- predict(model, dval)
thr_grid <- seq(0.05, 0.95, by = 0.01)

f1_at_thr <- function(p, y, thr) {
  yhat <- as.integer(p >= thr)
  tp <- sum(yhat == 1 & y == 1)
  fp <- sum(yhat == 1 & y == 0)
  fn <- sum(yhat == 0 & y == 1)
  if ((2*tp + fp + fn) == 0) return(NA_real_)
  2*tp / (2*tp + fp + fn)
}

f1s <- vapply(thr_grid, function(t) f1_at_thr(pred_val, y_val, t), numeric(1))
best_thr <- thr_grid[which.max(f1s)]

# Confusion matrix at best threshold (on test set)
yhat_test <- as.integer(pred_test >= best_thr)
tp <- sum(yhat_test == 1 & y_test == 1)
tn <- sum(yhat_test == 0 & y_test == 0)
fp <- sum(yhat_test == 1 & y_test == 0)
fn <- sum(yhat_test == 0 & y_test == 1)

metrics <- data.table(
  best_thr = best_thr,
  tp = tp, tn = tn, fp = fp, fn = fn,
  precision = ifelse(tp + fp == 0, NA_real_, tp/(tp+fp)),
  recall    = ifelse(tp + fn == 0, NA_real_, tp/(tp+fn)),
  f1        = ifelse(2*tp + fp + fn == 0, NA_real_, 2*tp/(2*tp + fp + fn)),
  accuracy  = (tp + tn) / (tp + tn + fp + fn)
)

fwrite(metrics, file.path(out_dir, "test_metrics_at_bestF1thr.csv"))

# Feature importance (gain)
imp <- xgb.importance(model = model, feature_names = feature_cols)
fwrite(imp, file.path(out_dir, "feature_importance_gain.csv"))

# Save a small predictions sample for QA
pred_sample <- data.table(
  y = y_test,
  p = pred_test
)
fwrite(pred_sample, file.path(out_dir, "test_pred_sample.csv"))

# ---------------------- NOTES ------------------------------------------
# 1) You should keep the tile-based split and report test performance on held-out tiles.
# 2) For mapping: load "feature_list.csv" and stack rasters in EXACT same order.
# 3) Avoid using x/y/tile as predictors to prevent spatial memorization.