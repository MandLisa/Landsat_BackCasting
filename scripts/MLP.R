# ======================================================================
# 0. Packages and TensorFlow / Keras setup
# ======================================================================

library(data.table)
library(dplyr)
library(reticulate)

Sys.unsetenv("RETICULATE_PYTHON")
Sys.setenv(RETICULATE_MINICONDA_PATH = "")
reticulate::py_config()  # should now warn that nothing is configured


# use the virtualenv you already created
use_virtualenv("r-tf-mlp", required = TRUE)

library(tensorflow)
library(keras)

# optional sanity check
tf$constant(1)
py_config()

set.seed(1234)

# ======================================================================
# 1. Load training data
# ======================================================================

TRAIN_CSV <- "/mnt/eo/EO4Backcasting/_intermediates/training_data.csv"
train_df  <- fread(TRAIN_CSV)

# Convert to data.table for lag operations
setDT(train_df)

# ======================================================================
# 2. Define 5-year bins (if not yet present or to be safe)
#    ysd1_5, ysd6_10, ysd11_15, ysd16_20, ysd_20
# ======================================================================

train_df[, bin_label := fcase(
  ysd >= 0  & ysd <= 5,   "ysd1_5",
  ysd >= 6  & ysd <= 10,  "ysd6_10",
  ysd >= 11 & ysd <= 15,  "ysd11_15",
  ysd >= 16 & ysd <= 20,  "ysd16_20",
  ysd > 20,               "ysd_20",
  default = NA_character_
)]

# Optional: check class balance
print(train_df[!is.na(bin_label) & state == "disturbed",
               table(bin_label)])

# ======================================================================
# 3. Add temporal features: 1-year lags and 1-year differences
#    per id, sorted by year
# ======================================================================

band_cols <- c("b1", "b2", "b3", "b4", "b5", "b6")

setorder(train_df, id, year)

# 1-year lag for each band
for (col in band_cols) {
  lag_name <- paste0(col, "_lag1")
  train_df[, (lag_name) := shift(get(col), 1L, type = "lag"), by = id]
}

# 1-year difference (current - previous year)
for (col in band_cols) {
  diff_name <- paste0(col, "_diff1")
  lag_name  <- paste0(col, "_lag1")
  train_df[, (diff_name) := get(col) - get(lag_name)]
}

# ======================================================================
# 4. Build training set: disturbed pixels with valid bin_label and lags
# ======================================================================

train_dist <- train_df[
  state == "disturbed" &
    bap_available == TRUE &
    !is.na(bin_label)
]

# keep only rows where all predictors are available
pred_cols <- c(
  band_cols,
  paste0(band_cols, "_lag1"),
  paste0(band_cols, "_diff1")
)

train_dist <- train_dist[complete.cases(train_dist[, ..pred_cols])]

cat("Training samples:", nrow(train_dist), "\n")

# ======================================================================
# 5. Predictor matrix X and target y (one-hot encoded)
# ======================================================================

X <- as.matrix(train_dist[, ..pred_cols])

# standardize features
X_mean <- colMeans(X, na.rm = TRUE)
X_sd   <- apply(X, 2, sd, na.rm = TRUE)
X_scaled <- scale(X, center = X_mean, scale = X_sd)

# target
y_factor     <- factor(train_dist$bin_label)
class_levels <- levels(y_factor)
n_classes    <- length(class_levels)  # should be 5

y_int <- as.integer(y_factor) - 1L
y_cat <- to_categorical(y_int, num_classes = n_classes)

# ======================================================================
# 6. Train–validation split
# ======================================================================

set.seed(1234)
n   <- nrow(X_scaled)
idx <- sample(seq_len(n))

train_frac <- 0.8
n_train    <- floor(train_frac * n)

idx_train <- idx[1:n_train]
idx_val   <- idx[(n_train + 1):n]

X_train <- X_scaled[idx_train, , drop = FALSE]
X_val   <- X_scaled[idx_val,   , drop = FALSE]

y_train <- y_cat[idx_train, , drop = FALSE]
y_val   <- y_cat[idx_val,   , drop = FALSE]

# ======================================================================
# 7. Define MLP
# ======================================================================

input_dim <- ncol(X_train)

model <- keras_model_sequential() %>%
  layer_dense(
    units = 64, activation = "relu",
    input_shape = input_dim
  ) %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = n_classes, activation = "softmax")

model %>% compile(
  loss      = "categorical_crossentropy",
  optimizer = optimizer_adam(learning_rate = 0.001),
  metrics   = "accuracy"
)

summary(model)

# ======================================================================
# 8. Fit model
# ======================================================================

history <- model %>% fit(
  x = X_train,
  y = y_train,
  epochs = 50,
  batch_size = 1024,
  validation_data = list(X_val, y_val),
  callbacks = list(
    callback_early_stopping(
      monitor = "val_loss",
      patience = 5,
      restore_best_weights = TRUE
    )
  )
)

plot(history)

# majority baseline accuracy
prop.table(table(train_dist$bin_label))
max(prop.table(table(train_dist$bin_label)))  # compare to ~0.445


# ======================================================================
# 9. Backcast for 1990 BAP (undisturbed pixels)
#    → need same temporal features: bands + 1-year lags + differences
# ======================================================================

# We already computed lag1 and diff1 for *all* rows in train_df above.
pred_1990 <- train_df[
  state == "undisturbed" &
    year == 1990 &
    bap_available == TRUE
]

# Require complete temporal features (have 1989 data)
pred_1990 <- pred_1990[complete.cases(pred_1990[, ..pred_cols])]

cat("Prediction samples (1990 undisturbed):", nrow(pred_1990), "\n")

X_new <- as.matrix(pred_1990[, ..pred_cols])
X_new_scaled <- scale(X_new, center = X_mean, scale = X_sd)

pred_prob <- model %>% predict(X_new_scaled)
pred_idx  <- apply(pred_prob, 1, which.max)
pred_class <- factor(class_levels[pred_idx], levels = class_levels)

pred_1990$pred_ysd_bin5 <- pred_class

head(pred_1990[, .(id, x, y, year, pred_ysd_bin5)])

# ======================================================================
# 10. Save model and metadata
# ======================================================================

save_model_hdf5(model, "mlp_backcast_ysd_bins5_temporal.h5")

saveRDS(
  list(
    mean      = X_mean,
    sd        = X_sd,
    levels    = class_levels,
    pred_cols = pred_cols
  ),
  file = "mlp_backcast_meta_bins5_temporal.rds"
)

#------------------------------------------------------------------------------
# 11. Apply model

# ======================================================================
# 0. Libraries + Model
# ======================================================================

library(terra)
library(keras)
library(data.table)

model_path <- "/mnt/eo/EO4Backcasting/_models/mlp_trajectory_ysd_bins5.h5"
meta_path  <- "/mnt/eo/EO4Backcasting/_models/mlp_trajectory_ysd_bins5_meta.rds"

model <- load_model_hdf5(model_path)
meta  <- readRDS(meta_path)

X_mean     <- meta$mean
X_sd       <- meta$sd
pred_cols  <- meta$pred_cols       # expected 40 names
ysd_sel    <- meta$ysd_sel         # e.g., c(1,3,5,10,15)
class_lvls <- meta$levels

message("Loaded model with ", length(pred_cols), " predictor variables.")


# ======================================================================
# 1. Tile setup
# ======================================================================

tile_dir   <- "/mnt/dss_europe/level3_interpolated/X0016_Y0020/"
out_raster <- "/mnt/eo/EO4Backcasting/_predictions/YSDbin_X0016_Y0020.tif"

dir.create(dirname(out_raster), recursive = TRUE, showWarnings = FALSE)


# ======================================================================
# 2. Load rasters (IBAP, NBR, EVI)
# ======================================================================

extract_year <- function(f) as.integer(substr(basename(f), 1, 4))

# IBAP
ibap_files <- list.files(tile_dir, pattern="IBAP\\.tif$", full.names=TRUE)
stopifnot(length(ibap_files) > 0)
ibap_years <- extract_year(ibap_files)
ord <- order(ibap_years)
ibap_files <- ibap_files[ord]
ibap_years <- ibap_years[ord]
ibap_list  <- lapply(ibap_files, rast)
names(ibap_list) <- as.character(ibap_years)

# NBR
nbr_files <- list.files(tile_dir, pattern="NBR\\.tif$", full.names=TRUE)
stopifnot(length(nbr_files) > 0)
nbr_years <- extract_year(nbr_files)
nbr_list  <- lapply(nbr_files, rast)
names(nbr_list) <- as.character(nbr_years)

# EVI
evi_files <- list.files(tile_dir, pattern="EVI\\.tif$", full.names=TRUE)
stopifnot(length(evi_files) > 0)
evi_years <- extract_year(evi_files)
evi_list  <- lapply(evi_files, rast)
names(evi_list) <- as.character(evi_years)


# ======================================================================
# 3. Hard alignment — this solves the NULL / empty block problem
# ======================================================================

t0  <- min(ibap_years)
ref <- ibap_list[[as.character(t0)]]   # full reference grid

align_to_ref <- function(r, ref) {
  r2 <- terra::resample(r, ref, method = "bilinear")  # match resolution + grid
  r2 <- terra::crop(r2, ref)                          # force same extent
  r2 <- terra::extend(r2, ref)                        # pad if needed
  r2 <- terra::mask(r2, ref)                          # enforce same NA mask
  return(r2)
}

ibap_list_aligned <- lapply(ibap_list, align_to_ref, ref = ref)
nbr_list_aligned  <- lapply(nbr_list,  align_to_ref, ref = ref)
evi_list_aligned  <- lapply(evi_list,  align_to_ref, ref = ref)


# ======================================================================
# 4. Determine required years for trajectory
# ======================================================================

years_needed <- t0 + ysd_sel

stopifnot(all(as.character(years_needed) %in% names(ibap_list_aligned)))
stopifnot(all(as.character(years_needed) %in% names(nbr_list_aligned)))
stopifnot(all(as.character(years_needed) %in% names(evi_list_aligned)))

message("Using t0 = ", t0, " → using years: ", paste(years_needed, collapse=", "))


# ======================================================================
# 5. Build predictor stack (aligned + named correctly)
# ======================================================================

message("Building predictor stack...")

pred_stack <- rast()

for (i in seq_along(ysd_sel)) {
  
  yr  <- as.character(years_needed[i])
  ysd <- ysd_sel[i]
  
  # IBAP = 6 bands
  ib <- ibap_list_aligned[[yr]]
  names(ib) <- sprintf("b%d_%d", 1:6, ysd)
  
  # NBR = 1 band
  nbr <- nbr_list_aligned[[yr]]
  names(nbr) <- sprintf("NBR_%d", ysd)
  
  # EVI = 1 band
  evi <- evi_list_aligned[[yr]]
  names(evi) <- sprintf("EVI_%d", ysd)
  
  pred_stack <- c(pred_stack, ib, nbr, evi)
}

message("Built stack with ", nlyr(pred_stack), " layers.")


# ======================================================================
# 6. Reorder to model input order
# ======================================================================

message("Reordering layers...")
pred_stack <- pred_stack[[pred_cols]]
stopifnot(all(names(pred_stack) == pred_cols))
message("Band order correct ✔")


# ======================================================================
# 7. Prepare output raster
# ======================================================================

template <- ref
out <- template[[1]]
values(out) <- NA
names(out) <- "pred_ysd_bin"


# ======================================================================
# 8. Block-wise prediction (safe)
# ======================================================================

message("Starting block-wise prediction...")

bs <- blocks(pred_stack, n = 20)

readStart(pred_stack)
out <- writeStart(out, out_raster, overwrite = TRUE)

for (i in seq_len(bs$n)) {
  
  cat("Block", i, "of", bs$n, "\n")
  
  M <- readValues(pred_stack, row = bs$row[i], nrows = bs$nrows[i])
  if (is.null(M) || ncol(M) == 0 || nrow(M) == 0) next
  
  X_new <- as.matrix(M)
  
  # scale
  X_scaled <- scale(X_new, center = X_mean, scale = X_sd)
  
  # predict
  pred_prob <- model %>% predict(X_scaled, verbose = 0)
  class_idx <- apply(pred_prob, 1, which.max)
  
  out <- writeValues(out, class_idx, bs$row[i], bs$nrows[i])
}

out <- writeStop(out)
readStop(pred_stack)

message("DONE ✔ Saved:\n", out_raster)



