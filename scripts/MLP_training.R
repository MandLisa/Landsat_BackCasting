# ======================================================================
# 0. Packages and TensorFlow / Keras setup
# ======================================================================

library(data.table)
library(dplyr)
library(reticulate)

library(reticulate)

# Reset reticulate completely
Sys.unsetenv("RETICULATE_PYTHON")
Sys.setenv(RETICULATE_MINICONDA_PATH = "")

# Prevent automatic Python selection
reticulate::py_discover_config(required_module = NULL)

use_virtualenv("~/.virtualenvs/r-tf-mlp", required = TRUE)
py_module_available("tensorflow")



library(tensorflow)
library(keras)

tf$constant(1)
py_config()
set.seed(1234)

# ======================================================================
# 0. Packages + TensorFlow / Keras setup
# ======================================================================

library(data.table)
library(dplyr)
library(reticulate)

# Make sure reticulate uses the correct Python env
use_virtualenv("~/.virtualenvs/r-tf-mlp", required = TRUE)

library(tensorflow)
library(keras)

tf$constant(1)     # quick sanity check
py_config()

set.seed(1234)

# ======================================================================
# 1. Load training data
# ======================================================================

TRAIN_CSV <- "/mnt/eo/EO4Backcasting/_intermediates/training_data.csv"
train_df  <- fread(TRAIN_CSV)
setDT(train_df)

# ======================================================================
# 2. Define 5-year bins
# ======================================================================

train_df[, bin_label := fcase(
  ysd >= 0  & ysd <= 5,   "ysd1_5",
  ysd >= 6  & ysd <= 10,  "ysd6_10",
  ysd >= 11 & ysd <= 15,  "ysd11_15",
  ysd >= 16 & ysd <= 20,  "ysd16_20",
  ysd > 20,               "ysd_20",
  default = NA_character_
)]

print(train_df[state=="disturbed" & !is.na(bin_label), table(bin_label)])

# ======================================================================
# 3. Add BAP temporal features (lag1 + diff1)
# ======================================================================

band_cols <- c("b1","b2","b3","b4","b5","b6")

setorder(train_df, id, year)

# --- 1-year lag ---
for (col in band_cols) {
  lag_name <- paste0(col, "_lag1")
  train_df[, (lag_name) := shift(get(col), 1L, type="lag"), by=id]
}

# --- 1-year difference ---
for (col in band_cols) {
  diff_name <- paste0(col, "_diff1")
  lag_name  <- paste0(col, "_lag1")
  train_df[, (diff_name) := get(col) - get(lag_name)]
}

# ======================================================================
# 4. Filter disturbed training samples with complete predictors
# ======================================================================

pred_cols <- c(
  band_cols,
  paste0(band_cols, "_lag1"),
  paste0(band_cols, "_diff1")
)

train_dist <- train_df[
  state == "disturbed" &
    bap_available &
    !is.na(bin_label)
]

train_dist <- train_dist[complete.cases(train_dist[, ..pred_cols])]

cat("Training samples:", nrow(train_dist), "\n")

# ======================================================================
# 5. Prepare X and y matrices
# ======================================================================

X <- as.matrix(train_dist[, ..pred_cols])

# standardize
X_mean <- colMeans(X)
X_sd   <- apply(X, 2, sd)
X_scaled <- scale(X, center=X_mean, scale=X_sd)

# target
y_factor     <- factor(train_dist$bin_label)
class_levels <- levels(y_factor)
n_classes    <- length(class_levels)

y_int <- as.integer(y_factor) - 1L
y_cat <- to_categorical(y_int, num_classes=n_classes)

# ======================================================================
# 6. Train / Validation split
# ======================================================================

set.seed(1234)
n   <- nrow(X_scaled)
idx <- sample(seq_len(n))

n_train <- floor(0.8 * n)
idx_train <- idx[1:n_train]
idx_val   <- idx[(n_train+1):n]

X_train <- X_scaled[idx_train, ]
X_val   <- X_scaled[idx_val, ]

y_train <- y_cat[idx_train, ]
y_val   <- y_cat[idx_val, ]

# ======================================================================
# 7. Define the MLP
# ======================================================================

input_dim <- ncol(X_train)

model <- keras_model_sequential() %>%
  layer_dense(units=64, activation="relu",
              input_shape=input_dim) %>%
  layer_dropout(0.3) %>%
  layer_dense(units=64, activation="relu") %>%
  layer_dropout(0.3) %>%
  layer_dense(units=n_classes, activation="softmax")

model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_adam(0.001),
  metrics = "accuracy"
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
      monitor="val_loss",
      patience=5,
      restore_best_weights=TRUE
    )
  )
)

plot(history)

message("Majority-class baseline: ",
        round(max(prop.table(table(train_dist$bin_label))), 3))

# ======================================================================
# 9. Backcast on undisturbed pixels (example: year=1990)
# ======================================================================

pred_1990 <- train_df[
  state=="undisturbed" &
    year==1990 &
    bap_available
]

pred_1990 <- pred_1990[complete.cases(pred_1990[, ..pred_cols])]

X_new <- as.matrix(pred_1990[, ..pred_cols])
X_new_scaled <- scale(X_new, center=X_mean, scale=X_sd)

pred_prob <- model %>% predict(X_new_scaled)
pred_idx  <- apply(pred_prob, 1, which.max)

pred_1990$pred_ysd_bin5 <- factor(class_levels[pred_idx],
                                  levels=class_levels)

head(pred_1990[, .(id, x, y, year, pred_ysd_bin5)])

# ======================================================================
# 10. Save model + metadata
# ======================================================================

save_model_hdf5(model, "mlp_backcast_ysd_bins5_BAPonly.h5")

saveRDS(
  list(
    mean      = X_mean,
    sd        = X_sd,
    levels    = class_levels,
    pred_cols = pred_cols
  ),
  "mlp_backcast_meta_BAPonly.rds"
)

message("Model & metadata saved.")
