library(keras)
library(dplyr)
library(tensorflow)
library(reticulate)
library(data.table)

#------------------------------------------------------------------------------
## ---- TensorFlow / keras setup ----
library(reticulate)
use_virtualenv("r-tf-mlp", required = TRUE)

# Quick test:
tf$constant(1)
py_config()


#-------------------------------------------------------------------------------
# for reproducibility
set.seed(1234)

# load training data
TRAIN_CSV <- "/mnt/eo/EO4Backcasting/_intermediates/training_data.csv"
train_df <- fread(TRAIN_CSV)

# ------------------------------------------------------------------------------
# 2. Prepare training data: disturbed pixels + 5-year ysd bins
# ------------------------------------------------------------------------------

train_dist <- train_df %>%
  filter(
    state == "disturbed",
    bap_available,
    !is.na(ysd)
  ) %>%
  mutate(
    # 5-year bins based on ysd
    bin_label = dplyr::case_when(
      ysd >= 0  & ysd <= 5   ~ "ysd1_5",
      ysd >= 6  & ysd <= 10  ~ "ysd6_10",
      ysd >= 11 & ysd <= 15  ~ "ysd11_15",
      ysd >= 16 & ysd <= 20  ~ "ysd16_20",
      ysd > 20               ~ "ysd_20",
      TRUE                   ~ NA_character_
    )
  ) %>%
  filter(!is.na(bin_label))

# ------------------------------------------------------------------------------
# 3. Build predictor matrix (bands only) and target (one-hot)
# ------------------------------------------------------------------------------

pred_cols <- c("b1", "b2", "b3", "b4", "b5", "b6")

X <- train_dist %>%
  select(all_of(pred_cols)) %>%
  as.matrix()

# standardize features
X_mean <- colMeans(X, na.rm = TRUE)
X_sd   <- apply(X, 2, sd, na.rm = TRUE)
X_scaled <- scale(X, center = X_mean, scale = X_sd)

# target: 5 bins as factor
y_factor     <- factor(train_dist$bin_label)
class_levels <- levels(y_factor)
n_classes    <- length(class_levels)  # should be 5

y_int <- as.integer(y_factor) - 1L
y_cat <- to_categorical(y_int, num_classes = n_classes)

# ------------------------------------------------------------------------------
# 4. Train–validation split
# ------------------------------------------------------------------------------

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

# ------------------------------------------------------------------------------
# 5. Define MLP
# ------------------------------------------------------------------------------

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

# ------------------------------------------------------------------------------
# 6. Fit model
# ------------------------------------------------------------------------------

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

# ------------------------------------------------------------------------------
# 7. Backcast for 1990 BAP (undisturbed pixels, bands only)
# ------------------------------------------------------------------------------

pred_1990 <- train_df %>%
  filter(
    state == "undisturbed",
    year == 1990,
    bap_available
  ) %>%
  select(id, x, y, year, all_of(pred_cols))

X_new <- pred_1990 %>%
  select(all_of(pred_cols)) %>%
  as.matrix()

X_new_scaled <- scale(X_new, center = X_mean, scale = X_sd)

pred_prob <- model %>% predict(X_new_scaled)
pred_idx  <- apply(pred_prob, 1, which.max)

pred_class <- factor(class_levels[pred_idx], levels = class_levels)

pred_1990$pred_ysd_bin5 <- pred_class

head(pred_1990)

# ------------------------------------------------------------------------------
# 8. Save model and scale metadata
# ------------------------------------------------------------------------------

save_model_hdf5(model, "mlp_backcast_ysd_bins5_bands_only.h5")

saveRDS(
  list(
    mean      = X_mean,
    sd        = X_sd,
    levels    = class_levels,
    pred_cols = pred_cols
  ),
  file = "mlp_backcast_meta_bins5_bands_only.rds"
)