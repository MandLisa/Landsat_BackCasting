# install.packages("keras")
library(keras)
library(dplyr)

# for reproducibility
set.seed(1234)

# load training data
TRAIN_CSV <- "/mnt/eo/EO4Backcasting/_intermediates/training_data.csv"
train_df <- fread(TRAIN_CSV)

### 2. Prepare training data
# restrict to disturbed pixels with valid class label
train_dist <- train_df %>%
  filter(
    state == "disturbed",
    bap_available,
    !is.na(class_label)
  )

### 3. Build predictor matrix and target (one-hot encoded)
# predictor columns
pred_cols <- c("EVI", "NBR", "b1", "b2", "b3", "b4", "b5", "b6")

X <- train_dist %>%
  select(all_of(pred_cols)) %>%
  as.matrix()

# standardize features (mean 0, sd 1)
X_mean <- colMeans(X, na.rm = TRUE)
X_sd   <- apply(X, 2, sd, na.rm = TRUE)

X_scaled <- scale(X, center = X_mean, scale = X_sd)

# target: ysd class as factor
y_factor <- factor(train_dist$class_label)
class_levels <- levels(y_factor)
n_classes <- length(class_levels)

# keras wants integers 0..(n_classes-1)
y_int <- as.integer(y_factor) - 1L
y_cat <- to_categorical(y_int, num_classes = n_classes)


### 4. Train - validation split
set.seed(1234)
n <- nrow(X_scaled)
idx <- sample(seq_len(n))

train_frac <- 0.8
n_train <- floor(train_frac * n)

idx_train <- idx[1:n_train]
idx_val   <- idx[(n_train + 1):n]

X_train <- X_scaled[idx_train, , drop = FALSE]
X_val   <- X_scaled[idx_val,   , drop = FALSE]

y_train <- y_cat[idx_train, , drop = FALSE]
y_val   <- y_cat[idx_val,   , drop = FALSE]


### 5. Define MLP
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
  loss = "categorical_crossentropy",
  optimizer = optimizer_adam(learning_rate = 0.001),
  metrics = "accuracy"
)

summary(model)


### 6. Fit model
history <- model %>% fit(
  x = X_train,
  y = y_train,
  epochs = 50,                  # adapt as needed
  batch_size = 1024,            # adapt to data size
  validation_data = list(X_val, y_val),
  callbacks = list(
    callback_early_stopping(
      monitor = "val_loss",
      patience = 5,
      restore_best_weights = TRUE
    )
  )
)

# optional: inspect training
plot(history)


### 7. Backcast for 1990 BAP
# data frame with undisturbed pixels in 1990
pred_1990 <- train_df %>%
  filter(
    state == "undisturbed",
    year == 1990,
    bap_available
  ) %>%
  # keep coordinates & predictors
  select(id, x, y, year, all_of(pred_cols))

# predictor matrix (apply same scaling as training!)
X_new <- pred_1990 %>%
  select(all_of(pred_cols)) %>%
  as.matrix()

X_new_scaled <- scale(X_new, center = X_mean, scale = X_sd)

# predict class probabilities
pred_prob <- model %>% predict(X_new_scaled)

# predicted class index per pixel
pred_idx <- apply(pred_prob, 1, which.max)

# map to factor levels (ysd bins)
pred_class <- factor(class_levels[pred_idx], levels = class_levels)

# attach to dataframe
pred_1990$pred_ysd_bin <- pred_class

head(pred_1990)


### 8. Save model and scale metadata
save_model_hdf5(model, "mlp_backcast_ysd_bins.h5")

saveRDS(
  list(
    mean = X_mean,
    sd = X_sd,
    levels = class_levels,
    pred_cols = pred_cols
  ),
  file = "mlp_backcast_meta.rds"
)

