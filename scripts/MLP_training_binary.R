# ======================================================================
# 0. Setup
# ======================================================================

library(data.table)
library(dplyr)
library(keras)
library(tensorflow)
library(pROC)
set.seed(1234)

# ======================================================================
# 1. Load training data
# ======================================================================

TRAIN_CSV <- "/mnt/eo/EO4Backcasting/_intermediates/training_data.csv"
train_df  <- fread(TRAIN_CSV)
setDT(train_df)

# ======================================================================
# 2. Define binary disturbance-age target (<15 vs >=15)
# ======================================================================

train_df[, y_binary := ifelse(ysd < 15, 1L, 0L)]
# restrict to disturbed for training
train_dist <- train_df[state == "disturbed" & bap_available == TRUE]

# ======================================================================
# 3. BAP-only predictor set
# ======================================================================

bap_cols <- c("b1","b2","b3","b4","b5","b6")

train_dist <- train_dist[complete.cases(train_dist[, ..bap_cols])]

cat("Training samples: ", nrow(train_dist), "\n")
print(table(train_dist$y_binary))

# Predictor matrix
X <- as.matrix(train_dist[, ..bap_cols])

# Standardization
X_mean <- colMeans(X)
X_sd   <- apply(X, 2, sd)
X_scaled <- scale(X, center = X_mean, scale = X_sd)

# Target vector
y <- train_dist$y_binary

# ======================================================================
# 4. Train/validation split
# ======================================================================

set.seed(1234)
n   <- nrow(X_scaled)
idx <- sample(seq_len(n))

n_train <- floor(0.8 * n)
idx_train <- idx[1:n_train]
idx_val   <- idx[(n_train+1):n]

X_train <- X_scaled[idx_train, , drop = FALSE]
X_val   <- X_scaled[idx_val, , drop = FALSE]

y_train <- y[idx_train]
y_val   <- y[idx_val]

# Convert to numeric {0,1} for Keras
y_train_tf <- as.numeric(y_train)
y_val_tf   <- as.numeric(y_val)

# ======================================================================
# 5. MLP model definition (simple + robust)
# ======================================================================

input_dim <- ncol(X_train)
model <- keras_model_sequential() %>%
  layer_dense(units = 32, activation = "relu", input_shape = input_dim) %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_adam(learning_rate = 0.001),
  metrics = "accuracy"
)

summary(model)

# ======================================================================
# 6. Fit model
# ======================================================================

history <- model %>% fit(
  x = X_train,
  y = y_train_tf,
  epochs = 50,
  batch_size = 1024,
  validation_data = list(X_val, y_val_tf),
  callbacks = list(
    callback_early_stopping(
      monitor = "val_loss",
      patience = 5,
      restore_best_weights = TRUE
    )
  )
)

plot(history)

# ======================================================================
# 7. Validation evaluation (ACC + ROC)
# ======================================================================

# Probabilities
val_prob <- model %>% predict(X_val)

# Accuracy with threshold 0.5
val_pred <- ifelse(val_prob > 0.5, 1, 0)
acc <- mean(val_pred == y_val)
cat("\nValidation accuracy:", acc, "\n")

# Confusion matrix
cat("\nConfusion matrix:\n")
print(table(pred = val_pred, true = y_val))

# ROC + AUC
roc_obj <- roc(response = y_val, predictor = as.numeric(val_prob))
auc_val <- auc(roc_obj)
cat("\nAUC =", auc_val, "\n")

plot(roc_obj, col="darkblue", lwd=3, main="ROC Curve (BAP-only MLP)")


# ======================================================================
# 8. SHAP analysis for Keras models (corrected)
# ======================================================================

library(iml)

cat("\nComputing SHAP (corrected version)…\n\n")

# ---- 1. Create prediction wrapper ----
predict_fun <- function(model, newdata) {
  as.numeric(model %>% predict(as.matrix(newdata)))
}

# ---- 2. Prepare SHAP data ----
df_scaled <- as.data.frame(X_scaled)
colnames(df_scaled) <- bap_cols

# subsample for speed
set.seed(42)
sel <- sample(nrow(df_scaled), 300)
df_small <- df_scaled[sel, ]
y_small  <- y[sel]

# ---- 3. Build Predictor ----
predictor <- Predictor$new(
  model = model,
  data  = df_small,
  y     = y_small,
  predict.function = predict_fun,
  type = "prob"
)

# ---- 4. Compute SHAP values per feature ----
shap_values_list <- lapply(1:200, function(i) {
  Shapley$new(predictor, x.interest = df_small[i, ])$results$phi
})

shap_matrix <- do.call(rbind, shap_values_list)

global_shap <- colMeans(abs(shap_matrix))
global_shap_df <- data.frame(
  feature = bap_cols,
  importance = global_shap
)

# ---- 5. Plot ----
barplot(
  global_shap_df$importance,
  names.arg = global_shap_df$feature,
  las = 2,
  cex.names = 0.9,
  ylab = "Mean |SHAP value|",
  main = "Global SHAP Importance"
)

# ======================================================================
# 8. Save model + metadata
# ======================================================================

save_model_hdf5(model, "/mnt/eo/EO4Backcasting/_models/mlp_bap_only_binary.h5")

saveRDS(
  list(
    mean      = X_mean,
    sd        = X_sd,
    pred_cols = bap_cols
  ),
  file = "/mnt/eo/EO4Backcasting/_models/mlp_bap_only_binary_meta.rds"
)

cat("\nModel + metadata saved.\n")





