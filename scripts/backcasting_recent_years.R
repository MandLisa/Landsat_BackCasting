#!/usr/bin/env R

# ===============================================================
# 0. Libraries
# ===============================================================
library(data.table)
library(ranger)
library(caret)
library(pROC)

# ===============================================================
# 1. Load long-format BAP dataset
#    Structure expected:
#    ID, year, blue, green, red, nir, swir1, swir2, yod, state, ...
# ===============================================================

DT <- fread("/mnt/eo/EO4Backcasting/_intermediates/training_healthy_disturbed_1911_final.csv")

# Ensure ordering
setorder(DT, ID, year)

# ==============================================================
# 1. DIST-Spalte erzeugen (binär!)
# ==============================================================

if (!"dist" %in% names(DT)) {
  message("Generating disturbance indicator 'dist' ...")
  DT[, dist := as.integer(year == yod)]
  DT[is.na(yod), dist := 0]   # undisturbed
}


# ===============================================================
# 2. Create multi-horizon targets: dist_t1 ... dist_t5
# ===============================================================
target_horizons <- 1:5

for (h in target_horizons) {
  colname <- paste0("dist_t", h)
  DT[, (colname) := as.integer(ysd == h)]
  DT[, (colname) := factor(get(colname), levels = c(0,1))]
}

# Predictor bands
band_cols <- c("blue","green","red","nir","swir1","swir2")

# ===============================================================
# 3. TRAIN/TEST split by pixel ID
# ===============================================================
set.seed(42)

IDs <- unique(DT$ID)
test_IDs  <- sample(IDs, size = 0.30 * length(IDs))  # 30% test data
train_IDs <- setdiff(IDs, test_IDs)

TRAIN <- DT[ID %in% train_IDs]
TEST  <- DT[ID %in% test_IDs]

message("TRAIN rows: ", nrow(TRAIN))
message("TEST rows : ", nrow(TEST))

# ===============================================================
# 4. TRAIN MODELS FOR EACH HORIZON
# ===============================================================
models <- list()

for (h in target_horizons) {
  
  target_col <- paste0("dist_t", h)
  message("\nTraining model for ", target_col)
  
  # Remove NA target rows
  TRAIN_h <- TRAIN[!is.na(get(target_col))]
  
  # Convert to factor
  TRAIN_h[[target_col]] <- factor(TRAIN_h[[target_col]], levels = c(0, 1))
  
  rf <- ranger(
    formula = as.formula(paste0(target_col, " ~ .")),
    data = TRAIN_h[, c(target_col, band_cols), with = FALSE],
    num.trees = 500,
    mtry = 3,
    probability = TRUE,
    importance = "impurity",
    seed = 42
  )
  
  models[[target_col]] <- rf
}

# ===============================================================
# 5. VALIDATION FUNCTION
# ===============================================================
validate_model <- function(model, test_df, target_col) {
  
  df <- copy(test_df[, c(target_col, band_cols), with=FALSE])
  df[[target_col]] <- factor(df[[target_col]], levels=c(0,1))
  
  preds <- predict(model, df[, ..band_cols])$predictions
  prob  <- preds[, "1"]
  pred_class <- factor(ifelse(prob > 0.5, 1, 0), levels=c(0,1))
  
  cm <- confusionMatrix(pred_class, df[[target_col]])
  auc_val <- auc(df[[target_col]], prob)
  
  list(
    cm = cm,
    auc = auc_val
  )
}

# ===============================================================
# 6. VALIDATE ALL MODELS
# ===============================================================
validation_results <- list()

for (h in target_horizons) {
  
  target_col <- paste0("dist_t", h)
  message("\nValidating ", target_col)
  
  res <- validate_model(models[[target_col]], TEST, target_col)
  validation_results[[target_col]] <- res
  
  print(res$cm)
  cat("AUC:", res$auc, "\n")
}

# ===============================================================
# 7. PERFORMANCE SUMMARY TABLE
# ===============================================================
summary_list <- list()

for (h in target_horizons) {
  
  target_col <- paste0("dist_t", h)
  res <- validation_results[[target_col]]
  
  cm <- res$cm
  auc_val <- res$auc
  
  summary_list[[target_col]] <- data.table(
    horizon = h,
    accuracy = cm$overall["Accuracy"],
    kappa = cm$overall["Kappa"],
    sensitivity = cm$byClass["Sensitivity"],
    specificity = cm$byClass["Specificity"],
    precision = cm$byClass["Pos Pred Value"],
    recall = cm$byClass["Sensitivity"],
    f1 = 2 * (cm$byClass["Pos Pred Value"] * cm$byClass["Sensitivity"]) /
      (cm$byClass["Pos Pred Value"] + cm$byClass["Sensitivity"]),
    AUC = auc_val
  )
}

perf_table <- rbindlist(summary_list)
print(perf_table)

# Save summary
fwrite(perf_table,
       "/mnt/eo/EO4Backcasting/_models/backcast_performance_summary.csv")

# ===============================================================
# 8. SAVE TRAINED MODELS
# ===============================================================
OUTDIR <- "/mnt/eo/EO4Backcasting/_models/"
dir.create(OUTDIR, showWarnings = FALSE)

for (h in target_horizons) {
  
  target_col <- paste0("dist_t", h)
  
  saveRDS(models[[target_col]],
          paste0(OUTDIR, "rf_model_", target_col, ".rds"))
}

message("\n=============================")
message(" Training + Validation DONE ")
message("=============================")