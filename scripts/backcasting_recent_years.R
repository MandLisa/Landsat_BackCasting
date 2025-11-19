#!/usr/bin/env R

# ===============================================================
# 0. Libraries
# ===============================================================
library(data.table)
library(ranger)
library(caret)

# ===============================================================
# 1. Load long-format BAP dataset
#    Structure expected:
#    ID, year, blue, green, red, nir, swir1, swir2, yod, state, ...
# ===============================================================

DT <- fread("/mnt/eo/EO4Backcasting/_intermediates/training_healthy_disturbed_1911_final.csv")

# Ensure ordering
setorder(DT, ID, year)

# ===============================================================
# 2. Define temporal window
#    Input: 6 consecutive years   (t0...t5)
#    Target: disturbances in the 5 years before window (t-1...t-5)
# ===============================================================

input_lags  <- 0:5      # 6 years of BAP
target_lags <- 1:5      # 5 disturbance years to predict
bands       <- c("blue","green","red","nir","swir1","swir2")


# ===============================================================
# 3. Add binary disturbance for each year: dist_year = (year == yod)
# ===============================================================

DT[, dist := as.integer(year == yod)]


# ===============================================================
# 4. BUILD SEQUENCES per pixel (core function)
#
# For each pixel:
#   - Create input features: BAP bands for t0..t5
#   - Create output labels: disturbance in t-1..t-5
# ===============================================================

make_sequences <- function(df) {
  
  n <- nrow(df)
  
  # Too short → no sequences possible
  if (n < max(input_lags) + max(target_lags) + 1) {
    return(NULL)
  }
  
  out <- list()
  
  for (i in seq_len(n)) {
    
    # Target years must exist
    if (i <= max(target_lags)) next
    
    # Input window indices
    idx_in <- (i - max(input_lags)) : i
    if (min(idx_in) < 1 || max(idx_in) > n) next
    
    chunk_in <- df[idx_in]
    
    if (nrow(chunk_in) != length(input_lags)) next
    
    # Target indices (t-1..t-5)
    idx_tar <- i - target_lags
    if (min(idx_tar) < 1) next
    
    chunk_tar <- df[idx_tar]
    
    # --- Build input feature vector ---
    Xvals <- as.vector(t(chunk_in[, ..bands]))
    
    # Safety check: skip if incomplete
    if (length(Xvals) != length(bands) * length(input_lags)) next
    
    X <- as.list(Xvals)
    names(X) <- paste0(
      rep(bands, each = length(input_lags)),
      "_t", rep(input_lags, times = length(bands))
    )
    
    # --- Build target labels ---
    Y <- as.list(chunk_tar$dist)
    names(Y) <- paste0("dist_t", target_lags)
    
    out[[length(out) + 1]] <- c(
      list(ID = df$ID[1], ref_year = df$year[i]),
      X, Y
    )
  }
  
  # If no valid sequences → return NULL
  if (length(out) == 0) return(NULL)
  
  # Filter out empty or NULL entries
  out <- out[sapply(out, function(x) !is.null(x) && length(x) > 0)]
  
  # Still empty?
  if (length(out) == 0) return(NULL)
  
  # Now all entries are valid rows
  rbindlist(out, fill = TRUE)
}

# Apply function to each pixel
SEQ <- DT[, make_sequences(.SD), by = ID]

# ===============================================================
# 5. TRAIN RANDOM FOREST MODELS for each target (t-1...t-5)
# ===============================================================

input_cols <- grep("_t[0-5]$", names(SEQ), value = TRUE)

models <- list()

for (lag in target_lags) {
  
  target_col <- paste0("dist_t", lag)
  
  message("Training model for ", target_col)
  
  dat <- SEQ[!is.na(get(target_col)), c(input_cols, target_col), with = FALSE]
  dat[, (target_col) := factor(get(target_col))]
  
  rf <- ranger(
    formula = as.formula(paste0(target_col, " ~ .")),
    data = dat,
    num.trees = 800,
    mtry = 12,
    probability = TRUE,
    importance = "impurity"
  )
  
  models[[target_col]] <- rf
}

# ===============================================================
# 6. EVALUATION: Predict back disturbance for recent years
#    (internal validation)
# ===============================================================

accuracy_results <- list()

for (lag in target_lags) {
  
  target_col <- paste0("dist_t", lag)
  message("Validating model ", target_col)
  
  dat <- SEQ[!is.na(get(target_col)), c(input_cols, target_col), with = FALSE]
  dat[, (target_col) := factor(get(target_col))]
  
  pred <- predict(models[[target_col]], dat[, input_cols, with=FALSE])$predictions
  dat[, pred_label := colnames(pred)[max.col(pred)]]
  
  cm <- confusionMatrix(dat$pred_label, dat[[target_col]])
  accuracy_results[[target_col]] <- cm
  
  print(cm)
}


# ===============================================================
# 7. SAVE MODELS (optional)
# ===============================================================

saveRDS(models, "/mnt/eo/EO4Backcasting/_models/rf_sequence_models.rds")


# ===============================================================
# DONE
# ===============================================================

message("Sequence-based RF training and validation finished.")
