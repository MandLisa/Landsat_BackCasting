# =====================================================================
# 0. Libraries
# =====================================================================
library(data.table)
library(dplyr)
library(ranger)
library(terra)
library(pROC)
library(tidyr)
library(ggplot2)

set.seed(1234)

# =====================================================================
# 1. Load data
# =====================================================================

TRAIN_CSV <- "/mnt/eo/EO4Backcasting/_intermediates/training_data_topo.csv"
df <- fread(TRAIN_CSV)

# Only keep rows with valid BAP
df <- df[df$bap_available == TRUE]

# =====================================================================
# 2. Create target variable (binary)
# =====================================================================

df <- df %>%
  mutate(
    target = case_when(
      state == "undisturbed"           ~ 0L,
      state == "disturbed" & ysd <= 15 ~ 1L,
      state == "disturbed" & ysd > 15  ~ 0L,
      TRUE ~ NA_integer_
    )
  )

df <- df %>% filter(!is.na(target))

pred_cols <- c("b1","b2","b3","b4","b5","b6")

df <- df %>% drop_na(all_of(pred_cols))

cat("Training rows:", nrow(df), "\n")
print(table(df$target))


# =====================================================================
# 3. Train/validation split
# =====================================================================

set.seed(1234)
idx <- sample(seq_len(nrow(df)))
n_train <- floor(0.8 * nrow(df))

train_df <- df[idx[1:n_train], ]
val_df   <- df[idx[(n_train+1):nrow(df)], ]


# =====================================================================
# 4. Ranger RF model
# =====================================================================

rf <- ranger(
  formula = as.factor(target) ~ b1 + b2 + b3 + b4 + b5 + b6,
  data    = train_df,
  num.trees   = 500,
  mtry        = 3,
  probability = TRUE,
  importance  = "impurity"
)

print(rf)


# =====================================================================
# 5. Validation
# =====================================================================

# probability of class "1" = recent disturbance
val_prob <- predict(rf, val_df)$predictions[,2]

# class threshold 0.5
val_pred <- ifelse(val_prob > 0.75, 1, 0)

acc <- mean(val_pred == val_df$target)
cat("\nValidation accuracy:", acc, "\n")

cat("\nConfusion matrix:\n")
print(table(pred=val_pred, true=val_df$target))

roc_obj <- roc(val_df$target, val_prob)
cat("\nAUC:", auc(roc_obj), "\n")


# =====================================================================
# 6. Save model
# =====================================================================

saveRDS(rf, "/mnt/eo/EO4Backcasting/_models/rf_bap_binary.rds")
saveRDS(list(pred_cols = pred_cols),
        "/mnt/eo/EO4Backcasting/_models/rf_bap_recentdist_binary_meta.rds")

cat("\nModel + metadata saved.\n")


# =====================================================================
# 7. Threshold evaluation 
# =====================================================================

cat("\n========================================\n")
cat("Threshold optimisation...\n")
cat("========================================\n")

y_val <- val_df$target
p_val <- val_prob

# Ensure correct length
stopifnot(length(y_val) == length(p_val))

# ---------------------------------------------------------------------
# Metrics function
# ---------------------------------------------------------------------
compute_metrics <- function(t, y, p) {
  pred <- ifelse(p >= t, 1, 0)
  
  TP <- sum(pred == 1 & y == 1)
  FP <- sum(pred == 1 & y == 0)
  FN <- sum(pred == 0 & y == 1)
  TN <- sum(pred == 0 & y == 0)
  
  precision <- ifelse((TP + FP) == 0, NA, TP / (TP + FP))
  recall    <- ifelse((TP + FN) == 0, NA, TP / (TP + FN))
  CE        <- ifelse((TP + FP) == 0, NA, FP / (TP + FP))   # Commission error
  OE        <- ifelse((TP + FN) == 0, NA, FN / (TP + FN))   # Omission error
  acc       <- (TP + TN) / (TP + FP + FN + TN)
  
  data.frame(
    threshold = t,
    precision = precision,
    recall    = recall,
    CE        = CE,
    OE        = OE,
    accuracy  = acc
  )
}

# ---------------------------------------------------------------------
# Evaluate thresholds 0.01–0.99
# ---------------------------------------------------------------------
thresholds <- seq(0.01, 0.99, by = 0.01)

metric_list <- purrr::map_df(
  thresholds,
  ~ compute_metrics(.x, y_val, p_val)
)

# ---------------------------------------------------------------------
# Plot curves
# ---------------------------------------------------------------------
metric_long <- metric_list %>%
  tidyr::pivot_longer(
    cols = c("precision", "recall", "CE", "OE"),
    names_to = "metric",
    values_to = "value"
  )

print(
  ggplot(metric_long, aes(x = threshold, y = value, color = metric)) +
    geom_line(size = 1.1) +
    theme_minimal(base_size = 14) +
    labs(
      title = "Threshold curves",
      y = "Value",
      x = "Threshold"
    ) +
    theme(legend.position = "bottom")
)


# -------------------------------------
# Load BAP raster (1990 example)
# -------------------------------------

library(terra)

# input BAP (now local)
INFILE <- "/mnt/eo/EO4Backcasting/_bap_local/19900801_LEVEL3_LNDLG_IBAP.tif"

r_bap <- rast(INFILE)
names(r_bap) <- pred_cols   

# --- random forest predict function ---
rf_fun <- function(d) {
  predict(rf, data.frame(d))$predictions[,2]
}

# --- probability output ---
prob_file <- "/mnt/eo/EO4Backcasting/_predictions/bap1990_prob.tif"
class_file <- "/mnt/eo/EO4Backcasting/_predictions/bap1990_class.tif"


rf_fun <- function(model, data, ...) {
  predict(rf, data.frame(data))$predictions[, 2]
}

prob_r <- terra::predict(
  r_bap,
  model = 1,           # <-- terra requires this!
  fun   = rf_fun,
  filename  = prob_file,
  overwrite = TRUE,
  wopt = list(datatype = "FLT4S")
)

threshold <- 0.75
class_r <- prob_r >= threshold


# --- classification output ---
class_file <- "/mnt/eo/EO4Backcasting/_predictions/bap1990_class_09.tif"

threshold <- 0.9
class_r <- prob_r >= threshold

writeRaster(class_r, class_file, overwrite = TRUE)

cat("Done.\n")
cat("Prob map:  ", prob_file, "\n")
cat("Class map: ", class_file, "\n")







#-------------------------------------------------------------------------------
# load forest mask
forest_mask <- rast("/mnt/eo/EFDA_v211/forest_landuse_aligned.tif")
forest_mask <- classify(forest_mask, cbind(0, NA))

# crop forest mask to r_bap extent
forest_crop <- crop(forest_mask, r_bap)

# mask the BAP with forest pixels only
r_bap_masked <- mask(r_bap, forest_crop)
#plot(r_bap_masked)

# prediction
rf_fun <- function(model, data, ...) {
  predict(rf, data.frame(data))$predictions[, 2]
}


prob_r <- terra::predict(
  r_bap_masked,
  model = 1,
  fun   = rf_fun,
  filename  = prob_file,
  overwrite = TRUE,
  wopt = list(datatype="FLT4S")
)



# ------------------------------------------------------------
# Probability raster already created:
# prob_r <- terra::predict(...)
# ------------------------------------------------------------

# Define multiple thresholds
thresholds <- c(0.5, 0.75,0.90)

# Output directory for threshold rasters
outdir <- "/mnt/eo/EO4Backcasting/_predictions/" 
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

# Loop over thresholds
for (thr in thresholds) {
  
  # classification raster
  class_r <- prob_r >= thr
  
  # build filename (e.g., class_thr_0.75.tif)
  outfile <- file.path(outdir, sprintf("class_thr_%0.2f.tif", thr))
  
  writeRaster(
    class_r,
    outfile,
    overwrite = TRUE,
    wopt = list(datatype = "INT1U")  # 0/1 raster
  )
  
  message(sprintf("Written: %s", outfile))
}


# classification
threshold <- 0.9
class_r <- prob_r >= threshold

writeRaster(class_r, class_file, overwrite=TRUE)


