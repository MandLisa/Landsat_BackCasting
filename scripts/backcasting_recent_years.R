#!/usr/bin/env R

# ===============================================================
# 0. Libraries
# ===============================================================
library(data.table)
library(ranger)
library(caret)
library(pROC)
library(terra)

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


# ---------------------------------------------------------------
# Load performance summary
# ---------------------------------------------------------------
perf <- fread("/mnt/eo/EO4Backcasting/_models/backcast_performance_summary.csv")

# ---------------------------------------------------------------
# Convert to long format for plotting
# ---------------------------------------------------------------
metrics <- c("accuracy", "sensitivity", "specificity", "precision", "f1", "AUC")

perf_long <- melt(
  perf,
  id.vars   = "horizon",
  measure.vars = metrics,
  variable.name = "metric",
  value.name = "value"
)

# Capitalize metric names for pretty plotting
pretty_names <- c(
  accuracy     = "Accuracy",
  sensitivity  = "Sensitivity",
  specificity  = "Specificity",
  precision    = "Precision",
  f1           = "F1-score",
  AUC          = "AUC"
)

perf_long[, metric := factor(metric, levels = metrics, labels = pretty_names)]


# ---------------------------------------------------------------
# Plot: Multi-panel validation metrics
# ---------------------------------------------------------------
p <- ggplot(perf_long, aes(x = horizon, y = value)) +
  geom_line(size = 1.2, color = "#1f77b4") +
  geom_point(size = 2.5, color = "#1f77b4") +
  scale_x_continuous(breaks = 1:5, labels = paste0("t-", 1:5)) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.1)) +
  facet_wrap(~ metric, ncol = 3, scales = "free_y") +
  labs(
    x = "Prediction Horizon",
    y = "Metric Value",
    title = "Backcasting Performance Across Horizons (t–1 … t–5)"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    strip.text = element_text(face = "bold", size = 12),
    plot.title = element_text(face = "bold", size = 16, margin = margin(0,0,10,0)),
    axis.title  = element_text(size = 12),
    axis.text   = element_text(size = 10)
  )

print(p)

# ---------------------------------------------------------------
# Save the plot
# ---------------------------------------------------------------
ggsave(
  "/mnt/eo/EO4Backcasting/_figs/backcast_validation_metrics.png",
  p,
  width = 10,
  height = 6,
  dpi = 300
)



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



#-------------------------------------------------------------------------------
### Model training

# ----------------------------------------------------
# Input paths
# ----------------------------------------------------
TILE <- "/mnt/dss_europe/level3_interpolated/X0016_Y0020/20100801_LEVEL3_LNDLG_IBAP.tif"
MASK <- "/mnt/eo/EFDA_v211/forest_landuse_aligned.tif"

OUT_TILE <- "/mnt/eo/EO4Backcasting/_tiles/X0016_Y0020_IBAP_forestonly.tif"
dir.create(dirname(OUT_TILE), showWarnings = FALSE)

# ----------------------------------------------------
# 1. Load tile + forest mask
# ----------------------------------------------------
r_tile <- rast(TILE)
r_mask <- rast(MASK)

# ----------------------------------------------------
# 2. Reproject/resample mask onto tile grid
# ----------------------------------------------------
# same CRS? If not, reproject
if (!same.crs(r_tile, r_mask)) {
  message("Reprojecting mask to tile CRS…")
  r_mask <- project(r_mask, r_tile, method = "near")
}

# same resolution and extent? If not, resample
if (!all(res(r_tile) == res(r_mask)) || !ext(r_tile) == ext(r_mask)) {
  message("Resampling mask to tile grid…")
  r_mask <- resample(r_mask, r_tile, method = "near")
}

# ----------------------------------------------------
# 3. Ensure mask is binary 1 (forest) / NA (non-forest)
# ----------------------------------------------------
r_mask01 <- classify(
  r_mask,
  rbind(
    c(-Inf, 0.5, NA),   # non-forest -> NA
    c(0.5,  Inf, 1)     # forest -> 1
  )
)

# ----------------------------------------------------
# 4. Apply forest mask to tile
# ----------------------------------------------------
# Multiply keeps only forest pixels; non-forest → NA
r_tile_forest <- r_tile * r_mask01

# ----------------------------------------------------
# 5. Save masked tile
# ----------------------------------------------------
writeRaster(
  r_tile_forest,
  OUT_TILE,
  overwrite = TRUE,
  wopt = list(
    datatype = "INT2S",
    gdal = c("COMPRESS=ZSTD", "PREDICTOR=2", "ZSTD_LEVEL=8")
  )
)

#-------------------------------------------------------------------------------
### Prediction


# -----------------------------------------------------
# PATHS
# -----------------------------------------------------
TILE <- "/mnt/eo/EO4Backcasting/_tiles/X0016_Y0020_IBAP_forestonly.tif"
MODEL_DIR <- "/mnt/eo/EO4Backcasting/_models"
OUT_DIR   <- "/mnt/eo/EO4Backcasting/_predictions_tiles"
dir.create(OUT_DIR, showWarnings = FALSE)

band_cols <- c("blue","green","red","nir","swir1","swir2")
horizons  <- 1:5

# -----------------------------------------------------
# LOAD MODELS
# -----------------------------------------------------
models <- list()
for (h in horizons) {
  models[[h]] <- readRDS(file.path(
    MODEL_DIR,
    paste0("rf_model_dist_t", h, ".rds")
  ))
}

# -----------------------------------------------------
# PREDICT FUNCTION (the ONLY one!)
# -----------------------------------------------------
predict_fun <- function(d, model) {
  df <- as.data.frame(d)
  p <- predict(model, df)$predictions[, "1"]
  return(p)
}

# -----------------------------------------------------
# PREDICTION
# -----------------------------------------------------
r <- rast(TILE)
names(r) <- band_cols

out_list <- list()

for (h in horizons) {
  
  message("Predicting horizon t-", h)
  
  out <- terra::predict(
    r,
    fun = function(d) predict_fun(d, models[[h]]),
    filename = file.path(OUT_DIR, paste0("X0016_Y0020_p_t", h, ".tif")),
    overwrite = TRUE,
    wopt = list(datatype="FLT4S")
  )
  
  names(out) <- paste0("p_dist_t", h)
  out_list[[h]] <- out
}

# stack
stack_out <- rast(out_list)
writeRaster(
  stack_out,
  file.path(OUT_DIR, "X0016_Y0020_prediction_stack.tif"),
  overwrite = TRUE
)

message("DONE.")