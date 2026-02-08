#!/usr/bin/env R

# ===============================================================
# Libraries
# ===============================================================
library(data.table)
library(ranger)
library(terra)
library(ggplot2)

# ===============================================================
# 1. Load training data
# ===============================================================
DT <- fread("/mnt/eo/EO4Backcasting/_intermediates/training_healthy_disturbed_1911_final.csv")
setorder(DT, ID, year)

band_cols <- c("blue","green","red","nir","swir1","swir2")

# Keep only rows with disturbance info
DT <- DT[!is.na(ysd)]

# ===============================================================
# 2. Create 5-class ysd response (only forward recovery info!)
# ===============================================================
DT[, class := fifelse(ysd >= 1  & ysd <= 5,   1,
                      fifelse(ysd >= 6  & ysd <= 10,  2,
                              fifelse(ysd >= 11 & ysd <= 15,  3,
                                      fifelse(ysd >= 16 & ysd <= 20,  4, 5))))]

DT$class <- factor(DT$class, levels = 1:5)

# Predictors = ONLY t0 (BAP at the observation year)
predictor_cols <- band_cols

# Remove incomplete rows
DT <- DT[complete.cases(DT[, predictor_cols, with=FALSE])]

# ===============================================================
# 3. Train/test split by ID
# ===============================================================
IDs <- unique(DT$ID)
set.seed(42)
test_IDs  <- sample(IDs, 0.30 * length(IDs))
train_IDs <- setdiff(IDs, test_IDs)

TRAIN <- DT[ID %in% train_IDs]
TEST  <- DT[ID %in% test_IDs]

cat("TRAINING rows:", nrow(TRAIN), "\n")
cat("TEST rows:", nrow(TEST), "\n")
print(table(TRAIN$class))

# ===============================================================
# 4. Train t0-only multiclass Random Forest
# ===============================================================
rf <- ranger(
  formula      = class ~ .,
  data         = TRAIN[, c("class", predictor_cols), with=FALSE],
  num.trees    = 600,
  mtry         = 3,
  importance   = "impurity",
  probability  = TRUE,
  seed         = 42
)

print(rf)

# ===============================================================
# 5. Validate model
# ===============================================================
pred_prob <- predict(rf, TEST[, predictor_cols, with=FALSE])$predictions
pred_class <- colnames(pred_prob)[max.col(pred_prob, ties.method="first")]
pred_class <- factor(pred_class, levels = levels(TEST$class))

acc <- mean(pred_class == TEST$class)
cat("\nOverall accuracy:", acc, "\n")

cm <- table(pred = pred_class, true = TEST$class)
print(cm)

# ===============================================================
# 6. Save model
# ===============================================================
saveRDS(rf, "/mnt/eo/EO4Backcasting/_models/rf_t0_ysdclass.rds")
saveRDS(predictor_cols, "/mnt/eo/EO4Backcasting/_models/rf_t0_predictors.rds")

# ===============================================================
# 7. Raster prediction (validation for 2010 → 2005–2009)
# ===============================================================
TILE <- "/mnt/dss_europe/level3_interpolated/X0016_Y0020/20100801_LEVEL3_LNDLG_IBAP.tif"
MASK <- "/mnt/eo/EFDA_v211/forest_landuse_aligned.tif"
OUT  <- "/mnt/eo/EO4Backcasting/_predictions/X0016_Y0020_ysdclass.tif"

r_tile <- rast(TILE)
names(r_tile) <- predictor_cols

r_mask <- rast(MASK)

if (!same.crs(r_tile, r_mask)) r_mask <- project(r_mask, r_tile, method="near")
if (!ext(r_mask)==ext(r_tile) || !all(res(r_mask)==res(r_tile)))
  r_mask <- resample(r_mask, r_tile, method="near")

r_mask01 <- classify(r_mask, rbind(c(-Inf,0.5,NA), c(0.5,Inf,1)))
r_forest <- mask(r_tile, r_mask01)

# prediction wrapper for terra
make_fun <- function(rf_model) {
  force(rf_model)
  function(model, data, ...) {
    df <- as.data.frame(data)
    
    # ranger braucht korrekte Spaltennamen
    colnames(df) <- rf_model$forest$independent.variable.names
    
    preds <- predict(rf_model, df)$predictions
    max.col(preds, ties.method = "first")
  }
}

# prediction call
fun_h <- make_fun(rf)

out <- terra::predict(
  r_forest,
  model = 1,                   # terra dummy model
  fun   = fun_h,               # must accept (model, data, ...)
  filename = OUT,
  overwrite = TRUE,
  wopt = list(
    datatype = "INT1U",
    gdal = c("COMPRESS=ZSTD","PREDICTOR=2","ZSTD_LEVEL=8")
  )
)

table(TRAIN$class)

cat("\nPrediction written to: ", OUT, "\n")




