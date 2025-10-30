# --- PACKAGES ----------------------------------------------------------------
library(data.table)
library(caret)
library(xgboost)
library(terra)   
library(stringr)


# --- INPUTS ------------------------------------------------------------------
# 1) Your training table
#    Must include: year, ysd, and the six reflectance bands as numeric columns.
#    Object name is 'pts' (as you said).
# 2) 1985 multiband BAP raster with the same 6 bands (order can be corrected).
bap1985_path <- "/mnt/dss_europe/level3_interpolated/X0021_Y0029/19850801_LEVEL3_LNDLG_IBAP.tif"
points_csv   <- "/mnt/eo/EO4Backcasting/_intermediates/training_data.csv"
pts <- fread(points_csv)[, .(id, x, y, yod, year, ysd, ysd_bin, class_label, EVI, NBR, b1, b2, b3, b4, b5, b6)]

# ======================= BACKCAST YSD FROM 1985 BAP ==========================
# Predict YSD on a 1985 BAP raster using a model trained on 1986–2024 samples.
# Uses ONLY the six reflectance bands b1..b6 (no EVI/NBR for this first run).
# ============================================================================

# --- USER INPUTS --------------------------------------------------------------
# 1) Training table (pick ONE of the three loading options below)
#    Must contain: year, (ysd or yod), and six band columns b1..b6.
TRAIN_CSV     <- "/mnt/eo/EO4Backcasting/_intermediates/training_data.csv"        # if CSV
TRAIN_PARQUET <- NULL                                      
TRAIN_RDS     <- NULL                                      

# 2) 1985 BAP raster (6-band stack; order doesn't matter, we'll rename)
BAP1985_PATH  <- "/mnt/dss_europe/level3_interpolated/X0021_Y0029/19850801_LEVEL3_LNDLG_IBAP.tif"

# 3) Output files
OUT_YSD_FLOAT <- "/mnt/eo/EO4Backcasting/_intermediates/ysd_1985_xgb.tif"
OUT_YSD_INT   <- "/mnt/eo/EO4Backcasting/_intermediates/ysd_1985_xgb_int.tif"
OUT_YOD_INT   <- "/mnt/eo/EO4Backcasting/_intermediates/yod_1985_xgb.tif"

# 4) Training year range and age range you trust
TRAIN_YEARS   <- 1986:2024
AGE_MIN       <- 1
AGE_MAX       <- 20

# 5) If needed, scale bands here (set to TRUE if pts has 0–10000 ints but raster is 0–1)
RESCALE_PTS_TO_0_1 <- FALSE

# --- LOAD TRAINING TABLE ------------------------------------------------------
message("Loading training data...")
if (!is.null(TRAIN_PARQUET)) {
  pts <- as.data.table(read_parquet(TRAIN_PARQUET))
} else if (!is.null(TRAIN_RDS)) {
  pts <- readRDS(TRAIN_RDS); pts <- as.data.table(pts)
} else {
  # default: CSV
  pts <- fread(TRAIN_CSV)
}

# --- SANITY CHECKS & PREP ----------------------------------------------------
required_bands <- c("b1","b2","b3","b4","b5","b6")
required_cols  <- c("year", required_bands)

stopifnot(all("year" %in% names(pts)))
stopifnot(any(c("ysd","yod") %in% names(pts)))

# create ysd if only yod present
if (!"ysd" %in% names(pts) && "yod" %in% names(pts)) {
  pts[, ysd := year - yod]
}

# ensure band columns exist and are numeric
missing_bands <- setdiff(required_bands, names(pts))
if (length(missing_bands) > 0) {
  stop("Training table is missing bands: ", paste(missing_bands, collapse = ", "))
}
for (nm in required_bands) {
  if (!is.numeric(pts[[nm]])) pts[, (nm) := as.numeric(get(nm))]
}

# optional rescaling to 0–1 reflectance
if (RESCALE_PTS_TO_0_1) {
  pts[, (required_bands) := lapply(.SD, function(z) z / 10000), .SDcols = required_bands]
}

# subset to training years and trusted age range
train <- copy(pts)[year %in% TRAIN_YEARS & ysd >= AGE_MIN & ysd <= AGE_MAX]

if (nrow(train) < 1000) warning("Training set seems small (n < 1000).")

# feature matrix / target
x_cols <- required_bands
x <- as.matrix(train[, ..x_cols])
y <- train$ysd

# balance uneven age distribution with inverse-frequency weights (optional but helpful)
age_tab <- table(y)
w <- 1 / as.numeric(age_tab[as.character(y)])
w <- w / mean(w)

# --- MODEL TRAINING (XGBoost regression) -------------------------------------
set.seed(42)
idx  <- caret::createDataPartition(y, p = 0.8, list = FALSE)
x_tr <- x[idx, ]; y_tr <- y[idx]; w_tr <- w[idx]
x_te <- x[-idx,]; y_te <- y[-idx]

ctrl <- trainControl(method = "cv", number = 5)

grid <- expand.grid(
  nrounds = seq(200, 800, 200),
  max_depth = c(3,4,5),
  eta = c(0.05, 0.1, 0.2),
  gamma = c(0, 1),
  colsample_bytree = c(0.8, 1.0),
  min_child_weight = c(1, 5),
  subsample = c(0.8, 1.0)
)

message("Tuning XGBoost (5-fold CV)...")
fit <- train(
  x = x_tr, y = y_tr,
  method = "xgbTree",
  trControl = ctrl,
  tuneGrid  = grid,
  metric    = "RMSE",
  weights   = w_tr
)

best <- fit$bestTune
message("Best hyperparameters:")
print(best)

# hold-out metrics
pred_te <- predict(fit, x_te)
rmse_val <- rmse(y_te, pred_te)
mae_val  <- mae(y_te, pred_te)
r2_val   <- 1 - sum((y_te - pred_te)^2) / sum((y_te - mean(y_te))^2)
message(sprintf("Hold-out RMSE=%.3f  MAE=%.3f  R^2=%.3f", rmse_val, mae_val, r2_val))

# final model on all training rows
dall <- xgb.DMatrix(x, label = y, weight = w)
params <- list(
  objective = "reg:squarederror",
  eval_metric = "rmse",
  eta = best$eta,
  max_depth = best$max_depth,
  gamma = best$gamma,
  subsample = best$subsample,
  colsample_bytree = best$colsample_bytree,
  min_child_weight = best$min_child_weight
)
set.seed(42)
final_xgb <- xgb.train(params, dall, nrounds = best$nrounds, verbose = 0)

# optional: feature importance
imp <- xgb.importance(feature_names = x_cols, model = final_xgb)
print(head(imp, 10))

# --- LOAD 1985 RASTER & ALIGN FEATURES ---------------------------------------
message("Loading 1985 BAP raster...")
r1985 <- rast(BAP1985_PATH)

# enforce b1..b6 names if 6 layers
if (nlyr(r1985) == 6) names(r1985) <- required_bands

# check required bands present; subset & order to x_cols
missing_1985 <- setdiff(x_cols, names(r1985))
if (length(missing_1985) > 0) {
  stop("1985 raster is missing bands: ", paste(missing_1985, collapse = ", "))
}
r_pred <- r1985[[x_cols]]  # ensures order matches training

# --- PREDICT YSD ON 1985; WRITE GEO-TIFFS ------------------------------------
message("Predicting YSD on 1985 raster...")
.predict_xgb <- function(m) predict(final_xgb, as.matrix(m))

ysd_1985 <- terra::predict(
  r_pred, model = .predict_xgb,
  filename = OUT_YSD_FLOAT, overwrite = TRUE,
  wopt = list(datatype = "FLT4S",
              gdal = "COMPRESS=DEFLATE,ZLEVEL=6,PREDICTOR=3")
)

message("Writing integer YSD and backcast YOD rasters...")
ysd_1985_int <- clamp(round(ysd_1985), lower = AGE_MIN, upper = AGE_MAX, values = TRUE)
writeRaster(
  ysd_1985_int, OUT_YSD_INT, overwrite = TRUE,
  wopt = list(datatype = "INT2S", gdal = "COMPRESS=DEFLATE,ZLEVEL=6")
)

yod_1985 <- 1985 - ysd_1985_int
writeRaster(
  yod_1985, OUT_YOD_INT, overwrite = TRUE,
  wopt = list(datatype = "INT2S", gdal = "COMPRESS=DEFLATE,ZLEVEL=6")
)

message("Done.")
message(sprintf("Outputs:\n  %s (float YSD)\n  %s (int YSD)\n  %s (int YOD)",
                OUT_YSD_FLOAT, OUT_YSD_INT, OUT_YOD_INT))

# ---------------------- OPTIONAL: EXTRA VALIDATION ---------------------------
# If you want a stricter validation that mimics predicting on an unseen year:
# - Do leave-one-year-out CV (train on all years but one, test on that year).
# - Check that 1985 band distributions are within the training range:
#     q_train <- sapply(as.data.table(x), quantile, probs=c(.01,.99), na.rm=TRUE)
#     q_1985  <- sapply(as.data.table(values(r_pred, na.rm=TRUE)), quantile, probs=c(.01,.99), na.rm=TRUE)
#   If 1985 falls outside, consider harmonizing or training on early-era years only.
