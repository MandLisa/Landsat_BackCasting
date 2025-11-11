# ========================== BACKCAST YSD FROM 1985 (FULL SCRIPT, TUNABLE RESOURCES) ==========================
# Global training on points; LOCAL inference on one BAP tile, restricted to forest (1/NA).
# Predictors for 1985: (1) BAP b1..b6, (2) NBR1985 (single band).
# Outputs: YSD (float + int), YOD (int), and CoE layers (median/agreement/spread).
# Resource profile lets you train heavy (e.g., 30 threads) and predict with moderate tiling cores.
# ============================================================================================================

suppressPackageStartupMessages({
  library(data.table)
  library(terra)
  library(caret)
  library(xgboost)
  library(ranger)
  library(Metrics)
})

# ------------------------------ USER I/O -------------------------------------
OUT_DIR <- "/mnt/eo/EO4Backcasting/_intermediates/predictions"
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

TRAIN_CSV     <- "/mnt/eo/EO4Backcasting/_intermediates/training_data.csv"
TRAIN_PARQUET <- NULL
TRAIN_RDS     <- NULL

BAP_TILE_PATH <- "/mnt/dss_europe/level3_interpolated/X0016_Y0020/19850801_LEVEL3_LNDLG_IBAP.tif"  # 6 bands
NBR1985_PATH  <- "/mnt/eo/eu_mosaics/NBR_comp/NBR_1985.tif"                                         # 1 band
FOREST_MASK   <- "/mnt/eo/EFDA_v211/forest_landuse_aligned.tif"                                     # 1 = forest, NA = non-forest

# --------------------------- TRAINING WINDOW ---------------------------------
TRAIN_YEARS <- 1986:2024
AGE_MIN     <- 1
AGE_MAX     <- 20

# ---------------------------- VALUE SCALING ----------------------------------
RESCALE_PTS_TO_0_1 <- FALSE   # if point bands are in 0..10000 but rasters are 0..1
SCALE_RASTERS_BY   <- 1       # multiply rasters if needed (e.g., 0.0001 or 10000)

# --------------------------- MODEL ENGINES -----------------------------------
ENGINE_BAP <- "rf"  # "rf" or "xgb" for 6-band BAP model
ENGINE_NBR <- "rf"  # "rf" or "xgb" for 1-band NBR model

set.seed(42)

# =========================== RESOURCE PROFILE ================================
# Machine: 48 cores available
# Strategy: TRAIN heavy; PREDICT moderate-cores + single-thread model predict to avoid oversubscription.
USE_FAST_PROFILE <- TRUE  # set FALSE to fall back to conservative safe mode

if (USE_FAST_PROFILE) {
  # Training
  RF_THREADS_TRAIN   <- 30
  XGB_THREADS_TRAIN  <- 30
  
  # Prediction
  TERRA_CORES_PRED   <- 8     # increase gradually (8–12) once stable
  RF_THREADS_PRED    <- 1
  XGB_THREADS_PRED   <- 1
  
  Sys.setenv(OMP_NUM_THREADS      = "2",
             MKL_NUM_THREADS      = "2",
             OPENBLAS_NUM_THREADS = "2",
             GDAL_NUM_THREADS     = "2")
  options(mc.cores = TERRA_CORES_PRED, ranger.num.threads = RF_THREADS_TRAIN)
  terraOptions(progress = 1, memfrac = 0.75)  # more aggressive than safe mode
  # terraOptions(tempdir = "/fast/tmp")      # optional: fast SSD temp
} else {
  RF_THREADS_TRAIN   <- 4
  XGB_THREADS_TRAIN  <- 4
  TERRA_CORES_PRED   <- 2
  RF_THREADS_PRED    <- 1
  XGB_THREADS_PRED   <- 1
  Sys.setenv(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", GDAL_NUM_THREADS="1")
  options(mc.cores = 1, ranger.num.threads = RF_THREADS_TRAIN)
  terraOptions(progress = 1, memfrac = 0.6)
}

# ----------------------------- WRITE OPTIONS ---------------------------------
wopt_flt <- list(datatype = "FLT4S", gdal = "COMPRESS=DEFLATE,ZLEVEL=6,PREDICTOR=3")
wopt_i16 <- list(datatype = "INT2S", gdal = "COMPRESS=DEFLATE,ZLEVEL=6")

# -------------------------------- HELPERS ------------------------------------
align_to_grid <- function(src, target, categorical = FALSE) {
  meth <- if (categorical) "near" else "bilinear"
  if (!same.crs(src, target)) project(src, target, method = meth) else resample(src, target, method = meth)
}
force_mask_1_NA <- function(mask_r) {
  u <- unique(na.omit(as.vector(values(mask_r))))
  if (length(u) == 0) return(mask_r)
  if (any(u == 0, na.rm = TRUE)) classify(mask_r, rbind(c(-Inf,0,NA), c(0.5, Inf,1))) else mask_r
}
to_int <- function(r, lo = AGE_MIN, hi = AGE_MAX) clamp(round(r), lower = lo, upper = hi, values = TRUE)
agree_pairs <- function(v, tol = 1) {
  v <- v[is.finite(v)]
  if (length(v) < 2) return(NA)
  combs <- combn(v, 2)
  sum(abs(combs[1, ] - combs[2, ]) <= tol)
}
PRED_FUN_RF <- function(model, data, ...) {
  as.numeric(predict(model, data = as.data.frame(data), num.threads = RF_THREADS_PRED)$predictions)
}
PRED_FUN_XGB <- function(model, data, ...) {
  predict(model, as.matrix(data), nthread = XGB_THREADS_PRED)
}

# ============================ LOAD TRAINING DATA =============================
message("Loading training data…")
if (!is.null(TRAIN_PARQUET)) {
  library(arrow); pts <- as.data.table(read_parquet(TRAIN_PARQUET))
} else if (!is.null(TRAIN_RDS)) {
  pts <- as.data.table(readRDS(TRAIN_RDS))
} else {
  pts <- fread(TRAIN_CSV)
}

required_bands <- c("b1","b2","b3","b4","b5","b6")
stopifnot("year" %in% names(pts))
if (!("ysd" %in% names(pts))) { stopifnot("yod" %in% names(pts)); pts[, ysd := year - yod] }
miss <- setdiff(required_bands, names(pts)); if (length(miss) > 0) stop("Missing: ", paste(miss, collapse=", "))
if (!("NBR" %in% names(pts))) stop("Training table requires column 'NBR' for NBR-only model.")

for (nm in required_bands) if (!is.numeric(pts[[nm]])) pts[, (nm) := as.numeric(get(nm))]
if (!is.numeric(pts$NBR)) pts[, NBR := as.numeric(NBR)]
if (RESCALE_PTS_TO_0_1) pts[, (required_bands) := lapply(.SD, function(z) z/10000), .SDcols = required_bands]

train <- pts[year %in% TRAIN_YEARS & is.finite(ysd)]
train <- train[ysd >= AGE_MIN & ysd <= AGE_MAX]
train <- train[complete.cases(train[, ..required_bands])]
if (nrow(train) < 1000) warning("Training set seems small (n < 1000).")

x_cols <- required_bands
X_bap  <- as.matrix(train[, ..x_cols])
Y      <- train$ysd

age_tab <- table(Y)
w <- 1 / as.numeric(age_tab[as.character(Y)])
w <- w / mean(w)

# ================================ TRAIN MODELS ===============================
if (ENGINE_BAP == "xgb") {
  message("Training XGBoost (BAP)…")
  idx  <- caret::createDataPartition(Y, p = 0.8, list = FALSE)
  X_tr <- X_bap[idx, ]; Y_tr <- Y[idx]; w_tr <- w[idx]
  X_te <- X_bap[-idx,]; Y_te <- Y[-idx]
  
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
  fit <- train(x = X_tr, y = Y_tr, method = "xgbTree",
               trControl = ctrl, tuneGrid = grid,
               metric = "RMSE", weights = w_tr)
  best <- fit$bestTune; message("Best params:"); print(best)
  
  pred_te <- predict(fit, X_te)
  message(sprintf("Hold-out RMSE=%.3f  MAE=%.3f  R^2=%.3f",
                  rmse(Y_te, pred_te), mae(Y_te, pred_te),
                  1 - sum((Y_te - pred_te)^2)/sum((Y_te - mean(Y_te))^2)))
  
  d_all <- xgb.DMatrix(X_bap, label = Y, weight = w)
  params <- list(
    objective = "reg:squarederror", eval_metric = "rmse",
    eta = best$eta, max_depth = best$max_depth, gamma = best$gamma,
    subsample = best$subsample, colsample_bytree = best$colsample_bytree,
    min_child_weight = best$min_child_weight, nthread = XGB_THREADS_TRAIN
  )
  model_bap <- xgb.train(params, d_all, nrounds = best$nrounds, verbose = 0)
  PRED_FUN_BAP <- PRED_FUN_XGB
  
} else if (ENGINE_BAP == "rf") {
  message("Training Random Forest (BAP)…")
  model_bap <- ranger(
    ysd ~ ., data = as.data.frame(train[, c("ysd", x_cols), with = FALSE]),
    num.trees = 1000, mtry = floor(sqrt(length(x_cols))),
    min.node.size = 5, importance = "impurity",
    num.threads = RF_THREADS_TRAIN, seed = 42
  )
  PRED_FUN_BAP <- PRED_FUN_RF
} else stop("ENGINE_BAP must be 'rf' or 'xgb'.")

if (ENGINE_NBR == "rf") {
  message("Training Random Forest (NBR-only)…")
  model_nbr <- ranger(ysd ~ NBR, data = as.data.frame(train[, .(ysd, NBR)]),
                      num.trees = 1000, min.node.size = 5,
                      num.threads = RF_THREADS_TRAIN, seed = 42)
  PRED_FUN_NBR <- PRED_FUN_RF
} else if (ENGINE_NBR == "xgb") {
  message("Training XGBoost (NBR-only)…")
  X_nbr <- matrix(train$NBR, ncol = 1)
  dtr   <- xgb.DMatrix(X_nbr, label = Y)   # optionally add weights = w
  params_nbr <- list(objective = "reg:squarederror", eval_metric = "rmse",
                     eta = 0.1, max_depth = 3, subsample = 0.9, colsample_bytree = 1.0,
                     nthread = XGB_THREADS_TRAIN)
  model_nbr <- xgb.train(params_nbr, dtr, nrounds = 400, verbose = 0)
  PRED_FUN_NBR <- PRED_FUN_XGB
} else stop("ENGINE_NBR must be 'rf' or 'xgb'.")

# Optionally persist:
# saveRDS(model_bap, file.path(OUT_DIR, "model_bap.rds"))
# saveRDS(model_nbr, file.path(OUT_DIR, "model_nbr.rds"))

# ============================ LOCAL TILE PREP ================================
message("Preparing BAP tile and forest mask…")
r_bap <- rast(BAP_TILE_PATH)
if (nlyr(r_bap) != 6) stop("BAP tile must have 6 bands.")
names(r_bap) <- c("b1","b2","b3","b4","b5","b6")

r_mask_full   <- rast(FOREST_MASK)
r_mask_aligned<- align_to_grid(r_mask_full, r_bap, categorical = TRUE)
r_mask_tile   <- crop(r_mask_aligned, r_bap)
r_mask_tile   <- force_mask_1_NA(r_mask_tile)

message("Aligning NBR to BAP grid…")
r_nbr         <- rast(NBR1985_PATH); names(r_nbr) <- "NBR"
r_nbr_bapgrid <- align_to_grid(r_nbr, r_bap, categorical = FALSE)
r_nbr_tile    <- crop(r_nbr_bapgrid, r_bap)

if (SCALE_RASTERS_BY != 1) {
  r_bap      <- r_bap * SCALE_RASTERS_BY
  r_nbr_tile <- r_nbr_tile * SCALE_RASTERS_BY
}

message("Masking predictors to forest…")
r_bap_masked <- mask(r_bap, r_mask_tile)
r_nbr_masked <- mask(r_nbr_tile, r_mask_tile)

writeRaster(r_bap_masked, file.path(OUT_DIR, "BAP_1985_tile_forestOnly.tif"),
            overwrite = TRUE, wopt = wopt_flt)
writeRaster(r_nbr_masked, file.path(OUT_DIR, "NBR_1985_tile_forestOnly.tif"),
            overwrite = TRUE, wopt = wopt_flt)

# Re-open from disk to force on-disk tiling
r_bap_masked <- rast(file.path(OUT_DIR, "BAP_1985_tile_forestOnly.tif"))
r_nbr_masked <- rast(file.path(OUT_DIR, "NBR_1985_tile_forestOnly.tif"))

stopifnot(terra::compareGeom(r_bap_masked, r_nbr_masked, stopOnError = TRUE))

# ================================ PREDICTION ================================
message("Predicting YSD on BAP tile (forest only)…")
ysd_bap_tile <- terra::predict(
  r_bap_masked[[c("b1","b2","b3","b4","b5","b6")]],
  model_bap, fun = PRED_FUN_BAP,
  cores = TERRA_CORES_PRED,
  filename = file.path(OUT_DIR, "ysd_1985_BAP_tile.tif"),
  overwrite = TRUE, wopt = wopt_flt
)

message("Predicting YSD on NBR tile (forest only)…")
ysd_nbr_tile <- terra::predict(
  r_nbr_masked,
  model_nbr, fun = PRED_FUN_NBR,
  cores = TERRA_CORES_PRED,
  filename = file.path(OUT_DIR, "ysd_1985_NBR_tile.tif"),
  overwrite = TRUE, wopt = wopt_flt
)

# ========================= INTEGERIZATION + YOD =============================
ysd_bap_i <- to_int(ysd_bap_tile)
writeRaster(ysd_bap_i, file.path(OUT_DIR, "ysd_1985_BAP_tile_int.tif"),
            overwrite = TRUE, wopt = wopt_i16)
writeRaster(1985 - ysd_bap_i, file.path(OUT_DIR, "yod_1985_BAP_tile.tif"),
            overwrite = TRUE, wopt = wopt_i16)

ysd_nbr_i <- to_int(ysd_nbr_tile)
writeRaster(ysd_nbr_i, file.path(OUT_DIR, "ysd_1985_NBR_tile_int.tif"),
            overwrite = TRUE, wopt = wopt_i16)
writeRaster(1985 - ysd_nbr_i, file.path(OUT_DIR, "yod_1985_NBR_tile.tif"),
            overwrite = TRUE, wopt = wopt_i16)

# ===================== CONVERGENCE OF EVIDENCE (LOCAL) ======================
message("Computing ensemble and uncertainty layers…")
ysd_stack <- c(ysd_bap_tile, ysd_nbr_tile)

ysd_ens_median <- app(
  ysd_stack, median, na.rm = TRUE, cores = TERRA_CORES_PRED,
  filename = file.path(OUT_DIR, "ysd_1985_tile_ensemble_median.tif"),
  overwrite = TRUE, wopt = wopt_flt
)

ysd_agree <- app(
  ysd_stack, agree_pairs, na.rm = TRUE, cores = TERRA_CORES_PRED,
  filename = file.path(OUT_DIR, "ysd_1985_tile_agreement_pairs.tif"),
  overwrite = TRUE
)

ysd_iqr <- app(
  ysd_stack, IQR, na.rm = TRUE, cores = TERRA_CORES_PRED,
  filename = file.path(OUT_DIR, "ysd_1985_tile_spread_IQR.tif"),
  overwrite = TRUE, wopt = wopt_flt
)

ysd_sd <- app(
  ysd_stack, sd, na.rm = TRUE, cores = TERRA_CORES_PRED,
  filename = file.path(OUT_DIR, "ysd_1985_tile_spread_SD.tif"),
  overwrite = TRUE, wopt = wopt_flt
)

message("Done.")
message(sprintf("Outputs written to: %s", OUT_DIR))

# ---------------------------- OPTIONAL CHECKS -------------------------------
# print(r_bap_masked); print(r_nbr_masked)
# q_bap <- as.data.frame(global(r_bap_masked, fun = quantile, na.rm = TRUE, probs = c(.01,.99)))
# q_nbr <- as.data.frame(global(r_nbr_masked, fun = quantile, na.rm = TRUE, probs = c(.01,.99)))
