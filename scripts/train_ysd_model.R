# ========================== BACKCAST YSD FROM 1985 ===========================
# Globales Training (Punkte) und lokale Anwendung NUR auf einem BAP-Tile.
# Vorhersage strikt auf (a) das Grid des BAP-Tiles und (b) nur innerhalb Wald (1/NA).
# Datenquellen für Vorhersage: (1) BAP b1..b6, (2) NBR1985
# CoE (Median/Agreement/Spread) auf Basis BAP+NBR.
# ============================================================================

# --- PACKAGES ----------------------------------------------------------------
suppressPackageStartupMessages({
  library(data.table)
  library(terra)
  library(caret)
  library(xgboost)
  library(ranger)
  library(Metrics)
})

# --- THREADING / PERFORMANCE -------------------------------------------------
options(ranger.num.threads = 30)
data.table::setDTthreads(30)
terraOptions(progress = 1)   # Fortschrittsbalken

# --- USER INPUTS -------------------------------------------------------------
# Trainingsdaten (global)
TRAIN_CSV     <- "/mnt/eo/EO4Backcasting/_intermediates/training_data.csv"
TRAIN_PARQUET <- NULL
TRAIN_RDS     <- NULL

# Lokale Anwendung: ein BAP-Tile + globales NBR 1985 + Waldmaske
BAP_TILE_PATH <- "/mnt/dss_europe/level3_interpolated/X0021_Y0029/19850801_LEVEL3_LNDLG_IBAP.tif"  # 6 Bänder
NBR1985_PATH  <- "/mnt/eo/eu_mosaics/NBR_comp/NBR_1985.tif"                                         # 1 Band
FOREST_MASK   <- "/mnt/eo/EFDA_v211/forest_landuse_aligned.tif"                                     # 1=Forest, NA=non-forest

# Ausgaben
OUT_DIR <- "/mnt/eo/EO4Backcasting/_intermediates/predictions"
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

# Trainings- und Altersparameter
TRAIN_YEARS <- 1986:2024       # Jahre, aus denen Punkte zum Training stammen
AGE_MIN     <- 1
AGE_MAX     <- 20

# Skalenabgleich: Punkte 0..10000 vs Raster 0..1?
RESCALE_PTS_TO_0_1 <- FALSE    # skaliert NUR die Punkt-Bänder b1..b6
SCALE_RASTERS_BY   <- 1        # setze ggf. 10000 oder 0.0001, um Raster anzupassen (optional)

# Modellwahl
ENGINE_BAP <- "rf"             # "rf" oder "xgb" (für BAP b1..b6)
ENGINE_NBR <- "rf"             # für NBR-Einzelband: i.d.R. "rf" (1D-XGB optional separat)

set.seed(42)

# --- HILFSFUNKTIONEN ---------------------------------------------------------
align_to_mask <- function(x, mask, method_bilinear = TRUE) {
  # gleiche Raster x exakt an das Grid der Maske an (CRS, Auflösung, Alignment)
  if (!same.crs(x, mask)) {
    x <- project(x, mask, method = if (method_bilinear) "bilinear" else "near")
  } else {
    x <- resample(x, mask, method = if (method_bilinear) "bilinear" else "near")
  }
  x
}

mask_to_forest <- function(x, mask_trimmed) {
  # behalte Werte NUR dort, wo Maske nicht NA (== Wald==1); sonst NA
  mask(x, mask_trimmed)  # Maske hat 1 (Wald) und NA (kein Wald)
}

to_int <- function(r, lo=AGE_MIN, hi=AGE_MAX) {
  clamp(round(r), lower = lo, upper = hi, values = TRUE)
}

# robustes Agreement (Paare ±tol)
agree_pairs <- function(v, tol = 1) {
  v <- v[is.finite(v)]
  if (length(v) < 2) return(NA)
  combs <- combn(v, 2)
  sum(abs(combs[1, ] - combs[2, ]) <= tol)
}

# --- TRAININGSDATEN LADEN (GLOBAL) ------------------------------------------
message("Loading training data...")
if (!is.null(TRAIN_PARQUET)) {
  library(arrow); pts <- as.data.table(read_parquet(TRAIN_PARQUET))
} else if (!is.null(TRAIN_RDS)) {
  pts <- as.data.table(readRDS(TRAIN_RDS))
} else {
  pts <- fread(TRAIN_CSV)
}

required_bands <- c("b1","b2","b3","b4","b5","b6")
stopifnot("year" %in% names(pts))
if (!("ysd" %in% names(pts))) {
  stopifnot("yod" %in% names(pts))
  pts[, ysd := year - yod]
}
miss <- setdiff(required_bands, names(pts))
if (length(miss) > 0) stop("Missing bands in training table: ", paste(miss, collapse=", "))

# NBR wird für das NBR-Modell benötigt
if (!("NBR" %in% names(pts))) {
  stop("Training table requires column 'NBR' for the NBR-only model.")
}

# Typen & ggf. Reskalierung (nur Punkte b1..b6)
for (nm in required_bands) if (!is.numeric(pts[[nm]])) pts[, (nm) := as.numeric(get(nm))]
if (RESCALE_PTS_TO_0_1) pts[, (required_bands) := lapply(.SD, \(z) z/10000), .SDcols = required_bands]

# Filter Training
train <- pts[year %in% TRAIN_YEARS & is.finite(ysd)]
train <- train[ysd >= AGE_MIN & ysd <= AGE_MAX]
train <- train[complete.cases(train[, ..required_bands])]
if (nrow(train) < 1000) warning("Training set seems small (n < 1000).")

# Design-Matrix (BAP b1..b6)
x_cols <- required_bands
x <- as.matrix(train[, ..x_cols])
y <- train$ysd

# Klassengewichtung (Altersverteilung)
age_tab <- table(y)
w <- 1 / as.numeric(age_tab[as.character(y)])
w <- w / mean(w)

# --- MODELLTRAINING (GLOBAL) -------------------------------------------------
PRED_FUN_RF <- function(model, data, ...) {
  as.numeric(predict(model, data = as.data.frame(data), num.threads = 30)$predictions)
}

if (ENGINE_BAP == "xgb") {
  message("Training XGBoost (BAP) with 5-fold CV...")
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
  fit <- train(x = x_tr, y = y_tr, method = "xgbTree",
               trControl = ctrl, tuneGrid = grid,
               metric = "RMSE", weights = w_tr)
  best <- fit$bestTune
  message("Best params:"); print(best)
  
  pred_te <- predict(fit, x_te)
  message(sprintf("Hold-out RMSE=%.3f  MAE=%.3f  R^2=%.3f",
                  rmse(y_te, pred_te), mae(y_te, pred_te),
                  1 - sum((y_te - pred_te)^2)/sum((y_te - mean(y_te))^2)))
  
  d_all <- xgb.DMatrix(x, label = y, weight = w)
  params <- list(
    objective = "reg:squarederror", eval_metric = "rmse",
    eta = best$eta, max_depth = best$max_depth, gamma = best$gamma,
    subsample = best$subsample, colsample_bytree = best$colsample_bytree,
    min_child_weight = best$min_child_weight, nthread = 30
  )
  model_bap <- xgb.train(params, d_all, nrounds = best$nrounds, verbose = 0)
  PRED_FUN_BAP <- function(model, data, ...) predict(model, as.matrix(data))
} else if (ENGINE_BAP == "rf") {
  message("Training Random Forest (BAP b1..b6, global)...")
  model_bap <- ranger(
    ysd ~ ., data = as.data.frame(train[, c("ysd", x_cols), with = FALSE]),
    num.trees = 1000, mtry = floor(sqrt(length(x_cols))),
    min.node.size = 5, importance = "impurity",
    num.threads = 30, seed = 42
  )
  PRED_FUN_BAP <- PRED_FUN_RF
} else stop("ENGINE_BAP must be 'rf' or 'xgb'.")

# NBR-Modell (einfaches RF standardmäßig)
if (ENGINE_NBR == "rf") {
  message("Training Random Forest (NBR-only, global)...")
  model_nbr <- ranger(ysd ~ NBR, data = as.data.frame(train[, .(ysd, NBR)]),
                      num.trees = 1000, min.node.size = 5,
                      num.threads = 30, seed = 42)
  PRED_FUN_NBR <- PRED_FUN_RF
} else if (ENGINE_NBR == "xgb") {
  message("Training XGBoost (NBR-only, global)...")
  x_nbr <- matrix(train$NBR, ncol = 1)
  dtr   <- xgb.DMatrix(x_nbr, label = y)   # optional: weights w
  params_nbr <- list(objective = "reg:squarederror", eval_metric = "rmse", nthread = 30,
                     eta = 0.1, max_depth = 3, subsample = 0.9, colsample_bytree = 1.0)
  model_nbr <- xgb.train(params_nbr, dtr, nrounds = 400, verbose = 0)
  PRED_FUN_NBR <- function(model, data, ...) predict(model, as.matrix(data))
} else stop("ENGINE_NBR must be 'rf' or 'xgb'.")

# --- WALDMASKE LADEN & AUF BAP-TILE ZUSCHNEIDEN ------------------------------
message("Loading forest mask and constraining it to the BAP tile grid & extent...")
r_bap  <- rast(BAP_TILE_PATH)              # 6 Bänder
if (nlyr(r_bap) != 6) stop("BAP tile must have 6 bands.")
names(r_bap) <- c("b1","b2","b3","b4","b5","b6")

r_mask_full <- rast(FOREST_MASK)           # 1 (forest) / NA (non-forest) bevorzugt
# auf BAP-Grid bringen (nearest für kategoriale Maske)
if (!same.crs(r_mask_full, r_bap)) {
  r_mask_aligned <- project(r_mask_full, r_bap, method = "near")
} else {
  r_mask_aligned <- resample(r_mask_full, r_bap, method = "near")
}
# auf BAP-Extent croppen
r_mask_tile <- crop(r_mask_aligned, r_bap)

# sicherstellen: 1/NA (falls 0/1 vorliegt, 0 -> NA)
vals <- unique(na.omit(as.vector(values(r_mask_tile))))
if (any(vals == 0, na.rm = TRUE)) {
  r_mask_tile <- classify(r_mask_tile, rbind(c(-Inf,0,NA), c(0.5, Inf,1)))
}

# --- RASTER LADEN & AUF BAP-GRID AUSRICHTEN ----------------------------------
message("Loading NBR and aligning to BAP tile grid...")
r_nbr <- rast(NBR1985_PATH); names(r_nbr) <- "NBR"
# NBR (kontinuierlich) bilinear auf BAP-Grid
if (!same.crs(r_nbr, r_bap)) {
  r_nbr_bapgrid <- project(r_nbr, r_bap, method = "bilinear")
} else {
  r_nbr_bapgrid <- resample(r_nbr, r_bap, method = "bilinear")
}
r_nbr_tile <- crop(r_nbr_bapgrid, r_bap)

# (optional) Skalenanpassung der Raster, falls nötig
if (SCALE_RASTERS_BY != 1) {
  r_bap  <- r_bap  * SCALE_RASTERS_BY
  r_nbr_tile <- r_nbr_tile * SCALE_RASTERS_BY
}

# --- MASKIEREN: NUR WALD-PIXEL BEHALTEN --------------------------------------
message("Masking BAP and NBR by the forest mask (tile)…")
r_bap_masked <- mask(r_bap, r_mask_tile)       # BAP b1..b6 nur im Wald
r_nbr_masked <- mask(r_nbr_tile, r_mask_tile)  # NBR nur im Wald

# (optional) schreibe die vorbereiteten Masken-Tiles
wopt_flt <- list(datatype = "FLT4S", gdal = "COMPRESS=DEFLATE,ZLEVEL=6,PREDICTOR=3")
writeRaster(r_bap_masked, file.path(OUT_DIR, "BAP_1985_tile_forestOnly.tif"),
            overwrite = TRUE, wopt = wopt_flt)
writeRaster(r_nbr_masked, file.path(OUT_DIR, "NBR_1985_tile_forestOnly.tif"),
            overwrite = TRUE, wopt = wopt_flt)

# --- VORHERSAGE (LOKAL AUF TILE) --------------------------------------------
message("Predicting YSD on BAP tile (forest only)…")
ysd_bap_tile <- terra::predict(
  r_bap_masked[[c("b1","b2","b3","b4","b5","b6")]], model_bap, fun = PRED_FUN_BAP,
  cores = 30,
  filename = file.path(OUT_DIR, "ysd_1985_BAP_tile.tif"),
  overwrite = TRUE, wopt = wopt_flt
)

message("Predicting YSD on NBR tile (forest only)…")
ysd_nbr_tile <- terra::predict(
  r_nbr_masked, model_nbr, fun = PRED_FUN_NBR,
  cores = 30,
  filename = file.path(OUT_DIR, "ysd_1985_NBR_tile.tif"),
  overwrite = TRUE, wopt = wopt_flt
)

# --- INTEGER YSD + YOD (1985, lokal) ----------------------------------------
wopt_i16 <- list(datatype = "INT2S", gdal = "COMPRESS=DEFLATE,ZLEVEL=6")

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

# --- CONVERGENCE OF EVIDENCE (BAP + NBR, lokal) ------------------------------
message("Building ensemble & uncertainty layers (tile, forest only)…")
ysd_stack <- c(ysd_bap_tile, ysd_nbr_tile)

ysd_ens_median <- app(
  ysd_stack, median, na.rm = TRUE, cores = 30,
  filename = file.path(OUT_DIR, "ysd_1985_tile_ensemble_median.tif"),
  overwrite = TRUE, wopt = wopt_flt
)

ysd_agree <- app(
  ysd_stack, agree_pairs, na.rm = TRUE, cores = 30,
  filename = file.path(OUT_DIR, "ysd_1985_tile_agreement_pairs.tif"),
  overwrite = TRUE
)

ysd_iqr <- app(
  ysd_stack, IQR, na.rm = TRUE, cores = 30,
  filename = file.path(OUT_DIR, "ysd_1985_tile_spread_IQR.tif"),
  overwrite = TRUE, wopt = wopt_flt
)

ysd_sd <- app(
  ysd_stack, sd, na.rm = TRUE, cores = 30,
  filename = file.path(OUT_DIR, "ysd_1985_tile_spread_SD.tif"),
  overwrite = TRUE, wopt = wopt_flt
)

message("Done.")
message(sprintf("Outputs written to: %s", OUT_DIR))

# --- OPTIONAL: QUICK SANITY CHECKS ------------------------------------------
# q_train <- sapply(as.data.table(x), quantile, probs = c(.01, .99), na.rm = TRUE)
# q_bap   <- as.data.frame(global(r_bap_masked, fun = quantile, na.rm = TRUE, probs = c(.01,.99)))
# q_nbr   <- as.data.frame(global(r_nbr_masked, fun = quantile, na.rm = TRUE, probs = c(.01,.99)))
