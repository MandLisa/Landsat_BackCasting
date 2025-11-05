# ========================== BACKCAST YSD FROM 1985 ===========================
# Train RF oder XGB auf Trainingspunkten und predicte YSD (Jahre seit Störung)
# im Jahr 1985 für: (1) BAP b1..b6, (2) NBR1985, (3) EVI1985.
# Danach: Ensemble (Median), Agreement (Paare ±1 Jahr), Unsicherheit (IQR/SD).
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
# Passe diese Werte an deine Maschine an
options(ranger.num.threads = 30)
data.table::setDTthreads(30)
terraOptions(progress = 1)   # Fortschritt anzeigen

# --- USER INPUTS -------------------------------------------------------------
# Trainingsdaten (müssen mind. 'year', ysd ODER yod, und b1..b6 enthalten)
TRAIN_CSV     <- "/mnt/eo/EO4Backcasting/_intermediates/training_data.csv"
TRAIN_PARQUET <- NULL    # z.B. "/path/train.parquet"
TRAIN_RDS     <- NULL    # z.B. "/path/train.rds"

# Rasterpfade
BAP1985_PATH  <- "/mnt/dss_europe/level3_interpolated/X0021_Y0029/19850801_LEVEL3_LNDLG_IBAP.tif" # 6 Bänder
NBR1985_PATH  <- "/mnt/eo/eu_mosaics/NBR_comp/NBR_1985.tif"  # 1 Band
EVI1985_PATH  <- "/mnt/eo/eu_mosaics/EVI_comp/EVI_1985.tif"  # 1 Band
FOREST_MASK   <- "/mnt/eo/EFDA_v211/forestlanduse_mask_EUmosaic3035.tif" # 1=Forest, 0/NA=Non-forest

# Ausgaben
OUT_DIR <- "/mnt/eo/EO4Backcasting/_intermediates/predictions"
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

# Training/Altersbereich
TRAIN_YEARS <- 1986:2024
AGE_MIN     <- 1
AGE_MAX     <- 20

# Skalierung angleichen? (TRUE, wenn Punkte 0–10000 und Raster 0–1 sind)
RESCALE_PTS_TO_0_1 <- FALSE   # skaliert NUR Punkte (falls nötig), Raster bleiben wie sind
SCALE_RASTERS_BY   <- 1       # setze auf 10000 oder 0.0001, falls du Raster aktiv skalieren willst

# Lernverfahren: "rf" (robust, einfach) oder "xgb" (dein ursprüngliches XGBoost)
ENGINE <- "rf"

set.seed(42)

# --- LOAD TRAIN TABLE --------------------------------------------------------
message("Loading training data...")
if (!is.null(TRAIN_PARQUET)) {
  library(arrow)
  pts <- as.data.table(read_parquet(TRAIN_PARQUET))
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

# numerisch erzwingen + optional reskalieren
for (nm in required_bands) if (!is.numeric(pts[[nm]])) pts[, (nm) := as.numeric(get(nm))]
if (RESCALE_PTS_TO_0_1) pts[, (required_bands) := lapply(.SD, \(z) z/10000), .SDcols = required_bands]

# Subset nach Jahren/Alter, NA droppen
train <- pts[year %in% TRAIN_YEARS & is.finite(ysd)]
train <- train[ysd >= AGE_MIN & ysd <= AGE_MAX]
train <- train[complete.cases(train[, ..required_bands])]
if (nrow(train) < 1000) warning("Training set seems small (n < 1000).")

# Feature-Matrix und Ziel
x_cols <- required_bands
x <- as.matrix(train[, ..x_cols])
y <- train$ysd

# Klassen-Weights gegen schiefe Altersverteilung
age_tab <- table(y)
w <- 1 / as.numeric(age_tab[as.character(y)])
w <- w / mean(w)

# --- TRAINING ---------------------------------------------------------------
if (ENGINE == "xgb") {
  message("Training XGBoost with 5-fold CV...")
  idx  <- caret::createDataPartition(y, p=0.8, list=FALSE)
  x_tr <- x[idx, ]; y_tr <- y[idx]; w_tr <- w[idx]
  x_te <- x[-idx,]; y_te <- y[-idx]
  
  ctrl <- trainControl(method="cv", number=5)
  grid <- expand.grid(
    nrounds = seq(200, 800, 200),
    max_depth = c(3,4,5),
    eta = c(0.05, 0.1, 0.2),
    gamma = c(0, 1),
    colsample_bytree = c(0.8, 1.0),
    min_child_weight = c(1, 5),
    subsample = c(0.8, 1.0)
  )
  
  fit <- train(x = x_tr, y = y_tr,
               method = "xgbTree",
               trControl = ctrl,
               tuneGrid  = grid,
               metric    = "RMSE",
               weights   = w_tr)
  
  best <- fit$bestTune
  message("Best params:"); print(best)
  
  pred_te <- predict(fit, x_te)
  message(sprintf("Hold-out RMSE=%.3f  MAE=%.3f  R^2=%.3f",
                  rmse(y_te, pred_te),
                  mae(y_te, pred_te),
                  1 - sum((y_te - pred_te)^2)/sum((y_te - mean(y_te))^2)))
  
  # final model on all data
  d_all <- xgb.DMatrix(x, label = y, weight = w)
  params <- list(
    objective = "reg:squarederror", eval_metric = "rmse",
    eta = best$eta, max_depth = best$max_depth, gamma = best$gamma,
    subsample = best$subsample, colsample_bytree = best$colsample_bytree,
    min_child_weight = best$min_child_weight,
    nthread = 48
  )
  final_model <- xgb.train(params, d_all, nrounds = best$nrounds, verbose = 0)
  PRED_FUN <- function(model, data, ...) predict(model, as.matrix(data))
  
} else if (ENGINE == "rf") {
  message("Training Random Forest (BAP b1..b6)...")
  rf_bap <- ranger(
    ysd ~ ., data = as.data.frame(train[, c("ysd", x_cols), with=FALSE]),
    num.trees = 1000, mtry = floor(sqrt(length(x_cols))),
    min.node.size = 5, importance = "impurity",
    num.threads = 48, seed = 42
  )
  # 1D-Modelle
  stopifnot(all(c("NBR","EVI") %in% names(train)))
  rf_nbr <- ranger(ysd ~ NBR, data = as.data.frame(train[, .(ysd, NBR)]),
                   num.trees=1000, min.node.size=5, num.threads=48, seed=42)
  rf_evi <- ranger(ysd ~ EVI, data = as.data.frame(train[, .(ysd, EVI)]),
                   num.trees=1000, min.node.size=5, num.threads=48, seed=42)
  
  PRED_FUN <- function(model, data, ...) {
    as.numeric(predict(model, data=as.data.frame(data), num.threads=48)$predictions)
  }
} else {
  stop("ENGINE must be 'rf' or 'xgb'.")
}

# --- LOAD RASTERS & FOREST MASK ---------------------------------------------
message("Loading rasters & aligning...")
r_mask <- rast(FOREST_MASK)                 # 1/0/NA
r_bap  <- rast(BAP1985_PATH)                # 6 bands
if (nlyr(r_bap) != 6) stop("BAP1985 must have 6 bands.")
names(r_bap) <- c("b1","b2","b3","b4","b5","b6")

r_nbr <- rast(NBR1985_PATH); names(r_nbr) <- "NBR"
r_evi <- rast(EVI1985_PATH); names(r_evi) <- "EVI"

# Auf BAP ausrichten
r_mask <- resample(r_mask, r_bap, method="near")
r_nbr  <- resample(r_nbr,  r_bap, method="bilinear")
r_evi  <- resample(r_evi,  r_bap, method="bilinear")

# (optional) Raster-Skalierung, falls nötig:
if (SCALE_RASTERS_BY != 1) {
  r_bap <- r_bap * SCALE_RASTERS_BY
  r_nbr <- r_nbr * SCALE_RASTERS_BY
  r_evi <- r_evi * SCALE_RASTERS_BY
}

# Nur Wald (Maske: 1 = Wald, NA = kein Wald)
r_bap_f <- mask(r_bap, r_mask)   
r_nbr_f <- mask(r_nbr, r_mask)
r_evi_f <- mask(r_evi, r_mask)

writeRaster(
  r_nbr_f,
  filename = file.path(OUT_DIR, "nbr_crop.tif"),
  overwrite = TRUE,
  wopt = list(
    datatype = "INT1U",               # 1 Byte, reicht für 0/1
    gdal = "COMPRESS=DEFLATE,ZLEVEL=6"
  )
)

# --- PREDICT YSD 1985 --------------------------------------------------------
wopt_flt <- list(datatype="FLT4S", gdal="COMPRESS=DEFLATE,ZLEVEL=6,PREDICTOR=3")
wopt_i16 <- list(datatype="INT2S", gdal="COMPRESS=DEFLATE,ZLEVEL=6")

message("Predicting YSD on 1985 BAP...")
if (ENGINE == "xgb") {
  ysd_bap <- terra::predict(
    r_bap_f[[x_cols]], final_model, fun = PRED_FUN,
    cores = 48,
    filename = file.path(OUT_DIR, "ysd_1985_BAP.tif"),
    overwrite = TRUE, wopt = wopt_flt
  )
} else {
  ysd_bap <- terra::predict(
    r_bap_f[[x_cols]], rf_bap, fun = PRED_FUN,
    cores = 48,
    filename = file.path(OUT_DIR, "ysd_1985_BAP.tif"),
    overwrite = TRUE, wopt = wopt_flt
  )
}

message("Predicting YSD on NBR1985...")
if (ENGINE == "xgb") stop("Für NBR/EVI-only bitte ENGINE='rf' oder eigene 1D-XGB-Modelle trainieren.")
ysd_nbr <- terra::predict(
  r_nbr_f, rf_nbr, fun = PRED_FUN,
  cores = 48,
  filename = file.path(OUT_DIR, "ysd_1985_NBR.tif"),
  overwrite = TRUE, wopt = wopt_flt
)

message("Predicting YSD on EVI1985...")
ysd_evi <- terra::predict(
  r_evi_f, rf_evi, fun = PRED_FUN,
  cores = 48,
  filename = file.path(OUT_DIR, "ysd_1985_EVI.tif"),
  overwrite = TRUE, wopt = wopt_flt
)

# --- INTEGER YSD + YOD (für 1985) -------------------------------------------
to_int <- function(r) clamp(round(r), lower = AGE_MIN, upper = AGE_MAX, values = TRUE)

ysd_bap_i <- to_int(ysd_bap)
writeRaster(ysd_bap_i, file.path(OUT_DIR, "ysd_1985_BAP_int.tif"), overwrite=TRUE, wopt=wopt_i16)
writeRaster(1985 - ysd_bap_i, file.path(OUT_DIR, "yod_1985_BAP.tif"), overwrite=TRUE, wopt=wopt_i16)

ysd_nbr_i <- to_int(ysd_nbr)
writeRaster(ysd_nbr_i, file.path(OUT_DIR, "ysd_1985_NBR_int.tif"), overwrite=TRUE, wopt=wopt_i16)
writeRaster(1985 - ysd_nbr_i, file.path(OUT_DIR, "yod_1985_NBR.tif"), overwrite=TRUE, wopt=wopt_i16)

ysd_evi_i <- to_int(ysd_evi)
writeRaster(ysd_evi_i, file.path(OUT_DIR, "ysd_1985_EVI_int.tif"), overwrite=TRUE, wopt=wopt_i16)
writeRaster(1985 - ysd_evi_i, file.path(OUT_DIR, "yod_1985_EVI.tif"), overwrite=TRUE, wopt=wopt_i16)

# --- CONVERGENCE-OF-EVIDENCE -------------------------------------------------
message("Building ensemble & uncertainty layers...")
ysd_stack <- c(ysd_bap, ysd_nbr, ysd_evi)

ysd_ens_median <- app(
  ysd_stack, median, na.rm=TRUE, cores=48,
  filename = file.path(OUT_DIR, "ysd_1985_ensemble_median.tif"),
  overwrite = TRUE, wopt = wopt_flt
)

agree_pairs <- function(v, tol=1) {
  v <- v[is.finite(v)]
  if (length(v) < 2) return(NA)
  combs <- combn(v, 2)
  sum(abs(combs[1,]-combs[2,]) <= tol)
}
ysd_agree <- app(
  ysd_stack, agree_pairs, na.rm=TRUE, cores=48,
  filename = file.path(OUT_DIR, "ysd_1985_agreement_pairs.tif"),
  overwrite = TRUE
)

ysd_iqr <- app(
  ysd_stack, IQR, na.rm=TRUE, cores=48,
  filename = file.path(OUT_DIR, "ysd_1985_spread_IQR.tif"),
  overwrite = TRUE, wopt = wopt_flt
)

ysd_sd  <- app(
  ysd_stack, sd, na.rm=TRUE, cores=48,
  filename = file.path(OUT_DIR, "ysd_1985_spread_SD.tif"),
  overwrite = TRUE, wopt = wopt_flt
)

message("Done.")
message(sprintf("Outputs written to: %s", OUT_DIR))

# --- OPTIONAL CHECKS ---------------------------------------------------------
# q_train <- sapply(as.data.table(x), quantile, probs=c(.01,.99), na.rm=TRUE)
# q_bap   <- as.data.frame(global(r_bap_f, fun=quantile, na.rm=TRUE, probs=c(.01,.99)))
# Vergleiche Verteilungsbereiche und skaliere ggf. Raster oder Punkte.
