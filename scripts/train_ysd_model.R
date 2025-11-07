# ========================== BACKCAST YSD FROM 1985 ===========================
# RF/XGB-Training auf Punkten; Vorhersage NUR auf Wald (1/NA) und NUR
# auf das Grid & die Ausdehnung der getrimmten Waldmaske.
# Karten: (1) BAP b1..b6, (2) NBR1985, (3) EVI1985 + CoE (Median/Agreement/Spread)
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
options(ranger.num.threads = 48)
data.table::setDTthreads(48)
terraOptions(progress = 1)   # Fortschrittsbalken
# terraOptions(memfrac = 0.8) # optional: RAM-Auslastung

# --- USER INPUTS -------------------------------------------------------------
TRAIN_CSV     <- "/mnt/eo/EO4Backcasting/_intermediates/training_data.csv"
TRAIN_PARQUET <- NULL
TRAIN_RDS     <- NULL

BAP1985_PATH  <- "/mnt/dss_europe/level3_interpolated/X0021_Y0029/19850801_LEVEL3_LNDLG_IBAP.tif" # 6 Bänder
NBR1985_PATH  <- "/mnt/eo/eu_mosaics/NBR_comp/NBR_1985.tif"  # 1 Band
EVI1985_PATH  <- "/mnt/eo/eu_mosaics/EVI_comp/EVI_1985.tif"  # 1 Band
FOREST_MASK   <- "/mnt/eo/EFDA_v211/forest_landuse_aligned.tif" 

OUT_DIR <- "/mnt/eo/EO4Backcasting/_intermediates/predictions"
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

TRAIN_YEARS <- 1986:2024
AGE_MIN     <- 1
AGE_MAX     <- 20

# Skalenabgleich: Punkte 0..10000 vs Raster 0..1?
RESCALE_PTS_TO_0_1 <- FALSE   # skaliert NUR die Punkt-Bänder b1..b6
SCALE_RASTERS_BY   <- 1       # setze ggf. 10000 oder 0.0001, um Raster anzupassen

ENGINE <- "rf" # "rf" oder "xgb"
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

# --- TRAININGSDATEN LADEN ----------------------------------------------------
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

for (nm in required_bands) if (!is.numeric(pts[[nm]])) pts[, (nm) := as.numeric(get(nm))]
if (RESCALE_PTS_TO_0_1) pts[, (required_bands) := lapply(.SD, \(z) z/10000), .SDcols = required_bands]

train <- pts[year %in% TRAIN_YEARS & is.finite(ysd)]
train <- train[ysd >= AGE_MIN & ysd <= AGE_MAX]
train <- train[complete.cases(train[, ..required_bands])]
if (nrow(train) < 1000) warning("Training set seems small (n < 1000).")

x_cols <- required_bands
x <- as.matrix(train[, ..x_cols])
y <- train$ysd

age_tab <- table(y)
w <- 1 / as.numeric(age_tab[as.character(y)])
w <- w / mean(w)

# --- MODELLTRAINING ----------------------------------------------------------
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
  fit <- train(x = x_tr, y = y_tr, method = "xgbTree",
               trControl = ctrl, tuneGrid  = grid,
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
    min_child_weight = best$min_child_weight, nthread = 48
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
  stopifnot(all(c("NBR","EVI") %in% names(train)))
  rf_nbr <- ranger(ysd ~ NBR, data = as.data.frame(train[, .(ysd, NBR)]),
                   num.trees=1000, min.node.size=5, num.threads=48, seed=42)
  rf_evi <- ranger(ysd ~ EVI, data = as.data.frame(train[, .(ysd, EVI)]),
                   num.trees=1000, min.node.size=5, num.threads=48, seed=42)
  
  PRED_FUN <- function(model, data, ...) {
    as.numeric(predict(model, data=as.data.frame(data), num.threads=48)$predictions)
  }
} else stop("ENGINE must be 'rf' or 'xgb'.")

# --- WALDMASKE LADEN & TRIMMEN ----------------------------------------------
message("Loading forest mask and trimming to valid forest extent...")
r_mask_full <- rast(FOREST_MASK)           # 1 = Wald, NA = kein Wald
# trim entfernt umgebende NA-Ränder -> kleineres Ausmaß, rechen- & IO-schonend
r_mask <- trim(r_mask_full)                # Referenzgrid (CRS/Res/Alignment!)


# --- RASTER LADEN & EXAKT AUF MASKEN-GRID AUSRICHTEN ------------------------
message("Loading rasters and aligning to trimmed mask grid...")
r_bap  <- rast(BAP1985_PATH)                # 6 Bänder
if (nlyr(r_bap) != 6) stop("BAP1985 must have 6 bands.")
names(r_bap) <- c("b1","b2","b3","b4","b5","b6")

r_nbr  <- rast(NBR1985_PATH); names(r_nbr) <- "NBR"
r_evi  <- rast(EVI1985_PATH); names(r_evi) <- "EVI"

# An Masken-Grid ausrichten (project/resample je nach CRS)
r_bap_a <- align_to_mask(r_bap, r_mask, method_bilinear = TRUE)
r_nbr_a <- align_to_mask(r_nbr, r_mask, method_bilinear = TRUE)
r_evi_a <- align_to_mask(r_evi, r_mask, method_bilinear = TRUE)

# (optional) Skalenanpassung der Raster, falls nötig
if (SCALE_RASTERS_BY != 1) {
  r_bap_a <- r_bap_a * SCALE_RASTERS_BY
  r_nbr_a <- r_nbr_a * SCALE_RASTERS_BY
  r_evi_a <- r_evi_a * SCALE_RASTERS_BY
}

# --- ZUSCHNEIDEN & NUR WALD-PIXEL BEHALTEN -----------------------------------
# Crop auf Masken-Ausmaß (nach trim), dann Maskierung (NA außerhalb Wald)
message("Cropping rasters to mask extent and keeping only forest pixels...")
r_bap_c <- crop(r_bap_a, r_mask)
r_nbr_c <- crop(r_nbr_a, r_mask)
r_evi_c <- crop(r_evi_a, r_mask)

r_bap_f <- mask_to_forest(r_bap_c, r_mask)
r_nbr_f <- mask_to_forest(r_nbr_c, r_mask)
r_evi_f <- mask_to_forest(r_evi_c, r_mask)

# --- PREDICT YSD 1985 (nur Wald, masken-grid, zugeschnitten) -----------------
wopt_flt <- list(datatype="FLT4S", gdal="COMPRESS=DEFLATE,ZLEVEL=6,PREDICTOR=3")
wopt_i16 <- list(datatype="INT2S", gdal="COMPRESS=DEFLATE,ZLEVEL=6")

message("Predicting YSD on 1985 BAP (forest only)...")
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

message("Predicting YSD on NBR1985 (forest only)...")
if (ENGINE == "xgb") stop("Für NBR/EVI-only bitte ENGINE='rf' oder 1D-XGB-Modelle trainieren.")
ysd_nbr <- terra::predict(
  r_nbr_f, rf_nbr, fun = PRED_FUN,
  cores = 48,
  filename = file.path(OUT_DIR, "ysd_1985_NBR.tif"),
  overwrite = TRUE, wopt = wopt_flt
)

message("Predicting YSD on EVI1985 (forest only)...")
ysd_evi <- terra::predict(
  r_evi_f, rf_evi, fun = PRED_FUN,
  cores = 48,
  filename = file.path(OUT_DIR, "ysd_1985_EVI.tif"),
  overwrite = TRUE, wopt = wopt_flt
)

# --- INTEGER YSD + YOD (1985) -----------------------------------------------
ysd_bap_i <- to_int(ysd_bap)
writeRaster(ysd_bap_i, file.path(OUT_DIR, "ysd_1985_BAP_int.tif"), overwrite=TRUE, wopt=wopt_i16)
writeRaster(1985 - ysd_bap_i, file.path(OUT_DIR, "yod_1985_BAP.tif"), overwrite=TRUE, wopt=wopt_i16)

ysd_nbr_i <- to_int(ysd_nbr)
writeRaster(ysd_nbr_i, file.path(OUT_DIR, "ysd_1985_NBR_int.tif"), overwrite=TRUE, wopt=wopt_i16)
writeRaster(1985 - ysd_nbr_i, file.path(OUT_DIR, "yod_1985_NBR.tif"), overwrite=TRUE, wopt=wopt_i16)

ysd_evi_i <- to_int(ysd_evi)
writeRaster(ysd_evi_i, file.path(OUT_DIR, "ysd_1985_EVI_int.tif"), overwrite=TRUE, wopt=wopt_i16)
writeRaster(1985 - ysd_evi_i, file.path(OUT_DIR, "yod_1985_EVI.tif"), overwrite=TRUE, wopt=wopt_i16)

# --- CONVERGENCE OF EVIDENCE -------------------------------------------------
message("Building ensemble & uncertainty layers (forest only)...")
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

# --- OPTIONAL: QUICK SANITY CHECKS ------------------------------------------
# Prüfe, ob Bänder & Wertebereiche zusammenpassen:
# q_train <- sapply(as.data.table(x), quantile, probs=c(.01,.99), na.rm=TRUE)
# q_bap   <- as.data.frame(global(r_bap_f, fun=quantile, na.rm=TRUE, probs=c(.01,.99)))
