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
bap1985_path <- "/mnt/dss_europe/mosaics_eu/mosaics_eu_baps/1985_mosaic_eu_cog.tif"
points_csv   <- "/mnt/eo/EO4Backcasting/_intermediates/training_data.csv"
pts <- fread(points_csv)[, .(id, x, y, yod, year, ysd, ysd_bin, class_label, EVI, NBR, b1, b2, b3, b4, b5, b6)]

# --- 0) PICK THE SIX BANDS FROM 'pts' ----------------------------------------
# Prefer canonical names, else fall back to any B[1-7] present (take 6).
preferred_bands <- c("B1","B2","B3","B4","B5","B7")
band_cols <- intersect(preferred_bands, names(pts))
if (length(band_cols) < 6) {
  cand <- grep("^B[0-9]+$", names(pts), value = TRUE)
  # keep a stable order like B1..B9
  cand <- cand[order(as.integer(sub("^B", "", cand)))]
  if (length(cand) >= 6) band_cols <- cand[1:6]
}
stopifnot(length(band_cols) == 6)

# --- 1) BUILD TRAINING SET (1986–2024, ages you trust) -----------------------
train <- copy(pts)[year >= 1986 & year <= 2024]
if (!"ysd" %in% names(train)) train[, ysd := year - yod]
train <- train[ysd >= 1 & ysd <= 20]

x <- as.matrix(train[, ..band_cols])
y <- train$ysd

# Optional: balance ages for more uniform learning
w <- {
  tab <- table(y); ww <- 1 / as.numeric(tab[as.character(y)])
  ww / mean(ww)
}

# --- 2) XGBOOST REGRESSION (CV-TUNED) ----------------------------------------
set.seed(42)
idx  <- caret::createDataPartition(y, p = 0.8, list = FALSE)
x_tr <- x[idx, ]; y_tr <- y[idx]; w_tr <- w[idx]
x_te <- x[-idx,]; y_te <- y[-idx]

ctrl <- trainControl(method = "cv", number = 5)
grid <- expand.grid(
  nrounds = seq(200, 800, 200),
  max_depth = c(3,4,5),
  eta = c(0.05, 0.1, 0.2),
  gamma = c(0,1),
  colsample_bytree = c(0.8, 1.0),
  min_child_weight = c(1,5),
  subsample = c(0.8, 1.0)
)

fit <- train(
  x = x_tr, y = y_tr,
  method = "xgbTree",
  trControl = ctrl, tuneGrid = grid, metric = "RMSE",
  weights = w_tr
)

best <- fit$bestTune
dall <- xgb.DMatrix(x, label = y, weight = w)
params <- list(
  objective="reg:squarederror", eval_metric="rmse",
  eta=best$eta, max_depth=best$max_depth, gamma=best$gamma,
  subsample=best$subsample, colsample_bytree=best$colsample_bytree,
  min_child_weight=best$min_child_weight
)
final_xgb <- xgb.train(params, dall, nrounds = best$nrounds, verbose = 0)

# --- 3) PREPARE THE 1985 RASTER (ONLY THE 6 BANDS) ---------------------------
r1985 <- rast(bap1985_path)

# Try to set canonical names if 6 layers:
if (nlyr(r1985) == 6) names(r1985) <- preferred_bands

# Check all required bands are available; subset & order to 'band_cols'
missing <- setdiff(band_cols, names(r1985))
if (length(missing) > 0) stop("1985 raster is missing bands: ", paste(missing, collapse = ", "))
r_pred <- r1985[[band_cols]]   # ensures order matches training

# --- 4) PREDICT YSD ON 1985; BACKCAST YOD ------------------------------------
.predict_xgb <- function(m) predict(final_xgb, as.matrix(m))

ysd_1985 <- terra::predict(
  r_pred, model = .predict_xgb,
  filename = "ysd_1985_xgb.tif", overwrite = TRUE,
  wopt = list(datatype="FLT4S", gdal="COMPRESS=DEFLATE,ZLEVEL=6,PREDICTOR=3")
)

ysd_1985_int <- clamp(round(ysd_1985), lower=1, upper=20, values=TRUE)
writeRaster(ysd_1985_int, "ysd_1985_xgb_int.tif", overwrite=TRUE,
            wopt = list(datatype="INT2S", gdal="COMPRESS=DEFLATE,ZLEVEL=6"))

yod_1985 <- 1985 - ysd_1985_int
writeRaster(yod_1985, "yod_1985_xgb_yod.tif", overwrite=TRUE,
            wopt = list(datatype="INT2S", gdal="COMPRESS=DEFLATE,ZLEVEL=6"))
