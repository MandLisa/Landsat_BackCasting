# ========================= BACKCAST YSD 1985 — FULL BLOCK-WISE, LOW-MEM SCRIPT =========================
# Global training: RF on BAP b1..b6; RF on NBR (1985).
# Local inference: one BAP tile, forest-only, streamed in blocks.
# Robust I/O: explicit start + nrows in writeValues(); never assign readStart()/writeStart().
# No EVI. Predict with safe wrapper for ranger (handles S3 dispatch).
# =======================================================================================================

suppressPackageStartupMessages({
  library(data.table)
  library(terra)
  library(ranger)
  library(Metrics)
})

# -------------------------------- USER PATHS --------------------------------
OUT_DIR        <- "/mnt/eo/EO4Backcasting/_intermediates/predictions"
TRAIN_CSV      <- "/mnt/eo/EO4Backcasting/_intermediates/training_data.csv"
BAP_TILE_PATH  <- "/mnt/dss_europe/level3_interpolated/X0016_Y0020/19850801_LEVEL3_LNDLG_IBAP.tif"  # 6 bands
NBR1985_PATH   <- "/mnt/eo/eu_mosaics/NBR_comp/NBR_1985.tif"                                         # 1 band
FOREST_MASK    <- "/mnt/eo/EFDA_v211/forest_landuse_aligned.tif"                                     # 1=forest, NA=non-forest

dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

# ------------------------------ HYPERPARAMS ---------------------------------
TRAIN_YEARS     <- 1986:2024
AGE_MIN         <- 1
AGE_MAX         <- 20
RF_THREADS_TRAIN<- 30
RF_THREADS_PRED <- 4
CHUNK_PRED_ROWS <- 1024L   # rows per prediction chunk
CHUNK_POST_ROWS <- 2048L   # rows per post-processing chunk

# ------------------------------ TERRA OPTIONS --------------------------------
Sys.setenv(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", GDAL_NUM_THREADS="2")
terraOptions(progress = 1, memfrac = 0.6)
# terraOptions(tempdir="/fast/tmp") # optional fast SSD temp

# ------------------------------ WRITE OPTIONS --------------------------------
wopt_flt <- list(datatype="FLT4S", gdal="COMPRESS=DEFLATE,ZLEVEL=6,PREDICTOR=3")
wopt_i16 <- list(datatype="INT2S",  gdal="COMPRESS=DEFLATE,ZLEVEL=6")

# --------------------------------- HELPERS -----------------------------------
align_to_grid <- function(src, target, categorical=FALSE) {
  meth <- if (categorical) "near" else "bilinear"
  if (!same.crs(src, target)) project(src, target, method=meth) else resample(src, target, method=meth)
}
force_mask_1_NA <- function(mask_r) {
  u <- unique(na.omit(as.vector(values(mask_r))))
  if (length(u) == 0) return(mask_r)
  if (any(u == 0, na.rm=TRUE)) classify(mask_r, rbind(c(-Inf,0,NA), c(0.5,Inf,1))) else mask_r
}
make_blocks <- function(r, chunk=1024L) {
  if (!inherits(r, "SpatRaster")) stop("make_blocks(): 'r' is not a SpatRaster, got: ", paste(class(r), collapse=", "))
  nr <- terra::nrow(r); if (!is.finite(nr) || nr <= 0) stop("make_blocks(): invalid nrow(r) = ", nr)
  chunk <- as.integer(chunk)[1]; if (is.na(chunk) || chunk <= 0) stop("'chunk' must be positive integer")
  rows  <- seq.int(1L, nr, by=chunk)
  nrows <- pmin.int(chunk, nr - rows + 1L)
  list(n=length(rows), row=rows, nrows=nrows)
}
to_int_vec <- function(v, lo=AGE_MIN, hi=AGE_MAX) { v <- round(v); v[v < lo] <- lo; v[v > hi] <- hi; v }
agree_pairs_vec <- function(a, b, tol=1) { ok <- is.finite(a) & is.finite(b); res <- rep(NA_real_, length(a)); res[ok] <- as.numeric(abs(a[ok]-b[ok]) <= tol); res }
iqr_vec <- function(a, b) { res <- rep(NA_real_, length(a)); ok <- is.finite(a) & is.finite(b); res[ok] <- abs(a[ok]-b[ok]); res }

# Safe prediction wrapper (works even if S3 dispatch is finicky)
predict_ranger_safe <- function(model, newdata, threads=1L) {
  out <- try(predict(model, data=newdata, num.threads=threads), silent=TRUE)
  if (inherits(out, "try-error")) {
    pr <- get("predict.ranger", envir=asNamespace("ranger"))
    out <- pr(model, data=newdata, num.threads=threads)
  }
  as.numeric(out$predictions)
}

# ============================ LOAD + PREP TRAIN DATA =========================
message("Loading training data…")
stopifnot(file.exists(TRAIN_CSV))
pts <- fread(TRAIN_CSV)

required_bands <- c("b1","b2","b3","b4","b5","b6")
stopifnot(all(required_bands %in% names(pts)), "year" %in% names(pts))
if (!("ysd" %in% names(pts))) { stopifnot("yod" %in% names(pts)); pts[, ysd := year - yod] }
stopifnot("NBR" %in% names(pts))

for (nm in c(required_bands, "ysd", "NBR")) if (!is.numeric(pts[[nm]])) pts[, (nm) := as.numeric(get(nm))]

train <- pts[year %in% TRAIN_YEARS & is.finite(ysd)]
train <- train[ysd >= AGE_MIN & ysd <= AGE_MAX]
train <- train[complete.cases(train[, ..required_bands])]
message(sprintf("Training rows: %d", nrow(train)))

# ================================ TRAIN MODELS ===============================
message("Training RF (BAP b1..b6)…")
model_bap <- ranger(
  ysd ~ ., data=as.data.frame(train[, c("ysd", required_bands), with=FALSE]),
  num.trees=1000, mtry=floor(sqrt(length(required_bands))),
  min.node.size=5, importance="impurity",
  num.threads=RF_THREADS_TRAIN, seed=42
)

message("Training RF (NBR-only)…")
model_nbr <- ranger(
  ysd ~ NBR, data=as.data.frame(train[, .(ysd, NBR)]),
  num.trees=1000, min.node.size=5,
  num.threads=RF_THREADS_TRAIN, seed=42
)

# ============================ PREP LOCAL TILE ================================
message("Preparing BAP tile, NBR, and forest mask…")
stopifnot(file.exists(BAP_TILE_PATH), file.exists(NBR1985_PATH), file.exists(FOREST_MASK))

r_bap <- rast(BAP_TILE_PATH); stopifnot(nlyr(r_bap) == 6); names(r_bap) <- required_bands

r_mask <- rast(FOREST_MASK)
r_mask <- align_to_grid(r_mask, r_bap, categorical=TRUE)
r_mask <- crop(r_mask, r_bap)
r_mask <- force_mask_1_NA(r_mask)

r_nbr <- rast(NBR1985_PATH); names(r_nbr) <- "NBR"
r_nbr <- align_to_grid(r_nbr, r_bap, categorical=FALSE)
r_nbr <- crop(r_nbr, r_bap)

# Forest-only
r_bap <- mask(r_bap, r_mask)
r_nbr <- mask(r_nbr, r_mask)

# Persist masked sources
bap_mask_path <- file.path(OUT_DIR, "BAP_1985_tile_forestOnly.tif")
nbr_mask_path <- file.path(OUT_DIR, "NBR_1985_tile_forestOnly.tif")
writeRaster(r_bap, bap_mask_path, overwrite=TRUE, wopt=wopt_flt)
writeRaster(r_nbr, nbr_mask_path, overwrite=TRUE, wopt=wopt_flt)
stopifnot(file.exists(bap_mask_path), file.exists(nbr_mask_path))

# Re-open for streaming I/O
r_bap <- rast(bap_mask_path)
r_nbr <- rast(nbr_mask_path)
message(sprintf("BAP dims: %s rows x %s cols x %s bands", terra::nrow(r_bap), terra::ncol(r_bap), terra::nlyr(r_bap)))
message(sprintf("NBR dims: %s rows x %s cols x %s bands", terra::nrow(r_nbr), terra::ncol(r_nbr), terra::nlyr(r_nbr)))

# ============================ BLOCK-WISE PREDICTION ==========================
ysd_bap_path <- file.path(OUT_DIR, "ysd_1985_BAP_tile.tif")
ysd_nbr_path <- file.path(OUT_DIR, "ysd_1985_NBR_tile.tif")

ysd_bap_out <- rast(r_bap, nlyrs=1); names(ysd_bap_out) <- "ysd_bap"
ysd_nbr_out <- rast(r_nbr, nlyrs=1); names(ysd_nbr_out) <- "ysd_nbr"

# Start writing (do NOT assign return values)
writeStart(ysd_bap_out, ysd_bap_path, overwrite=TRUE, wopt=wopt_flt)
writeStart(ysd_nbr_out, ysd_nbr_path, overwrite=TRUE, wopt=wopt_flt)

# Open inputs for streaming reads (do NOT assign)
readStart(r_bap); readStart(r_nbr)

bs <- make_blocks(r_bap, chunk=CHUNK_PRED_ROWS)
message(sprintf("Predicting in %d blocks…", bs$n))

# Precompute number of columns to verify block sizes
ncols_bap <- ncol(r_bap); ncols_nbr <- ncol(r_nbr)

for (i in seq_len(bs$n)) {
  # ---- BAP block ----
  bap_blk <- terra::readValues(r_bap, row=bs$row[i], nrows=bs$nrows[i], mat=TRUE)  # (ncells x 6)
  pred_bap <- rep(NA_real_, nrow(bap_blk))
  ok_bap <- rowSums(is.finite(bap_blk)) == ncol(bap_blk)
  if (any(ok_bap)) {
    pred_bap[ok_bap] <- predict_ranger_safe(
      model_bap,
      newdata = as.data.frame(bap_blk[ok_bap, , drop=FALSE]),
      threads = RF_THREADS_PRED
    )
  }
  # write as numeric vector; explicitly supply start and nrows
  terra::writeValues(ysd_bap_out, pred_bap, start=bs$row[i], nrows=bs$nrows[i])
  
  # ---- NBR block ----
  nbr_blk <- terra::readValues(r_nbr, row=bs$row[i], nrows=bs$nrows[i], mat=TRUE)  # (ncells x 1)
  pred_nbr <- rep(NA_real_, nrow(nbr_blk))
  ok_nbr <- is.finite(nbr_blk[,1])
  if (any(ok_nbr)) {
    pred_nbr[ok_nbr] <- predict_ranger_safe(
      model_nbr,
      newdata = data.frame(NBR = nbr_blk[ok_nbr,1]),
      threads = RF_THREADS_PRED
    )
  }
  terra::writeValues(ysd_nbr_out, pred_nbr, start=bs$row[i], nrows=bs$nrows[i])
  
  if (i %% 10 == 0) message(sprintf("  block %d/%d", i, bs$n))
}

# Close outputs and inputs
writeStop(ysd_bap_out); writeStop(ysd_nbr_out)
readStop(r_bap); readStop(r_nbr)

# ============================ INTEGER + YOD (BLOCK-WISE) =====================
ysd_bap_i_path <- file.path(OUT_DIR, "ysd_1985_BAP_tile_int.tif")
yod_bap_path   <- file.path(OUT_DIR, "yod_1985_BAP_tile.tif")
ysd_nbr_i_path <- file.path(OUT_DIR, "ysd_1985_NBR_tile_int.tif")
yod_nbr_path   <- file.path(OUT_DIR, "yod_1985_NBR_tile.tif")

for (src_path in c(ysd_bap_path, ysd_nbr_path)) {
  r_src <- rast(src_path)
  r_int <- rast(r_src, nlyrs=1); names(r_int) <- "ysd_int"
  r_yod <- rast(r_src, nlyrs=1); names(r_yod) <- "yod"
  
  writeStart(r_int, outname <- if (grepl("BAP", src_path)) ysd_bap_i_path else ysd_nbr_i_path,
             overwrite=TRUE, wopt=wopt_i16)
  writeStart(r_yod, outname_y <- if (grepl("BAP", src_path)) yod_bap_path else yod_nbr_path,
             overwrite=TRUE, wopt=wopt_i16)
  
  readStart(r_src)
  
  bs2 <- make_blocks(r_src, chunk=CHUNK_POST_ROWS)
  for (i in seq_len(bs2$n)) {
    v <- terra::readValues(r_src, row=bs2$row[i], nrows=bs2$nrows[i], mat=TRUE)[,1]
    vint <- to_int_vec(v); vyod <- 1985 - vint
    terra::writeValues(r_int, vint, start=bs2$row[i], nrows=bs2$nrows[i])
    terra::writeValues(r_yod, vyod, start=bs2$row[i], nrows=bs2$nrows[i])
  }
  
  readStop(r_src)
  writeStop(r_int); writeStop(r_yod)
}

# ===================== CONVERGENCE OF EVIDENCE (BLOCK-WISE) ==================
r_bap_pred <- rast(ysd_bap_path)
r_nbr_pred <- rast(ysd_nbr_path)

ens_median_path <- file.path(OUT_DIR, "ysd_1985_tile_ensemble_median.tif")
agree_pairs_path<- file.path(OUT_DIR, "ysd_1985_tile_agreement_pairs.tif")
spread_iqr_path <- file.path(OUT_DIR, "ysd_1985_tile_spread_IQR.tif")
spread_sd_path  <- file.path(OUT_DIR, "ysd_1985_tile_spread_SD.tif")

r_med <- rast(r_bap_pred, nlyrs=1); names(r_med) <- "median"
r_agr <- rast(r_bap_pred, nlyrs=1); names(r_agr) <- "agree_pairs"
r_iqr <- rast(r_bap_pred, nlyrs=1); names(r_iqr) <- "IQR"
r_sd  <- rast(r_bap_pred, nlyrs=1); names(r_sd)  <- "SD"

writeStart(r_med, ens_median_path, overwrite=TRUE, wopt=wopt_flt)
writeStart(r_agr, agree_pairs_path, overwrite=TRUE, wopt=wopt_flt)
writeStart(r_iqr, spread_iqr_path, overwrite=TRUE, wopt=wopt_flt)
writeStart(r_sd,  spread_sd_path,  overwrite=TRUE, wopt=wopt_flt)

readStart(r_bap_pred); readStart(r_nbr_pred)

bs3 <- make_blocks(r_bap_pred, chunk=CHUNK_POST_ROWS)
message(sprintf("Computing CoE in %d blocks…", bs3$n))
for (i in seq_len(bs3$n)) {
  a <- terra::readValues(r_bap_pred, row=bs3$row[i], nrows=bs3$nrows[i], mat=TRUE)[,1]
  b <- terra::readValues(r_nbr_pred, row=bs3$row[i], nrows=bs3$nrows[i], mat=TRUE)[,1]
  
  med <- ifelse(is.finite(a) & is.finite(b), (a+b)/2, ifelse(is.finite(a), a, ifelse(is.finite(b), b, NA_real_)))
  agr <- agree_pairs_vec(a, b, tol=1)
  iqr <- iqr_vec(a, b)
  sdv <- rep(NA_real_, length(a)); ok <- is.finite(a) & is.finite(b)
  sdv[ok] <- sqrt(((a[ok]-med[ok])^2 + (b[ok]-med[ok])^2)/2)
  
  terra::writeValues(r_med, med, start=bs3$row[i], nrows=bs3$nrows[i])
  terra::writeValues(r_agr, agr, start=bs3$row[i], nrows=bs3$nrows[i])
  terra::writeValues(r_iqr, iqr, start=bs3$row[i], nrows=bs3$nrows[i])
  terra::writeValues(r_sd,  sdv, start=bs3$row[i], nrows=bs3$nrows[i])
  
  if (i %% 10 == 0) message(sprintf("  CoE block %d/%d", i, bs3$n))
}

readStop(r_bap_pred); readStop(r_nbr_pred)
writeStop(r_med); writeStop(r_agr); writeStop(r_iqr); writeStop(r_sd)

message("Done.")
message(sprintf("Outputs written to: %s", OUT_DIR))
