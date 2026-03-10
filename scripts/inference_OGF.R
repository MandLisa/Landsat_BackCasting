# ======================================================================
# TILE-WISE INFERENCE (t0 = 1985)
# Predict P(undisturbed in the 20 years before 1985) for each tile
#
# Features (must match training):
#   - IBAP median per band over 1985–1987 (t0..t0+2)
#   - NBR metrics over 1985–1995 (t0..t0+10), incl. Theil–Sen slope
#
# Masks:
#   - forest max extent (from European mosaic forest_landuse_aligned.tif)
#   - no disturbance in 1985–1995 (feature hygiene; from yod_aligned.tif)
#
# Inputs:
#   /mnt/dss_europe/level3_interpolated/<TILE>/*_IBAP*.tif
#   /mnt/dss_europe/level3_interpolated/<TILE>/*_NBR*.tif
#   /mnt/eo/EFDA_v211/yod_aligned.tif               (European mosaic)
#   /mnt/eo/EFDA_v211/forest_landuse_aligned.tif    (European mosaic)
#   /mnt/eo/EO4Backcasting/model_xgb/xgb_model.json
#   /mnt/eo/EO4Backcasting/model_xgb/feature_list.csv
# ======================================================================

suppressPackageStartupMessages({
  library(terra)
  library(data.table)
  library(xgboost)
})

# ---------------------------- SETTINGS ---------------------------------

root_dir_interp <- "/mnt/dss_europe/level3_interpolated"

model_dir <- "/mnt/eo/EO4Backcasting/model_xgb"
model_file <- file.path(model_dir, "xgb_model.json")
feature_list_file <- file.path(model_dir, "feature_list.csv")

yod_mosaic_path   <- "/mnt/eo/EFDA_v211/yod_aligned.tif"
fmask_mosaic_path <- "/mnt/eo/EFDA_v211/forest_landuse_aligned.tif"

out_dir_pred <- "/mnt/eo/EO4Backcasting/predictions_t0_1985"
dir.create(out_dir_pred, showWarnings = FALSE, recursive = TRUE)

ibap_suffix_regex <- "_IBAP.*\\.tif$"
nbr_suffix_regex  <- "_NBR.*\\.tif$"

# Fixed inference reference
t0 <- 1985
ibap_years <- 1985:1987
nbr_years  <- 1985:1995

# Forest mask coding (adjust if needed)
forest_values <- c(1)

# NoData values used in your products
nodata_values <- c(-10000, -9999, -32768)

# Optional: run only on a subset first
# tiles_subset <- c("X0001_Y0024", "X0002_Y0024")
tiles_subset <- NULL

# -------------------------- HELPERS ------------------------------------

parse_year <- function(x) as.integer(substr(basename(x), 1, 4))

pick_one_file <- function(files) {
  if (length(files) == 0) return(NA_character_)
  files <- sort(files)
  files[1]
}

list_year_files <- function(tile_dir, years, suffix_regex) {
  f <- list.files(tile_dir, pattern = suffix_regex, full.names = TRUE)
  if (length(f) == 0) return(setNames(rep(NA_character_, length(years)), years))
  y <- parse_year(f)
  out <- vapply(years, function(yy) {
    ff <- f[y == yy]
    pick_one_file(ff)
  }, FUN.VALUE = character(1))
  names(out) <- years
  out
}

clean_nodata_rast <- function(r, nodata = nodata_values) {
  for (v in nodata) r <- ifel(r == v, NA, r)
  r
}

forest_bool <- function(fmask_r, forest_values) {
  m <- fmask_r == forest_values[1]
  if (length(forest_values) > 1) {
    for (v in forest_values[-1]) m <- m | (fmask_r == v)
  }
  m
}

# Vectorized Theil–Sen slope for a fixed-length vector v (length = 11 here)
# Uses precomputed pair indices for speed and ignores NA pairs automatically.
make_sen_fun <- function(k, years_vec) {
  stopifnot(length(years_vec) == k)
  idx <- which(upper.tri(matrix(FALSE, k, k)), arr.ind = TRUE)
  i1 <- idx[, 1]
  i2 <- idx[, 2]
  dt <- years_vec[i2] - years_vec[i1]
  
  function(v) {
    # v length k
    if (all(is.na(v))) return(NA_real_)
    slopes <- (v[i2] - v[i1]) / dt
    if (all(is.na(slopes))) return(NA_real_)
    stats::median(slopes, na.rm = TRUE)
  }
}

# Build NBR metric rasters from an 11-layer stack (1985..1995)
nbr_metric_stack <- function(nbr_stack, years_vec) {
  k <- length(years_vec)
  stopifnot(nlyr(nbr_stack) == k)
  
  sen_fun <- make_sen_fun(k, years_vec)
  
  fun <- function(v) {
    n_valid <- sum(!is.na(v))
    if (n_valid == 0) return(rep(NA_real_, 9))
    t0v <- v[1]
    meanv <- mean(v, na.rm = TRUE)
    minv  <- min(v,  na.rm = TRUE)
    maxv  <- max(v,  na.rm = TRUE)
    sdv   <- stats::sd(v,  na.rm = TRUE)
    madv  <- stats::mad(v, na.rm = TRUE)
    rangev <- maxv - minv
    slopev <- sen_fun(v)
    c(n_valid, t0v, meanv, minv, maxv, sdv, madv, rangev, slopev)
  }
  
  out <- app(nbr_stack, fun = fun)
  names(out) <- c(
    "nbr_n_valid", "nbr_t0", "nbr_mean", "nbr_min", "nbr_max",
    "nbr_sd", "nbr_mad", "nbr_range", "nbr_sen_slope"
  )
  out
}

# Build IBAP median per band from 3 annual multi-band rasters
# Output names: ibap_B1_med ... ibap_Bn_med
ibap_median_stack <- function(ibap_files) {
  ibap_stack <- rast(ibap_files)     # concatenates layers from all files
  ibap_stack <- clean_nodata_rast(ibap_stack)
  
  # Determine number of bands per year
  n_total <- nlyr(ibap_stack)
  n_years <- length(ibap_files)
  n_bands <- n_total / n_years
  if (n_bands != round(n_bands)) stop("IBAP stack has unexpected number of layers.")
  n_bands <- as.integer(n_bands)
  
  band_names <- paste0("B", seq_len(n_bands))
  out_list <- vector("list", n_bands)
  
  for (b in seq_len(n_bands)) {
    # layer indices for band b across the 3 years
    idx <- c(b, b + n_bands, b + 2 * n_bands)
    rb <- ibap_stack[[idx]]
    out_list[[b]] <- app(rb, fun = function(v) stats::median(v, na.rm = TRUE))
  }
  
  out <- rast(out_list)
  names(out) <- paste0("ibap_", band_names, "_med")
  out
}

# Optionally compute IBAP completeness (count non-NA across all IBAP layers)
ibap_n_valid_all_raster <- function(ibap_files) {
  ibap_stack <- rast(ibap_files)
  ibap_stack <- clean_nodata_rast(ibap_stack)
  app(ibap_stack, fun = function(v) sum(!is.na(v)))
}

# Block-wise prediction: reads feature raster values in chunks and writes output
predict_xgb_raster <- function(feature_r, model, out_file) {
  bs <- blocks(feature_r)
  
  out_r <- rast(feature_r[[1]])
  out_r <- writeStart(out_r, filename = out_file, overwrite = TRUE)
  
  for (i in 1:bs$n) {
    v <- readValues(feature_r, row = bs$row[i], nrows = bs$nrows[i], mat = TRUE)
    if (is.null(v) || nrow(v) == 0) {
      out_r <- writeValues(out_r, values = numeric(0), start = bs$row[i])
      next
    }
    
    # keep NA rows as NA
    all_na <- apply(v, 1, function(x) all(is.na(x)))
    pred <- rep(NA_real_, nrow(v))
    
    if (any(!all_na)) {
      dm <- xgb.DMatrix(data = v[!all_na, , drop = FALSE])
      pred[!all_na] <- predict(model, dm)
    }
    
    out_r <- writeValues(out_r, pred, start = bs$row[i])
  }
  
  writeStop(out_r)
  invisible(out_r)
}

# ---------------------- LOAD MODEL + FEATURE LIST -----------------------

stopifnot(file.exists(model_file), file.exists(feature_list_file))
model <- xgb.load(model_file)
feat_list <- fread(feature_list_file)$feature

# ---------------------- LOAD MOSAICS ONCE ------------------------------

stopifnot(file.exists(yod_mosaic_path), file.exists(fmask_mosaic_path))
yod_mosaic   <- rast(yod_mosaic_path)
fmask_mosaic <- rast(fmask_mosaic_path)

# ----------------------------- RUN -------------------------------------

tiles <- list.dirs(root_dir_interp, full.names = FALSE, recursive = FALSE)
tiles <- tiles[grepl("^X\\d{4}_Y\\d{4}$", tiles)]
if (length(tiles) == 0) stop("No tiles found under root_dir_interp.")

if (!is.null(tiles_subset)) {
  tiles <- intersect(tiles, tiles_subset)
  if (length(tiles) == 0) stop("tiles_subset does not match any tiles on disk.")
}

for (tile in tiles) {
  message("Tile: ", tile)
  tile_dir <- file.path(root_dir_interp, tile)
  
  # Template grid from any IBAP/NBR file
  any_ibap <- pick_one_file(list.files(tile_dir, pattern = ibap_suffix_regex, full.names = TRUE))
  any_nbr  <- pick_one_file(list.files(tile_dir, pattern = nbr_suffix_regex,  full.names = TRUE))
  template_file <- if (!is.na(any_ibap)) any_ibap else any_nbr
  if (is.na(template_file)) {
    message("  No IBAP/NBR files found -> skip")
    next
  }
  template1 <- rast(template_file)[[1]]
  
  # Annual inputs
  ibap_files <- list_year_files(tile_dir, ibap_years, ibap_suffix_regex)
  nbr_files  <- list_year_files(tile_dir, nbr_years,  nbr_suffix_regex)
  
  if (anyNA(ibap_files) || anyNA(nbr_files)) {
    message("  Missing required years (IBAP or NBR) -> skip")
    next
  }
  
  # Crop mosaics to tile extent and align to tile grid
  yod_t   <- crop(yod_mosaic,   ext(template1), snap = "out")
  fmask_t <- crop(fmask_mosaic, ext(template1), snap = "out")
  
  if (ncell(yod_t) == 0 || ncell(fmask_t) == 0) {
    message("  Tile outside mosaic coverage -> skip")
    next
  }
  
  if (!compareGeom(yod_t, template1, stopOnError = FALSE))   yod_t <- resample(yod_t, template1, method = "near")
  if (!compareGeom(fmask_t, template1, stopOnError = FALSE)) fmask_t <- resample(fmask_t, template1, method = "near")
  
  yod_t   <- clean_nodata_rast(yod_t)
  fmask_t <- clean_nodata_rast(fmask_t)
  
  # Inference mask:
  # - within forest max extent
  # - no disturbance in 1985–1995 (feature hygiene for NBR metrics)
  forest_ok <- forest_bool(fmask_t, forest_values)
  post_ok   <- is.na(yod_t) | yod_t == 0 | yod_t < min(nbr_years) | yod_t > max(nbr_years)
  infer_ok  <- forest_ok & post_ok
  
  # Convert to 1/NA mask for terra::mask
  infer_m <- ifel(infer_ok, 1, NA)
  
  # Build feature rasters
  ibap_med <- ibap_median_stack(unname(ibap_files))
  
  nbr_stack <- rast(unname(nbr_files))
  nbr_stack <- clean_nodata_rast(nbr_stack)
  nbr_feat  <- nbr_metric_stack(nbr_stack, years_vec = nbr_years)
  
  feat_r <- c(ibap_med, nbr_feat)
  
  # Add IBAP completeness only if model expects it
  if ("ibap_n_valid_all" %in% feat_list && !("ibap_n_valid_all" %in% names(feat_r))) {
    ibap_nv <- ibap_n_valid_all_raster(unname(ibap_files))
    names(ibap_nv) <- "ibap_n_valid_all"
    feat_r <- c(feat_r, ibap_nv)
  }
  
  # Apply inference mask (outside -> NA)
  feat_r <- mask(feat_r, infer_m)
  
  # Enforce exact feature set + order required by the trained model
  missing <- setdiff(feat_list, names(feat_r))
  if (length(missing) > 0) {
    stop("Missing features for tile ", tile, ": ", paste(missing, collapse = ", "))
  }
  feat_r <- feat_r[[feat_list]]
  
  # Predict and write
  out_file <- file.path(out_dir_pred, paste0("p_undisturbed20y_t0_1985_", tile, ".tif"))
  message("  Writing: ", out_file)
  predict_xgb_raster(feat_r, model, out_file)
  
  rm(ibap_med, nbr_stack, nbr_feat, feat_r, yod_t, fmask_t, infer_ok, infer_m)
  gc()
}