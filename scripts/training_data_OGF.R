# ======================================================================
# Training data preparation for backcasting forest continuity
# (tile-wise, RAM-safe, reproducible sampling)
#
# CHANGE vs. previous version:
#   - YOD raster and forest mask are read as EUROPEAN MOSAICS (single files)
#   - For each tile, we crop/resample those mosaics to the tile grid
# ======================================================================

suppressPackageStartupMessages({
  library(terra)
  library(data.table)
})

# ---------------------------- SETTINGS ---------------------------------

# Root directory containing all tile folders
root_dir_interp <- "/mnt/dss_europe/level3_interpolated"

# European mosaic disturbance year raster (YOD) and forest mask
# Provide either a direct .tif path OR a directory containing the mosaic tif.
yod_mosaic_path   <- "/mnt/dss_europe/disturbance_yod"  # <- directory or file
fmask_mosaic_path <- "/mnt/dss_europe/forest_mask"      # <- directory or file

# Output directory
out_dir <- "/mnt/dss_europe/training_tables"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# File patterns inside each tile folder
ibap_suffix_regex <- "_IBAP.*\\.tif$"
nbr_suffix_regex  <- "_NBR.*\\.tif$"

# Training reference years t0
t0_years <- 2005:2014

# Time windows
lookback_years <- 20
post_years     <- 10

# IBAP window definition: 3 years starting at t0 => [t0 .. t0+2]
ibap_start_offset <- 0
ibap_end_offset   <- 2

# Sampling size (after QC) per tile and t0
n_per_class <- 50
oversample_factor <- 4
seed_base <- 42

# Nodata handling
nodata_values <- c(-10000, -9999, -32768)

# Minimum valid observations
min_valid_ibap_years <- 2   # out of 3
min_valid_nbr_years  <- 8   # out of 11 (t0..t0+10)

# -------------------------- HELPERS ------------------------------------

parse_year <- function(x) as.integer(substr(basename(x), 1, 4))

pick_one_file <- function(files) {
  if (length(files) == 0) return(NA_character_)
  files <- sort(files)
  files[1]
}

get_mosaic_rast <- function(path_or_dir) {
  if (file.exists(path_or_dir) && !dir.exists(path_or_dir)) {
    return(rast(path_or_dir))
  }
  if (dir.exists(path_or_dir)) {
    ff <- pick_one_file(list.files(path_or_dir, pattern = "\\.tif$", full.names = TRUE))
    if (is.na(ff)) stop("No .tif found in: ", path_or_dir)
    return(rast(ff))
  }
  stop("Path does not exist: ", path_or_dir)
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

clean_nodata <- function(x, nodata = nodata_values) {
  for (v in nodata) x[x == v] <- NA_real_
  x
}

row_median_fast <- function(mat) {
  if (requireNamespace("matrixStats", quietly = TRUE)) {
    return(matrixStats::rowMedians(mat, na.rm = TRUE))
  }
  apply(mat, 1, median, na.rm = TRUE)
}

theil_sen_slope <- function(y_mat, t_vec) {
  k <- length(t_vec)
  idx <- which(upper.tri(matrix(NA_real_, k, k)), arr.ind = TRUE)
  i1 <- idx[, 1]; i2 <- idx[, 2]
  dt <- t_vec[i2] - t_vec[i1]
  slopes <- (y_mat[, i2, drop = FALSE] - y_mat[, i1, drop = FALSE]) / rep(dt, each = nrow(y_mat))
  row_median_fast(slopes)
}

nbr_metrics <- function(nbr_mat, years_vec) {
  out <- data.table(
    nbr_n_valid = rowSums(!is.na(nbr_mat)),
    nbr_t0      = nbr_mat[, 1],
    nbr_mean    = rowMeans(nbr_mat, na.rm = TRUE),
    nbr_min     = apply(nbr_mat, 1, min, na.rm = TRUE),
    nbr_max     = apply(nbr_mat, 1, max, na.rm = TRUE),
    nbr_sd      = apply(nbr_mat, 1, sd,  na.rm = TRUE),
    nbr_mad     = apply(nbr_mat, 1, mad, na.rm = TRUE)
  )
  out[, nbr_range := nbr_max - nbr_min]
  out[, nbr_sen_slope := theil_sen_slope(nbr_mat, years_vec)]
  out
}

ibap_median_features <- function(ibap_extract_mat, n_years, band_names) {
  n_bands <- length(band_names)
  stopifnot(ncol(ibap_extract_mat) == n_years * n_bands)
  
  feats <- vector("list", n_bands)
  names(feats) <- band_names
  
  for (b in seq_len(n_bands)) {
    cols <- seq(from = b, by = n_bands, length.out = n_years)
    feats[[b]] <- row_median_fast(ibap_extract_mat[, cols, drop = FALSE])
  }
  
  out <- as.data.table(feats)
  setnames(out, paste0("ibap_", make.names(band_names), "_med"))
  out
}

sample_points_from_mask <- function(mask_r, n, seed) {
  set.seed(seed)
  terra::spatSample(mask_r, size = n, method = "random",
                    na.rm = TRUE, xy = TRUE, as.points = TRUE, values = FALSE)
}

# ---------------------- READ MOSAICS ONCE ------------------------------

yod_mosaic   <- get_mosaic_rast(yod_mosaic_path)
fmask_mosaic <- get_mosaic_rast(fmask_mosaic_path)

# ----------------------- MAIN PER TILE ---------------------------------

make_training_for_tile <- function(tile) {
  message("Tile: ", tile)
  tile_dir <- file.path(root_dir_interp, tile)
  
  # Build a tile template grid from any IBAP/NBR file (only for geometry)
  any_ibap <- pick_one_file(list.files(tile_dir, pattern = ibap_suffix_regex, full.names = TRUE))
  any_nbr  <- pick_one_file(list.files(tile_dir, pattern = nbr_suffix_regex,  full.names = TRUE))
  template_file <- if (!is.na(any_ibap)) any_ibap else any_nbr
  if (is.na(template_file)) {
    warning("No IBAP/NBR files found in tile folder ", tile, " -> skipping.")
    return(NULL)
  }
  
  template <- rast(template_file)
  template1 <- template[[1]]
  
  # Crop mosaics to tile extent and align to tile grid (RAM-safe window reads)
  yod_t   <- crop(yod_mosaic,   ext(template1), snap = "out")
  fmask_t <- crop(fmask_mosaic, ext(template1), snap = "out")
  
  if (ncell(yod_t) == 0 || ncell(fmask_t) == 0) {
    warning("Tile extent outside mosaic coverage for tile ", tile, " -> skipping.")
    return(NULL)
  }
  
  # Ensure same geometry as tile grid (nearest neighbour for categorical rasters)
  if (!compareGeom(yod_t, template1, stopOnError = FALSE)) {
    yod_t <- resample(yod_t, template1, method = "near")
  }
  if (!compareGeom(fmask_t, template1, stopOnError = FALSE)) {
    fmask_t <- resample(fmask_t, template1, method = "near")
  }
  
  tile_dt_list <- list()
  
  for (t0 in t0_years) {
    message("  t0 = ", t0)
    
    ibap_years <- (t0 + ibap_start_offset):(t0 + ibap_end_offset)  # [t0 .. t0+2]
    nbr_years  <- t0:(t0 + post_years)                             # [t0 .. t0+10]
    
    ibap_files <- list_year_files(tile_dir, ibap_years, ibap_suffix_regex)
    nbr_files  <- list_year_files(tile_dir, nbr_years,  nbr_suffix_regex)
    
    if (anyNA(ibap_files) || anyNA(nbr_files)) {
      message("    Missing IBAP/NBR files for this t0 -> skipping t0.")
      next
    }
    
    # Base valid area: forest mask (on tile-aligned mosaic crop)
    valid <- (fmask_t == 1)
    
    # Exclude disturbances in the post window [t0 .. t0+10]
    post_ok <- is.na(yod_t) | yod_t == 0 | yod_t < t0 | yod_t > (t0 + post_years)
    valid <- valid & post_ok
    
    # Class masks based on lookback [t0-20 .. t0-1]
    disturbed <- valid & (yod_t >= (t0 - lookback_years) & yod_t <= (t0 - 1))
    undist    <- valid & (is.na(yod_t) | yod_t == 0 | yod_t < (t0 - lookback_years))
    
    disturbed_m <- ifel(disturbed, 1, NA)
    undist_m    <- ifel(undist,    1, NA)
    
    # Oversample candidates to survive QC filtering
    n_need <- n_per_class * oversample_factor
    
    pts_d <- try(sample_points_from_mask(disturbed_m, n_need, seed_base + t0 * 10 + 1), silent = TRUE)
    pts_u <- try(sample_points_from_mask(undist_m,    n_need, seed_base + t0 * 10 + 2), silent = TRUE)
    
    if (inherits(pts_d, "try-error") || inherits(pts_u, "try-error")) {
      message("    Sampling failed (too few valid cells) -> skipping t0.")
      next
    }
    
    pts_d$state <- "disturbed"
    pts_u$state <- "undisturbed"
    pts <- rbind(pts_d, pts_u)
    
    # Extract YOD at points (metadata)
    yod_val <- terra::extract(yod_t, pts, ID = FALSE)[, 1]
    yod_val <- clean_nodata(yod_val)
    ysd_val <- ifelse(!is.na(yod_val) & yod_val >= (t0 - lookback_years) & yod_val <= (t0 - 1),
                      t0 - yod_val, NA_real_)
    
    # -------------------- Extract IBAP features --------------------
    ibap_r <- rast(unname(ibap_files))
    ibap1  <- rast(unname(ibap_files)[1])
    
    band_names <- names(ibap1)
    if (is.null(band_names) || anyNA(band_names)) band_names <- paste0("B", seq_len(nlyr(ibap1)))
    
    ibap_ex <- as.data.table(terra::extract(ibap_r, pts, ID = FALSE))
    ibap_ex[] <- lapply(ibap_ex, clean_nodata)
    ibap_mat <- as.matrix(ibap_ex)
    
    ibap_n_valid_all <- rowSums(!is.na(ibap_mat))
    
    ibap_feat <- ibap_median_features(
      ibap_extract_mat = ibap_mat,
      n_years = length(ibap_years),
      band_names = band_names
    )
    
    # -------------------- Extract NBR metrics ----------------------
    nbr_r  <- rast(unname(nbr_files))
    nbr_ex <- as.data.table(terra::extract(nbr_r, pts, ID = FALSE))
    nbr_ex[] <- lapply(nbr_ex, clean_nodata)
    nbr_mat <- as.matrix(nbr_ex)
    
    nbr_feat <- nbr_metrics(nbr_mat, years_vec = nbr_years)
    
    # -------------------- Assemble table ---------------------------
    xy <- crds(pts, df = TRUE)
    
    dt <- data.table(
      point_id = sprintf("%s_t0%04d_%06d", tile, t0, seq_len(nrow(xy))),
      x = xy[, 1], y = xy[, 2],
      tile = tile, t0 = t0,
      state = pts$state,
      label_undisturbed20y = as.integer(pts$state == "undisturbed"),
      yod = yod_val,
      ysd = ysd_val,
      ibap_n_valid_all = ibap_n_valid_all
    )
    
    dt <- cbind(dt, ibap_feat, nbr_feat)
    
    # QC filters
    dt <- dt[
      nbr_n_valid >= min_valid_nbr_years &
        ibap_n_valid_all >= (min_valid_ibap_years * length(band_names))
    ]
    
    # Balance classes after QC
    dt_u <- dt[state == "undisturbed"]
    dt_d <- dt[state == "disturbed"]
    
    if (nrow(dt_u) == 0 || nrow(dt_d) == 0) {
      message("    After QC: one class empty -> skipping t0.")
      next
    }
    
    set.seed(seed_base + t0)
    dt_u <- dt_u[sample.int(nrow(dt_u), min(n_per_class, nrow(dt_u)))]
    dt_d <- dt_d[sample.int(nrow(dt_d), min(n_per_class, nrow(dt_d)))]
    
    tile_dt_list[[as.character(t0)]] <- rbind(dt_u, dt_d)
    
    rm(ibap_r, ibap1, nbr_r, ibap_ex, nbr_ex, ibap_mat, nbr_mat, dt, dt_u, dt_d)
    gc()
  }
  
  if (length(tile_dt_list) == 0) return(NULL)
  rbindlist(tile_dt_list, use.names = TRUE, fill = TRUE)
}

# ----------------------------- RUN -------------------------------------

tiles <- list.dirs(root_dir_interp, full.names = FALSE, recursive = FALSE)
tiles <- tiles[grepl("^X\\d{4}_Y\\d{4}$", tiles)]
if (length(tiles) == 0) stop("No tiles found under root_dir_interp.")

# Quick test:
# tiles <- "X0002_Y0024"

for (tile in tiles) {
  dt_tile <- make_training_for_tile(tile)
  if (is.null(dt_tile)) next
  
  out_file <- file.path(out_dir, paste0("training_", tile, ".csv"))
  fwrite(dt_tile, out_file)
  message("Wrote: ", out_file)
  
  rm(dt_tile)
  gc()
}