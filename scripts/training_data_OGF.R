# ======================================================================
# Backcasting training data preparation (tile-wise, RAM-safe)
# Using:
#   /mnt/dss_europe/level3_interpolated/<TILE>/*_IBAP*.tif
#   /mnt/dss_europe/level3_interpolated/<TILE>/*_NBR*.tif
# ======================================================================

suppressPackageStartupMessages({
  library(terra)
  library(data.table)
})

# ---------------------------- SETTINGS ---------------------------------

root_dir_interp <- "/mnt/dss_europe/level3_interpolated"   # contains tile folders
root_dir_yod    <- "/mnt/dss_europe/disturbance_yod"       # year-of-disturbance raster per tile (1 band)
root_dir_fmask  <- "/mnt/dss_europe/forest_mask"           # forest mask per tile (1=forest)

out_dir <- "/mnt/dss_europe/training_tables"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# File patterns inside each tile folder
ibap_suffix_regex <- "_IBAP.*\\.tif$"
nbr_suffix_regex  <- "_NBR.*\\.tif$"

# Reference years for training (must allow full lookback + post window)
t0_years <- 2005:2014

lookback_years <- 20
post_years     <- 10

# IBAP window around t0: [t0-1 ... t0+3] mirrors 1984–1988 for t0=1985
ibap_pre  <- 1
ibap_post <- 3

# Sampling per tile per t0 (after QC, per class)
n_per_class <- 50            # start small; scale to 100–150 later
oversample_factor <- 4       # sample more candidates, keep n_per_class after QC
seed_base <- 42

# Nodata handling
nodata_values <- c(-10000, -9999, -32768)

# Minimum valid observations required
min_valid_ibap_years <- 3    # out of 5 years
min_valid_nbr_years  <- 8    # out of 11 years

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

# ----------------------- MAIN PER TILE ---------------------------------

make_training_for_tile <- function(tile) {
  message("Tile: ", tile)
  
  tile_dir <- file.path(root_dir_interp, tile)
  
  yod_file   <- pick_one_file(list.files(file.path(root_dir_yod, tile), full.names = TRUE))
  fmask_file <- pick_one_file(list.files(file.path(root_dir_fmask, tile), full.names = TRUE))
  
  if (is.na(yod_file) || is.na(fmask_file)) {
    warning("Missing YOD or forest mask for tile ", tile, " -> skipping.")
    return(NULL)
  }
  
  yod   <- rast(yod_file)
  fmask <- rast(fmask_file)
  
  if (!compareGeom(yod, fmask, stopOnError = FALSE)) {
    stop("Geometry mismatch between YOD and forest mask for tile ", tile)
  }
  
  tile_dt_list <- list()
  
  for (t0 in t0_years) {
    message("  t0 = ", t0)
    
    ibap_years <- (t0 - ibap_pre):(t0 + ibap_post)
    nbr_years  <- t0:(t0 + post_years)
    
    ibap_files <- list_year_files(tile_dir, ibap_years, ibap_suffix_regex)
    nbr_files  <- list_year_files(tile_dir, nbr_years,  nbr_suffix_regex)
    
    if (anyNA(ibap_files) || anyNA(nbr_files)) {
      message("    Missing IBAP/NBR files for this t0 -> skipping t0.")
      next
    }
    
    # Valid forest area
    valid <- (fmask == 1)
    
    # Exclude disturbances in post window [t0 .. t0+post_years]
    post_ok <- is.na(yod) | yod == 0 | yod < t0 | yod > (t0 + post_years)
    valid <- valid & post_ok
    
    # Class masks based on lookback [t0-20 .. t0-1]
    disturbed <- valid & (yod >= (t0 - lookback_years) & yod <= (t0 - 1))
    undist    <- valid & (is.na(yod) | yod == 0 | yod < (t0 - lookback_years) | yod > (t0 - 1))
    
    disturbed_m <- ifel(disturbed, 1, NA)
    undist_m    <- ifel(undist,    1, NA)
    
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
    
    # Extract YOD at points for metadata + ysd (not for predictors at inference)
    yod_val <- terra::extract(yod, pts, ID = FALSE)[, 1]
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
    
    ibap_feat <- ibap_median_features(ibap_mat, n_years = length(ibap_years), band_names = band_names)
    
    # -------------------- Extract NBR metrics ----------------------
    nbr_r  <- rast(unname(nbr_files))
    nbr_ex <- as.data.table(terra::extract(nbr_r, pts, ID = FALSE))
    nbr_ex[] <- lapply(nbr_ex, clean_nodata)
    nbr_mat <- as.matrix(nbr_ex)
    
    nbr_feat <- nbr_metrics(nbr_mat, years_vec = nbr_years)
    
    # -------------------- Assemble table --------------------------
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

# Start with a single test tile first:
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