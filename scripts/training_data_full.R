suppressPackageStartupMessages({
  library(terra)
  library(data.table)
})

# ---------------------------- SETTINGS ---------------------------------

root_dir_interp <- "/mnt/dss_europe/level3_interpolated"

yod_mosaic_path   <- "/mnt/eo/EFDA_v211/yod_aligned.tif"
fmask_mosaic_path <- "/mnt/eo/EFDA_v211/forest_landuse_aligned.tif"

out_dir <- "/mnt/eo/EO4Backcasting/training_data"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

ibap_suffix_regex <- "_IBAP.*\\.tif$"
nbr_suffix_regex  <- "_NBR.*\\.tif$"

t0_years <- 2005:2014

lookback_years <- 20
post_years     <- 10

# IBAP window: 3 years starting at t0 => [t0..t0+2]
ibap_start_offset <- 0
ibap_end_offset   <- 2

n_per_class <- 50
oversample_factor <- 4
seed_base <- 42

nodata_values <- c(-10000, -9999, -32768)

min_valid_ibap_years <- 2
min_valid_nbr_years  <- 8

forest_values <- c(1)

enforce_unique_cells_across_t0 <- TRUE
max_sampling_attempts <- 8

write_meta_csv <- TRUE

# optional: _n Spalten mitschreiben?
write_ibap_n <- TRUE

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

clean_nodata_vec <- function(x, nodata = nodata_values) {
  for (v in nodata) x[x == v] <- NA_real_
  x
}

clean_nodata_rast <- function(r, nodata = nodata_values) {
  for (v in nodata) r <- ifel(r == v, NA, r)
  r
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

safe_iqr_row <- function(mat) {
  apply(mat, 1, function(x) {
    x <- x[is.finite(x)]
    if (length(x) == 0) return(NA_real_)
    IQR(x, na.rm = TRUE, type = 7)
  })
}

safe_mad_row <- function(mat) {
  apply(mat, 1, function(x) {
    x <- x[is.finite(x)]
    if (length(x) == 0) return(NA_real_)
    mad(x, center = median(x), constant = 1, na.rm = TRUE)
  })
}

safe_q_row <- function(mat, prob) {
  apply(mat, 1, function(x) {
    x <- x[is.finite(x)]
    if (length(x) == 0) return(NA_real_)
    as.numeric(quantile(x, probs = prob, names = FALSE, type = 7, na.rm = TRUE))
  })
}

safe_n_row <- function(mat) {
  rowSums(is.finite(mat))
}

# NEU: mehrere IBAP-Metriken statt nur Median
ibap_summary_features <- function(ibap_extract_mat, n_years, band_names, write_n = TRUE) {
  n_bands <- length(band_names)
  stopifnot(ncol(ibap_extract_mat) == n_years * n_bands)
  
  out_list <- vector("list", length = 0)
  
  for (b in seq_len(n_bands)) {
    cols <- seq(from = b, by = n_bands, length.out = n_years)
    band_mat <- ibap_extract_mat[, cols, drop = FALSE]
    band_name <- band_names[b]
    
    out_list[[paste0("ibap_", band_name, "_med")]] <- row_median_fast(band_mat)
    out_list[[paste0("ibap_", band_name, "_iqr")]] <- safe_iqr_row(band_mat)
    out_list[[paste0("ibap_", band_name, "_mad")]] <- safe_mad_row(band_mat)
    out_list[[paste0("ibap_", band_name, "_p10")]] <- safe_q_row(band_mat, 0.10)
    out_list[[paste0("ibap_", band_name, "_p90")]] <- safe_q_row(band_mat, 0.90)
    
    if (write_n) {
      out_list[[paste0("ibap_", band_name, "_n")]] <- safe_n_row(band_mat)
    }
  }
  
  as.data.table(out_list)
}

forest_bool <- function(fmask_r, forest_values) {
  m <- fmask_r == forest_values[1]
  if (length(forest_values) > 1) {
    for (v in forest_values[-1]) m <- m | (fmask_r == v)
  }
  m
}

sample_unique_points <- function(mask_r, n_target, seed, template_r, used_cells = integer(0)) {
  if (n_target <= 0) return(list(pts = NULL, cell = integer(0)))
  
  collected_pts <- NULL
  collected_cells <- integer(0)
  
  attempt <- 0
  while (length(collected_cells) < n_target && attempt < max_sampling_attempts) {
    attempt <- attempt + 1
    set.seed(seed + attempt)
    
    n_try <- max(n_target * 2, 200)
    pts_try <- try(
      terra::spatSample(mask_r, size = n_try, method = "random",
                        na.rm = TRUE, xy = TRUE, as.points = TRUE, values = FALSE),
      silent = TRUE
    )
    if (inherits(pts_try, "try-error")) next
    
    xy_try <- crds(pts_try, df = TRUE)
    cell_try <- terra::cellFromXY(template_r, xy_try)
    
    keep <- !(cell_try %in% used_cells) & !(cell_try %in% collected_cells)
    if (!any(keep)) next
    
    pts_keep <- pts_try[keep, ]
    cell_keep <- cell_try[keep]
    
    n_need <- n_target - length(collected_cells)
    if (length(cell_keep) > n_need) {
      sel <- seq_len(n_need)
      pts_keep <- pts_keep[sel, ]
      cell_keep <- cell_keep[sel]
    }
    
    if (is.null(collected_pts)) collected_pts <- pts_keep else collected_pts <- rbind(collected_pts, pts_keep)
    collected_cells <- c(collected_cells, cell_keep)
  }
  
  list(pts = collected_pts, cell = collected_cells)
}

# ---------------------- READ MOSAICS ONCE ------------------------------

yod_mosaic   <- get_mosaic_rast(yod_mosaic_path)
fmask_mosaic <- get_mosaic_rast(fmask_mosaic_path)

# ----------------------- MAIN PER TILE ---------------------------------

make_training_for_tile <- function(tile) {
  message("Tile: ", tile)
  tile_dir <- file.path(root_dir_interp, tile)
  
  any_ibap <- pick_one_file(list.files(tile_dir, pattern = ibap_suffix_regex, full.names = TRUE))
  any_nbr  <- pick_one_file(list.files(tile_dir, pattern = nbr_suffix_regex,  full.names = TRUE))
  template_file <- if (!is.na(any_ibap)) any_ibap else any_nbr
  if (is.na(template_file)) {
    warning("No IBAP/NBR files found in tile folder ", tile, " -> skipping.")
    return(NULL)
  }
  
  template <- rast(template_file)
  template1 <- template[[1]]
  
  yod_t   <- crop(yod_mosaic,   ext(template1), snap = "out")
  fmask_t <- crop(fmask_mosaic, ext(template1), snap = "out")
  
  if (ncell(yod_t) == 0 || ncell(fmask_t) == 0) {
    warning("Tile extent outside mosaic coverage for tile ", tile, " -> skipping.")
    return(NULL)
  }
  
  if (!compareGeom(yod_t, template1, stopOnError = FALSE)) {
    yod_t <- resample(yod_t, template1, method = "near")
  }
  if (!compareGeom(fmask_t, template1, stopOnError = FALSE)) {
    fmask_t <- resample(fmask_t, template1, method = "near")
  }
  
  yod_t   <- clean_nodata_rast(yod_t,   nodata_values)
  fmask_t <- clean_nodata_rast(fmask_t, nodata_values)
  
  used_cells <- integer(0)
  tile_dt_list <- list()
  
  for (t0 in t0_years) {
    message("  t0 = ", t0)
    
    ibap_years <- (t0 + ibap_start_offset):(t0 + ibap_end_offset)
    nbr_years  <- t0:(t0 + post_years)
    
    ibap_files <- list_year_files(tile_dir, ibap_years, ibap_suffix_regex)
    nbr_files  <- list_year_files(tile_dir, nbr_years,  nbr_suffix_regex)
    
    if (anyNA(ibap_files) || anyNA(nbr_files)) {
      message("    Missing IBAP/NBR files for this t0 -> skipping t0.")
      next
    }
    
    valid <- forest_bool(fmask_t, forest_values)
    post_ok <- is.na(yod_t) | yod_t == 0 | yod_t < t0 | yod_t > (t0 + post_years)
    valid <- valid & post_ok
    
    disturbed <- valid & (yod_t >= (t0 - lookback_years) & yod_t <= (t0 - 1))
    undist <- valid & (is.na(yod_t) | yod_t == 0 | yod_t < (t0 - lookback_years) | yod_t > (t0 - 1))
    
    disturbed_m <- ifel(disturbed, 1, NA)
    undist_m    <- ifel(undist,    1, NA)
    
    n_need <- n_per_class * oversample_factor
    
    if (enforce_unique_cells_across_t0) {
      s_d <- sample_unique_points(disturbed_m, n_need, seed_base + t0 * 10 + 1, template1, used_cells)
      pts_d <- s_d$pts
      cells_d <- s_d$cell
      
      s_u <- sample_unique_points(undist_m, n_need, seed_base + t0 * 10 + 2, template1, c(used_cells, cells_d))
      pts_u <- s_u$pts
      cells_u <- s_u$cell
    } else {
      pts_d <- try(terra::spatSample(disturbed_m, size = n_need, method = "random",
                                     na.rm = TRUE, xy = TRUE, as.points = TRUE, values = FALSE), silent = TRUE)
      pts_u <- try(terra::spatSample(undist_m, size = n_need, method = "random",
                                     na.rm = TRUE, xy = TRUE, as.points = TRUE, values = FALSE), silent = TRUE)
      if (!inherits(pts_d, "try-error")) cells_d <- terra::cellFromXY(template1, crds(pts_d, df = TRUE)) else cells_d <- integer(0)
      if (!inherits(pts_u, "try-error")) cells_u <- terra::cellFromXY(template1, crds(pts_u, df = TRUE)) else cells_u <- integer(0)
    }
    
    if (is.null(pts_d) || is.null(pts_u) || nrow(pts_d) == 0 || nrow(pts_u) == 0) {
      message("    Sampling failed (too few valid cells) -> skipping t0.")
      next
    }
    
    pts_d$state <- "disturbed"
    pts_u$state <- "undisturbed"
    pts <- rbind(pts_d, pts_u)
    cells <- c(cells_d, cells_u)
    
    yod_val <- terra::extract(yod_t, pts, ID = FALSE)[, 1]
    yod_val <- clean_nodata_vec(yod_val)
    
    ysd_val <- ifelse(!is.na(yod_val) & yod_val >= (t0 - lookback_years) & yod_val <= (t0 - 1),
                      t0 - yod_val, NA_real_)
    
    # -------------------- Extract IBAP features --------------------
    ibap_r <- rast(unname(ibap_files))
    n_bands <- nlyr(ibap_r) / length(ibap_years)
    if (n_bands != round(n_bands)) stop("IBAP stack has unexpected number of layers in tile ", tile)
    n_bands <- as.integer(n_bands)
    
    band_names <- paste0("B", seq_len(n_bands))
    
    ibap_ex <- as.data.table(terra::extract(ibap_r, pts, ID = FALSE))
    ibap_ex[] <- lapply(ibap_ex, clean_nodata_vec)
    ibap_mat <- as.matrix(ibap_ex)
    
    ibap_n_valid_all <- rowSums(!is.na(ibap_mat))
    
    ibap_feat <- ibap_summary_features(
      ibap_extract_mat = ibap_mat,
      n_years = length(ibap_years),
      band_names = band_names,
      write_n = write_ibap_n
    )
    
    # -------------------- Extract NBR metrics ----------------------
    nbr_r  <- rast(unname(nbr_files))
    nbr_ex <- as.data.table(terra::extract(nbr_r, pts, ID = FALSE))
    nbr_ex[] <- lapply(nbr_ex, clean_nodata_vec)
    nbr_mat <- as.matrix(nbr_ex)
    
    nbr_feat <- nbr_metrics(nbr_mat, years_vec = nbr_years)
    
    # -------------------- Assemble table ---------------------------
    xy <- crds(pts, df = TRUE)
    
    dt <- data.table(
      point_id = sprintf("%s_t0%04d_%06d", tile, t0, seq_len(nrow(xy))),
      x = xy[, 1], y = xy[, 2],
      tile = tile,
      t0 = t0,
      cell_id = cells,
      state = pts$state,
      label_undisturbed_20y = as.integer(pts$state == "undisturbed"),
      yod = yod_val,
      ysd = ysd_val,
      ibap_n_valid_all = ibap_n_valid_all
    )
    
    dt <- cbind(dt, ibap_feat, nbr_feat)
    
    dt <- dt[
      nbr_n_valid >= min_valid_nbr_years &
        ibap_n_valid_all >= (min_valid_ibap_years * length(band_names))
    ]
    
    dt_u <- dt[state == "undisturbed"]
    dt_d <- dt[state == "disturbed"]
    
    if (nrow(dt_u) == 0 || nrow(dt_d) == 0) {
      message("    After QC: one class empty -> skipping t0.")
      next
    }
    
    set.seed(seed_base + t0)
    dt_u <- dt_u[sample.int(nrow(dt_u), min(n_per_class, nrow(dt_u)))]
    dt_d <- dt_d[sample.int(nrow(dt_d), min(n_per_class, nrow(dt_d)))]
    
    dt_final <- rbind(dt_u, dt_d)
    
    if (enforce_unique_cells_across_t0) {
      used_cells <- unique(c(used_cells, dt_final$cell_id))
    }
    
    tile_dt_list[[as.character(t0)]] <- dt_final
    
    rm(ibap_r, nbr_r, ibap_ex, nbr_ex, ibap_mat, nbr_mat, dt, dt_u, dt_d, dt_final)
    gc()
  }
  
  if (length(tile_dt_list) == 0) return(NULL)
  
  dt_tile <- rbindlist(tile_dt_list, use.names = TRUE, fill = TRUE)
  
  ibap_cols <- grep("^ibap_", names(dt_tile), value = TRUE)
  nbr_cols  <- grep("^nbr_",  names(dt_tile), value = TRUE)
  
  dt_features <- dt_tile[, c("point_id", "x", "y", "tile", "label_undisturbed_20y", ibap_cols, nbr_cols), with = FALSE]
  dt_meta <- dt_tile
  
  list(features = dt_features, meta = dt_meta)
}

# ----------------------------- RUN -------------------------------------

tiles <- list.dirs(root_dir_interp, full.names = FALSE, recursive = FALSE)
tiles <- tiles[grepl("^X\\d{4}_Y\\d{4}$", tiles)]
if (length(tiles) == 0) stop("No tiles found under root_dir_interp.")

# tiles <- "X0001_Y0024"

for (tile in tiles) {
  res <- make_training_for_tile(tile)
  if (is.null(res)) next
  
  out_features <- file.path(out_dir, paste0("training_", tile, "_features.csv"))
  fwrite(res$features, out_features)
  message("Wrote: ", out_features)
  
  if (write_meta_csv) {
    out_meta <- file.path(out_dir, paste0("training_", tile, "_meta.csv"))
    fwrite(res$meta, out_meta)
    message("Wrote: ", out_meta)
  }
  
  rm(res)
  gc()
}
