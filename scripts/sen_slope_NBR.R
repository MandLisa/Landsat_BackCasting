# ======================================================================
# EO data cube: Sen's slope (Theil–Sen) on NBR for 1985–1995
#
# Folder structure:
#   root_dir/
#     X0001_Y0024/
#       19910801_LEVEL3_LNDLG_NBR.tif
#       ...
#
# Output:
#   - per-tile slope rasters:  slope_NBR_1985_1995_<tile>.tif
#   - final mosaic:            mosaic_slope_NBR_1985_1995.tif
#
# Key assumptions / design:
# - YEAR is the first 4 characters of the filename (YYYY)
# - NBR rasters are float in ~[-1, 1] and missing values are native NA/NaN
# - RAM efficient: process one tile at a time; write outputs to disk
# - Mosaic is incremental (pairwise), with periodic checkpoint writes
# ======================================================================

suppressPackageStartupMessages({
  library(terra)
  library(data.table)
})

# ---------------------------- SETTINGS --------------------------------
root_dir <- "/mnt/dss_europe/level3_interpolated"
years    <- 1985:1995

# output
out_dir <- "/mnt/eo/EO4Backcasting/_sen_slope_1985_1995/"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

out_tile_dir <- file.path(out_dir, "tile_slopes")
dir.create(out_tile_dir, showWarnings = FALSE, recursive = TRUE)

out_mosaic <- file.path(out_dir, "mosaic_slope_NBR_1985_1995.tif")

# tempdir: must exist BEFORE terraOptions()
# Prefer a fast local scratch if available; fall back to out_dir/tmp
scratch_base <- Sys.getenv("SCRATCH")
if (nzchar(scratch_base) && dir.exists(scratch_base)) {
  tmp_dir <- file.path(scratch_base, "terra_tmp_sen_slope_1985_1995")
} else {
  tmp_dir <- file.path(out_dir, "tmp")
}
dir.create(tmp_dir, showWarnings = FALSE, recursive = TRUE)

terraOptions(
  tempdir = tmp_dir,
  memfrac = 0.7,
  todisk  = TRUE
)

# ------------------------- HELPER FUNCTIONS ---------------------------

# Extract year from basename: first 4 chars
get_year <- function(f) suppressWarnings(as.integer(substr(basename(f), 1, 4)))

# Choose exactly one file per year; if multiple exist, pick the first after sorting
pick_one_per_year <- function(files, years) {
  dt <- data.table(file = files)
  dt[, year := get_year(file)]
  dt <- dt[!is.na(year) & year %in% years]
  if (nrow(dt) == 0) return(character(0))
  
  setorder(dt, year, file)
  
  dup_years <- dt[duplicated(year), unique(year)]
  if (length(dup_years) > 0) {
    message("  Note: multiple NBR files found; keeping first for years: ",
            paste(dup_years, collapse = ", "))
  }
  
  dt <- dt[!duplicated(year)]
  dt <- dt[order(match(year, years))]
  dt$file
}

# Precompute pair indices + dt once per tile, then compute median pairwise slope per pixel
make_theil_sen_fun <- function(t_vec) {
  idx <- utils::combn(seq_along(t_vec), 2)
  dt  <- t_vec[idx[2, ]] - t_vec[idx[1, ]]
  
  function(y) {
    # y is the pixel time series (length = nlayers)
    ok <- is.finite(y)
    if (sum(ok) < 2) return(NA_real_)
    
    y2 <- y
    y2[!ok] <- NA_real_
    
    a <- y2[idx[1, ]]
    b <- y2[idx[2, ]]
    s <- (b - a) / dt
    
    stats::median(s, na.rm = TRUE)
  }
}

# ------------------------- INDEX TILE FOLDERS --------------------------
tile_dirs <- list.dirs(root_dir, recursive = FALSE, full.names = TRUE)
tile_dirs <- tile_dirs[grepl("^X\\d{4}_Y\\d{4}$", basename(tile_dirs))]

if (length(tile_dirs) == 0) {
  stop("No tile folders found under root_dir. Expected names like X0001_Y0024.")
}

message("Found ", length(tile_dirs), " tile folders.")

# ------------------------- PROCESS EACH TILE ---------------------------
tile_slope_files <- character(0)

for (tile_path in tile_dirs) {
  tile_id <- basename(tile_path)
  message("\nProcessing tile: ", tile_id)
  
  # Find all NBR tifs in this tile folder
  nbr_files <- list.files(tile_path, pattern = "NBR.*\\.tif$", full.names = TRUE, ignore.case = TRUE)
  if (length(nbr_files) == 0) {
    message("  Skipping (no NBR tif found).")
    next
  }
  
  # Pick one file per year (1985..1995)
  yr_files <- pick_one_per_year(nbr_files, years)
  if (length(yr_files) < 2) {
    message("  Skipping (need at least 2 years, found ", length(yr_files), ").")
    next
  }
  
  found_years <- get_year(yr_files)
  missing_years <- setdiff(years, found_years)
  if (length(missing_years) > 0) {
    message("  Missing years: ", paste(missing_years, collapse = ", "),
            " (slope uses available years only).")
  }
  
  out_tile <- file.path(out_tile_dir, paste0("slope_NBR_1985_1995_", tile_id, ".tif"))
  if (file.exists(out_tile)) {
    message("  Output exists, skipping computation: ", out_tile)
    tile_slope_files <- c(tile_slope_files, out_tile)
    next
  }
  
  # Load small stack (<= 11 layers). NA are native NA/NaN (no recoding needed).
  r <- rast(yr_files)
  
  # Ensure layer order corresponds to the time vector
  t_vec <- get_year(yr_files)
  
  # Build per-pixel Sen slope function with precomputed pair indices
  sen_fun <- make_theil_sen_fun(t_vec)
  
  # Compute Sen slope pixel-wise (block-wise)
  slope_r <- app(r, sen_fun, cores = 1)
  names(slope_r) <- "sen_slope"
  
  # Write immediately to disk
  writeRaster(
    slope_r,
    filename = out_tile,
    overwrite = TRUE,
    wopt = list(gdal = c("COMPRESS=DEFLATE", "PREDICTOR=3", "ZLEVEL=6"))
  )
  
  tile_slope_files <- c(tile_slope_files, out_tile)
  
  rm(r, slope_r)
  gc()
}

if (length(tile_slope_files) == 0) {
  stop("No per-tile slope rasters were produced. Check patterns and years.")
}

message("\nPer-tile slope rasters: ", length(tile_slope_files))

# ------------------------- MOSAIC INCREMENTALLY ------------------------
# Assumes tiles are aligned (same CRS/resolution). If not, resample/warp first.

if (file.exists(out_mosaic)) {
  message("Mosaic exists, skipping: ", out_mosaic)
} else {
  message("\nBuilding mosaic incrementally...")
  
  mos <- rast(tile_slope_files[1])
  
  for (i in 2:length(tile_slope_files)) {
    message("  Mosaicking ", i, "/", length(tile_slope_files), ": ", basename(tile_slope_files[i]))
    r_i <- rast(tile_slope_files[i])
    
    # If overlaps exist: mean is conservative; change if you have a preferred rule
    mos <- mosaic(mos, r_i, fun = "mean")
    
    # Periodic checkpoint write to keep memory stable for very many tiles
    if (i %% 10 == 0) {
      tmp_mos <- file.path(out_dir, "tmp_mosaic.tif")
      writeRaster(
        mos, tmp_mos, overwrite = TRUE,
        wopt = list(gdal = c("COMPRESS=DEFLATE", "PREDICTOR=3", "ZLEVEL=6"))
      )
      mos <- rast(tmp_mos)
      gc()
    }
  }
  
  writeRaster(
    mos, out_mosaic, overwrite = TRUE,
    wopt = list(gdal = c("COMPRESS=DEFLATE", "PREDICTOR=3", "ZLEVEL=6"))
  )
  
  message("Final mosaic written: ", out_mosaic)
}

message("\nDone.")