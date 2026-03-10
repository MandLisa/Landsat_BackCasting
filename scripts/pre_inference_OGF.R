# ======================================================================
# Pre-inference checks for mosaic vs tile consistency
# Checks:
#   1) CRS match (tile vs mosaics)
#   2) After crop+resample: geometry match (extent/resolution/origin)
#   3) Required annual files exist for t0=1985 inference windows
#   4) Basic sanity: forest mask has forest pixels in tile; YOD has coverage
# ======================================================================

suppressPackageStartupMessages({
  library(terra)
  library(data.table)
})

# ---------------------------- SETTINGS ---------------------------------

root_dir_interp <- "/mnt/dss_europe/level3_interpolated"

yod_mosaic_path   <- "/mnt/eo/EFDA_v211/yod_aligned.tif"
fmask_mosaic_path <- "/mnt/eo/EFDA_v211/forest_landuse_aligned.tif"

ibap_suffix_regex <- "_IBAP.*\\.tif$"
nbr_suffix_regex  <- "_NBR.*\\.tif$"

t0 <- 1985
ibap_years <- 1985:1987
nbr_years  <- 1985:1995

forest_values <- c(1)
nodata_values <- c(-10000, -9999, -32768)

# How many tiles to check (set NA for all)
n_tiles_check <- NA_integer_  # e.g., 20

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

geom_key <- function(r) {
  list(
    crs = as.character(crs(r)),
    res = res(r),
    ext = ext(r),
    nrow = nrow(r),
    ncol = ncol(r)
  )
}

# ---------------------- LOAD MOSAICS ONCE ------------------------------

stopifnot(file.exists(yod_mosaic_path), file.exists(fmask_mosaic_path))
yod_mosaic   <- rast(yod_mosaic_path)
fmask_mosaic <- rast(fmask_mosaic_path)

# ---------------------- SELECT TILES -----------------------------------

tiles <- list.dirs(root_dir_interp, full.names = FALSE, recursive = FALSE)
tiles <- tiles[grepl("^X\\d{4}_Y\\d{4}$", tiles)]
if (length(tiles) == 0) stop("No tiles found under root_dir_interp.")

if (!is.na(n_tiles_check) && length(tiles) > n_tiles_check) {
  set.seed(42)
  tiles <- sample(tiles, n_tiles_check)
}

# ---------------------- RUN CHECKS -------------------------------------

results <- rbindlist(lapply(tiles, function(tile) {
  tile_dir <- file.path(root_dir_interp, tile)
  
  any_ibap <- pick_one_file(list.files(tile_dir, pattern = ibap_suffix_regex, full.names = TRUE))
  any_nbr  <- pick_one_file(list.files(tile_dir, pattern = nbr_suffix_regex,  full.names = TRUE))
  template_file <- if (!is.na(any_ibap)) any_ibap else any_nbr
  
  if (is.na(template_file)) {
    return(data.table(tile = tile, ok = FALSE, reason = "No IBAP/NBR template file found"))
  }
  
  template1 <- rast(template_file)[[1]]
  
  # Check required years exist
  ibap_files <- list_year_files(tile_dir, ibap_years, ibap_suffix_regex)
  nbr_files  <- list_year_files(tile_dir, nbr_years,  nbr_suffix_regex)
  
  miss_ibap <- names(ibap_files)[is.na(ibap_files)]
  miss_nbr  <- names(nbr_files)[is.na(nbr_files)]
  
  # CRS check (mosaics vs tile)
  crs_tile  <- as.character(crs(template1))
  crs_yod   <- as.character(crs(yod_mosaic))
  crs_fmask <- as.character(crs(fmask_mosaic))
  
  crs_ok <- (crs_tile == crs_yod) && (crs_tile == crs_fmask)
  
  # Crop mosaics to tile and align
  yod_t   <- crop(yod_mosaic,   ext(template1), snap = "out")
  fmask_t <- crop(fmask_mosaic, ext(template1), snap = "out")
  
  if (ncell(yod_t) == 0 || ncell(fmask_t) == 0) {
    return(data.table(tile = tile, ok = FALSE, reason = "Tile outside mosaic coverage",
                      crs_ok = crs_ok,
                      miss_ibap = paste(miss_ibap, collapse = ","),
                      miss_nbr = paste(miss_nbr, collapse = ",")))
  }
  
  if (!compareGeom(yod_t, template1, stopOnError = FALSE))   yod_t <- resample(yod_t, template1, method = "near")
  if (!compareGeom(fmask_t, template1, stopOnError = FALSE)) fmask_t <- resample(fmask_t, template1, method = "near")
  
  yod_t   <- clean_nodata_rast(yod_t)
  fmask_t <- clean_nodata_rast(fmask_t)
  
  geom_ok_yod   <- compareGeom(yod_t, template1, stopOnError = FALSE)
  geom_ok_fmask <- compareGeom(fmask_t, template1, stopOnError = FALSE)
  
  # Sanity: forest pixels exist
  forest_ok <- forest_bool(fmask_t, forest_values)
  n_forest  <- global(forest_ok, "sum", na.rm = TRUE)[1, 1]
  
  # Sanity: yod coverage exists (some non-NA)
  n_yod_non_na <- global(!is.na(yod_t), "sum", na.rm = TRUE)[1, 1]
  
  ok <- TRUE
  reasons <- character(0)
  
  if (!crs_ok) reasons <- c(reasons, "CRS mismatch (tile vs mosaics)")
  if (length(miss_ibap) > 0) reasons <- c(reasons, paste0("Missing IBAP years: ", paste(miss_ibap, collapse = ",")))
  if (length(miss_nbr)  > 0) reasons <- c(reasons, paste0("Missing NBR years: ",  paste(miss_nbr,  collapse = ",")))
  if (!geom_ok_yod) reasons <- c(reasons, "YOD not aligned to tile after crop/resample")
  if (!geom_ok_fmask) reasons <- c(reasons, "Forest mask not aligned to tile after crop/resample")
  if (is.na(n_forest) || n_forest == 0) reasons <- c(reasons, "No forest pixels in mask for this tile")
  if (is.na(n_yod_non_na) || n_yod_non_na == 0) reasons <- c(reasons, "No YOD coverage (all NA) in this tile")
  
  if (length(reasons) > 0) ok <- FALSE
  
  data.table(
    tile = tile,
    ok = ok,
    reason = ifelse(ok, "", paste(reasons, collapse = " | ")),
    crs_ok = crs_ok,
    geom_ok_yod = geom_ok_yod,
    geom_ok_fmask = geom_ok_fmask,
    forest_pixels = n_forest,
    yod_non_na_pixels = n_yod_non_na
  )
}), fill = TRUE)

# Print summary
cat("\nSUMMARY\n")
print(results[, .(
  n_tiles = .N,
  n_ok = sum(ok, na.rm=TRUE),
  n_fail = sum(!ok, na.rm=TRUE)
)])

cat("\nFAILED TILES (first 30)\n")
print(results[ok == FALSE][1:min(30, .N)])

# Save full report
out_report <- file.path(dirname(root_dir_interp), "pre_inference_checks_t0_1985.csv")
fwrite(results, out_report)
cat("\nWrote report:", out_report, "\n")