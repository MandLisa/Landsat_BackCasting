library(terra)
library(data.table)
library(stringr)
# optional for a short hash of the CRS (helps grouping WKT strings)
safelabel <- function(x) substr(digest::digest(x), 1, 8)

# --- INPUTS -------------------------------------------------------------------
bap_dir <- "/mnt/dss_europe/mosaics_eu/mosaics_eu_baps"
bap_pat <- "^(\\d{4})_mosaic_eu_cog\\.tif$"

files <- list.files(bap_dir, pattern = bap_pat, full.names = TRUE)
years <- as.integer(str_match(basename(files), bap_pat)[,2])
tbl <- data.table(file = files, year = years)[!is.na(year)][order(year)]

# --- PROBE GEOMETRY PER FILE --------------------------------------------------
probe_one <- function(f) {
  r <- rast(f)
  e <- ext(r)
  data.table(
    file = f,
    crs_wkt   = crs(r, proj = TRUE),
    crs_hash  = safelabel(crs(r, proj = TRUE)),
    res_x     = round(xres(r), 9),
    res_y     = round(yres(r), 9),
    xmin      = round(e$xmin, 3),
    xmax      = round(e$xmax, 3),
    ymin      = round(e$ymin, 3),
    ymax      = round(e$ymax, 3),
    ncol      = ncol(r),
    nrow      = nrow(r),
    nbands    = nlyr(r)
  )
}

geom <- rbindlist(lapply(tbl$file, probe_one))
geom[, year := tbl$year]

# --- FIND THE REFERENCE GEOMETRY (most common combo) --------------------------
geom_key <- c("crs_hash","res_x","res_y","xmin","xmax","ymin","ymax","ncol","nrow")
geom[, combo := do.call(paste, c(.SD, sep="|")), .SDcols = geom_key]

ref_combo <- geom[, .N, by = combo][order(-N)][1, combo]
ref <- geom[combo == ref_combo][1]

# --- REPORT SUMMARY -----------------------------------------------------------
cat("Most common geometry (reference):\n")
print(ref[, c("crs_hash","res_x","res_y","xmin","xmax","ymin","ymax","ncol","nrow","nbands")])

cat("\nUnique geometry combos and counts:\n")
print(geom[, .N, by = combo][order(-N)])

# --- LIST DEVIATIONS ----------------------------------------------------------
devs <- geom[combo != ref_combo][order(year)]
if (nrow(devs)) {
  cat("\nFiles that deviate from the reference grid:\n")
  print(devs[, .(year, file, crs_hash, res_x, res_y, xmin, xmax, ymin, ymax, ncol, nrow, nbands)])
  
  # Pinpoint *what* differs for each deviant file
  diff_cols <- function(row, refrow, keys = geom_key) {
    bad <- keys[vapply(keys, function(k) !identical(row[[k]], refrow[[k]]), logical(1))]
    paste(bad, collapse = ", ")
  }
  msg <- devs[, .(year, file, differs = vapply(seq_len(.N),
                                               function(i) diff_cols(devs[i], ref), character(1)))]
  cat("\nWhat differs (per file):\n")
  print(msg)
} else {
  cat("\nAll BAPs share the same grid. ✅\n")
}

# --- OPTIONAL: quick table by year -------------------------------------------
geom_by_year <- geom[, .(crs_hash, res_x, res_y, xmin, xmax, ymin, ymax, ncol, nrow, nbands), by=year]




library(terra)
library(data.table)
library(stringr)

# --- INPUTS -------------------------------------------------------------------
bap_dir <- "/mnt/dss_europe/mosaics_eu/mosaics_eu_baps"
bap_pat <- "^(\\d{4})_mosaic_eu_cog\\.tif$"
out_dir <- "/mnt/dss_europe/temp_lm"
bad_years <- c(1995, 2003)   # from your check

dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# Files & years
files <- list.files(bap_dir, pattern = bap_pat, full.names = TRUE)
years <- as.integer(stringr::str_match(basename(files), bap_pat)[,2])
tbl   <- data.table(file = files, year = years)[!is.na(year)][order(year)]

# Reference grid (pick a good, non-deviating year; 1985 is common)
ref_file <- tbl[year == 1985, file][1]
ref_r    <- rast(ref_file)

# Pad & write aligned copies for deviators into out_dir
for (yy in bad_years) {
  f <- tbl[year == yy, file][1]
  r <- rast(f)
  
  if (!compareGeom(r, ref_r, stopOnError = FALSE)) {
    r_fix <- extend(r, ext(ref_r))  # pad missing north rows as NA
    out_f <- file.path(out_dir, sprintf("%d_mosaic_eu_cog_aligned.tif", yy))
    writeRaster(
      r_fix, out_f, overwrite = TRUE,
      gdal = c("TILED=YES", "COMPRESS=LZW", "PREDICTOR=2")
    )
    # point table to the aligned copy
    tbl[year == yy, file := out_f]
  }
}

# Quick sanity check: stacking should now work
S_test <- rast(tbl$file)  # if this succeeds, the grid is consistent


