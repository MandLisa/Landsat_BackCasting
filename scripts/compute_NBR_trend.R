#!/usr/bin/env Rscript
# ============================================================
# Efficient NBR 1985–1990 summary stats + trend (slope)
# Forest mask: forest = 1, non-forest = NA (NoData)
# - reads VRT lazily (no full load)
# - crops + masks to forest first (major speed-up)
# - uses vectorised terra ops for mean / SD where available
# - computes slope via closed-form regression (no per-pixel lm())
# - writes outputs block-wise to GeoTIFF (BigTIFF + compression)
# ============================================================

suppressPackageStartupMessages({
  library(terra)
})

# -----------------------------
# 0) INPUTS
# -----------------------------
in_dir      <- "/mnt/eo/eu_mosaics/NBR_comp"
vrt_file    <- file.path(in_dir, "NBR_1985_1990.vrt")

forest_file <- "/mnt/eo/EFDA_v211/forest_landuse_aligned.tif"  # <-- CHANGE THIS

# If your NBR nodata is encoded (e.g., -10000), set here. Otherwise set to NULL.
# Values <= nodata_threshold become NA.
nodata_threshold <- -9990   # set NULL to skip

# Output filenames
out_mean  <- file.path(in_dir, "NBR_1985_1990_mean_forest.tif")
out_sd    <- file.path(in_dir, "NBR_1985_1990_sd_forest.tif")
out_slope <- file.path(in_dir, "NBR_1985_1990_slope_forest.tif")

# -----------------------------
# 1) terra options (limit RAM)
# -----------------------------
terraOptions(memfrac = 0.25, progress = 1)

# -----------------------------
# 2) Read rasters (lazy)
# -----------------------------
message("Reading VRT stack...")
r <- rast(vrt_file)

years <- 1985:1990
if (nlyr(r) != length(years)) {
  stop("Expected 6 layers (1985–1990). Found: ", nlyr(r))
}
names(r) <- paste0("NBR_", years)

message("Reading forest mask (forest=1, non-forest=NA)...")
f0 <- rast(forest_file)

# Align forest mask to r (do this ONCE; cheaper than resampling time stack)
if (!compareGeom(r, f0, stopOnError = FALSE)) {
  message("Forest mask not aligned to NBR stack -> projecting mask to NBR grid (nearest neighbour).")
  f0 <- project(f0, r, method = "near")
}

# Build boolean forest mask:
# - TRUE where forest==1
# - FALSE elsewhere (including NA)
f <- (f0 == 1)
f[is.na(f)] <- FALSE

# -----------------------------
# 3) Crop + mask FIRST (huge I/O saving)
# -----------------------------
message("Cropping NBR stack to forest extent...")
r_crop <- crop(r, ext(f))

message("Masking NBR stack to forest pixels...")
# keep only TRUE cells; FALSE -> NA
r_for <- mask(r_crop, f, maskvalues = FALSE)

# -----------------------------
# 4) Optional nodata -> NA (do after masking to touch fewer pixels)
# -----------------------------
if (!is.null(nodata_threshold)) {
  message("Classifying NBR nodata values <= ", nodata_threshold, " to NA (block-wise)...")
  r_for <- classify(
    r_for,
    rcl = matrix(c(-Inf, nodata_threshold, NA), ncol = 3, byrow = TRUE)
  )
}

# -----------------------------
# 5) Write options
# -----------------------------
wopt <- list(
  datatype = "FLT4S",
  gdal = c("COMPRESS=DEFLATE", "PREDICTOR=2", "BIGTIFF=YES")
)

# -----------------------------
# 6) Mean + SD (prefer specialised ops)
# -----------------------------
message("Computing mean (block-wise) -> ", out_mean)
mean_r <- mean(r_for, na.rm = TRUE, filename = out_mean, overwrite = TRUE, wopt = wopt)

message("Computing SD (block-wise) -> ", out_sd)
if ("stdev" %in% ls(getNamespace("terra"))) {
  sd_r <- stdev(r_for, na.rm = TRUE, filename = out_sd, overwrite = TRUE, wopt = wopt)
} else {
  message("terra::stdev() not available; falling back to app(sd).")
  sd_r <- app(r_for, fun = sd, na.rm = TRUE, filename = out_sd, overwrite = TRUE, wopt = wopt)
}

# -----------------------------
# 7) Slope (closed-form; avoids lm per pixel)
# -----------------------------
message("Computing slope (per-year) -> ", out_slope)

x <- years

slope_fun_fast <- function(v) {
  ok <- is.finite(v)
  n  <- sum(ok)
  if (n < 2) return(NA_real_)
  
  xv <- x[ok]
  yv <- v[ok]
  
  xb <- mean(xv)
  yb <- mean(yv)
  
  sxx <- sum((xv - xb)^2)
  if (sxx == 0) return(NA_real_)
  
  sum((xv - xb) * (yv - yb)) / sxx
}

slope_r <- app(
  r_for,
  fun = slope_fun_fast,
  filename = out_slope,
  overwrite = TRUE,
  wopt = wopt
)

message("Done.")
message("Wrote:")
message(" - ", out_mean)
message(" - ", out_sd)
message(" - ", out_slope)
