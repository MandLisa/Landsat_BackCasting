# ================================================================
# Pixel-wise NBR trend (slope) for 1985–1990
#
# Purpose:
#   - quantify short-term spectral recovery signal
#   - support convergence-of-evidence for backcasting (~1987)
#
# Output:
#   - single raster with NBR slope (units: NBR per year)
# ================================================================

library(terra)

# ----------------------------------------------------------------
# 1. Load annual NBR rasters
# ----------------------------------------------------------------
nbr_dir <- "/mnt/dss_europe/level3_interpolated/X0016_Y0020"

# Regex: Jahr 1985–1990 am Dateianfang
pattern <- "^(1985|1986|1987|1988|1989|1990)0801_LEVEL3_LNDLG_NBR\\.tif$"

nbr_files <- list.files(
  nbr_dir,
  pattern     = pattern,
  full.names = TRUE
)

# Jahr aus Dateinamen extrahieren (erste 4 Zeichen)
nbr_years <- as.integer(substr(basename(nbr_files), 1, 4))

# Sicherheit
stopifnot(all(nbr_years %in% 1985:1990))

# Nach Jahr sortieren (sehr wichtig!)
ord <- order(nbr_years)
nbr_files <- nbr_files[ord]
nbr_years <- nbr_years[ord]

# Kontrolle
print(nbr_years)
print(basename(nbr_files))

# stack
nbr_stack <- rast(nbr_files)

# extract years from filenames
years <- as.integer(substr(basename(nbr_files), 1, 4))

stopifnot(length(years) == nlyr(nbr_stack))


# ----------------------------------------------------------------
# 2. Define slope function
# ----------------------------------------------------------------
# Linear trend of NBR over time
# Requires at least 4 valid observations

nbr_trend_fun <- function(v, yrs) {
  if (sum(!is.na(v)) < 4) return(NA_real_)
  coef(lm(v ~ yrs))[2]
}



# ----------------------------------------------------------------
# 3. Compute NBR slope raster
# ----------------------------------------------------------------
nbr_slope <- app(
  nbr_stack,
  fun   = nbr_trend_fun,
  yrs   = years,   # <-- DAS ist der entscheidende Punkt
  cores = 4
)


names(nbr_slope) <- "nbr_slope_1985_1990"


# ----------------------------------------------------------------
# 4. Write output
# ----------------------------------------------------------------

writeRaster(
  nbr_slope,
  "/mnt/eo/EO4Backcasting/_output/nbr_slope_1985_1990.tif",
  datatype  = "FLT4S",
  gdal      = c("COMPRESS=LZW", "TILED=YES"),
  overwrite = TRUE
)
