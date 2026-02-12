# =====================================================================
# Sample 5000 pixels per zone and class from XGB probability rasters
# Directory: /mnt/eo/EO4Backcasting/_preds_Feb_3c
# Bands: 1=healthy, 2=ysd1-10, 3=ysd>10
# Condition: prob > 0.7
# Output CSV: zone, prob, class, x, y (centroids)
# =====================================================================

suppressPackageStartupMessages({
  library(terra)
  library(stringr)
  library(data.table)
})

in_dir  <- "/mnt/eo/EO4Backcasting/_preds_Feb_3c"
out_csv <- file.path(in_dir, "xgb_zone_samples_prob_gt_0p7__n5000_per_class.csv")

thr <- 0.8
k   <- 5000

zones <- c("boreal", "temperate", "mediterranean")
band_classes <- c("healthy", "ysd1-10", "ysd>10")  # bands 1..3

set.seed(42)

# ---- 1) Find rasters ----
files <- list.files(in_dir, pattern = "xgb.*\\.(tif|tiff)$", full.names = TRUE, ignore.case = TRUE)

infer_zone <- function(fp) {
  nm <- tolower(basename(fp))
  z <- zones[str_detect(nm, zones)]
  if (length(z) == 0) return(NA_character_)
  z[[1]]
}

zone_of_file <- vapply(files, infer_zone, character(1))
files <- files[!is.na(zone_of_file)]
zone_of_file <- zone_of_file[!is.na(zone_of_file)]

if (length(files) == 0) stop("No rasters with 'xgb' and zone keyword found in: ", in_dir)

# ---- 2) Helper: sample from a single band of one raster (thresholded) ----
sample_band_from_raster <- function(r, band, n, thr) {
  rb <- r[[band]]
  
  # read all values (terra streams from disk, but returns a vector)
  v <- terra::values(rb, mat = FALSE)
  
  # remove NA/NaN and apply threshold
  ok <- is.finite(v) & (v > thr)
  idx <- which(ok)
  
  if (length(idx) == 0) {
    return(data.table(prob = numeric(), x = numeric(), y = numeric()))
  }
  
  # sample cell indices (no replacement)
  if (length(idx) > n) idx <- sample(idx, n)
  
  probs <- v[idx]
  xy    <- terra::xyFromCell(rb, idx)
  
  data.table(prob = as.numeric(probs),
             x    = as.numeric(xy[, 1]),
             y    = as.numeric(xy[, 2]))
}


# ---- 3) For each zone, pool across ALL rasters of that zone, then sample 5000 per class ----
# Strategy: create a virtual mosaic per zone (stacked list), sample from each file,
# and then downsample to exactly k. This keeps memory manageable.
out_list <- list()

for (z in zones) {
  z_files <- files[zone_of_file == z]
  if (length(z_files) == 0) next
  
  message("\n=== Zone: ", z, " | files: ", length(z_files), " ===")
  
  for (b in 1:3) {
    cls <- band_classes[b]
    message("  - Class: ", cls, " (band ", b, ")")
    
    # collect candidates by sampling up to (k * 3) from each file (cap),
    # then combine and sample to k. This avoids reading all candidates everywhere.
    # If you want closer-to-uniform over all qualifying pixels, see note below.
    per_file_cap <- k * 3L
    
    dt_all <- rbindlist(lapply(z_files, function(fp) {
      r <- rast(fp)
      if (nlyr(r) < 3) stop("Raster has <3 bands: ", fp)
      
      dt <- sample_band_from_raster(r, band = b, n = per_file_cap, thr = thr)
      dt
    }), use.names = TRUE, fill = TRUE)
    
    if (nrow(dt_all) == 0) {
      out_dt <- data.table(zone = z, class = cls, prob = numeric(), x = numeric(), y = numeric())
    } else if (nrow(dt_all) <= k) {
      out_dt <- cbind(data.table(zone = z, class = cls), dt_all)
    } else {
      out_dt <- cbind(data.table(zone = z, class = cls), dt_all[sample.int(nrow(dt_all), k)])
    }
    
    out_list[[paste(z, cls, sep = "_")]] <- out_dt
    message("    sampled: ", nrow(out_dt))
  }
}

out <- rbindlist(out_list, use.names = TRUE, fill = TRUE)

# ---- 4) Report counts + write ----
counts <- out[, .N, by = .(zone, class)][order(zone, class)]
message("\nSample counts (should be 5000 if enough pixels existed):")
print(counts)

fwrite(out, out_csv)
message("\nWrote: ", out_csv)



suppressPackageStartupMessages({
  library(terra)
  library(data.table)
})

# --- paths ---
in_dir   <- "/mnt/eo/EO4Backcasting/_preds_Feb_3c"
in_csv   <- file.path(in_dir, "xgb_zone_samples_prob_gt_0p7__n5000_per_class.csv")
trend_fp <- "/mnt/eo/eu_mosaics/NBR_comp/NBR_1985_1990_slope_forest.tif"
out_csv  <- file.path(in_dir, "xgb_zone_samples_prob_gt_0p7__n5000_per_class__with_NBR_trend.csv")

# --- read samples ---
dt <- fread(in_csv)
stopifnot(all(c("zone","class","prob","x","y") %in% names(dt)))

# --- read trend raster ---
trend <- rast(trend_fp)

# If trend raster has multiple layers, keep the first (adjust if needed)
if (nlyr(trend) > 1) trend <- trend[[1]]

# --- build points (assume x/y are in the CRS of the probability rasters) ---
# If you know the CRS of x/y explicitly, set it here. Otherwise we will still
# reproject only if necessary based on trend CRS, but we need a starting CRS.
# Best practice: read one XGB raster and use its CRS as the source CRS.
xgb_files <- list.files(in_dir, pattern="xgb.*\\.(tif|tiff)$", full.names=TRUE, ignore.case=TRUE)
if (length(xgb_files) == 0) stop("No XGB rasters found in: ", in_dir)

src_crs <- crs(rast(xgb_files[1]))
if (is.na(src_crs) || src_crs == "") {
  warning("Could not infer CRS from XGB raster. Assuming x/y are already in the NBR trend raster CRS.")
  src_crs <- crs(trend)
}

pts <- vect(dt[, .(x, y)], geom = c("x","y"), crs = src_crs)

# --- reproject points to trend CRS if needed ---
trend_crs <- crs(trend)
if (!is.na(trend_crs) && trend_crs != "" && !same.crs(pts, trend)) {
  pts <- project(pts, trend_crs)
}

# --- extract NBR trend values ---
# terra::extract returns data.frame with ID + value column(s)
ext <- terra::extract(trend, pts)

# ext has one row per point, first column is ID
# second column is the trend value
dt[, NBR_trend := ext[[2]]]

# --- write ---
fwrite(dt, out_csv)
message("Wrote: ", out_csv)


library(data.table)
library(ggplot2)

# read data (or skip if dt already exists)
dt <- fread("/mnt/eo/EO4Backcasting/_preds_Feb_3c/xgb_zone_samples_prob_gt_0p7__n5000_per_class__with_NBR_trend.csv")

# factor ordering (important for clean plots)
dt[, zone := factor(zone, levels = c("boreal", "temperate", "mediterranean"))]
dt[, class := factor(class, levels = c("healthy", "ysd1-10", "ysd>10"))]

# optional: remove NA trend values
dt_plot <- dt[!is.na(NBR_trend)]

ggplot(dt_plot, aes(x = class, y = NBR_trend, fill = class)) +
  geom_boxplot(outlier.size = 0.4, alpha = 0.85) +
  facet_wrap(~ zone, nrow = 1) +
  theme_bw(base_size = 15) +
  labs(
    x = "",
    y = "NBR trend",
    title = ""
  ) +
  theme(legend.position = "none")


ggplot(dt_plot, aes(x = zone, y = NBR_trend, fill = zone)) +
  geom_boxplot(outlier.size = 0.4) +
  facet_wrap(~ class, nrow = 1) +
  theme_bw(base_size = 13) +
  labs(
    x = "Zone",
    y = "NBR trend",
    title = ""
  ) +
  theme(legend.position = "none")


