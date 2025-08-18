# ============================================================
# Balanced sampling from a single Year-of-Disturbance raster
# Binary classes: disturbed (YoD ∈ [1985,2004]) vs undisturbed (no disturbance 1985..2024)
# Efficient: per-hex window reads; no continent-scale stacks in RAM
# Outputs: GPKG + CSV in /mnt/eo/EO4Backcasting/_output
# ============================================================

library(terra)
library(sf)
library(dplyr)


# ----------------------------
# 0) CONFIG
# ----------------------------
# INPUTS
yod_path    <- "/mnt/eo/EFDA_v211/latest_disturbance_eu_v211_2_3035.tif"
forest_path <- "/mnt/eo/EFDA_v211/forestlanduse_mask_EUmosaic3035.tif"  # forest mask; any positive value treated as forest

# OUTPUTS (non-NAS)
out_dir <- "/mnt/eo/EO4Backcasting/_output"
tmp_dir <- "/mnt/eo/EO4Backcasting/_tmp"   # prefer local scratch to avoid NAS I/O


terraOptions(tempdir = tmp_dir, memfrac = 0.7, progress = 1)
Sys.setenv("TMPDIR" = tmp_dir, "TEMP" = tmp_dir, "TMP" = tmp_dir, "GDAL_CACHEMAX" = "2048")

# SAMPLING DESIGN
year_min <- 1985L
year_max <- 2004L
final_year <- 2024L             # horizon for "undisturbed" definition
hex_km <- 50                    # hex diameter for spatial balancing (25–50 km typical)
target_total <- 200000          # total samples (disturbed + undisturbed)
balance_by_year <- TRUE         # equalize disturbed counts per YoD (soft)
min_spacing_m <- 750            # approximate within-hex thinning
set.seed(42)

# ENCODING ASSUMPTIONS for "undisturbed"
# Treat as undisturbed if: is.na(YoD) OR YoD == 0 OR YoD > final_year
# IMPORTANT: YoD < year_min (e.g., pre-1985) is NOT undisturbed → excluded from undisturbed pool
undist_is_true <- function(y) { is.na(y) | (y == 0L) | (y > final_year) }

# ----------------------------
# 1) Load rasters; align
# ----------------------------
yod <- rast(yod_path)
forest <- rast(forest_path)
if (!compareGeom(forest, yod, stopOnError = FALSE)) {
  forest <- project(forest, yod, method = "near")
}
# Binarize forest (some masks are 100/255/etc.)
forest <- ifel(!is.na(forest) & forest == 1, 1, 0)

# ----------------------------
# 2) Build hex grid over extent (SpatVector) + forest screening
# ----------------------------
bb <- st_as_sfc(st_bbox(forest)) |> st_set_crs(crs(forest, proj = TRUE))
hex_all <- st_make_grid(bb, cellsize = hex_km * 1000, square = FALSE)
hex_sf  <- st_sf(hex_id = seq_along(hex_all), geometry = hex_all)

forest_coarse <- aggregate(forest, fact = 200, fun = "max")
has_forest <- terra::extract(forest_coarse, vect(hex_sf), fun = "max", ID = FALSE)[,1]
hex_sf <- hex_sf[!is.na(has_forest) & has_forest > 0, , drop = FALSE]
stopifnot(nrow(hex_sf) > 0)

# Precompute SpatVector hex grid (more robust than converting each loop)
hex_v <- vect(hex_sf)
n_hex <- nrow(hex_sf)

# Per-class quota per hex
k_per_hex <- ceiling(target_total / (2 * n_hex))
k_per_hex <- max(1, min(k_per_hex, 60))  # sanity cap
k_per_year <- if (isTRUE(balance_by_year)) max(1, floor(k_per_hex / length(year_min:year_max))) else NA_integer_

message("Active hexes (with forest): ", n_hex)
message("Quota: ", k_per_hex, " / class / hex",
        if (isTRUE(balance_by_year)) paste0(" (≈", k_per_year, "/year)") else "")

# ----------------------------
# 3) Utility: thin points by coarse grid (≈ min spacing)
# ----------------------------
thin_by_grid <- function(sf_pts, spacing_m = 750) {
  if (nrow(sf_pts) <= 1) return(sf_pts)
  bb <- st_as_sfc(st_bbox(sf_pts))
  grd <- st_make_grid(bb, cellsize = spacing_m, what = "polygons")
  idx <- st_intersects(sf_pts, grd)
  keep <- tapply(seq_len(nrow(sf_pts)), sapply(idx, function(i) if (length(i)) i[1] else NA), function(ix) ix[1])
  sf_pts[na.omit(unlist(keep)), ]
}

# ----------------------------
# 4) Core sampler: read once per hex (SpatVector), sample from vectors
# ----------------------------
sample_hex <- function(hx_v, k_hex, k_year = NA_integer_) {
  # Window reads (crop+mask to hex polygon)
  y_h <- try(mask(crop(yod,   hx_v), hx_v), silent = TRUE)
  f_h <- try(mask(crop(forest, hx_v), hx_v), silent = TRUE)
  if (inherits(y_h, "try-error") || inherits(f_h, "try-error") || ncell(y_h) == 0) {
    return(list(dist = NULL, und = NULL))
  }
  
  vy <- values(y_h, mat = FALSE)
  vf <- values(f_h, mat = FALSE)
  if (is.null(vy) || is.null(vf)) return(list(dist = NULL, und = NULL))
  
  # on-forest indices
  on_forest <- which(vf > 0)
  if (!length(on_forest)) return(list(dist = NULL, und = NULL))
  
  idx_dist_all <- on_forest[ vy[on_forest] >= year_min & vy[on_forest] <= year_max ]
  idx_und_all  <- on_forest[ undist_is_true(vy[on_forest]) ]
  
  # --- Disturbed sampling ---
  dist_sf <- NULL
  if (length(idx_dist_all) > 0) {
    take <- integer(0)
    if (!is.na(k_year)) {
      years <- year_min:year_max
      for (yy in years) {
        idx_y <- idx_dist_all[ vy[idx_dist_all] == yy ]
        if (length(idx_y)) take <- c(take, sample(idx_y, min(length(idx_y), k_year)))
      }
      if (length(take) < k_hex) {
        rem <- setdiff(idx_dist_all, take)
        if (length(rem)) take <- c(take, sample(rem, min(length(rem), k_hex - length(take))))
      } else if (length(take) > k_hex) {
        take <- sample(take, k_hex)
      }
    } else {
      take <- sample(idx_dist_all, min(length(idx_dist_all), k_hex))
    }
    
    if (length(take)) {
      xy <- xyFromCell(y_h, take)
      dist_sf <- st_as_sf(
        data.frame(x = xy[,1], y = xy[,2],
                   label = rep("disturbed", length(take)),
                   yod   = as.integer(vy[take])),
        coords = c("x","y"),
        crs = crs(y_h, proj = TRUE)
      )
      if (nrow(dist_sf) > 1 && min_spacing_m > 0) dist_sf <- thin_by_grid(dist_sf, min_spacing_m)
    }
  }
  
  # --- Undisturbed sampling ---
  und_sf <- NULL
  if (length(idx_und_all) > 0) {
    n_u <- min(length(idx_und_all), k_hex)
    take_u <- sample(idx_und_all, n_u)
    xy_u <- xyFromCell(y_h, take_u)
    und_sf <- st_as_sf(
      data.frame(x = xy_u[,1], y = xy_u[,2],
                 label = rep("undisturbed", n_u),
                 yod   = NA_integer_),
      coords = c("x","y"),
      crs = crs(y_h, proj = TRUE)
    )
    if (nrow(und_sf) > 1 && min_spacing_m > 0) und_sf <- thin_by_grid(und_sf, min_spacing_m)
  }
  
  list(dist = dist_sf, und = und_sf)
}

# ----------------------------
# 5) Loop hexes and sample (assemble safely)
# ----------------------------
samples <- vector("list", length = n_hex)
for (i in seq_len(n_hex)) {
  if (i %% 50 == 0) message("… hex ", i, "/", n_hex)
  res <- sample_hex(hex_v[i], k_hex = k_per_hex,
                    k_year = if (isTRUE(balance_by_year)) k_per_year else NA_integer_)
  pieces <- Filter(Negate(is.null), list(res$dist, res$und))
  samples[[i]] <- if (length(pieces)) do.call(rbind, pieces) else NULL
}

samples <- Filter(Negate(is.null), samples)
if (!length(samples)) stop("No samples drawn — check masks and year coding.")
samp <- do.call(rbind, samples)

# ----------------------------
# 6) Global 1:1 balance and bookkeeping
# ----------------------------
n_dist <- sum(samp$label == "disturbed")
n_und  <- sum(samp$label == "undisturbed")
if (min(n_dist, n_und) == 0) stop("One class empty after sampling.")
if (n_dist != n_und) {
  if (n_dist > n_und) {
    drop <- sample(which(samp$label=="disturbed"), n_dist - n_und)
  } else {
    drop <- sample(which(samp$label=="undisturbed"), n_und - n_dist)
  }
  samp <- samp[-drop,]
}
samp$pt_id <- seq_len(nrow(samp))

# Optional: keep native CRS or reproject
# samp <- st_transform(samp, 4326)

# ----------------------------
# 7) Save
# ----------------------------
gpkg_path <- file.path(out_dir, "eu_train_points_binary.gpkg")
csv_path  <- file.path(out_dir, "eu_train_points_binary.csv")
st_write(samp, gpkg_path, delete_dsn = TRUE, quiet = TRUE)
write.csv(st_drop_geometry(samp), csv_path, row.names = FALSE)

message("Done. Wrote ", nrow(samp), " points → ",
        sum(samp$label=='disturbed'), " disturbed + ",
        sum(samp$label=='undisturbed'), " undisturbed.",
        "\nFiles:\n - ", gpkg_path, "\n - ", csv_path)
