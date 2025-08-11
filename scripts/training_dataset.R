# ============================================================
# Balanced sampling from a single Year-of-Disturbance raster
# Binary classes: disturbed (YoD ∈ [1985,2004]) vs undisturbed (no disturbance 1985..2024)
# Efficient: per-hex window reads; no continent-wide stacks loaded into RAM
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
forest_path <- "/mnt/eo/EFDA_v211/forestlanduse_mask_EUmosaic3035.tif"       # 1 = forest, same grid or projectable to YoD raster

# OUTPUTS (non-NAS)
out_dir <- "/mnt/eo/EO4Backcasting/_output"
tmp_dir <- "/mnt/dss_europe/temp_lm"

terraOptions(tempdir = tmp_dir, memfrac = 0.7, progress = 1)
Sys.setenv("TMPDIR" = tmp_dir, "TEMP" = tmp_dir, "TMP" = tmp_dir, "GDAL_CACHEMAX" = "2048")

# SAMPLING DESIGN
year_min <- 1985L
year_max <- 2004L
final_year <- 2023L                # horizon for "undisturbed" definition
hex_km <- 50                       # hex diameter for spatial balancing (try 25–50 km)
target_total <- 200000             # total points (disturbed + undisturbed)
balance_by_year <- TRUE            # optionally equalize disturbed counts across years 1985..2004
min_spacing_m <- 750               # approximate within-hex thinning
set.seed(42)

# ENCODING ASSUMPTIONS for "undisturbed"
# Treat as undisturbed if: is.na(YoD) OR YoD == 0 OR YoD > final_year
# IMPORTANT: YoD < year_min (e.g., pre-1985) is *NOT* undisturbed → exclude from undisturbed pool
undist_is_true <- function(y) { is.na(y) | (y == 0L) | (y > final_year) }

# ----------------------------
# 1) Load rasters; align forest to YoD
# ----------------------------
yod <- rast(yod_path)
forest <- rast(forest_path)
if (!compareGeom(forest, yod, stopOnError = FALSE))
  forest <- project(forest, yod, method = "near")

# ----------------------------
# 2) Build hex grid over forest extent (YoD CRS)
# ----------------------------
bb <- st_as_sfc(st_bbox(forest)) |> st_set_crs(crs(forest, proj=TRUE))
hex_all <- st_make_grid(bb, cellsize = hex_km * 1000, square = FALSE)
hex <- st_sf(hex_id = seq_along(hex_all), geometry = hex_all)

# Keep only hexes that actually intersect forest (coarse filter via aggregated mask)
forest_coarse <- aggregate(forest, fact = 200, fun = "max")
has_forest <- terra::extract(forest_coarse, vect(hex), fun = "max", ID = FALSE)[,1] == 1
hex <- hex[has_forest, , drop = FALSE]
n_hex <- nrow(hex)
stopifnot(n_hex > 0)
message("Active hexes (with forest): ", n_hex)

# Per-class quota per hex
k_per_hex <- ceiling(target_total / (2 * n_hex))
k_per_hex <- max(1, min(k_per_hex, 60))  # sanity cap
k_per_year <- if (balance_by_year) max(1, floor(k_per_hex / length(year_min:year_max))) else NA_integer_
message("Quota: ", k_per_hex, " / class / hex",
        if (balance_by_year) paste0(" (≈", k_per_year, "/year)") else "")

# ----------------------------
# 3) Utility: thin points by coarse grid (≈ min spacing)
# ----------------------------
thin_by_grid <- function(sf_pts, spacing_m = 750) {
  if (nrow(sf_pts) <= 1) return(sf_pts)
  bb <- st_as_sfc(st_bbox(sf_pts))
  grd <- st_make_grid(bb, cellsize = spacing_m, what = "polygons")
  idx <- st_intersects(sf_pts, grd)
  keep <- tapply(seq_len(nrow(sf_pts)), sapply(idx, function(i) if (length(i)) i[1] else NA),
                 function(ix) ix[1])
  sf_pts[na.omit(unlist(keep)), ]
}

# ----------------------------
# 4) Core sampler: read once per hex, sample from value vectors
# ----------------------------
sample_hex <- function(hx, k_hex, k_year = NA_integer_) {
  # Crop YoD & forest to hex (small window reads)
  y_h <- mask(crop(yod, vect(hx)),   vect(hx))
  f_h <- mask(crop(forest, vect(hx)), vect(hx))
  
  # Pull values into RAM (small; area = one hex)
  vy <- values(y_h, mat = FALSE)
  vf <- values(f_h, mat = FALSE)
  
  # Valid (on-forest) cells
  on_forest <- which(vf == 1)
  if (!length(on_forest)) return(list(dist = NULL, und = NULL))
  
  # Candidate indices
  idx_dist_all <- on_forest[ vy[on_forest] >= year_min & vy[on_forest] <= year_max ]
  idx_und_all  <- on_forest[ undist_is_true(vy[on_forest]) ]
  
  # --- Disturbed sampling ---
  if (length(idx_dist_all) > 0) {
    if (!is.na(k_year)) {
      # per-year soft balancing
      years <- year_min:year_max
      take <- integer(0)
      for (yy in years) {
        idx_y <- idx_dist_all[ vy[idx_dist_all] == yy ]
        if (length(idx_y) > 0) {
          take <- c(take, sample(idx_y, min(length(idx_y), k_year)))
        }
      }
      # if still under quota, top up at random from remaining candidates
      if (length(take) < k_hex) {
        rem <- setdiff(idx_dist_all, take)
        if (length(rem) > 0) {
          take <- c(take, sample(rem, min(length(rem), k_hex - length(take))))
        }
      } else if (length(take) > k_hex) {
        take <- sample(take, k_hex)
      }
    } else {
      take <- sample(idx_dist_all, min(length(idx_dist_all), k_hex))
    }
    xy <- xyFromCell(y_h, take)
    dist_sf <- st_as_sf(data.frame(x = xy[,1], y = xy[,2],
                                   label = "disturbed",
                                   yod = as.integer(vy[take])),
                        coords = c("x","y"),
                        crs = crs(y_h, proj=TRUE))
  } else {
    dist_sf <- NULL
  }
  
  # --- Undisturbed sampling ---
  if (length(idx_und_all) > 0) {
    take_u <- sample(idx_und_all, min(length(idx_und_all), k_hex))
    xy_u <- xyFromCell(y_h, take_u)
    und_sf <- st_as_sf(data.frame(x = xy_u[,1], y = xy_u[,2],
                                  label = "undisturbed",
                                  yod = NA_integer_),
                       coords = c("x","y"),
                       crs = crs(y_h, proj=TRUE))
  } else {
    und_sf <- NULL
  }
  
  # Optional spacing within hex
  if (!is.null(dist_sf) && nrow(dist_sf) > 1 && min_spacing_m > 0)
    dist_sf <- thin_by_grid(dist_sf, min_spacing_m)
  if (!is.null(und_sf) && nrow(und_sf) > 1 && min_spacing_m > 0)
    und_sf <- thin_by_grid(und_sf,  min_spacing_m)
  
  list(dist = dist_sf, und = und_sf)
}

# ----------------------------
# 5) Loop hexes and sample
# ----------------------------
samples <- vector("list", length = n_hex)
for (i in seq_len(n_hex)) {
  if (i %% 50 == 0) message("… hex ", i, "/", n_hex)
  res <- sample_hex(hex[i,], k_hex = k_per_hex,
                    k_year = if (isTRUE(balance_by_year)) k_per_year else NA_integer_)
  samples[[i]] <- rbind(res$dist, res$und)
}

samp <- do.call(rbind, samples)
if (is.null(samp) || nrow(samp) == 0) stop("No samples drawn; check inputs and definitions.")

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

# Optional: keep in native CRS, or reproject for portability
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
