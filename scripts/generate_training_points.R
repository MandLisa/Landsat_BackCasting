# ============================================================
# Set A only — Systematic 2.5 km lattice (Canada-style)
# - One lattice point per 2.5 km cell center across the r_dist extent
# - Labels via intersection:
#     disturbed (has year) vs undisturbed_forest (forest==1 & no year)
# - Edge exclusion with on-disk focal ("range") to avoid boundary noise
# - Forest mask is resampled (nearest) to match r_dist pixel grid
# - Optional 50-km tile 80/20 split
# ============================================================


library(terra)
library(sf)
library(dplyr)


# Speed & stability knobs 
terraOptions(memfrac = 0.6, progress = 1)  

# or import it
r_dist   <- rast("/mnt/eo/EFDA_v211/latest_disturbance_eu_v211_2_3035.tif")
r_forest <- rast("/mnt/eo/EFDA_v211/forestlanduse_mask_EUmosaic3035.tif")
r_forest_aligned <- rast("/mnt/eo/EO4Backcasting/_intermediates/r_forest_aligned.tif")
dist_mask <- rast("/mnt/eo/EO4Backcasting/_intermediates/dist_mask_11.tif")

# -------------------------
# USER INPUTS
# -------------------------
path_dist   <- "/mnt/eo/EFDA_v211/latest_disturbance_eu_v211_2_3035.tif"
path_forest <- "/mnt/eo/EFDA_v211/forestlanduse_mask_EUmosaic3035.tif"

out_dir <- "/mnt/eo/EO4Backcasting/_output"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

crs_europe     <- "EPSG:3035"
na_code        <- NA    
grid_spacing_m <- 2500  
edge_px        <- 1      
do_tile_split  <- TRUE    
tile_km        <- 50
train_frac     <- 0.8
set.seed(42)

# -------------------------
# 1) READ & PREP RASTERS
# -------------------------
message("Reading rasters ...")
r_dist   <- rast(path_dist)
r_forest <- rast(path_forest)

# If multi-band, take first band (common for year or mask stacks)
if (nlyr(r_dist)   > 1) { message("r_dist has multiple layers; using the first.");   r_dist   <- r_dist[[1]] }
if (nlyr(r_forest) > 1) { message("r_forest has multiple layers; using the first."); r_forest <- r_forest[[1]] }

# Normalize special NA codes if present
if (!is.na(na_code)) {
  r_dist[r_dist == na_code]     <- NA
  r_forest[r_forest == na_code] <- NA
}
stopifnot(!is.na(crs(r_dist)), !is.na(crs(r_forest)))

# Project to EPSG:3035 if needed (categorical → 'near')
if (crs(r_dist) != crs(crs_europe)) {
  message("Projecting disturbance to EPSG:3035 ...")
  r_dist <- project(r_dist, crs_europe, method = "near")
}
if (crs(r_forest) != crs(crs_europe)) {
  message("Projecting forest to EPSG:3035 ...")
  r_forest <- project(r_forest, crs_europe, method = "near")
}

# Resample forest → disturbance grid (nearest). If your mask isn’t 1/NA, binarize before.
# r_forest[r_forest > 0] <- 1; r_forest[r_forest <= 0] 
r_forest_aligned <- resample(r_forest, r_dist, method = "near")
if (!isTRUE(compareGeom(r_dist, r_forest_aligned, stopOnError = FALSE)))
  stop("Forest mask is not aligned to disturbance grid after resampling.")

writeRaster(
  r_forest_aligned, "/mnt/eo/EO4Backcasting/_intermediates/r_forest_aligned.tif",
  wopt = list(datatype = "INT1U", gdal = c("COMPRESS=LZW","TILED=YES", "BIGTIFF = YES")),
  overwrite = TRUE
)

# or import it
r_forest_aligned <- rast("/mnt/eo/EO4Backcasting/_intermediates/r_forest_aligned.tif")

# -------------------------
# 2) EDGE EXCLUSION (one-pass focal range, on-disk)
# -------------------------
# where to save intermediate rasters
scratch_dir <- "/mnt/eo/EO4Backcasting/_intermediates"
if (!dir.exists(scratch_dir)) dir.create(scratch_dir, recursive = TRUE)

# helper to build full paths inside that folder
.out <- function(name) file.path(scratch_dir, name)

message("Computing edge exclusion mask ...")
# 0/1 disturbed presence (1 if pixel has a year; 0 otherwise)
# 0/1 disturbance presence (1 if pixel has a year; 0 otherwise)
m01 <- ifel(
  is.na(r_dist), 0L, 1L,
  filename  = .out("dist_mask_11.tif"),
  wopt      = list(datatype = "INT1U", gdal = "COMPRESS=LZW"),
  overwrite = TRUE
)

# 0) Runtime: stream to disk + give GDAL a bigger cache
terraOptions(todisk = TRUE, memfrac = 0.5, progress = 1)
# cache in MB; adjust if you have more/less RAM
Sys.setenv(GDAL_CACHEMAX = "2048")

# 1) Ensure m01 is a tiled, compressed INT1U GeoTIFF on disk
#    (0/1 only; NA treated as 0 downstream)
m01 <- ifel(dist_mask >= 1, 1L, 0L)
m01 <- writeRaster(
  m01,
  filename = .out("m01_uint1.tif"),
  wopt = list(datatype = "INT1U",
              gdal = c("COMPRESS=LZW", "TILED=YES", "BLOCKXSIZE=256", "BLOCKYSIZE=256", "BIGTIFF=YES")),
  overwrite = TRUE
)

# 2) 3×3 kernel
w3 <- matrix(1, 3, 3)

# 3) Clean any stale outputs from previous runs
unlink(.out("edge_sum3.tif"))

# 4) Try the fast C++ 'sum' reducer first
sum3_try <- try(
  focal(
    m01, w = w3, fun = "sum", na.policy = "omit", fillvalue = 0,
    filename  = .out("edge_sum3.tif"),
    wopt      = list(datatype = "INT2U",  # 0..9 fits in UINT16
                     gdal = c("COMPRESS=LZW", "TILED=YES", "BLOCKXSIZE=256", "BLOCKYSIZE=256", "BIGTIFF=YES")),
    overwrite = TRUE
  ),
  silent = TRUE
)

if (inherits(sum3_try, "try-error")) {
  message("focal(fun='sum') failed on this build; falling back to mean*9 (equivalent for 0/1).")
  # Fallback: compute mean over 3×3 and multiply by 9 -> identical to sum on {0,1}
  mean3 <- focal(
    m01, w = w3, fun = "mean", na.policy = "omit", fillvalue = 0,
    filename  = .out("edge_mean3.tif"),
    wopt      = list(datatype = "FLT4S",
                     gdal = c("COMPRESS=LZW", "TILED=YES", "BLOCKXSIZE=256", "BLOCKYSIZE=256", "BIGTIFF=YES")),
    overwrite = TRUE
  )
  sum3 <- round(mean3 * 9)
  sum3 <- writeRaster(
    sum3, filename = .out("edge_sum3.tif"),
    wopt = list(datatype = "INT2U",
                gdal = c("COMPRESS=LZW", "TILED=YES", "BLOCKXSIZE=256", "BLOCKYSIZE=256", "BIGTIFF=YES")),
    overwrite = TRUE
  )
} else {
  sum3 <- sum3_try
}

# 5) Your original edge criterion
edge_raw <- (sum3 > 0) & (sum3 < 9)



# ---------------------------------------------------------------

# or import it
dist_mask <- rast("/mnt/eo/EO4Backcasting/_intermediates/dist_mask_11.tif")

# Stabilize terra's IO/tiling
terraOptions(todisk = TRUE, memfrac = 0.6)        # stream to disk, limit RAM
terraOptions(tempdir = tempdir())                 # ensure a valid temp path

# 3×3 kernel implies 8-neighborhood (Moore neighborhood)
w3 <- matrix(1, 3, 3)

# Build a helper raster where non-disturbance (0) is non-NA; disturbance (1) is NA
# (distance() computes distance to non-NA cells)
nd <- ifel(dist_mask == 0, 1, NA)

# Euclidean distance (in map units) to the nearest non-disturbance cell
d <- distance(nd, filename = "dist_to_nondist_native.grd", overwrite = TRUE)

# One-pixel erosion (8-neighborhood): threshold at one diagonal cell size
cell_diag <- sqrt(res(dist_mask)[1]^2 + res(dist_mask)[2]^2)
interior_1px <- d > cell_diag

writeRaster(
  interior_1px, "/mnt/eo/EO4Backcasting/_intermediates/dist_mask_interior_1px_new.tif",
  wopt = list(datatype = "INT1U", gdal = c("COMPRESS=LZW","TILED=YES","BIGTIFF=YES")),
  overwrite = TRUE
)


# 2) Keep only cells where all 9 are disturbance (strict interior)
interior_1px <- n_sum == 9

# 3) Write final mask (now you can compress)
writeRaster(
  interior_1px, "dist_mask_interior_1px.tif",
  wopt = list(datatype = "INT1U",
              gdal = c("COMPRESS=LZW","TILED=YES","BIGTIFF=YES")),
  overwrite = TRUE
)

# -------------------------
# 3) SYSTEMATIC 2.5 km LATTICE
# -------------------------
message("Generating systematic lattice points ...")
# Robust way: copy the disturbance raster, then change only the resolution
r_grid <- rast(r_dist)                 # copies extent + CRS safely
res(r_grid) <- grid_spacing_m          # set both x/y resolution (meters in EPSG:3035)
r_grid <- init(r_grid, fun = "cell")   # unique id per coarse cell

p_grid <- as.points(r_grid)               # one point at each cell center (SpatVector)

# -------------------------
# 4) CLASSIFY LATTICE POINTS
# -------------------------
message("Extracting labels at lattice points ...")
v_year   <- terra::extract(r_dist, p_grid)[, 2]
v_forest <- terra::extract(r_forest_aligned, p_grid)[, 2]
v_edge   <- terra::extract(edge_exclude01,   p_grid)[, 2]; v_edge[is.na(v_edge)] <- 0

is_disturbed <- !is.na(v_year) & (v_edge == 0)
is_undist    <-  is.na(v_year) & !is.na(v_forest) & (v_forest == 1) & (v_edge == 0)

p_dist <- p_grid[is_disturbed, ]
p_und  <- p_grid[is_undist,   ]

# Build Set A (systematic)
SetA <- dplyr::bind_rows(
  { if (nrow(p_dist)) { sfd <- st_as_sf(p_dist); sfd$type <- "disturbed"; sfd$year <- v_year[is_disturbed]; sfd } },
  { if (nrow(p_und))  { sfu <- st_as_sf(p_und);  sfu$type <- "undisturbed_forest"; sfu$year <- NA_integer_; sfu } }
)

message(sprintf("Set A (systematic 2.5 km): %s disturbed, %s undisturbed_forest",
                sum(is_disturbed), sum(is_undist)))

# -------------------------
# 5) OPTIONAL: 50-km TILE 80/20 SPLIT
# -------------------------
if (do_tile_split) {
  message("Splitting Set A into 80/20 by 50-km tiles ...")
  r_tile <- rast(ext(r_dist), crs = crs(r_dist), resolution = tile_km * 1000)
  r_tile <- init(r_tile, fun = "cell")
  tid <- terra::extract(r_tile, vect(SetA))[, 2]
  SetA$tile_id <- tid
  SetA <- SetA[!is.na(SetA$tile_id), ]
  
  tiles   <- unique(SetA$tile_id)
  n_train <- ceiling(train_frac * length(tiles))
  train_tiles <- sample(tiles, size = n_train)
  
  SetA$split <- ifelse(SetA$tile_id %in% train_tiles, "train", "val")
  
  SetA_train <- dplyr::filter(SetA, split == "train")
  SetA_val   <- dplyr::filter(SetA, split == "val")
}

# -------------------------
# 6) WRITE OUTPUTS
# -------------------------
message("Writing outputs ...")
st_write(SetA, file.path(out_dir, "SetA_systematic_2p5km_all.gpkg"),
         delete_dsn = TRUE, quiet = TRUE)
if (do_tile_split) {
  if (nrow(SetA_train)) st_write(SetA_train, file.path(out_dir, "SetA_systematic_2p5km_train.gpkg"),
                                 delete_dsn = TRUE, quiet = TRUE)
  if (nrow(SetA_val))   st_write(SetA_val,   file.path(out_dir, "SetA_systematic_2p5km_val.gpkg"),
                                 delete_dsn = TRUE, quiet = TRUE)
}

message("Done (Set A only).")
