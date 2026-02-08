library(terra)

# or import it
r_dist   <- rast("/mnt/eo/EFDA_v211/latest_disturbance_eu_v211_2_3035.tif")
r_forest <- rast("/mnt/eo/EFDA_v211/forestlanduse_mask_EUmosaic3035.tif")
r_forest_aligned <- rast("/mnt/eo/EO4Backcasting/_intermediates/r_forest_aligned.tif")
dist_mask <- rast("/mnt/eo/EO4Backcasting/_intermediates/dist_mask_11.tif")
target <- rast("/mnt/eo/EO4Backcasting/_intermediates/dist_mask_core_1985_2008.tif")

# dist_mask: 0/1, projected CRS in meters (or any linear units)

# Distance to the nearest 0-cell (streamed to disk)
bg <- ifel(dist_mask == 0, 1, NA)
d  <- distance(bg, filename = "__tmp_d2zero.grd", overwrite = TRUE)

# Cell diagonal threshold: include diagonals (8-neighbour interior)
thr <- sqrt(sum(res(dist_mask)^2))

# Interior-only mask: center==1 and all 8 neighbours also 1
interior <- ifel(dist_mask == 1 & d > thr, 1, NA)

writeRaster(
  interior, "/mnt/eo/EO4Backcasting/_intermediates/dist_mask_interior.tif",
  wopt = list(datatype = "INT1U", gdal = c("COMPRESS=LZW","TILED=YES","BIGTIFF=YES")),
  overwrite = TRUE
)

plot(interior)

interior <- rast("/mnt/eo/EO4Backcasting/_intermediates/dist_mask_interior.tif")

lower <- 1986
upper <- 2008  # 15y post-window to 2023

# Eligible years mask
eligible <- ifel(r_dist >= lower & r_dist <= upper, 1, NA)

# Intersect with your interior mask
target <- mask(interior, eligible)

writeRaster(
  target, "/mnt/eo/EO4Backcasting/_intermediates/dist_mask_core_1985_2008.tif",
  wopt = list(datatype = "INT1U", gdal = c("COMPRESS=LZW","TILED=YES","BIGTIFF=YES")),
  overwrite = TRUE
)

# Count and sample
n_ok <- as.integer(global(!is.na(target), "sum", na.rm = TRUE)[1,1])
set.seed(123)

# sample 750k from the stratum "1" only
set.seed(123)
samp_points <- spatSample(
  x         = target,
  strata    = target,             # use target values as strata
  size      = c("1" = 750000L),   # request only from class/value 1
  method    = "random",
  as.points = TRUE,
  na.rm     = TRUE,
  values    = FALSE
)

nrow(samp_points)



# 5) Save to file (streamed write)
writeVector(samp_points, "/mnt/eo/EO4Backcasting/_intermediates/sample_points_core_1985_2005.gpkg",
            filetype = "GPKG", overwrite = TRUE)


#-------------------------------------------------------------------------------
### extract distrbance year
samp_points <- vect("/mnt/eo/EO4Backcasting/_intermediates/sample_points_core_1985_2005_yod.gpkg")
# Ensure the raster layer is clearly named (helps downstream)
names(r_dist) <- "yod"

# Reproject points only if CRS differs (works across terra versions)
if (!same.crs(r_dist, samp_points)) {
  samp_points <- project(samp_points, r_dist)  # use raster's CRS
}

# Streamed extraction; bind the raster value into the point attributes
# Returns a SpatVector with a new "yod" column
samp_points <- extract(r_dist, samp_points, bind = TRUE, ID = FALSE)

# (Optional) sanity check: confirm these are interior mask cells
# stopifnot(all(extract(dist_mask, samp_points)[[2]] == 1, na.rm = TRUE))

# Save
writeVector(
  samp_points,
  "/mnt/eo/EO4Backcasting/_intermediates/sample_points_core_1985_2005_yod.gpkg",
  filetype = "GPKG",
  overwrite = TRUE
)

y <- samp_points$yod
y <- y[!is.na(y)]

# integer-aligned 1-year bins: [..., 1984.5, 1985.5, ...]
brks <- seq(floor(min(y)) - 0.5, ceiling(max(y)) + 0.5, by = 1)

hist(
  y, breaks = brks, right = FALSE,
  main = "Distribution of Year of Disturbance (YOD)",
  xlab = "Year of disturbance", ylab = "Count", col = "grey"
)


