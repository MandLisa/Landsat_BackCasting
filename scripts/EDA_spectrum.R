suppressPackageStartupMessages(library(data.table))
suppressPackageStartupMessages(library(terra))

# ---- set your path ----
TRAIN_CSV <- "/mnt/eo/EO4Backcasting/_intermediates/training_data.csv"

# ---- read ----
pts <- fread(TRAIN_CSV)

# ---- add/overwrite column 'state' ----
pts[, state := "disturbed"]              # character column with constant value
# If you prefer a factor:
# pts[, state := factor("disturbed", levels = c("healthy", "disturbed"))]


# ---- write back (overwrite original) ----
fwrite(pts, TRAIN_CSV)

message("Done. Added column 'state' = 'disturbed' for all rows.")
message(sprintf("Backup written to: %s", bk))



suppressPackageStartupMessages(library(terra))

# --------- paths ---------
FOREST_MASK_PATH <- "/mnt/eo/EFDA_v211/forest_landuse_aligned.tif"   # 1 = forest, 0/NA = non-forest
DIST_YOD_PATH    <- "/mnt/eo/EFDA_v211/latest_disturbance_eu_v211_2_3035.tif"  # year of first disturbance; NA = none
OUT_PATH         <- "/mnt/eo/EFDA_v211/undisturbed_forest.tif"

# --------- load ---------
r_forest <- rast(FOREST_MASK_PATH)
r_yod    <- rast(DIST_YOD_PATH)

# --------- align grids (nearest for categorical) ---------
if (!same.crs(r_yod, r_forest)) {
  r_yod <- project(r_yod, r_forest, method = "near")
} else if (!all.equal(res(r_yod), res(r_forest)) || !ext(r_yod) == ext(r_forest)) {
  r_yod <- resample(r_yod, r_forest, method = "near")
}

# Write to disk
writeRaster(r_yod, "/mnt/eo/EFDA_v211/yod_aligned.tif")

# --------- normalize forest mask to 1 / NA ---------
# (treat any value >0.5 as forest = 1; else NA)
r_forest01 <- classify(r_forest, rbind(c(-Inf,0.5,NA), c(0.5,Inf,1)))

# --------- keep only forest pixels with NO disturbance mapped ---------
# mask(..., inverse=TRUE) keeps cells where 'r_yod' is NA
undist_forest <- mask(r_forest01, r_yod, inverse = TRUE)

# --------- write ---------
wopt_u8 <- list(datatype = "INT1U", gdal = "COMPRESS=DEFLATE,ZLEVEL=6")
writeRaster(undist_forest, OUT_PATH, overwrite = TRUE, wopt = wopt_u8)

message("Wrote undisturbed forest raster to: ", OUT_PATH)


# plot spectrum

# load long_plot
# read
long_plot <- readRDS("/mnt/eo/EO4Backcasting/_intermediates/long_plot.rds")

ggplot(long_plot, aes(x = band, y = val, fill = group)) +
  geom_boxplot(outlier.alpha = 0.2, width = 0.75,
               position = position_dodge2(preserve = "single")) +
  scale_fill_manual(values = pal, breaks = lvl) +
  labs(x = "Band", y = "BAP", fill = "Group",
       title = "BAP distributions by band: healthy vs. YSD bins") +
  theme_minimal(base_size = 12) +
  theme(legend.position = "right")
