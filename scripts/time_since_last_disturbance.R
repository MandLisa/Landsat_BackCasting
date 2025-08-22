yod <- rast("/mnt/eo/EFDA_v211/latest_disturbance_eu_v211_2_3035.tif")

ref_year    <- 2023L                            # reference year for TSLD
min_year_ok <- 1986L                            # set to the first valid year in your map (adjust!)

# 1) Clean/validate the disturbance-year layer
#    - If your map uses 0 for "no disturbance", convert those to NA.
#    - Clamp to a plausible range so future/invalid years become NA.
yod_clean <- yod
yod_clean <- subst(yod_clean, from = 0, to = NA)                  # drop "no disturbance" if coded as 0
yod_clean <- clamp(yod_clean, lower = min_year_ok, upper = ref_year, values = TRUE)


# 2) Compute TSLD = reference year – year of disturbance
#    (undisturbed pixels remain NA; disturbance in 2023 → TSLD = 0)
tsld <- ref_year - yod_clean
names(tsld) <- paste0("tsld_", ref_year)

# 3) (Optional) Ensure integer storage
tsld <- round(tsld)
plot(tsld)

# 4) Write to disk as a tiled, compressed GeoTIFF
# set your output folder
outdir  <- "/mnt/eo/EFDA_v211/"   # <-- adapt this to your system
# fixed filename
outfile <- file.path(outdir, "time_since_last_disturbance.tif")

writeRaster(
  tsld, outfile, overwrite = TRUE,
  datatype = "INT2U",
  wopt = list(gdal = c("TILED=YES", "COMPRESS=LZW", "BIGTIFF=YES"))
)



