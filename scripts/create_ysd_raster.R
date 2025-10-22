library(terra)

yod <- rast("/mnt/eo/EFDA_v211/latest_disturbance_eu_v211_2_3035.tif")

# einfache Rasterarithmetik: NA bleiben NA
ysd_2023 <- 2023 - yod

writeRaster(
  ysd_2023,
  "/mnt/eo/EFDA_v211/ysd_2023.tif",
  overwrite = TRUE,
  wopt = list(
    datatype = "INT2S",
    NAflag   = -32768,
    gdal     = c("COMPRESS=ZSTD","TILED=YES","BIGTIFF=YES","BLOCKXSIZE=512","BLOCKYSIZE=512")
  )
)
