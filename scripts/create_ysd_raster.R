library(terra)
library(dplyr)
library(readr)

yod <- rast("/mnt/eo/EFDA_v211/latest_disturbance_eu_v211_2_3035.tif")

# compute yod raster
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

# --- paths (edit if needed) --------------------------------------------------
ysd_path <- "/mnt/eo/EFDA_v211/ysd_2023.tif"
pt_path  <- "/mnt/eo/EO4Backcasting/_intermediates/sample_points_core_1985_2005_yod.gpkg"
out_path <- "/mnt/eo/EO4Backcasting/_intermediates/sample_points_core_1985_2005_yod_ysd.gpkg"

# --- read data ---------------------------------------------------------------
ysd  <- rast(ysd_path)
names(ysd) <- "ysd"   # ensure the extracted column is named 'ysd'

pts  <- vect(pt_path) # SpatVector (points)

# --- align CRS if needed -----------------------------------------------------
if (!identical(crs(ysd), crs(pts))) {
  pts <- project(pts, crs(ysd))
}

# --- extract YSD to points (order-preserving) --------------------------------
vals <- extract(ysd, pts, ID = FALSE)  # returns a data.frame with column 'ysd'

# add as new attribute; same row order as 'pts'
pts$ysd <- as.integer(vals$ysd)

# (optional) drop points where ysd is NA
# pts <- pts[!is.na(pts$ysd), ]

# --- write result ------------------------------------------------------------
writeVector(pts, out_path, overwrite = TRUE)

# turn into a df
# Get attributes + x/y columns in one go
df <- as.data.frame(pts, geom = "XY")

# Ensure the coord column names are 'x' and 'y' (they usually already are)
coord_cols <- tail(names(df), 2)
names(df)[names(df) %in% coord_cols] <- c("x","y")

# Preview & save
head(df)
write.csv(df, "/mnt/eo/EO4Backcasting/_intermediates/sample_points_core_1985_2005_yod_ysd_xy.csv",
          row.names = FALSE)

# extract BAPs




