library(data.table)
library(terra)

#-------------------------------------------------------------------------------
# 0. Input objects
#-------------------------------------------------------------------------------
# df  ... your training data frame shown in the screenshot
#      must contain columns: x, y, year

train_csv <- "/mnt/eo/EO4Backcasting/_intermediates/training_healthy_disturbed_2711_final.csv"
dt <- fread(train_csv)
#dt <- as.data.table(DT)          # if it is not yet a data.table

nbr_dir  <- "/mnt/eo/eu_mosaics/NBR_comp/"
pattern  <- "^NBR_(\\d{4})\\.tif$"

#-------------------------------------------------------------------------------
# 1. List NBR rasters and extract year from file name
#-------------------------------------------------------------------------------
nbr_files <- list.files(nbr_dir, pattern = pattern, full.names = TRUE)

nbr_years <- as.integer(sub("NBR_(\\d{4})\\.tif$", "\\1",
                            basename(nbr_files)))

# name the files by their year for easy lookup
names(nbr_files) <- nbr_years

#-------------------------------------------------------------------------------
# 2. Prepare data table and CRS
#-------------------------------------------------------------------------------
dt[, NBR := NA_real_]

years_in_data  <- sort(unique(dt$year))
nbr_years      <- as.integer(names(nbr_files))
years_common   <- intersect(years_in_data, nbr_years)

terraOptions(memfrac = 0.5, todisk = TRUE)  # conservative RAM usage

for (yr in years_common) {
  
  message("Processing year: ", yr)
  
  # 1. Open the NBR raster for this year (disk-backed, not all in RAM)
  r <- rast(nbr_files[as.character(yr)])
  
  # 2. Indices for this year in the training table
  idx <- which(dt$year == yr)
  
  if (length(idx) == 0L) next
  
  # 3. Build a small data.frame/data.table with explicit column names
  coords <- dt[idx, .(x = x, y = y)]   # ensure names "x" and "y"
  
  # 4. Create a SpatVector; explicitly specify geometry columns
  v <- vect(coords,
            geom = c("x", "y"),
            crs  = crs_nbr)
  
  # 5. Extract NBR values; only one layer, so take column 1
  vals <- terra::extract(r, v, ID = FALSE)[, 1]
  
  # 6. Write back to the NBR column
  dt[idx, NBR := vals]
}

# result
df_with_NBR <- as.data.frame(dt)

write.csv(df_with_NBR,
          file     = "/mnt/eo/EO4Backcasting/_intermediates/training_healthy_disturbed_2711_final.csv",
          row.names = FALSE,
          na       = "")

