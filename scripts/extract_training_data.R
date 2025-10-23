# --- PACKAGES -----------------------------------------------------------------
library(terra)      
library(data.table) 
library(arrow)       
library(stringr)

# --- USER INPUTS --------------------------------------------------------------
points_csv   <- "/mnt/eo/EO4Backcasting/_intermediates/sample_points_core_1985_2005_yod_ysd_xy.csv"         # columns: x, y, yod
ras_dir      <- "/mnt/eo/eu_mosaics/NBR_comp/"        
pattern      <- "^NBR_(\\d{4})\\.tif$"       
out_csv      <- "nbr_extracted.csv"         
#out_parquet  <- "nbr_extracted.parquet"      
include_yod_year <- FALSE                   

# --- PREP: READ POINTS & MAKE STABLE IDs --------------------------------------
include_yod_year <- FALSE  # FALSE => keep only year > YOD (YSD >= 1); TRUE => include YSD = 0

# --- READ POINTS & CREATE STABLE IDS -----------------------------------------
pts <- fread(points_csv)                     # expects x, y, yod
stopifnot(all(c("x","y","yod") %in% names(pts)))
setDT(pts)
pts[, id := .I]                              # stable pixel ID

# If your points have a known CRS different from rasters, set it here, e.g.:
# crs_pts <- "EPSG:4326"  # WGS84 lon/lat

# --- FIND RASTERS & YEARS -----------------------------------------------------
files <- list.files(ras_dir, pattern = pattern, full.names = TRUE)
years <- as.integer(stringr::str_match(basename(files), pattern)[,2])
tbl_files <- data.table(file = files, year = years)
tbl_files <- tbl_files[!is.na(year)][order(year)]
tbl_files <- tbl_files[year >= 1985 & year <= 2024]  # guardrails

# --- INIT OUTPUT --------------------------------------------------------------
if (file.exists(out_csv)) file.remove(out_csv)

# --- MAIN LOOP: PROCESS YEAR BY YEAR ------------------------------------------
for (k in seq_len(nrow(tbl_files))) {
  yr <- tbl_files$year[k]
  f  <- tbl_files$file[k]
  
  # Subset to eligible points for this year
  pts_y <- if (isTRUE(include_yod_year)) pts[yod <= yr] else pts[yod < yr]
  if (nrow(pts_y) == 0L) next
  
  # Read just this year's raster (terra streams from disk)
  r <- rast(f)
  
  # Build vector for extraction; assume same CRS as raster.
  v <- vect(pts_y, geom = c("x","y"), crs = crs(r))
  # If points are in different CRS, uncomment the next two lines:
  # v <- vect(pts_y, geom = c("x","y"), crs = crs_pts)
  # v <- project(v, crs(r))
  
  # Extract NBR
  vals <- terra::extract(r, v, ID = FALSE)[,1]
  
  # Assemble results
  res <- pts_y[, .(id, x, y, yod)][, `:=`(
    year = yr,
    NBR  = vals
  )]
  
  # Drop NAs and compute YSD
  res <- res[!is.na(NBR)]
  res[, ysd := year - yod]
  
  # Enforce YSD bounds (handles the include_yod_year choice)
  res <- if (isTRUE(include_yod_year)) res[ysd >= 0L] else res[ysd >= 1L]
  
  # Append to CSV (header written only once because we removed any old file)
  if (nrow(res)) fwrite(res, out_csv, append = file.exists(out_csv))
}

# --- OPTIONAL: add bins / labels afterward -----------------------------------
dt <- fread(out_csv)
dt[, ysd_bin := fifelse(ysd <= 20, as.character(ysd),
                        fifelse(ysd <= 25, "21-25",
                                fifelse(ysd <= 30, "26-30", "31-38+")))]
# Example multi-class label (if you need it later):
dt[, class_label := ifelse(ysd >= 0, paste0("ysd_", ysd_bin), "undisturbed")]
fwrite(dt, sub("\\.csv$", "_with_bins.csv", out_csv))