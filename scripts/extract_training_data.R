library(terra)
library(data.table)
library(stringr)

# --- USER INPUTS --------------------------------------------------------------
points_csv   <- "/mnt/eo/EO4Backcasting/_intermediates/sample_points_core_1985_2005_yod_ysd_xy.csv"
ras_dir      <- "/mnt/eo/eu_mosaics/NBR_comp/"
pattern      <- "^NBR_(\\d{4})\\.tif$"
out_csv      <- "/mnt/eo/EO4Backcasting/_intermediates/training_data.csv"
include_yod_year <- FALSE   # FALSE => keep only year > YOD (YSD >= 1)

# --- READ POINTS --------------------------------------------------------------
pts <- fread(points_csv)[, .(x, y, yod)]
pts[, id := .I]  # stable pixel ID

# --- LIST FILES & BUILD MULTI-LAYER RASTER -----------------------------------
files <- list.files(ras_dir, pattern = pattern, full.names = TRUE)
years <- as.integer(str_match(basename(files), pattern)[,2])
tbl_files <- data.table(file = files, year = years)[!is.na(year)][order(year)]
tbl_files <- tbl_files[year >= 1985 & year <= 2024]

# A single SpatRaster with one layer per year (names become file basenames)
S <- rast(tbl_files$file)
names(S) <- paste0("Y", tbl_files$year)  # cleaner names like Y1985, ...

# --- BUILD VECTOR ONCE (assumes same CRS as rasters; reproject if needed) ----
# after you built S and pts
v_all <- vect(pts[, .(x, y)], geom = c("x","y"), crs = crs(S))


# --- SINGLE EXTRACT CALL ------------------------------------------------------
vals <- terra::extract(S, v_all, ID = FALSE)  # matrix: n_points x n_years

# --- LONG TABLE + JOIN METADATA ----------------------------------------------
DT <- as.data.table(vals)
DT[, id := pts$id]
setcolorder(DT, c("id", setdiff(names(DT), "id")))

# melt to long: columns Y1985..Y2024 -> rows
long <- melt(
  DT,
  id.vars = "id",
  variable.name = "layer",
  value.name = "NBR",
  variable.factor = FALSE
)

# add coordinates & yod
long[pts, c("x","y","yod") := .(x, y, yod), on = "id"]

# parse year from layer name "Y1985"
long[, year := as.integer(sub("^Y", "", layer))]
long[, layer := NULL]

# compute YSD and filter
long <- long[!is.na(NBR)]
long[, ysd := year - yod]
long <- if (isTRUE(include_yod_year)) long[ysd >= 0L] else long[ysd >= 1L]

# optional bins
long[, ysd_bin := fifelse(ysd <= 20, as.character(ysd),
                          fifelse(ysd <= 25, "21-25",
                                  fifelse(ysd <= 30, "26-30", "31-38+")))]
long[, class_label := paste0("ysd_", ysd_bin)]

# write once
fwrite(long[, .(id, x, y, yod, year, ysd, NBR, ysd_bin, class_label)], out_csv)


### add EVI
# --- USER INPUTS FOR EVI ------------------------------------------------------
evi_dir   <- "/mnt/eo/eu_mosaics/EVI_comp/"
evi_pat   <- "^EVI_(\\d{4})\\.tif$"
yr_min    <- 1985
yr_max    <- 2024

# --- LIST EVI FILES & BUILD MULTI-LAYER STACK --------------------------------
evi_files <- list.files(evi_dir, pattern = evi_pat, full.names = TRUE)
evi_years <- as.integer(str_match(basename(evi_files), evi_pat)[,2])

tbl_evi <- data.table(file = evi_files, year = evi_years)
tbl_evi <- tbl_evi[!is.na(year)][order(year)][year >= yr_min & year <= yr_max]

stopifnot(nrow(tbl_evi) > 0)

S_evi <- rast(tbl_evi$file)
names(S_evi) <- paste0("Y", tbl_evi$year)

# --- BUILD/REUSE THE VECTOR OF POINTS ----------------------------------------
# If you still have v_all from the NBR step, reuse it. Otherwise:
if (!exists("v_all")) {
  v_all <- vect(pts[, .(x, y)], geom = c("x","y"), crs = crs(S_evi))
  # If pts are in a different CRS, do:
  # v_all <- vect(pts[, .(x, y)], geom = c("x","y"), crs = "EPSG:4326")
  # v_all <- project(v_all, crs(S_evi))
}

# --- SINGLE EXTRACT CALL FOR EVI ---------------------------------------------
evi_vals <- terra::extract(S_evi, v_all, ID = FALSE)  # matrix: n_points x n_years

# Convert to long and attach id + year
EVI_dt <- as.data.table(evi_vals)
EVI_dt[, id := pts$id]
EVI_long <- melt(
  EVI_dt,
  id.vars = "id",
  variable.name = "layer",
  value.name = "EVI",
  variable.factor = FALSE
)
EVI_long[, year := as.integer(sub("^Y", "", layer))]
EVI_long[, layer := NULL]

# Treat coded NoData if present (optional; only if your EVI uses -10000)
EVI_long[EVI == -10000, EVI := NA_real_]

# --- JOIN INTO YOUR EXISTING LONG TABLE --------------------------------------
# long currently has columns: id, x, y, yod, year, ysd, NBR, (ysd_bin, class_label, …)
setkey(long,    id, year)
setkey(EVI_long, id, year)

# Append EVI column in place; rows with missing EVI remain NA
long[EVI_long, EVI := i.EVI]

# (Optional) If you want to enforce the same YSD filter as for NBR (already applied),
# nothing else is needed. If you want to drop rows where EVI is NA:
# long <- long[!is.na(EVI)]

# --- WRITE OUT (if desired) ---------------------------------------------------
fwrite(long, "/mnt/eo/EO4Backcasting/_intermediates/training_data/nbr_evi_extracted.csv")

#-------------------------------------------------------------------------------
### read in df containing EVI and NBR values to extract band-wise BAP values
points_csv   <- "/mnt/eo/EO4Backcasting/_intermediates/training_data/nbr_evi_extracted.csv"

pts <- fread(points_csv)[, .(id, x, y, yod, year, ysd, ysd_bin, class_label, EVI, NBR)]


### extract band-wise BAP values
# --- USER INPUTS --------------------------------------------------------------
bap_dir   <- "/mnt/dss_europe/mosaics_eu/mosaics_eu_baps"
bap_pat   <- "^(\\d{4})_mosaic_eu_cog\\.tif$"
yr_min    <- 1985
yr_max    <- 2024
nodata    <- -10000      # set to NULL if your BAPs don't use coded NoData
out_csv   <- "/mnt/eo/EO4Backcasting/_intermediates/training_data/nbr_evi_bap_extracted.csv"

# --- LIST BAP FILES -----------------------------------------------------------
bap_files <- list.files(bap_dir, pattern = bap_pat, full.names = TRUE)
bap_years <- as.integer(str_match(basename(bap_files), bap_pat)[,2])
tbl_bap <- data.table(file = bap_files, year = bap_years)[
  !is.na(year)][order(year)][year >= yr_min & year <= yr_max]
stopifnot(nrow(tbl_bap) > 0)

# --- POINTS VECTOR: ensure valid CRS -----------------------------------------
# Build once (include 'id' so we can join back)
v_all <- terra::vect(pts[, .(x, y, id)], geom = c("x","y"))

# Use the first BAP as CRS reference
R0 <- terra::rast(tbl_bap$file[1])

# If points have no CRS, *assign* the BAP CRS (not a reprojection yet)
if (is.na(terra::crs(v_all)) || terra::crs(v_all) == "") {
  terra::crs(v_all) <- terra::crs(R0)
}

# If CRS still differs, now we can safely project
if (!terra::same.crs(v_all, R0)) {
  v_all <- terra::project(v_all, terra::crs(R0))
}

# --- YEAR-BY-YEAR EXTRACTION (no stacking, tolerant to edge diffs) -----------
out_list <- vector("list", nrow(tbl_bap))

for (i in seq_len(nrow(tbl_bap))) {
  f  <- tbl_bap$file[i]
  yy <- tbl_bap$year[i]
  
  R <- terra::rast(f)  # multiband BAP for one year
  
  # If a particular year uses a different CRS (unlikely), project points on the fly
  v_use <- if (!terra::same.crs(v_all, R)) terra::project(v_all, terra::crs(R)) else v_all
  
  # Keep only points inside this raster's extent
  v_in <- try(terra::crop(v_use, R), silent = TRUE)
  if (inherits(v_in, "try-error") || is.null(v_in) || nrow(v_in) == 0) {
    out_list[[i]] <- data.table(id = integer(), year = integer())
    next
  }
  
  # Extract band values (n_points_in × n_bands)
  vals <- terra::extract(R, v_in, ID = FALSE)
  dt   <- as.data.table(vals)
  
  # Convert coded NoData to NA (if applicable)
  if (!is.null(nodata) && ncol(dt) > 0) {
    for (cc in names(dt)) {
      set(dt, which(dt[[cc]] == nodata), cc, NA_real_)
    }
  }
  
  # Name bands b1..bK
  if (ncol(dt) > 0) setnames(dt, paste0("b", seq_len(ncol(dt))))
  
  # Attach ids and year
  dt[, `:=`(id = v_in$id, year = yy)]
  setcolorder(dt, c("id","year", setdiff(names(dt), c("id","year"))))
  
  # Availability flag (useful later)
  dt[, bap_available := TRUE]
  
  out_list[[i]] <- dt
}

# Combine across years; varying band counts are handled with fill=TRUE
BAP_wide <- rbindlist(out_list, use.names = TRUE, fill = TRUE)

# Ensure flag exists even if no rows had coverage
if (!"bap_available" %in% names(BAP_wide)) BAP_wide[, bap_available := FALSE]

# --- JOIN INTO YOUR EXISTING 'long' (id × year) -------------------------------
long <- pts
setkey(long, id, year)
setkey(BAP_wide, id, year)

band_cols <- setdiff(names(BAP_wide), c("id","year","bap_available"))
if (length(band_cols)) {
  long[BAP_wide, (band_cols) := mget(paste0("i.", band_cols))]
}
long[BAP_wide, bap_available := i.bap_available]

# Optional: drop rows with no BAP coverage (edge outside 1995/2003)
# long <- long[bap_available == TRUE]

# Optional: rescale reflectance if scaled (e.g., divide by 10000)
# long[, (band_cols) := lapply(.SD, function(x) x / 10000), .SDcols = band_cols]

# --- WRITE OUT ---------------------------------------------------------------
fwrite(long, out_csv)
