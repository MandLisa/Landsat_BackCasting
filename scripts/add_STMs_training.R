suppressPackageStartupMessages({
  library(data.table)
  library(terra)
})

# ======================================================================
# Add STM features to existing training dataset
# - RAM-efficient
# - tile-wise and year-wise extraction
# - uses metadata table to retrieve t0 per point
# - extracts BLUE, GREEN, RED, NIR, SWIR1, SWIR2
# - extracts all STM bands: Q10, Q25, Q50, Q75, Q90, AVG, STD
# ======================================================================

# ----------------------------- SETTINGS --------------------------------

# Root directory with one folder per tile
stm_root_dir <- "/mnt/dss_europe/level3_repr/STM"

# Existing model-ready training dataset
train_file <- "/mnt/eo/EO4Backcasting/model_input/2303/training_selected_ibap_nbr_NNH.csv"

# Metadata file containing t0 per point_id
meta_file  <- "/mnt/eo/EO4Backcasting/model_input/2303/training_meta_all.csv"

# Output file
out_file   <- "/mnt/eo/EO4Backcasting/model_input/2303/training_data_stm.csv"

# Temporary directory for terra
terra_tmpdir <- "/mnt/dss_europe/temp_lm/stm"
dir.create(terra_tmpdir, recursive = TRUE, showWarnings = FALSE)
terraOptions(tempdir = terra_tmpdir, progress = 1)

# STM variables to extract
stm_vars <- c("BLUE", "GREEN", "RED", "NIR", "SWIR1", "SWIR2")

# Expected STM band names / aggregations
stm_band_names <- c("Q10", "Q25", "Q50", "Q75", "Q90", "AVG", "STD")

# If your files may contain nodata values that should become NA
nodata_values <- c(-10000, -9999, -32768)

# File naming assumption:
# YYYY0101-YYYY1231_001-365_HL_TSA_LNDLG_<VAR>_STM.tif
# where first 4 digits indicate the year
stm_suffix_regex <- "_STM\\.tif$"

# ----------------------------- HELPERS ---------------------------------

clean_values <- function(x, nodata_values = NULL) {
  if (!is.null(nodata_values) && length(nodata_values) > 0) {
    x[x %in% nodata_values] <- NA
  }
  x
}

get_year_from_filename <- function(x) {
  # first 4 chars of basename
  as.integer(substr(basename(x), 1, 4))
}

get_var_from_filename <- function(x) {
  # extract token before _STM.tif
  sub(".*_([A-Z0-9_]+)_STM\\.tif$", "\\1", basename(x))
}

make_stm_colnames <- function(var_name, band_names) {
  paste0("stm_", var_name, "_", band_names)
}

extract_one_stm_file <- function(file, pts_df, band_names, nodata_values = NULL) {
  r <- rast(file)
  
  # enforce band names
  if (nlyr(r) != length(band_names)) {
    stop(sprintf(
      "Unexpected number of bands in file:\n%s\nExpected: %d, found: %d",
      file, length(band_names), nlyr(r)
    ))
  }
  
  names(r) <- make_stm_colnames(get_var_from_filename(file), band_names)
  
  # create points with raster CRS
  pts <- vect(
    pts_df[, .(x, y)],
    geom = c("x", "y"),
    crs = crs(r)
  )
  
  vals <- terra::extract(r, pts, ID = FALSE)
  vals <- as.data.table(vals)
  
  # clean nodata if necessary
  for (j in seq_along(vals)) {
    set(vals, j = j, value = clean_values(vals[[j]], nodata_values))
  }
  
  vals
}

# ----------------------------- READ DATA -------------------------------

cat("Reading training data ...\n")
train_dt <- fread(train_file)

cat("Reading metadata ...\n")
meta_dt <- fread(meta_file)

# Keep only what is needed from metadata
meta_keep <- unique(meta_dt[, .(point_id, t0)])

# Basic checks
if (!"point_id" %in% names(train_dt)) stop("train_file must contain 'point_id'")
if (!all(c("point_id", "t0") %in% names(meta_keep))) stop("meta_file must contain 'point_id' and 't0'")
if (!all(c("x", "y", "tile") %in% names(train_dt))) stop("train_file must contain x, y, tile")

# Join t0 into training data
cat("Joining t0 from metadata into training data ...\n")
setkey(train_dt, point_id)
setkey(meta_keep, point_id)
train_dt <- meta_keep[train_dt]

# Check for missing t0
n_missing_t0 <- train_dt[is.na(t0), .N]
if (n_missing_t0 > 0) {
  warning(sprintf("%d rows have missing t0 after join. They will be skipped for STM extraction.", n_missing_t0))
}

# Prepare result table with only point_id initially
result_list <- vector("list", length = 0)

# Process only rows with t0 available
work_dt <- train_dt[!is.na(t0)]

# ------------------------- PROCESS TILE-WISE ---------------------------

tiles <- unique(work_dt$tile)
cat(sprintf("Processing %d tiles ...\n", length(tiles)))

for (tile_i in seq_along(tiles)) {
  tile_name <- tiles[tile_i]
  cat(sprintf("\n[%d/%d] Tile: %s\n", tile_i, length(tiles), tile_name))
  
  tile_dt <- work_dt[tile == tile_name]
  if (nrow(tile_dt) == 0) next
  
  tile_dir <- file.path(stm_root_dir, tile_name)
  if (!dir.exists(tile_dir)) {
    warning(sprintf("Tile directory not found: %s", tile_dir))
    next
  }
  
  all_files <- list.files(tile_dir, pattern = stm_suffix_regex, full.names = TRUE)
  if (length(all_files) == 0) {
    warning(sprintf("No STM files found in: %s", tile_dir))
    next
  }
  
  file_index <- data.table(
    file = all_files,
    year = get_year_from_filename(all_files),
    var  = get_var_from_filename(all_files)
  )
  
  # keep only requested variables
  file_index <- file_index[var %in% stm_vars]
  
  if (nrow(file_index) == 0) {
    warning(sprintf("No requested STM variables found in: %s", tile_dir))
    next
  }
  
  years_tile <- sort(unique(tile_dt$t0))
  
  tile_res_list <- vector("list", length(years_tile))
  names(tile_res_list) <- as.character(years_tile)
  
  for (yy in years_tile) {
    cat(sprintf("  Year %s\n", yy))
    
    pts_year <- tile_dt[t0 == yy, .(point_id, x, y)]
    if (nrow(pts_year) == 0) next
    
    files_year <- file_index[year == yy & var %in% stm_vars]
    if (nrow(files_year) == 0) {
      warning(sprintf("No STM files for tile %s and year %s", tile_name, yy))
      next
    }
    
    # ensure one file per requested variable
    missing_vars <- setdiff(stm_vars, files_year$var)
    if (length(missing_vars) > 0) {
      warning(sprintf(
        "Missing STM vars for tile %s year %s: %s",
        tile_name, yy, paste(missing_vars, collapse = ", ")
      ))
    }
    
    files_year <- files_year[match(stm_vars, var, nomatch = 0)]
    files_year <- files_year[!duplicated(var)]
    
    # start with point ids
    year_out <- copy(pts_year[, .(point_id)])
    
    # extract each variable separately to stay memory-safe
    for (k in seq_len(nrow(files_year))) {
      f <- files_year$file[k]
      v <- files_year$var[k]
      
      cat(sprintf("    extracting %s\n", v))
      
      vals <- extract_one_stm_file(
        file = f,
        pts_df = pts_year,
        band_names = stm_band_names,
        nodata_values = nodata_values
      )
      
      year_out <- cbind(year_out, vals)
      rm(vals)
      gc()
    }
    
    tile_res_list[[as.character(yy)]] <- year_out
    rm(year_out, pts_year, files_year)
    gc()
  }
  
  tile_res <- rbindlist(tile_res_list, use.names = TRUE, fill = TRUE)
  result_list[[tile_name]] <- tile_res
  
  rm(tile_dt, tile_res, tile_res_list, all_files, file_index)
  gc()
}

# -------------------------- MERGE BACK --------------------------------

cat("\nCombining extracted STM features ...\n")
stm_dt <- rbindlist(result_list, use.names = TRUE, fill = TRUE)

# Check duplicate point_ids in extracted result
dup_n <- stm_dt[, .N, by = point_id][N > 1, .N]
if (dup_n > 0) {
  warning(sprintf("There are %d duplicated point_id entries in STM result. Keeping first.", dup_n))
  setorder(stm_dt, point_id)
  stm_dt <- stm_dt[!duplicated(point_id)]
}

setkey(stm_dt, point_id)
setkey(train_dt, point_id)

final_dt <- stm_dt[train_dt]

# Reorder columns:
# existing columns first, then new stm_* columns at the end
old_cols <- names(train_dt)
stm_cols <- setdiff(names(final_dt), old_cols)
setcolorder(final_dt, c(old_cols, stm_cols))

# --------------------------- WRITE OUTPUT ------------------------------

cat(sprintf("\nWriting output to:\n%s\n", out_file))
fwrite(final_dt, out_file)

cat("\nDone.\n")
cat(sprintf("Rows in output: %d\n", nrow(final_dt)))
cat(sprintf("Number of added STM columns: %d\n", length(stm_cols)))