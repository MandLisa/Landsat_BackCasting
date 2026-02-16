# ======================================================================
# Build an index of annual IBAP files in /mnt/dss_europe/level3_interpolated
# - Recursively finds files that contain "IBAP" in the filename
# - Extracts year from the first 4 characters of the basename
# - Extracts tile ID from the path (X####_Y####)
# - Checks duplicates and missing years (optional)
# - Writes index to CSV + RDS for fast reuse
#
# Output:
#   - ibap_file_index.csv
#   - ibap_file_index.rds
# ======================================================================

suppressPackageStartupMessages({
  library(data.table)
})

# -------------------- USER SETTINGS -----------------------------------
root_dir <- "/mnt/dss_europe/level3_interpolated"

# years you care about (adjust as needed)
year_min <- 1985
year_max <- 2024

# output
out_dir <- "/mnt/eo/EO4Backcasting/_indices/"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

out_csv <- file.path(out_dir, "ibap_file_index.csv")
out_rds <- file.path(out_dir, "ibap_file_index.rds")

# file filter:
# - includes: "...IBAP.tif" anywhere in the basename
# - excludes: "EV2.tif" etc automatically by pattern
pattern <- "IBAP.*\\.tif$"

# -------------------- HELPERS -----------------------------------------
get_year_from_basename <- function(p) {
  bn <- basename(p)
  y  <- substr(bn, 1, 4)
  if (!grepl("^\\d{4}$", y)) return(NA_integer_)
  as.integer(y)
}

get_tile_from_path <- function(p) {
  m <- regmatches(p, regexpr("X\\d{4}_Y\\d{4}", p))
  if (length(m) == 0 || is.na(m) || m == "") return(NA_character_)
  m
}

# Optional: parse "YYYYMMDD" date prefix if you want it
get_date_prefix <- function(p) {
  bn <- basename(p)
  d <- substr(bn, 1, 8)
  if (!grepl("^\\d{8}$", d)) return(NA_character_)
  d
}

# -------------------- FIND FILES --------------------------------------
message("Searching IBAP files under: ", root_dir)

ibap_files <- list.files(
  root_dir,
  pattern = pattern,
  recursive = TRUE,
  full.names = TRUE
)

if (length(ibap_files) == 0) {
  stop("No IBAP files found. Check root_dir and pattern.")
}

message("Found IBAP files: ", length(ibap_files))

# -------------------- BUILD INDEX -------------------------------------
dt <- data.table(file = ibap_files)

dt[, `:=`(
  basename = basename(file),
  year     = vapply(file, get_year_from_basename, integer(1)),
  date8    = vapply(file, get_date_prefix, character(1)),
  tile     = vapply(file, get_tile_from_path, character(1)),
  size_MB  = file.info(file)$size / 1024^2
)]

# keep only valid years
dt <- dt[!is.na(year)]
dt <- dt[year >= year_min & year <= year_max]

# normalize paths (useful for consistency)
dt[, file := normalizePath(file, winslash = "/", mustWork = TRUE)]

# sort
setorder(dt, tile, year, file)

# -------------------- QA CHECKS ---------------------------------------
message("\nQA checks:")

# 1) duplicates per tile-year
dups <- dt[, .N, by = .(tile, year)][N > 1]
if (nrow(dups) > 0) {
  message("WARNING: Duplicate files for some (tile, year). Showing first 20:")
  print(head(dups, 20))
} else {
  message("OK: No duplicates per (tile, year).")
}

# 2) missing years per tile (optional quick check)
#    This can be heavy if you have many tiles; keep it optional.
do_missing_check <- TRUE
if (do_missing_check) {
  yrs <- year_min:year_max
  tiles <- unique(na.omit(dt$tile))
  message("Checking missing years per tile (this may take a moment for many tiles)...")
  
  # Build a presence table: for each tile, which years exist
  pres <- dt[, .(has = TRUE), by = .(tile, year)]
  setkey(pres, tile, year)
  
  # Count missing years per tile
  miss_cnt <- rbindlist(lapply(tiles, function(tl) {
    have <- pres[J(tl, yrs), has]
    data.table(tile = tl, missing_n = sum(is.na(have)))
  }))
  setorder(miss_cnt, -missing_n)
  
  message("Tiles with most missing years (top 20):")
  print(head(miss_cnt, 20))
  
  # If you want to list missing years for a specific tile:
  # tl <- tiles[1]
  # have <- pres[J(tl, yrs), has]
  # missing_years <- yrs[is.na(have)]
  # print(data.table(tile=tl, missing_years=list(missing_years)))
}

# -------------------- SAVE --------------------------------------------
fwrite(dt, out_csv)
saveRDS(dt, out_rds)

message("\nSaved index:")
message(" - CSV: ", out_csv)
message(" - RDS: ", out_rds)

# -------------------- EXAMPLE USAGE -----------------------------------
# Fast lookup: file for given tile + year
# dt <- readRDS(out_rds)
# setkey(dt, tile, year)
# dt[J("X0001_Y0024", 2019), file]
