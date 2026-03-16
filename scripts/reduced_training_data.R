# ======================================================================
# Consolidate per-tile training tables and create a reduced dataset
# for a simple first model:
#   predictors = IBAP/BAP medians + NBR Theil-Sen trend
#
# Keeps for bookkeeping:
#   point_id, x, y, tile, label_undisturbed_20y
#
# Model features:
#   ibap_B*_med + nbr_sen_slope
# ======================================================================

suppressPackageStartupMessages({
  library(data.table)
})

# ---------------------------- SETTINGS ---------------------------------

in_dir  <- "/mnt/eo/EO4Backcasting/training_data"
out_dir <- "/mnt/eo/EO4Backcasting/model_input"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

feature_pattern <- "^training_X\\d{4}_Y\\d{4}_features\\.csv$"
meta_pattern    <- "^training_X\\d{4}_Y\\d{4}_meta\\.csv$"

out_reduced <- file.path(out_dir, "training_reduced_bap_nbrtrend.csv")
out_tilesum <- file.path(out_dir, "training_reduced_bap_nbrtrend_tile_summary.csv")
out_metasum <- file.path(out_dir, "training_meta_all.csv")   # optional

# -------------------------- FIND FILES ---------------------------------

feature_files <- list.files(in_dir, pattern = feature_pattern, full.names = TRUE)
if (length(feature_files) == 0) {
  stop("No training feature files found in: ", in_dir)
}

meta_files <- list.files(in_dir, pattern = meta_pattern, full.names = TRUE)

cat("Found", length(feature_files), "feature files\n")
cat("Found", length(meta_files), "meta files\n")

# ---------------------- DEFINE TARGET COLUMNS --------------------------

ref_names <- names(fread(feature_files[1], nrows = 0))

ibap_cols <- grep("^ibap_B\\d+_med$", ref_names, value = TRUE)
ibap_cols <- ibap_cols[order(as.integer(sub("^ibap_B(\\d+)_med$", "\\1", ibap_cols)))]

required_bookkeeping <- c("point_id", "x", "y", "tile", "label_undisturbed_20y")
required_model       <- c("nbr_sen_slope")

missing_ref <- setdiff(c(required_bookkeeping, required_model), ref_names)
if (length(missing_ref) > 0) {
  stop("Reference file is missing required columns: ", paste(missing_ref, collapse = ", "))
}
if (length(ibap_cols) == 0) {
  stop("No ibap_B*_med columns found.")
}

keep_cols <- c(required_bookkeeping, ibap_cols, "nbr_sen_slope")

cat("\nReduced dataset will keep these columns:\n")
print(keep_cols)

# ---------------------- CHECK COLUMN CONSISTENCY -----------------------

check_one_file <- function(f, keep_cols, ibap_cols_ref) {
  nm <- names(fread(f, nrows = 0))
  
  missing_keep <- setdiff(keep_cols, nm)
  
  ibap_here <- grep("^ibap_B\\d+_med$", nm, value = TRUE)
  ibap_extra <- setdiff(ibap_here, ibap_cols_ref)
  ibap_missing <- setdiff(ibap_cols_ref, ibap_here)
  
  data.table(
    file = basename(f),
    ok = length(missing_keep) == 0 && length(ibap_extra) == 0 && length(ibap_missing) == 0,
    missing_keep = paste(missing_keep, collapse = ","),
    ibap_missing = paste(ibap_missing, collapse = ","),
    ibap_extra = paste(ibap_extra, collapse = ",")
  )
}

checks <- rbindlist(lapply(feature_files, check_one_file, keep_cols = keep_cols, ibap_cols_ref = ibap_cols))

if (any(!checks$ok)) {
  print(checks[ok == FALSE])
  stop("Not all feature files have consistent reduced columns.")
}

# ---------------------- READ + BIND REDUCED TABLES ---------------------

read_reduced <- function(f, keep_cols) {
  dt <- fread(f, select = keep_cols)
  setcolorder(dt, keep_cols)
  dt[, source_file := basename(f)]
  dt
}

dt_reduced <- rbindlist(lapply(feature_files, read_reduced, keep_cols = keep_cols), use.names = TRUE, fill = TRUE)

# Put source_file at the end
setcolorder(dt_reduced, c(keep_cols, "source_file"))

# ---------------------- BASIC SANITY CHECKS ----------------------------

if (anyDuplicated(dt_reduced$point_id) > 0) {
  dup_n <- anyDuplicated(dt_reduced$point_id)
  warning("point_id is not globally unique. First duplicate index: ", dup_n)
}

model_cols <- c(ibap_cols, "nbr_sen_slope")

na_summary <- dt_reduced[, lapply(.SD, function(x) sum(is.na(x))), .SDcols = model_cols]
na_summary_long <- melt(na_summary, measure.vars = names(na_summary),
                        variable.name = "feature", value.name = "n_na")[order(feature)]

cat("\nRows in reduced dataset:", nrow(dt_reduced), "\n")
cat("Tiles in reduced dataset:", uniqueN(dt_reduced$tile), "\n")

cat("\nClass balance:\n")
print(dt_reduced[, .N, by = label_undisturbed_20y][order(label_undisturbed_20y)])

cat("\nFirst NA summary entries:\n")
print(na_summary_long[1:min(10, .N)])

# ---------------------- TILE SUMMARY -----------------------------------

tile_summary <- dt_reduced[, .(
  n = .N,
  n_undisturbed = sum(label_undisturbed_20y == 1, na.rm = TRUE),
  n_disturbed   = sum(label_undisturbed_20y == 0, na.rm = TRUE)
), by = tile][order(tile)]

# ---------------------- OPTIONAL: CONSOLIDATE META ---------------------

if (length(meta_files) > 0) {
  dt_meta_all <- rbindlist(lapply(meta_files, fread), use.names = TRUE, fill = TRUE)
  fwrite(dt_meta_all, out_metasum)
  cat("\nWrote meta master table:", out_metasum, "\n")
}

# ---------------------- WRITE OUTPUTS ----------------------------------

fwrite(dt_reduced, out_reduced)
fwrite(tile_summary, out_tilesum)

cat("\nWrote reduced training table:", out_reduced, "\n")
cat("Wrote tile summary:", out_tilesum, "\n")


### Import reduced training dataset with biome info


