# ======================================================================
# 03_thresholds_xgb_fast_compare_precisionPriority_vs_F1Balanced_plusAccuracy.R
# (FULL STAND-ALONE SCRIPT — FAST, XGB ONLY, NO REJECT OPTION)
#
# Outputs
#   - thresholds_xgb_zone_objectives.csv
#   - thresholds_xgb_zone_objectives.html (for Viewer)
#
# Optional
#   - apply thresholds to XGB probability rasters to create binary rasters
#     (with NEW names so nothing is overwritten)
# ======================================================================

suppressPackageStartupMessages({
  library(data.table)
  library(terra)
  library(xgboost)
  library(gt)
  library(scales)
})

# ======================================================================
# 0) USER SETTINGS
# ======================================================================

train_csv      <- "/mnt/eo/EO4Backcasting/_intermediates/training_healthy_disturbed_2711_final.csv"
out_dir_models <- "/mnt/eo/EO4Backcasting/_models_comparison"
split_file     <- file.path(out_dir_models, "train_val_split_ids.csv")
xgb_path       <- file.path(out_dir_models, "xgb_ysd_bin2_prob.rds")

base_pred  <- c("blue", "green", "red", "nir", "swir1", "swir2")
ysd_levels <- c("ysd1_10", "ysd>10")
zones      <- c("mediterranean", "temperate", "boreal")

out_dir_eval <- file.path(out_dir_models, "_internal_validation")
dir.create(out_dir_eval, showWarnings = FALSE, recursive = TRUE)

out_thr_csv  <- file.path(out_dir_eval, "thresholds_xgb_zone_objectives.csv")
out_thr_html <- file.path(out_dir_eval, "thresholds_xgb_zone_objectives.html")

# ---- optional raster application ----
apply_to_rasters <- FALSE  # set TRUE to write binary rasters
out_dir_pred     <- "/mnt/eo/EO4Backcasting/_preds_Feb"  # folder with your XGB prob rasters
# Pattern should match YOUR XGB prob rasters; adjust if needed.
# Example expected: ysd_probs_xgb_mediterranean.tif
prob_pattern_template <- "^ysd_probs_xgb_%s.*\\.tif$"

# output names (do NOT overwrite existing probability rasters)
suffix_prec <- "_precision_priority"
suffix_f1   <- "_F1_balanced"

# ======================================================================
# 1) FAST threshold scan for 2-class decision (exact, O(n log n))
# ======================================================================

scan_threshold_metrics_fast <- function(y_true, p_pos, ysd_levels) {
  y_true <- factor(y_true, levels = ysd_levels)
  p_pos  <- as.numeric(p_pos)
  
  ok <- is.finite(p_pos) & !is.na(y_true)
  y_true <- y_true[ok]
  p_pos  <- p_pos[ok]
  n <- length(p_pos)
  if (n == 0) stop("No valid samples for threshold scanning.")
  
  # positive class = ysd>10
  y <- as.integer(y_true == "ysd>10")
  
  ord <- order(p_pos, decreasing = TRUE)
  p <- p_pos[ord]
  y <- y[ord]
  
  P <- sum(y == 1)  # total >10
  N <- n - P        # total ≤10
  
  tp <- cumsum(y == 1)
  fp <- cumsum(y == 0)
  tn <- N - fp
  fn <- P - tp
  
  accuracy <- (tp + tn) / n
  
  # precision for both classes
  precision_gt10 <- tp / (tp + fp)                         # predicted >10
  denom_le10     <- (tn + fn)                              # predicted ≤10
  precision_le10 <- ifelse(denom_le10 > 0, tn / denom_le10, NA_real_)
  
  # recall for both classes
  recall_gt10 <- ifelse(P > 0, tp / (tp + fn), NA_real_)   # true >10 captured
  recall_le10 <- ifelse(N > 0, tn / (tn + fp), NA_real_)   # true ≤10 captured
  
  f1_gt10 <- ifelse((precision_gt10 + recall_gt10) > 0,
                    2 * precision_gt10 * recall_gt10 / (precision_gt10 + recall_gt10), NA_real_)
  f1_le10 <- ifelse((precision_le10 + recall_le10) > 0,
                    2 * precision_le10 * recall_le10 / (precision_le10 + recall_le10), NA_real_)
  
  dt <- data.table(
    threshold = p,  # predict >10 if p >= threshold
    accuracy  = accuracy,
    precision_le10 = precision_le10,
    precision_gt10 = precision_gt10,
    recall_le10    = recall_le10,
    recall_gt10    = recall_gt10,
    f1_le10        = f1_le10,
    f1_gt10        = f1_gt10
  )
  
  dt[, obj_min_precision := pmin(precision_le10, precision_gt10, na.rm = TRUE)]
  dt[, obj_macro_f1      := rowMeans(.SD, na.rm = TRUE), .SDcols = c("f1_le10", "f1_gt10")]
  dt[, mean_precision    := rowMeans(.SD, na.rm = TRUE), .SDcols = c("precision_le10", "precision_gt10")]
  dt[, min_recall        := pmin(recall_le10, recall_gt10, na.rm = TRUE)]
  
  dt
}

pick_best_threshold <- function(scan_dt, objective = c("precision_priority", "F1_balanced")) {
  objective <- match.arg(objective)
  
  if (objective == "precision_priority") {
    # tie-breakers: mean precision, then accuracy, then min recall, then higher threshold
    scan_dt <- scan_dt[order(-obj_min_precision, -mean_precision, -accuracy, -min_recall, -threshold)]
    best <- scan_dt[1]
    best[, objective := "precision_priority"]
  } else {
    # tie-breakers: mean precision, then accuracy, then min recall, then higher threshold
    scan_dt <- scan_dt[order(-obj_macro_f1, -mean_precision, -accuracy, -min_recall, -threshold)]
    best <- scan_dt[1]
    best[, objective := "F1_balanced"]
  }
  best
}

# ======================================================================
# 2) Load validation set + zones
# ======================================================================

cat("\nLoading training CSV...\n")
dt <- fread(train_csv)

req_cols <- c("ID", "x", "y", "ysd", "state", base_pred)
stopifnot(all(req_cols %in% names(dt)))

dt <- dt[state == "disturbed"]

dt[, ysd_bin2 := NA_character_]
dt[ysd >= 1 & ysd <= 10, ysd_bin2 := "ysd1_10"]
dt[ysd > 10,             ysd_bin2 := "ysd>10"]
dt <- dt[!is.na(ysd_bin2)]
dt[, ysd_bin2 := factor(ysd_bin2, levels = ysd_levels)]

dt <- dt[complete.cases(dt[, ..base_pred])]

cat("Loading split file...\n")
stopifnot(file.exists(split_file))
split_dt <- fread(split_file)[, .(ID, set)]
stopifnot(all(c("ID","set") %in% names(split_dt)))
if (split_dt[, anyDuplicated(ID)] > 0) stop("Split file contains duplicate IDs.")

dt[, set := NA_character_]
dt[split_dt, on = "ID", set := i.set]
dt_val <- dt[set == "val"]
cat("Validation rows:", nrow(dt_val), "\n")

cat("Assigning zones (EPSG:3035 -> lat)...\n")
pts_3035 <- terra::vect(dt_val[, .(x, y)], geom = c("x","y"), crs = "EPSG:3035")
pts_ll   <- terra::project(pts_3035, "EPSG:4326")
ll       <- terra::crds(pts_ll)
dt_val[, lat := ll[,2]]
dt_val[, zone := fifelse(
  lat < 45, "mediterranean",
  fifelse(lat < 58, "temperate", "boreal")
)]
dt_val[, zone := factor(zone, levels = zones)]

cat("\nValidation counts by zone × class:\n")
print(dt_val[, .N, by = .(zone, ysd_bin2)][order(zone, ysd_bin2)])

# ======================================================================
# 3) Load XGB + predict validation probabilities
# ======================================================================

cat("\nLoading XGBoost model...\n")
stopifnot(file.exists(xgb_path))
xgb_model <- readRDS(xgb_path)

cat("Predicting p(ysd>10) on validation set...\n")
Xv <- as.matrix(dt_val[, ..base_pred])
dt_val[, ppos_xgb := as.numeric(predict(xgb_model, xgboost::xgb.DMatrix(Xv)))]

# ======================================================================
# 4) Optimise thresholds per zone (binary objectives only)
# ======================================================================

cat("\nOptimising thresholds per zone...\n")
rows <- list()

for (z in zones) {
  d <- dt_val[zone == z & is.finite(ppos_xgb)]
  if (nrow(d) == 0) next
  
  scan_dt   <- scan_threshold_metrics_fast(d$ysd_bin2, d$ppos_xgb, ysd_levels)
  best_prec <- pick_best_threshold(scan_dt, "precision_priority")
  best_f1   <- pick_best_threshold(scan_dt, "F1_balanced")
  
  best_prec[, `:=`(model="xgb", zone=z)]
  best_f1[,   `:=`(model="xgb", zone=z)]
  
  rows[[paste0(z, "_precision_priority")]] <- best_prec
  rows[[paste0(z, "_F1_balanced")]]        <- best_f1
}

thr_export <- rbindlist(rows, fill = TRUE)
thr_export[, zone := factor(zone, levels = zones)]
setorder(thr_export, zone, objective)

# export columns
thr_export <- thr_export[, .(
  model, zone, objective,
  threshold,
  accuracy,
  obj_min_precision, obj_macro_f1,
  precision_le10, precision_gt10,
  recall_le10, recall_gt10,
  f1_le10, f1_gt10,
  mean_precision, min_recall
)]

cat("\nThreshold comparison table:\n")
print(thr_export)

fwrite(thr_export, out_thr_csv)
cat("\nWrote CSV:\n", out_thr_csv, "\n")

# ======================================================================
# 5) Nicely formatted GT table (HTML + display)
# ======================================================================

gt_thr <- gt(thr_export) |>
  fmt_number(columns = c(threshold), decimals = 3) |>
  fmt_percent(
    columns = c(
      accuracy,
      obj_min_precision, obj_macro_f1,
      precision_le10, precision_gt10,
      recall_le10, recall_gt10,
      f1_le10, f1_gt10,
      mean_precision, min_recall
    ),
    decimals = 1
  ) |>
  cols_label(
    model = "Model",
    zone = "Zone",
    objective = "Objective",
    threshold = "Threshold",
    accuracy = "Accuracy",
    obj_min_precision = "Min precision",
    obj_macro_f1 = "Macro F1",
    precision_le10 = "≤10y",
    precision_gt10 = ">10y",
    recall_le10 = "≤10y",
    recall_gt10 = ">10y",
    f1_le10 = "≤10y",
    f1_gt10 = ">10y",
    mean_precision = "Mean precision",
    min_recall = "Min recall"
  ) |>
  tab_spanner(label = "Objectives", columns = c(obj_min_precision, obj_macro_f1)) |>
  tab_spanner(label = "Precision", columns = c(precision_le10, precision_gt10)) |>
  tab_spanner(label = "Recall", columns = c(recall_le10, recall_gt10)) |>
  tab_spanner(label = "F1 score", columns = c(f1_le10, f1_gt10)) |>
  tab_header(
    title = "Threshold optimisation comparison (XGBoost)",
    subtitle = "Per-zone thresholds for precision_priority vs F1_balanced (binary classification)"
  ) |>
  tab_options(table.font.size = 14)

gtsave(gt_thr, out_thr_html)
cat("\nWrote HTML table:\n", out_thr_html, "\n")
cat("Displaying table in Viewer...\n\n")
gt_thr

# ======================================================================
# 6) OPTIONAL: Apply thresholds to probability rasters (binary outputs)
# ======================================================================

apply_binary_thr_to_prob_raster <- function(prob_file, thr, out_file, overwrite = TRUE) {
  r <- rast(prob_file)
  idx <- grep("prob_ysd>10", names(r), fixed = TRUE)
  if (length(idx) != 1) stop("Band 'prob_ysd>10' not found uniquely in ", prob_file)
  ppos <- r[[idx]]
  cls  <- ifel(ppos >= thr, 1, 0)
  names(cls) <- "ysd_gt10"
  writeRaster(cls, out_file,
              datatype = "INT1U",
              gdal = c("COMPRESS=LZW","TILED=YES"),
              overwrite = overwrite)
  invisible(out_file)
}

if (isTRUE(apply_to_rasters)) {
  
  cat("\nApplying thresholds to XGB probability rasters...\n")
  
  for (z in zones) {
    pat  <- sprintf(prob_pattern_template, z)
    cand <- list.files(out_dir_pred, pattern = pat, full.names = TRUE)
    
    if (length(cand) == 0) {
      warning("No probability raster found for zone '", z, "' in ", out_dir_pred,
              "\nExpected pattern: ", pat)
      next
    }
    
    # choose newest if multiple
    if (length(cand) > 1) {
      cand <- cand[order(file.info(cand)$mtime, decreasing = TRUE)]
      message("Multiple prob rasters for zone ", z, ". Using newest:\n  ", cand[1])
    }
    prob_file <- cand[1]
    
    thr_prec <- thr_export[zone == z & objective == "precision_priority", threshold][1]
    thr_f1   <- thr_export[zone == z & objective == "F1_balanced", threshold][1]
    
    if (is.finite(thr_prec)) {
      out_prec <- file.path(out_dir_pred, paste0("ysd_class_xgb_", z, suffix_prec, ".tif"))
      apply_binary_thr_to_prob_raster(prob_file, thr_prec, out_prec, overwrite = TRUE)
    }
    
    if (is.finite(thr_f1)) {
      out_f1 <- file.path(out_dir_pred, paste0("ysd_class_xgb_", z, suffix_f1, ".tif"))
      apply_binary_thr_to_prob_raster(prob_file, thr_f1, out_f1, overwrite = TRUE)
    }
  }
  
  cat("\nDONE. Classified rasters written to:\n", out_dir_pred, "\n")
}
