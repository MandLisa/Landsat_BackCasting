# ======================================================================
# 03_thresholds_xgb_fast_compare_F1_vs_precision.R  (STAND-ALONE)
#
# - Threshold optimisation per zone for XGB ONLY (FAST exact scan)
# - Compare two objectives:
#     (A) Precision-first:  maximise min(precision_le10, precision_gt10)
#     (B) F1-balanced:      maximise macro-F1 = mean(F1_le10, F1_gt10)
# - Write threshold comparison table
# - Optionally apply thresholds to XGB probability rasters to create binary rasters
#
# Binary raster encoding:
#   1 = ysd>10
#   0 = ysd<=10
# ======================================================================

suppressPackageStartupMessages({
  library(data.table)
  library(terra)
  library(xgboost)
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

# outputs
out_dir_eval <- file.path(out_dir_models, "_internal_validation")
dir.create(out_dir_eval, showWarnings = FALSE, recursive = TRUE)
out_thr_csv  <- file.path(out_dir_eval, "thresholds_xgb_compare_precision_vs_F1.csv")

# probability rasters directory (for applying thresholds)
out_dir_pred <- "/mnt/eo/EO4Backcasting/_preds_Feb"
prob_pattern_template <- "^ysd_probs_xgb_%s.*\\.tif$"  # expects ysd_probs_xgb_<zone>*.tif

# apply thresholds to rasters?
apply_to_rasters <- TRUE

# do NOT overwrite probability rasters; we write new binary rasters
suffix_prec <- "_thrMaxMinPrec"
suffix_f1   <- "_thrMaxMacroF1"

# ======================================================================
# 1) FAST scan: compute metrics for all thresholds in O(n log n)
# ======================================================================
# Threshold rule: predict gt10 if p >= t, else le10.

scan_threshold_metrics_fast <- function(y_true, p_pos) {
  y_true <- factor(y_true, levels = ysd_levels)
  p_pos  <- as.numeric(p_pos)
  
  ok <- is.finite(p_pos) & !is.na(y_true)
  y_true <- y_true[ok]
  p_pos  <- p_pos[ok]
  n <- length(p_pos)
  if (n == 0) stop("No valid samples for threshold scanning.")
  
  # positive class = gt10
  y <- as.integer(y_true == "ysd>10")
  
  ord <- order(p_pos, decreasing = TRUE)
  p <- p_pos[ord]
  y <- y[ord]
  
  P <- sum(y == 1)       # total gt10
  N <- n - P             # total le10
  
  k  <- seq_len(n)       # predicted positive count at each cut
  tp <- cumsum(y == 1)
  fp <- cumsum(y == 0)
  
  # Pred pos: k ; Pred neg: n-k
  tn <- N - fp
  fn <- P - tp
  
  # Precision
  prec_gt10 <- tp / (tp + fp)             # = tp/k
  denom_neg <- (tn + fn)                  # = n-k
  prec_le10 <- ifelse(denom_neg > 0, tn / denom_neg, NA_real_)
  
  # Recall
  rec_gt10 <- ifelse(P > 0, tp / (tp + fn), NA_real_)   # = tp/P
  rec_le10 <- ifelse(N > 0, tn / (tn + fp), NA_real_)   # = tn/N
  
  # F1 per class
  f1_gt10 <- ifelse(is.finite(prec_gt10 + rec_gt10) & (prec_gt10 + rec_gt10) > 0,
                    2 * prec_gt10 * rec_gt10 / (prec_gt10 + rec_gt10), NA_real_)
  f1_le10 <- ifelse(is.finite(prec_le10 + rec_le10) & (prec_le10 + rec_le10) > 0,
                    2 * prec_le10 * rec_le10 / (prec_le10 + rec_le10), NA_real_)
  
  dt <- data.table(
    threshold = p,   # predict gt10 if p >= threshold
    precision_gt10 = prec_gt10,
    precision_le10 = prec_le10,
    recall_gt10    = rec_gt10,
    recall_le10    = rec_le10,
    f1_gt10        = f1_gt10,
    f1_le10        = f1_le10
  )
  
  # objectives
  dt[, obj_min_precision := pmin(precision_gt10, precision_le10, na.rm = TRUE)]
  dt[, obj_macro_f1      := rowMeans(.SD, na.rm = TRUE), .SDcols = c("f1_gt10", "f1_le10")]
  dt[, mean_precision    := rowMeans(.SD, na.rm = TRUE), .SDcols = c("precision_gt10", "precision_le10")]
  dt[, min_recall        := pmin(recall_gt10, recall_le10, na.rm = TRUE)]
  
  dt
}

pick_best_threshold <- function(scan_dt, objective = c("min_precision", "macro_f1")) {
  objective <- match.arg(objective)
  
  if (objective == "min_precision") {
    # tie-breakers: mean precision, then min recall, then higher threshold
    scan_dt <- scan_dt[order(-obj_min_precision, -mean_precision, -min_recall, -threshold)]
  } else {
    # macro-F1 tie-breakers: mean precision, then min recall, then threshold
    scan_dt <- scan_dt[order(-obj_macro_f1, -mean_precision, -min_recall, -threshold)]
  }
  
  scan_dt[1]
}

# ======================================================================
# 2) Load validation data (disturbed only) + split + zones
# ======================================================================

cat("\nLoading CSV...\n")
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

cat("Loading split...\n")
stopifnot(file.exists(split_file))
split_dt <- fread(split_file)[, .(ID, set)]
stopifnot(all(c("ID","set") %in% names(split_dt)))
if (split_dt[, anyDuplicated(ID)] > 0) stop("Split file has duplicate IDs.")

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
# 3) Load XGB + predict p(ysd>10) on validation set
# ======================================================================

cat("\nLoading XGB model...\n")
stopifnot(file.exists(xgb_path))
xgb_model <- readRDS(xgb_path)

cat("Predicting p(ysd>10) on validation set...\n")
Xv <- as.matrix(dt_val[, ..base_pred])
dt_val[, ppos_xgb := as.numeric(predict(xgb_model, xgboost::xgb.DMatrix(Xv)))]

# ======================================================================
# 4) Optimise thresholds per zone for BOTH objectives
# ======================================================================

cat("\nOptimising thresholds per zone (FAST) for:\n  A) max-min precision\n  B) max macro-F1\n")

thr_rows <- list()

for (z in zones) {
  d <- dt_val[zone == z & is.finite(ppos_xgb)]
  if (nrow(d) == 0) next
  
  scan_dt <- scan_threshold_metrics_fast(d$ysd_bin2, d$ppos_xgb)
  
  best_prec <- pick_best_threshold(scan_dt, "min_precision")
  best_f1   <- pick_best_threshold(scan_dt, "macro_f1")
  
  thr_rows[[paste0(z, "_prec")]] <- cbind(
    data.table(zone = z, objective = "max_min_precision"),
    best_prec[, .(threshold,
                  obj_min_precision, mean_precision, min_recall,
                  precision_gt10, precision_le10,
                  recall_gt10, recall_le10,
                  f1_gt10, f1_le10,
                  obj_macro_f1)]
  )
  
  thr_rows[[paste0(z, "_f1")]] <- cbind(
    data.table(zone = z, objective = "max_macro_F1"),
    best_f1[, .(threshold,
                obj_macro_f1, mean_precision, min_recall,
                precision_gt10, precision_le10,
                recall_gt10, recall_le10,
                f1_gt10, f1_le10,
                obj_min_precision)]
  )
}

thr_cmp <- rbindlist(thr_rows, fill = TRUE)
thr_cmp[, model := "xgb"]
setcolorder(thr_cmp, c("model","zone","objective","threshold",
                       "obj_min_precision","obj_macro_f1",
                       "precision_le10","precision_gt10",
                       "recall_le10","recall_gt10",
                       "f1_le10","f1_gt10",
                       "mean_precision","min_recall"))
print(thr_cmp)

fwrite(thr_cmp, out_thr_csv)
cat("\nWrote threshold comparison table:\n", out_thr_csv, "\n")

# ======================================================================
# 5) Apply thresholds to XGB probability rasters (optional)
# ======================================================================

apply_thr_to_prob_raster <- function(prob_file, thr, out_file, overwrite = TRUE) {
  r <- rast(prob_file)
  
  idx <- grep("prob_ysd>10", names(r), fixed = TRUE)
  if (length(idx) != 1) {
    stop("Could not uniquely find band 'prob_ysd>10' in: ", prob_file,
         "\nBands: ", paste(names(r), collapse = ", "))
  }
  
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
  
  cat("\nApplying BOTH threshold sets to XGB probability rasters...\n")
  
  for (z in zones) {
    # find prob raster for this zone
    pat  <- sprintf(prob_pattern_template, z)
    cand <- list.files(out_dir_pred, pattern = pat, full.names = TRUE)
    if (length(cand) == 0) stop("No probability raster found for zone ", z, " (pattern: ", pat, ")")
    
    if (length(cand) > 1) {
      cand <- cand[order(file.info(cand)$mtime, decreasing = TRUE)]
      message("Multiple rasters for zone ", z, ". Using newest:\n  ", cand[1])
    }
    prob_file <- cand[1]
    
    # thresholds
    thr_prec <- thr_cmp[zone == z & objective == "max_min_precision", threshold][1]
    thr_f1   <- thr_cmp[zone == z & objective == "max_macro_F1",     threshold][1]
    
    if (!length(thr_prec) || is.na(thr_prec)) stop("Missing precision threshold for ", z)
    if (!length(thr_f1)   || is.na(thr_f1))   stop("Missing F1 threshold for ", z)
    
    out_prec <- file.path(out_dir_pred, paste0("ysd_class_xgb_", z, suffix_prec, ".tif"))
    out_f1   <- file.path(out_dir_pred, paste0("ysd_class_xgb_", z, suffix_f1,   ".tif"))
    
    cat("\n------------------------------------------------------------\n")
    cat("Zone: ", z, "\n")
    cat("Prob: ", prob_file, "\n")
    cat("thr (precision): ", thr_prec, " -> ", out_prec, "\n")
    cat("thr (macroF1):   ", thr_f1,   " -> ", out_f1,   "\n")
    
    apply_thr_to_prob_raster(prob_file, thr_prec, out_prec, overwrite = TRUE)
    apply_thr_to_prob_raster(prob_file, thr_f1,   out_f1,   overwrite = TRUE)
  }
  
  cat("\nDONE: binary rasters written to:\n", out_dir_pred, "\n")
}




### make a nice table where I compare thresholds
library(gt)
library(scales)

# reorder rows for readability
thr_cmp[, zone := factor(zone, levels = c("mediterranean","temperate","boreal"))]
thr_cmp[, objective := factor(objective, levels = c("max_min_precision","max_macro_F1"))]
setorder(thr_cmp, zone, objective)

gt_thr <- gt(thr_cmp) |>
  fmt_number(columns = "threshold", decimals = 3) |>
  fmt_percent(
    columns = c(
      precision_le10, precision_gt10,
      recall_le10, recall_gt10,
      f1_le10, f1_gt10,
      obj_min_precision, obj_macro_f1,
      mean_precision, min_recall
    ),
    decimals = 1
  ) |>
  cols_label(
    zone = "Zone",
    objective = "Optimization",
    threshold = "Threshold",
    precision_le10 = "≤10y",
    precision_gt10 = ">10y",
    recall_le10 = "≤10y",
    recall_gt10 = ">10y",
    f1_le10 = "≤10y",
    f1_gt10 = ">10y",
    obj_min_precision = "Min precision",
    obj_macro_f1 = "Macro F1",
    mean_precision = "Mean precision",
    min_recall = "Min recall"
  ) |>
  tab_spanner(
    label = "Precision",
    columns = c(precision_le10, precision_gt10)
  ) |>
  tab_spanner(
    label = "Recall",
    columns = c(recall_le10, recall_gt10)
  ) |>
  tab_spanner(
    label = "F1 score",
    columns = c(f1_le10, f1_gt10)
  ) |>
  tab_spanner(
    label = "Objectives",
    columns = c(obj_min_precision, obj_macro_f1)
  ) |>
  tab_header(
    title = "Threshold optimisation comparison (XGBoost)",
    subtitle = "Per-zone comparison: precision-first vs F1 optimisation"
  )

gt_thr

