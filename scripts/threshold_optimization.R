# ======================================================================
# 03_thresholds_xgb_by_zone_single_threshold.R  (FULL STAND-ALONE SCRIPT)
#
# XGBoost only, INTERNAL validation only
# - one threshold per zone
# - two optimisation objectives:
#     (1) F1_balanced        : maximise macro-F1 (mean of class F1s)
#     (2) precision_priority : maximise min(precision_≤10, precision_>10)
#
# Outputs
#   - thresholds_xgb_by_zone.csv
#   - thresholds_xgb_by_zone.html   (open in Viewer)
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

out_csv  <- file.path(out_dir_eval, "thresholds_xgb_by_zone.csv")
out_html <- file.path(out_dir_eval, "thresholds_xgb_by_zone.html")

# ======================================================================
# 1) FAST THRESHOLD SCAN (exact, O(n log n))
# ======================================================================

scan_threshold_metrics_fast <- function(y_true, p_pos,
                                        ysd_levels = c("ysd1_10","ysd>10")) {
  
  y_true <- factor(y_true, levels = ysd_levels)
  p_pos  <- as.numeric(p_pos)
  
  ok <- is.finite(p_pos) & !is.na(y_true)
  y_true <- y_true[ok]
  p_pos  <- p_pos[ok]
  n <- length(p_pos)
  if (n == 0) stop("No valid samples to scan.")
  
  # pos class = ysd>10
  y <- as.integer(y_true == "ysd>10")
  
  # sort decreasing probabilities
  ord <- order(p_pos, decreasing = TRUE)
  p <- p_pos[ord]
  y <- y[ord]
  
  P <- sum(y == 1)    # true >10
  N <- n - P          # true ≤10
  
  tp <- cumsum(y == 1)
  fp <- cumsum(y == 0)
  tn <- N - fp
  fn <- P - tp
  
  accuracy <- (tp + tn) / n
  
  precision_gt10 <- tp / (tp + fp)                    # predicted >10 precision
  denom_le10     <- (tn + fn)                         # predicted ≤10 count
  precision_le10 <- ifelse(denom_le10 > 0, tn / denom_le10, NA_real_)
  
  recall_gt10 <- ifelse(P > 0, tp / (tp + fn), NA_real_)  # sensitivity for >10
  recall_le10 <- ifelse(N > 0, tn / (tn + fp), NA_real_)  # sensitivity for ≤10
  
  f1_gt10 <- ifelse((precision_gt10 + recall_gt10) > 0,
                    2 * precision_gt10 * recall_gt10 / (precision_gt10 + recall_gt10), NA_real_)
  f1_le10 <- ifelse((precision_le10 + recall_le10) > 0,
                    2 * precision_le10 * recall_le10 / (precision_le10 + recall_le10), NA_real_)
  
  dt <- data.table(
    threshold = p,  # rule: predict >10 if p >= threshold
    accuracy  = accuracy,
    precision_le10 = precision_le10,
    precision_gt10 = precision_gt10,
    recall_le10    = recall_le10,
    recall_gt10    = recall_gt10,
    f1_le10        = f1_le10,
    f1_gt10        = f1_gt10
  )
  
  dt[, min_precision := pmin(precision_le10, precision_gt10, na.rm = TRUE)]
  dt[, macro_f1      := rowMeans(.SD, na.rm = TRUE), .SDcols = c("f1_le10", "f1_gt10")]
  dt[, mean_precision := rowMeans(.SD, na.rm = TRUE), .SDcols = c("precision_le10","precision_gt10")]
  dt[, min_recall     := pmin(recall_le10, recall_gt10, na.rm = TRUE)]
  
  dt
}

pick_best <- function(scan_dt, objective = c("precision_priority","F1_balanced")) {
  objective <- match.arg(objective)
  
  if (objective == "precision_priority") {
    # best min-precision (conservative: avoids false positives in both classes),
    # then best mean precision, then accuracy, then min recall
    scan_dt <- scan_dt[order(-min_precision, -mean_precision, -accuracy, -min_recall, -threshold)]
  } else {
    # best macro-F1, then best mean precision, then accuracy, then min recall
    scan_dt <- scan_dt[order(-macro_f1, -mean_precision, -accuracy, -min_recall, -threshold)]
  }
  
  best <- scan_dt[1]
  best[, objective := objective]
  best
}

# ======================================================================
# 2) LOAD DATA (validation only) + zones
# ======================================================================

cat("\nLoading training CSV...\n")
dt <- fread(train_csv)
req_cols <- c("ID","x","y","ysd","state", base_pred)
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
if (split_dt[, anyDuplicated(ID)] > 0) stop("Split file contains duplicate IDs.")
dt[, set := NA_character_]
dt[split_dt, on="ID", set := i.set]
dt_val <- dt[set == "val"]
cat("Validation rows:", nrow(dt_val), "\n")

cat("Assigning zones (EPSG:3035 -> lat)...\n")
pts_3035 <- terra::vect(dt_val[, .(x,y)], geom=c("x","y"), crs="EPSG:3035")
pts_ll   <- terra::project(pts_3035, "EPSG:4326")
ll       <- terra::crds(pts_ll)
dt_val[, lat := ll[,2]]
dt_val[, zone := fifelse(lat < 45, "mediterranean",
                         fifelse(lat < 58, "temperate", "boreal"))]
dt_val[, zone := factor(zone, levels = zones)]

cat("\nValidation counts by zone × class:\n")
print(dt_val[, .N, by=.(zone, ysd_bin2)][order(zone, ysd_bin2)])

# ======================================================================
# 3) LOAD XGB + predict validation probabilities p(ysd>10)
# ======================================================================

cat("\nLoading XGBoost model...\n")
stopifnot(file.exists(xgb_path))
xgb_model <- readRDS(xgb_path)

cat("Predicting validation probabilities...\n")
Xv <- as.matrix(dt_val[, ..base_pred])
dt_val[, ppos_xgb := as.numeric(predict(xgb_model, xgboost::xgb.DMatrix(Xv)))]

# ======================================================================
# 4) OPTIMISE THRESHOLDS PER ZONE (two objectives)
# ======================================================================

cat("\nOptimising thresholds per zone...\n")

out_list <- list()
for (z in zones) {
  
  d <- dt_val[zone == z & is.finite(ppos_xgb)]
  if (nrow(d) == 0) next
  
  scan_dt <- scan_threshold_metrics_fast(d$ysd_bin2, d$ppos_xgb, ysd_levels)
  
  best_prec <- pick_best(scan_dt, "precision_priority")
  best_f1   <- pick_best(scan_dt, "F1_balanced")
  
  best_prec[, `:=`(model="xgb", zone=z)]
  best_f1[,   `:=`(model="xgb", zone=z)]
  
  out_list[[paste0(z,"_prec")]] <- best_prec
  out_list[[paste0(z,"_f1")]]   <- best_f1
}

thr_tbl <- rbindlist(out_list, fill=TRUE)
thr_tbl[, zone := factor(zone, levels = zones)]
thr_tbl[, objective := factor(objective, levels = c("precision_priority","F1_balanced"))]
setorder(thr_tbl, zone, objective)

thr_tbl <- thr_tbl[, .(
  model, zone, objective,
  threshold,
  accuracy,
  min_precision, macro_f1,
  precision_le10, precision_gt10,
  recall_le10, recall_gt10,
  f1_le10, f1_gt10,
  mean_precision, min_recall
)]

cat("\nThreshold table:\n")
print(thr_tbl)

fwrite(thr_tbl, out_csv)
cat("\nWrote CSV:\n", out_csv, "\n")

# ======================================================================
# 5) NICE TABLE (Viewer) — HTML export
# ======================================================================

gt_thr <- gt(thr_tbl) |>
  fmt_number(columns = c(threshold), decimals = 3) |>
  fmt_percent(
    columns = c(
      accuracy,
      min_precision, macro_f1,
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
    min_precision = "Min precision",
    macro_f1 = "Macro F1",
    precision_le10 = "≤10y",
    precision_gt10 = ">10y",
    recall_le10 = "≤10y",
    recall_gt10 = ">10y",
    f1_le10 = "≤10y",
    f1_gt10 = ">10y",
    mean_precision = "Mean precision",
    min_recall = "Min recall"
  ) |>
  tab_spanner(label = "Objectives", columns = c(min_precision, macro_f1)) |>
  tab_spanner(label = "Precision", columns = c(precision_le10, precision_gt10)) |>
  tab_spanner(label = "Recall", columns = c(recall_le10, recall_gt10)) |>
  tab_spanner(label = "F1 score", columns = c(f1_le10, f1_gt10)) |>
  tab_header(
    title = "Threshold optimisation comparison (XGBoost)",
    subtitle = "One threshold per zone: precision_priority vs F1_balanced (binary classification)"
  ) |>
  tab_options(table.font.size = 14)

gtsave(gt_thr, out_html)
cat("\nWrote HTML table for Viewer:\n", out_html, "\n")
cat("Open it in RStudio Viewer (or browser):\n", out_html, "\n\n")

gt_thr
