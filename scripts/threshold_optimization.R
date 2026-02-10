# ============================================================
# SELF-CONTAINED SETUP
# Loads data, split, models, and computes validation probabilities
# ============================================================

suppressPackageStartupMessages({
  library(data.table)
  library(terra)
  library(ranger)
  library(xgboost)
  library(nnet)
  library(LiblineaR)
})

# ------------------------------------------------------------
# USER SETTINGS
# ------------------------------------------------------------

train_csv      <- "/mnt/eo/EO4Backcasting/_intermediates/training_healthy_disturbed_2711_final.csv"
out_dir_models <- "/mnt/eo/EO4Backcasting/_models_comparison"
split_file     <- file.path(out_dir_models, "train_val_split_ids.csv")

base_pred  <- c("blue", "green", "red", "nir", "swir1", "swir2")
ysd_levels <- c("ysd1_10", "ysd>10")

run_mlp <- TRUE

# ------------------------------------------------------------
# Helper: LiblineaR margin extraction
# ------------------------------------------------------------

extract_margin <- function(pred_obj, n_expected) {
  dv <- NULL
  if (!is.null(attr(pred_obj, "decisionValues"))) {
    dv <- attr(pred_obj, "decisionValues")
  } else if (is.list(pred_obj) && !is.null(pred_obj$decisionValues)) {
    dv <- pred_obj$decisionValues
  } else if (is.list(pred_obj) && !is.null(pred_obj$scores)) {
    dv <- pred_obj$scores
  }
  if (is.null(dv)) return(numeric(0))
  
  dv_mat <- as.matrix(dv)
  
  if (nrow(dv_mat) == n_expected) return(as.numeric(dv_mat[, 1]))
  if (ncol(dv_mat) == n_expected) return(as.numeric(t(dv_mat)[, 1]))
  
  dv_vec <- as.numeric(dv_mat)
  if (length(dv_vec) == n_expected) return(dv_vec)
  if (length(dv_vec) == 2L * n_expected) return(matrix(dv_vec, ncol = 2)[, 1])
  
  numeric(0)
}

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

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

# ------------------------------------------------------------
# LOAD SPLIT FILE
# ------------------------------------------------------------

cat("Loading split file...\n")
split_dt <- fread(split_file)[, .(ID, set)]
dt[, set := NA_character_]
dt[split_dt, on = "ID", set := i.set]

dt_val <- dt[set == "val"]

cat("Validation rows:", nrow(dt_val), "\n")

# ------------------------------------------------------------
# ZONE ASSIGNMENT
# ------------------------------------------------------------

pts_3035 <- terra::vect(dt_val[, .(x, y)], geom = c("x", "y"), crs = "EPSG:3035")
pts_ll   <- terra::project(pts_3035, "EPSG:4326")
ll       <- terra::crds(pts_ll)

dt_val[, lat := ll[, 2]]

dt_val[, zone := fifelse(
  lat < 45, "mediterranean",
  fifelse(lat < 58, "temperate", "boreal")
)]
dt_val[, zone := factor(zone, levels = c("mediterranean","temperate","boreal"))]

# ------------------------------------------------------------
# LOAD MODELS
# ------------------------------------------------------------

cat("Loading models...\n")

rf_model   <- readRDS(file.path(out_dir_models, "rf_ysd_bin2_prob.rds"))
svm_bundle <- readRDS(file.path(out_dir_models, "svm_linear_liblinear_platt.rds"))
xgb_model  <- readRDS(file.path(out_dir_models, "xgb_ysd_bin2_prob.rds"))

mlp_bundle <- NULL
mlp_path <- file.path(out_dir_models, "mlp_nnet_ysd_bin2_prob_scaled.rds")
if (run_mlp && file.exists(mlp_path)) {
  mlp_bundle <- readRDS(mlp_path)
}

# ------------------------------------------------------------
# PREDICT ON VALIDATION SET
# ------------------------------------------------------------

cat("Predicting validation probabilities...\n")

Xv <- as.matrix(dt_val[, ..base_pred])

# RF
p_rf <- predict(rf_model, data = as.data.frame(Xv))$predictions
p_rf <- p_rf[, ysd_levels, drop = FALSE]

# SVM
Xs <- scale(Xv, center = svm_bundle$mean, scale = svm_bundle$sd)
pr_svm <- predict(svm_bundle$model, Xs, decisionValues = TRUE)
dec <- extract_margin(pr_svm, n_expected = nrow(Xv))
p1 <- predict(svm_bundle$cal, newdata = data.frame(dec = dec), type = "response")
p_svm <- cbind(`ysd1_10` = 1 - p1, `ysd>10` = p1)

# XGB
p1_xgb <- predict(xgb_model, xgboost::xgb.DMatrix(Xv))
p_xgb <- cbind(`ysd1_10` = 1 - p1_xgb, `ysd>10` = p1_xgb)

# MLP
if (!is.null(mlp_bundle)) {
  Xs2 <- scale(Xv, center = mlp_bundle$mean, scale = mlp_bundle$sd)
  p_mlp <- predict(mlp_bundle$model, Xs2, type = "raw")
  colnames(p_mlp) <- ysd_levels
}

cat("Validation prediction complete.\n")


# ============================================================
# A) Threshold optimisation per zone: maximise min-precision
# ============================================================

pos_class <- "ysd>10"
ysd_levels <- c("ysd1_10", "ysd>10")

precisions_at_t <- function(y_true, p_pos, t) {
  pred <- ifelse(p_pos >= t, "ysd>10", "ysd1_10")
  pred <- factor(pred, levels = ysd_levels)
  y_true <- factor(y_true, levels = ysd_levels)
  
  cm <- table(y_true, pred)
  
  # precision for ysd>10
  tp_gt10  <- cm["ysd>10", "ysd>10"]
  col_gt10 <- sum(cm[, "ysd>10"])
  prec_gt10 <- if (col_gt10 == 0) NA_real_ else tp_gt10 / col_gt10
  
  # precision for ysd1_10
  tp_le10  <- cm["ysd1_10", "ysd1_10"]
  col_le10 <- sum(cm[, "ysd1_10"])
  prec_le10 <- if (col_le10 == 0) NA_real_ else tp_le10 / col_le10
  
  # recall (diagnostics)
  row_gt10 <- sum(cm["ysd>10", ])
  rec_gt10 <- if (row_gt10 == 0) NA_real_ else tp_gt10 / row_gt10
  row_le10 <- sum(cm["ysd1_10", ])
  rec_le10 <- if (row_le10 == 0) NA_real_ else tp_le10 / row_le10
  
  list(cm=cm, prec_gt10=prec_gt10, prec_le10=prec_le10, rec_gt10=rec_gt10, rec_le10=rec_le10)
}

opt_threshold_maxmin_precision <- function(y_true, p_pos) {
  grid <- sort(unique(p_pos))
  grid <- grid[is.finite(grid)]
  grid <- unique(c(0, grid, 1))
  
  res <- rbindlist(lapply(grid, function(t) {
    m <- precisions_at_t(y_true, p_pos, t)
    obj <- min(m$prec_gt10, m$prec_le10, na.rm = TRUE)
    
    data.table(
      threshold = t,
      obj_min_precision = obj,
      precision_gt10 = m$prec_gt10,
      precision_le10 = m$prec_le10,
      recall_gt10 = m$rec_gt10,
      recall_le10 = m$rec_le10
    )
  }))
  
  res[, mean_precision := rowMeans(.SD, na.rm = TRUE), .SDcols = c("precision_gt10","precision_le10")]
  res[, min_recall := pmin(recall_gt10, recall_le10, na.rm = TRUE)]
  
  # tie-breakers: best min-precision, then best mean precision, then best min recall
  res <- res[order(-obj_min_precision, -mean_precision, -min_recall, threshold)]
  res[1]
}

# attach p(ysd>10) for each model
dt_val[, ppos_rf  := p_rf[,  pos_class]]
dt_val[, ppos_svm := p_svm[, pos_class]]
dt_val[, ppos_xgb := p_xgb[, pos_class]]
if (exists("p_mlp")) dt_val[, ppos_mlp := p_mlp[, pos_class]]

prob_cols <- c(rf="ppos_rf", svm="ppos_svm", xgb="ppos_xgb")
if ("ppos_mlp" %in% names(dt_val)) prob_cols <- c(prob_cols, mlp="ppos_mlp")

thr_tbl <- rbindlist(lapply(names(prob_cols), function(m) {
  pc <- prob_cols[[m]]
  rbindlist(lapply(levels(dt_val$zone), function(z) {
    d <- dt_val[zone == z & is.finite(get(pc))]
    if (nrow(d) == 0) return(NULL)
    best <- opt_threshold_maxmin_precision(d$ysd_bin2, d[[pc]])
    cbind(data.table(model=m, zone=z), best)
  }))
}))

print(thr_tbl)

# save thresholds (choose your folder)
out_dir_eval <- file.path(out_dir_models, "_internal_validation")
dir.create(out_dir_eval, showWarnings = FALSE, recursive = TRUE)
fwrite(thr_tbl, file.path(out_dir_eval, "thresholds_maxmin_precision_by_zone.csv"))



# ============================================================
# B) Select XGBoost thresholds + apply to XGB probability rasters
#     → create binary class rasters (precision-oriented)
# ============================================================

library(data.table)
library(terra)

# --- where the probability rasters are (from your prediction script)
out_dir_pred <- "/mnt/eo/EO4Backcasting/_preds_Feb_2.0"

# --- IMPORTANT: naming suffix so you do NOT overwrite older outputs
# change this to whatever you want
bin_suffix <- "_thrMaxMinPrec"

# --- select XGB thresholds per zone
xgb_thr <- thr_tbl[model == "xgb", .(zone, threshold, obj_min_precision,
                                     precision_gt10, precision_le10,
                                     recall_gt10, recall_le10)]

stopifnot(nrow(xgb_thr) > 0)
setorder(xgb_thr, zone)
print(xgb_thr)

# --- helper: apply threshold to one raster file
apply_thr_to_xgb_raster <- function(prob_file, thr, out_file, overwrite = TRUE) {
  
  r <- terra::rast(prob_file)
  
  # identify the p(ysd>10) band robustly
  idx <- grep("prob_ysd>10", names(r), fixed = TRUE)
  if (length(idx) != 1) {
    stop("Could not uniquely find band 'prob_ysd>10' in: ", prob_file,
         "\nBands found: ", paste(names(r), collapse = ", "))
  }
  
  ppos <- r[[idx]]
  
  # Binary classification:
  # 1 = ysd>10
  # 0 = ysd<=10
  cls <- terra::ifel(ppos >= thr, 1, 0)
  names(cls) <- "ysd_gt10"
  
  terra::writeRaster(
    cls, out_file,
    datatype = "INT1U",
    gdal = c("COMPRESS=LZW", "TILED=YES"),
    overwrite = overwrite
  )
  
  invisible(out_file)
}

# --- apply per zone
zones <- as.character(levels(dt_val$zone))  # mediterranean / temperate / boreal

for (z in zones) {
  
  thr <- xgb_thr[zone == z, threshold]
  if (length(thr) != 1 || is.na(thr)) stop("No threshold found for zone: ", z)
  
  # find the probability raster for this zone
  # If your filenames are exactly 'ysd_probs_xgb_<zone>.tif', you can directly build the path instead.
  prob_pattern <- paste0("^ysd_probs_xgb_", z, ".*\\.tif$")
  cand <- list.files(out_dir_pred, pattern = prob_pattern, full.names = TRUE)
  
  if (length(cand) == 0) {
    stop("No XGB probability raster found for zone '", z, "' in ", out_dir_pred,
         "\nExpected pattern: ", prob_pattern)
  }
  if (length(cand) > 1) {
    # If you have multiple candidates (e.g., multiple runs), pick the newest by mtime
    cand <- cand[order(file.info(cand)$mtime, decreasing = TRUE)]
    message("Multiple rasters for zone ", z, ". Using newest:\n  ", cand[1])
  }
  
  prob_file <- cand[1]
  
  # output name (unique; will not overwrite probability raster)
  out_file <- file.path(
    out_dir_pred,
    paste0("ysd_class_xgb_", z, bin_suffix, ".tif")
  )
  
  cat("\n------------------------------------------------------------\n")
  cat("Zone:      ", z, "\n")
  cat("Threshold: ", thr, "\n")
  cat("Input:     ", prob_file, "\n")
  cat("Output:    ", out_file, "\n")
  
  apply_thr_to_xgb_raster(prob_file, thr, out_file, overwrite = TRUE)
}

cat("\nDONE: Binary XGB class rasters written to:\n", out_dir_pred, "\n")









