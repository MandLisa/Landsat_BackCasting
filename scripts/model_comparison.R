# ======================================================================
# EO4Backcasting — 2-class YSD mapping (ysd1_10 vs ysd>10)
# Train once (global training data) and apply to three BAP tiles
# (Mediterranean / Temperate / Boreal), outputting ONLY probability rasters.
#
# Author: Lisa Mandl
# Project: EO4Backcasting
#
# Models (probability rasters, 2 bands):
#   - Random Forest (ranger)                  -> native class probabilities
#   - Linear SVM (LiblineaR type=1)           -> decision values + Platt scaling
#   - Gradient Boosting (xgboost)             -> binary:logistic probability
#   - Optional MLP (nnet, scaled, softmax)    -> 2-class softmax probabilities
#
# Outputs per tile & model:
#   - forest mask (tile-aligned; optional erosion)
#   - probability raster: prob_ysd1_10, prob_ysd>10
#
# Notes:
#   - No probability thresholding and no MMU filtering in this version.
#   - This script writes predictions only (no accuracy evaluation).
#   - Per-algorithm skipping: if the output file exists, it is skipped.
# ======================================================================

suppressPackageStartupMessages({
  library(data.table)
  library(terra)
  library(ranger)
  library(xgboost)
  library(nnet)
  library(LiblineaR)
})

# ======================================================================
# 0) USER SETTINGS
# ======================================================================

# --- training data (single-year observations) ---
train_csv <- "/mnt/eo/EO4Backcasting/_intermediates/training_healthy_disturbed_2711_final.csv"

# --- global forest mask source (forest coded as 1; non-forest NA/0) ---
forest_mask_global_file <- "/mnt/eo/EFDA_v211/forest_landuse_aligned.tif"

# --- output dirs ---
out_dir_models <- "/mnt/eo/EO4Backcasting/_models_comparison"
out_dir_pred   <- "/mnt/eo/EO4Backcasting/_preds_Feb"
dir.create(out_dir_models, showWarnings = FALSE, recursive = TRUE)
dir.create(out_dir_pred,   showWarnings = FALSE, recursive = TRUE)

# --- predictors (must match BAP band order you assign below) ---
base_pred <- c("blue", "green", "red", "nir", "swir1", "swir2")

# --- target (2 classes) ---
ysd_levels <- c("ysd1_10", "ysd>10")
K <- length(ysd_levels)

# --- forest mask erosion (in pixels); set 0 to disable ---
erosion_px <- 2

# --- models ---
run_mlp <- TRUE

# --- reproducibility ---
set.seed(42)

# --- tiles ---
tiles <- list(
  mediterranean = "/mnt/dss_europe/level3_interpolated/X0004_Y0027/19900801_LEVEL3_LNDLG_IBAP.tif",
  temperate     = "/mnt/dss_europe/level3_interpolated/X0011_Y0021/19900801_LEVEL3_LNDLG_IBAP.tif",
  boreal        = "/mnt/dss_europe/level3_interpolated/X0016_Y0006/19900801_LEVEL3_LNDLG_IBAP.tif"
)

# ======================================================================
# 0.1) GLOBAL HELPER: extract ONE margin (decision value) per sample
#      from LiblineaR predictions (handles n×1, n×2, transposed, flattened)
# ======================================================================

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
  
  # rows = n_expected (most common)
  if (nrow(dv_mat) == n_expected) return(as.numeric(dv_mat[, 1]))
  
  # sometimes returned transposed
  if (ncol(dv_mat) == n_expected) return(as.numeric(t(dv_mat)[, 1]))
  
  # flattened vector cases
  dv_vec <- as.numeric(dv_mat)
  if (length(dv_vec) == n_expected) return(dv_vec)
  if (length(dv_vec) == 2L * n_expected) return(matrix(dv_vec, ncol = 2)[, 1])
  
  numeric(0)
}

# ======================================================================
# 1) HELPER: PER-TILE FOREST MASK (ALIGN + OPTIONAL EROSION)
# ======================================================================

make_forest_mask_for_tile <- function(global_mask, tile_ref,
                                      erosion_px = 2,
                                      save_file = NULL,
                                      overwrite = TRUE) {
  
  # 1) crop first
  m <- terra::crop(global_mask, terra::ext(tile_ref))
  
  # 2) align to tile grid
  same_crs <- tryCatch(terra::same.crs(m, tile_ref), error = function(e) FALSE)
  if (same_crs) {
    m <- terra::resample(m, tile_ref, method = "near")
  } else {
    m <- terra::project(m, tile_ref, method = "near")
  }
  
  # 3) keep only forest==1 (everything else NA)
  m <- terra::classify(
    m,
    rcl = matrix(c(-Inf, 0.9999, NA,
                   0.9999, 1.0001, 1,
                   1.0001, Inf, NA),
                 ncol = 3, byrow = TRUE),
    include.lowest = TRUE
  )
  
  # 4) optional erosion via focal minimum
  if (!is.null(erosion_px) && erosion_px > 0) {
    bin <- terra::ifel(!is.na(m), 1, 0)
    w <- matrix(1, nrow = 2 * erosion_px + 1, ncol = 2 * erosion_px + 1)
    bin_er <- terra::focal(bin, w = w, fun = "min", na.policy = "omit", fillvalue = 0)
    m <- terra::ifel(bin_er == 1, 1, NA)
  }
  
  # 5) optional write-out
  if (!is.null(save_file)) {
    terra::writeRaster(
      m, save_file,
      datatype = "INT1U",
      gdal = c("COMPRESS=LZW", "TILED=YES"),
      overwrite = overwrite
    )
  }
  
  m
}

# ======================================================================
# 2) TRAINING DATA PREP (GLOBAL)
# ======================================================================

dt <- data.table::fread(train_csv)

req_cols <- c("ID", "ysd", "state", base_pred)
stopifnot(all(req_cols %in% names(dt)))

dt <- dt[state == "disturbed"]

dt[, ysd_bin2 := NA_character_]
dt[ysd >= 1 & ysd <= 10, ysd_bin2 := "ysd1_10"]
dt[ysd > 10,             ysd_bin2 := "ysd>10"]
dt <- dt[!is.na(ysd_bin2)]
dt[, ysd_bin2 := factor(ysd_bin2, levels = ysd_levels)]

dt <- dt[complete.cases(dt[, ..base_pred])]
setorder(dt, ID)

freq <- dt[, .N, by = ysd_bin2][order(ysd_bin2)]
w_vec <- freq$N
names(w_vec) <- as.character(freq$ysd_bin2)
class_weights <- max(w_vec) / w_vec

cat("\nClass frequency (training):\n"); print(freq)
cat("\nClass weights (inverse frequency):\n"); print(class_weights)

X <- as.matrix(dt[, ..base_pred])
y_factor <- dt$ysd_bin2
y_int <- as.integer(y_factor) - 1L
y01 <- ifelse(y_factor == "ysd>10", 1L, 0L)

x_mean <- colMeans(X)
x_sd   <- apply(X, 2, sd)
x_sd[x_sd == 0] <- 1
X_sc <- scale(X, center = x_mean, scale = x_sd)

w_row <- class_weights[as.character(y_factor)]

# ======================================================================
# 3) TRAIN MODELS (GLOBAL) + SAVE
# ======================================================================

# --- RF ---
cat("\nTraining RF...\n")
rf_formula <- as.formula(paste("ysd_bin2 ~", paste(base_pred, collapse = " + ")))
rf_model <- ranger::ranger(
  formula        = rf_formula,
  data           = dt[, c(base_pred, "ysd_bin2"), with = FALSE],
  num.trees      = 500,
  mtry           = max(1, floor(sqrt(length(base_pred)))),
  min.node.size  = 5,
  importance     = "impurity",
  probability    = TRUE,
  classification = TRUE,
  class.weights  = class_weights,
  num.threads    = 30
)
saveRDS(rf_model, file.path(out_dir_models, "rf_ysd_bin2_prob.rds"))

# --- SVM (LiblineaR SVM + Platt scaling) ---
cat("\nTraining SVM (linear, LiblineaR type=1) + Platt scaling calibration...\n")

set.seed(42)
cal_frac <- 0.10
n_all <- nrow(X_sc)
cal_idx <- sample.int(n_all, size = floor(cal_frac * n_all))
trn_idx <- setdiff(seq_len(n_all), cal_idx)

svm_lin <- LiblineaR::LiblineaR(
  data   = X_sc[trn_idx, , drop = FALSE],
  target = y01[trn_idx],
  type   = 1,
  cost   = 1,
  wi     = c(`0` = unname(class_weights["ysd1_10"]),
             `1` = unname(class_weights["ysd>10"])),
  verbose = FALSE
)

cal_pred <- predict(svm_lin, X_sc[cal_idx, , drop = FALSE], decisionValues = TRUE)
dec_cal <- extract_margin(cal_pred, n_expected = length(cal_idx))

if (length(dec_cal) == 0) {
  cat("\n--- DEBUG INFO: LiblineaR predict() output on calibration set ---\n")
  print(str(cal_pred))
  print(attributes(cal_pred))
  stop("LiblineaR did not return usable decision values; see debug output above.")
}
if (length(dec_cal) != length(cal_idx)) {
  stop("Calibration margin length mismatch: ", length(dec_cal), " vs ", length(cal_idx))
}

cal_df  <- data.frame(y = y01[cal_idx], dec = dec_cal)
cal_glm <- glm(y ~ dec, data = cal_df, family = binomial())

svm_bundle <- list(
  model = svm_lin,
  cal   = cal_glm,
  mean  = x_mean,
  sd    = x_sd,
  ysd_levels = ysd_levels
)
saveRDS(svm_bundle, file.path(out_dir_models, "svm_linear_liblinear_platt.rds"))

# --- XGBoost ---
cat("\nTraining XGBoost (binary logistic)...\n")
dtrain <- xgboost::xgb.DMatrix(data = X, label = y_int, weight = w_row)

params <- list(
  booster = "gbtree",
  objective = "binary:logistic",
  eval_metric = "logloss",
  eta = 0.05,
  max_depth = 6,
  min_child_weight = 5,
  subsample = 0.8,
  colsample_bytree = 0.8,
  lambda = 1,
  alpha = 0,
  tree_method = "hist",
  nthread = 30
)

xgb_model <- xgboost::xgb.train(
  params  = params,
  data    = dtrain,
  nrounds = 800,
  verbose = 1
)
saveRDS(xgb_model, file.path(out_dir_models, "xgb_ysd_bin2_prob.rds"))

# --- MLP (optional) ---
mlp_bundle <- NULL
if (isTRUE(run_mlp)) {
  cat("\nTraining MLP (nnet, scaled)...\n")
  mlp_model <- nnet::nnet(
    x = X_sc,
    y = nnet::class.ind(y_factor),
    size = 8,
    decay = 1e-4,
    maxit = 300,
    softmax = TRUE,
    trace = FALSE,
    weights = w_row
  )
  mlp_bundle <- list(model = mlp_model, mean = x_mean, sd = x_sd, ysd_levels = ysd_levels)
  saveRDS(mlp_bundle, file.path(out_dir_models, "mlp_nnet_ysd_bin2_prob_scaled.rds"))
}

cat("\nModel training complete.\n")

# ======================================================================
# 4) terra::predict() WRAPPERS (RETURN [n_pixels x 2] PROB MATRICES)
# ======================================================================

rf_fun_probs <- function(model, x, ...) {
  x_df <- as.data.frame(x)
  n <- nrow(x_df)
  out <- matrix(NA_real_, nrow = n, ncol = K)
  if (n == 0) return(out)
  
  idx <- stats::complete.cases(x_df)
  if (any(idx)) {
    p <- predict(model, data = x_df[idx, , drop = FALSE])$predictions
    p <- p[, ysd_levels, drop = FALSE]
    out[idx, ] <- as.matrix(p)
  }
  out
}

svm_fun_probs <- function(svm_bundle, x, ...) {
  x_df <- as.data.frame(x)
  n <- nrow(x_df)
  out <- matrix(NA_real_, nrow = n, ncol = K)
  if (n == 0) return(out)
  
  idx <- stats::complete.cases(x_df)
  if (!any(idx)) return(out)
  
  X0 <- as.matrix(x_df[idx, , drop = FALSE])
  Xs <- scale(X0, center = svm_bundle$mean, scale = svm_bundle$sd)
  
  pr <- predict(svm_bundle$model, Xs, decisionValues = TRUE)
  dec <- extract_margin(pr, n_expected = sum(idx))
  if (length(dec) == 0) stop("No usable decision values returned by LiblineaR in svm_fun_probs().")
  if (length(dec) != sum(idx)) stop("Decision length mismatch in svm_fun_probs().")
  
  p1 <- stats::predict(svm_bundle$cal, newdata = data.frame(dec = dec), type = "response")
  p0 <- 1 - p1
  
  out[idx, ] <- cbind(p0, p1)
  out
}

xgb_fun_probs <- function(model, x, ...) {
  x_df <- as.data.frame(x)
  n <- nrow(x_df)
  out <- matrix(NA_real_, nrow = n, ncol = K)
  if (n == 0) return(out)
  
  idx <- stats::complete.cases(x_df)
  if (any(idx)) {
    X0 <- as.matrix(x_df[idx, , drop = FALSE])
    p1 <- predict(model, xgboost::xgb.DMatrix(X0))
    p0 <- 1 - p1
    out[idx, ] <- cbind(p0, p1)
  }
  out
}

mlp_fun_probs <- function(mlp_bundle, x, ...) {
  x_df <- as.data.frame(x)
  n <- nrow(x_df)
  out <- matrix(NA_real_, nrow = n, ncol = K)
  if (n == 0) return(out)
  
  idx <- stats::complete.cases(x_df)
  if (any(idx)) {
    X0 <- as.matrix(x_df[idx, , drop = FALSE])
    Xs <- scale(X0, center = mlp_bundle$mean, scale = mlp_bundle$sd)
    p <- predict(mlp_bundle$model, Xs, type = "raw")
    out[idx, ] <- p
  }
  out
}

# ======================================================================
# 5) LOAD MODELS (OPTIONAL) — useful if you re-run only predictions
# ======================================================================

rf_path  <- file.path(out_dir_models, "rf_ysd_bin2_prob.rds")
svm_path <- file.path(out_dir_models, "svm_linear_liblinear_platt.rds")
xgb_path <- file.path(out_dir_models, "xgb_ysd_bin2_prob.rds")
mlp_path <- file.path(out_dir_models, "mlp_nnet_ysd_bin2_prob_scaled.rds")

cat("\nLoading trained models (if not already in memory)...\n")

if (!exists("rf_model"))   { stopifnot(file.exists(rf_path));  rf_model   <- readRDS(rf_path);  cat("  ✓ RF loaded\n") }
if (!exists("svm_bundle")) { stopifnot(file.exists(svm_path)); svm_bundle <- readRDS(svm_path); cat("  ✓ SVM loaded\n") }
if (!exists("xgb_model"))  { stopifnot(file.exists(xgb_path)); xgb_model  <- readRDS(xgb_path); cat("  ✓ XGBoost loaded\n") }

if (isTRUE(run_mlp) && !exists("mlp_bundle")) {
  if (file.exists(mlp_path)) {
    mlp_bundle <- readRDS(mlp_path); cat("  ✓ MLP loaded\n")
  } else {
    mlp_bundle <- NULL
  }
}

cat("All required models are available.\n")

# ======================================================================
# 6) PER-TILE PREDICTION FUNCTION (MASK + PROBABILITY OUTPUTS)
# ======================================================================

predict_probs_on_tile <- function(tile_name, tile_path,
                                  forest_mask_global,
                                  rf_model, svm_bundle, xgb_model, mlp_bundle,
                                  out_dir_pred,
                                  erosion_px = 2,
                                  overwrite = TRUE) {
  
  cat("\n============================================================\n")
  cat("Tile:", tile_name, "\n")
  cat("Path:", tile_path, "\n")
  
  bap <- terra::rast(tile_path)
  names(bap) <- base_pred
  
  fm_out <- file.path(out_dir_pred, paste0("forest_mask_", tile_name, "_eroded", erosion_px, "px.tif"))
  fm <- make_forest_mask_for_tile(
    global_mask = forest_mask_global,
    tile_ref    = bap,
    erosion_px  = erosion_px,
    save_file   = fm_out,
    overwrite   = overwrite
  )
  
  if (!isTRUE(terra::compareGeom(bap, fm, stopOnError = FALSE))) {
    stop("Forest mask and BAP are not aligned after projection — check CRS/resolution/extent.")
  }
  
  bap_forest <- terra::mask(bap, fm)
  rm(bap, fm); gc()
  
  write_probs <- function(model_tag, model_obj, fun_probs) {
    
    out_file <- file.path(out_dir_pred, paste0("ysd_probs_", model_tag, "_", tile_name, ".tif"))
    
    if (file.exists(out_file)) {
      cat("  ->", model_tag, "already exists — skipping\n")
      return(invisible(NULL))
    }
    
    cat("  -> predicting", model_tag, "...\n")
    
    pr <- terra::predict(
      bap_forest,
      model_obj,
      fun_probs,
      filename  = out_file,
      overwrite = overwrite
    )
    
    names(pr) <- paste0("prob_", ysd_levels)
    invisible(pr)
  }
  
  cat("Predicting RF probabilities...\n")
  pr_rf <- write_probs("rf", rf_model, rf_fun_probs)
  
  cat("Predicting SVM probabilities...\n")
  pr_svm <- write_probs("svm", svm_bundle, svm_fun_probs)
  
  cat("Predicting XGB probabilities...\n")
  pr_xgb <- write_probs("xgb", xgb_model, xgb_fun_probs)
  
  if (!is.null(mlp_bundle)) {
    cat("Predicting MLP probabilities...\n")
    pr_mlp <- write_probs("mlp", mlp_bundle, mlp_fun_probs)
    rm(pr_mlp)
  }
  
  rm(pr_rf, pr_svm, pr_xgb, bap_forest); gc()
  cat("Done tile:", tile_name, "\n")
}

# ======================================================================
# 7) RUN: APPLY TO THREE TILES
# ======================================================================

for (nm in names(tiles)) {
  if (!file.exists(tiles[[nm]])) {
    stop(paste0("Tile file not found for '", nm, "': ", tiles[[nm]]))
  }
}

forest_mask_global <- terra::rast(forest_mask_global_file)

for (nm in names(tiles)) {
  predict_probs_on_tile(
    tile_name = nm,
    tile_path = tiles[[nm]],
    forest_mask_global = forest_mask_global,
    rf_model  = rf_model,
    svm_bundle = svm_bundle,
    xgb_model = xgb_model,
    mlp_bundle = mlp_bundle,
    out_dir_pred = out_dir_pred,
    erosion_px = erosion_px,
    overwrite = TRUE
  )
}

cat("\nAll tiles processed. Probability rasters written to:\n", out_dir_pred, "\n")



# ======================================================================
# 9) ACCURACY PER CLASS × ZONE × MODEL  (requires labeled validation data)
# ======================================================================

# --- Provide validation data per zone (must contain ysd_bin2 + base_pred) ---
val_files <- list(
  mediterranean = "/mnt/eo/EO4Backcasting/_validation/val_mediterranean.csv",
  temperate     = "/mnt/eo/EO4Backcasting/_validation/val_temperate.csv",
  boreal        = "/mnt/eo/EO4Backcasting/_validation/val_boreal.csv"
)

# ---- helper: per-class accuracies (= recall per class) + overall/balanced ----
zone_metrics <- function(truth, pred, levels = ysd_levels) {
  truth <- factor(truth, levels = levels)
  pred  <- factor(pred,  levels = levels)
  
  cm <- table(truth = truth, pred = pred)
  overall <- mean(truth == pred)
  
  per_class <- diag(prop.table(cm, 1))
  # ensure named outputs even if one class missing in a zone subset
  acc_ysd1_10 <- unname(per_class["ysd1_10"])
  acc_ysd_gt10 <- unname(per_class["ysd>10"])
  bal <- mean(c(acc_ysd1_10, acc_ysd_gt10), na.rm = TRUE)
  
  list(
    overall_acc = overall,
    bal_acc = bal,
    acc_ysd1_10 = acc_ysd1_10,
    acc_ysd_gt10 = acc_ysd_gt10
  )
}

# ---- helper: predict hard class per model (using the trained objects) ----
predict_class_rf <- function(rf_model, Xdf) {
  p <- predict(rf_model, data = Xdf)$predictions
  factor(colnames(p)[max.col(p, ties.method = "first")], levels = ysd_levels)
}

predict_class_svm <- function(svm_bundle, Xmat) {
  # scale using training mean/sd stored in svm_bundle
  Xs <- scale(Xmat, center = svm_bundle$mean, scale = svm_bundle$sd)
  
  pr <- predict(svm_bundle$model, Xs, decisionValues = TRUE)
  dec <- extract_margin(pr, n_expected = nrow(Xs))
  stopifnot(length(dec) == nrow(Xs))
  
  p1 <- stats::predict(svm_bundle$cal, newdata = data.frame(dec = dec), type = "response")
  factor(ifelse(p1 >= 0.5, "ysd>10", "ysd1_10"), levels = ysd_levels)
}

predict_class_xgb <- function(xgb_model, Xmat) {
  p1 <- predict(xgb_model, xgboost::xgb.DMatrix(Xmat))
  factor(ifelse(p1 >= 0.5, "ysd>10", "ysd1_10"), levels = ysd_levels)
}

predict_class_mlp <- function(mlp_bundle, Xmat) {
  Xs <- scale(Xmat, center = mlp_bundle$mean, scale = mlp_bundle$sd)
  p  <- predict(mlp_bundle$model, Xs, type = "raw")  # n x 2
  factor(ysd_levels[max.col(p, ties.method = "first")], levels = ysd_levels)
}

# ---- loop zones and build the table ----
res <- list()

for (zn in names(val_files)) {
  
  f <- val_files[[zn]]
  stopifnot(file.exists(f))
  
  v <- data.table::fread(f)
  
  # requirements
  stopifnot(all(c("ysd_bin2", base_pred) %in% names(v)))
  v[, ysd_bin2 := factor(ysd_bin2, levels = ysd_levels)]
  v <- v[complete.cases(v[, ..base_pred])]
  
  Xmat <- as.matrix(v[, ..base_pred])
  Xdf  <- as.data.frame(Xmat)   # ranger likes data.frame
  
  truth <- v$ysd_bin2
  
  # RF
  pr_rf <- predict_class_rf(rf_model, Xdf)
  m_rf  <- zone_metrics(truth, pr_rf)
  res[[length(res)+1]] <- data.table::data.table(zone = zn, model = "RF", m_rf)
  
  # SVM (LiblineaR + Platt)
  pr_svm <- predict_class_svm(svm_bundle, Xmat)
  m_svm  <- zone_metrics(truth, pr_svm)
  res[[length(res)+1]] <- data.table::data.table(zone = zn, model = "SVM", m_svm)
  
  # XGB
  pr_xgb <- predict_class_xgb(xgb_model, Xmat)
  m_xgb  <- zone_metrics(truth, pr_xgb)
  res[[length(res)+1]] <- data.table::data.table(zone = zn, model = "XGB", m_xgb)
  
  # MLP (optional)
  if (!is.null(mlp_bundle)) {
    pr_mlp <- predict_class_mlp(mlp_bundle, Xmat)
    m_mlp  <- zone_metrics(truth, pr_mlp)
    res[[length(res)+1]] <- data.table::data.table(zone = zn, model = "MLP", m_mlp)
  }
}

acc_by_zone_model <- data.table::rbindlist(res, use.names = TRUE, fill = TRUE)
data.table::setorder(acc_by_zone_model, zone, model)

print(acc_by_zone_model)





# ======================================================================
# 8) INTERNAL VALIDATION (HOLD-OUT) + NICE PLOTS (PER ZONE × MODEL)
#    Append this block after model training in your script.
# ======================================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(pROC)
  library(scales)
})

# -----------------------------
# 8.0) Detect / define "zone"
# -----------------------------
zone_candidates <- c("zone", "biome", "tile", "region", "domain", "ecozone")
zone_col <- zone_candidates[zone_candidates %in% names(dt)][1]

if (is.na(zone_col) || length(zone_col) == 0) {
  message("\n[Validation] No zone column found in training table dt. ",
          "Proceeding with a single zone: 'global'.\n",
          "If you want per-zone validation, add a column such as 'zone' to the CSV.")
  dt[, zone_val := "global"]
  zone_col <- "zone_val"
} else {
  dt[, (zone_col) := as.factor(get(zone_col))]
}

# Make sure the response is in the expected factor levels
stopifnot("ysd_bin2" %in% names(dt))
dt[, ysd_bin2 := factor(ysd_bin2, levels = ysd_levels)]

# -----------------------------
# 8.1) Stratified hold-out split within each zone × class
# -----------------------------
set.seed(42)
val_frac <- 0.20

dt[, row_id__ := .I]
dt[, u__ := runif(.N)]

# within each zone × class: assign validation indices
dt[, is_val__ := FALSE]
dt[, is_val__ := {
  n_grp <- .N
  n_val <- max(1L, floor(val_frac * n_grp))
  idx <- order(u__)[seq_len(n_val)]
  out <- rep(FALSE, n_grp); out[idx] <- TRUE
  out
}, by = c(zone_col, "ysd_bin2")]

trn <- dt[is_val__ == FALSE]
val <- dt[is_val__ == TRUE]

message("\n[Validation] Split summary:")
print(trn[, .N, by = c(zone_col, "ysd_bin2")][order(get(zone_col), ysd_bin2)])
print(val[, .N, by = c(zone_col, "ysd_bin2")][order(get(zone_col), ysd_bin2)])

# -----------------------------
# 8.2) Prediction helpers on data.frame (not terra)
# -----------------------------
predict_rf_df <- function(model, new_dt) {
  xdf <- new_dt[, ..base_pred]
  idx <- complete.cases(xdf)
  out <- matrix(NA_real_, nrow = nrow(new_dt), ncol = length(ysd_levels))
  colnames(out) <- ysd_levels
  if (any(idx)) {
    p <- predict(model, data = xdf[idx, , drop = FALSE])$predictions
    p <- p[, ysd_levels, drop = FALSE]
    out[idx, ] <- as.matrix(p)
  }
  out
}

predict_svm_df <- function(bundle, new_dt) {
  xdf <- as.data.frame(new_dt[, ..base_pred])
  idx <- complete.cases(xdf)
  out <- matrix(NA_real_, nrow = nrow(new_dt), ncol = length(ysd_levels))
  colnames(out) <- ysd_levels
  if (!any(idx)) return(out)
  
  X0 <- as.matrix(xdf[idx, , drop = FALSE])
  Xs <- scale(X0, center = bundle$mean, scale = bundle$sd)
  
  pr <- predict(bundle$model, Xs, decisionValues = TRUE)
  dec <- extract_margin(pr, n_expected = sum(idx))
  if (length(dec) != sum(idx)) stop("[SVM] Decision length mismatch in validation prediction.")
  
  p1 <- predict(bundle$cal, newdata = data.frame(dec = dec), type = "response")  # prob(ysd>10)
  p0 <- 1 - p1
  out[idx, ] <- cbind(p0, p1)
  out
}

predict_xgb_df <- function(model, new_dt) {
  xdf <- as.data.frame(new_dt[, ..base_pred])
  idx <- complete.cases(xdf)
  out <- matrix(NA_real_, nrow = nrow(new_dt), ncol = length(ysd_levels))
  colnames(out) <- ysd_levels
  if (any(idx)) {
    X0 <- as.matrix(xdf[idx, , drop = FALSE])
    p1 <- predict(model, xgboost::xgb.DMatrix(X0))  # prob(ysd>10)
    out[idx, ] <- cbind(1 - p1, p1)
  }
  out
}

predict_mlp_df <- function(bundle, new_dt) {
  xdf <- as.data.frame(new_dt[, ..base_pred])
  idx <- complete.cases(xdf)
  out <- matrix(NA_real_, nrow = nrow(new_dt), ncol = length(ysd_levels))
  colnames(out) <- ysd_levels
  if (any(idx)) {
    X0 <- as.matrix(xdf[idx, , drop = FALSE])
    Xs <- scale(X0, center = bundle$mean, scale = bundle$sd)
    p <- predict(bundle$model, Xs, type = "raw")
    out[idx, ] <- p
  }
  out
}

# -----------------------------
# 8.3) Run validation predictions
# -----------------------------
models_to_eval <- list(
  rf  = function(new_dt) predict_rf_df(rf_model, new_dt),
  svm = function(new_dt) predict_svm_df(svm_bundle, new_dt),
  xgb = function(new_dt) predict_xgb_df(xgb_model, new_dt)
)

if (!is.null(mlp_bundle)) {
  models_to_eval$mlp <- function(new_dt) predict_mlp_df(mlp_bundle, new_dt)
}

# Build a long table with probabilities and hard class (argmax)
make_long_pred <- function(model_tag, P, new_dt) {
  stopifnot(nrow(P) == nrow(new_dt))
  # hard prediction = max prob
  pred_idx <- apply(P, 1, function(v) if (all(is.na(v))) NA_integer_ else which.max(v))
  pred_lab <- ifelse(is.na(pred_idx), NA_character_, ysd_levels[pred_idx])
  
  out <- data.table(
    model = model_tag,
    zone  = as.character(new_dt[[zone_col]]),
    truth = as.character(new_dt$ysd_bin2),
    prob_ysd1_10 = P[, "ysd1_10"],
    prob_ysd_gt10 = P[, "ysd>10"],
    pred = pred_lab
  )
  out
}

pred_long_list <- list()
for (m in names(models_to_eval)) {
  P <- models_to_eval[[m]](val)
  pred_long_list[[m]] <- make_long_pred(m, P, val)
}
pred_long <- rbindlist(pred_long_list, use.names = TRUE, fill = TRUE)
pred_long <- pred_long[!is.na(truth) & !is.na(pred)]  # remove incomplete rows

# Ensure factor order
pred_long[, truth := factor(truth, levels = ysd_levels)]
pred_long[, pred  := factor(pred,  levels = ysd_levels)]
pred_long[, model := factor(model, levels = names(models_to_eval))]

# -----------------------------
# 8.4) Metrics: overall + per-class (per zone × model)
# -----------------------------
safe_div <- function(a, b) ifelse(b == 0, NA_real_, a / b)

compute_metrics_one <- function(dd) {
  # confusion matrix
  cm <- table(dd$truth, dd$pred)
  # ensure full 2x2 layout
  cm2 <- matrix(0, nrow = 2, ncol = 2, dimnames = list(ysd_levels, ysd_levels))
  cm2[rownames(cm), colnames(cm)] <- cm
  
  # treat "ysd>10" as positive class for binary metrics
  TP <- cm2["ysd>10", "ysd>10"]
  TN <- cm2["ysd1_10", "ysd1_10"]
  FP <- cm2["ysd1_10", "ysd>10"]
  FN <- cm2["ysd>10", "ysd1_10"]
  
  acc  <- safe_div(TP + TN, TP + TN + FP + FN)
  tpr  <- safe_div(TP, TP + FN)  # recall pos
  tnr  <- safe_div(TN, TN + FP)  # specificity
  bacc <- mean(c(tpr, tnr), na.rm = TRUE)
  prec <- safe_div(TP, TP + FP)
  f1   <- ifelse(is.na(prec) | is.na(tpr) | (prec + tpr) == 0, NA_real_, 2 * prec * tpr / (prec + tpr))
  
  # per-class recalls
  rec_1_10 <- safe_div(cm2["ysd1_10", "ysd1_10"], sum(cm2["ysd1_10", ]))
  rec_gt10 <- safe_div(cm2["ysd>10", "ysd>10"], sum(cm2["ysd>10", ]))
  
  data.frame(
    accuracy = acc,
    balanced_accuracy = bacc,
    precision_pos = prec,
    recall_pos = tpr,
    specificity = tnr,
    f1_pos = f1,
    recall_ysd1_10 = rec_1_10,
    recall_ysd_gt10 = rec_gt10
  )
}

metrics_by_zone_model <- pred_long[, {
  mm <- compute_metrics_one(.SD)
  as.list(mm)
}, by = .(zone, model)]

# Per-class “accuracy-like” summaries (precision/recall/F1 for each class)
per_class_metrics <- function(dd, cls) {
  # one-vs-rest for a given class
  y_true <- dd$truth == cls
  y_pred <- dd$pred  == cls
  TP <- sum(y_true & y_pred)
  TN <- sum(!y_true & !y_pred)
  FP <- sum(!y_true & y_pred)
  FN <- sum(y_true & !y_pred)
  prec <- safe_div(TP, TP + FP)
  rec  <- safe_div(TP, TP + FN)
  f1   <- ifelse(is.na(prec) | is.na(rec) | (prec + rec) == 0, NA_real_, 2 * prec * rec / (prec + rec))
  sup  <- sum(y_true)
  data.frame(class = cls, precision = prec, recall = rec, f1 = f1, support = sup)
}

metrics_by_zone_model_class <- pred_long[, {
  rbind(
    per_class_metrics(.SD, "ysd1_10"),
    per_class_metrics(.SD, "ysd>10")
  )
}, by = .(zone, model)]

# -----------------------------
# 8.5) ROC/AUC per zone × model (positive class = ysd>10)
# -----------------------------
roc_tbl <- pred_long[, {
  y <- as.integer(truth == "ysd>10")
  p <- prob_ysd_gt10
  keep <- is.finite(p)
  if (sum(keep) < 10 || length(unique(y[keep])) < 2) {
    list(auc = NA_real_)
  } else {
    rr <- pROC::roc(response = y[keep], predictor = p[keep], quiet = TRUE)
    list(auc = as.numeric(pROC::auc(rr)))
  }
}, by = .(zone, model)]

# Build ROC curve points for plotting
roc_curve_pts <- pred_long[, {
  y <- as.integer(truth == "ysd>10")
  p <- prob_ysd_gt10
  keep <- is.finite(p)
  if (sum(keep) < 10 || length(unique(y[keep])) < 2) {
    data.frame(fpr = NA_real_, tpr = NA_real_)
  } else {
    rr <- pROC::roc(response = y[keep], predictor = p[keep], quiet = TRUE)
    data.frame(
      fpr = 1 - rr$specificities,
      tpr = rr$sensitivities
    )
  }
}, by = .(zone, model)]

# -----------------------------
# 8.6) Confusion matrix table for heatmaps
# -----------------------------
cm_long <- pred_long[, .N, by = .(zone, model, truth, pred)]

# Ensure all 2×2 cells exist per facet (fill missing with 0)
all_cells <- CJ(
  zone = unique(cm_long$zone),
  model = unique(cm_long$model),
  truth = factor(ysd_levels, levels = ysd_levels),
  pred  = factor(ysd_levels, levels = ysd_levels),
  unique = TRUE
)
cm_long <- merge(all_cells, cm_long, by = c("zone", "model", "truth", "pred"), all.x = TRUE)
cm_long[is.na(N), N := 0L]

# Also compute row-wise proportions (recall view)
cm_long[, row_sum := sum(N), by = .(zone, model, truth)]
cm_long[, prop := ifelse(row_sum == 0, NA_real_, N / row_sum)]

# -----------------------------
# 8.7) Calibration (reliability) curves
# -----------------------------
make_calibration <- function(dd, bins = 10) {
  tmp <- copy(dd)
  tmp <- tmp[is.finite(prob_ysd_gt10)]
  if (nrow(tmp) == 0) return(data.table(bin = integer(), p_mean = numeric(), y_mean = numeric(), n = integer()))
  tmp[, bin := cut(prob_ysd_gt10, breaks = seq(0, 1, length.out = bins + 1), include.lowest = TRUE, labels = FALSE)]
  tmp[, .(
    p_mean = mean(prob_ysd_gt10),
    y_mean = mean(truth == "ysd>10"),
    n = .N
  ), by = .(bin)]
}

cal_tbl <- pred_long[, make_calibration(.SD, bins = 10), by = .(zone, model)]

# ======================================================================
# 8.8) PLOTS
# ======================================================================

# (1) Balanced accuracy by zone × model
p_bacc <- ggplot(metrics_by_zone_model, aes(x = model, y = balanced_accuracy)) +
  geom_col() +
  facet_wrap(~ zone, nrow = 1) +
  scale_y_continuous(limits = c(0, 1), labels = percent_format(accuracy = 1)) +
  labs(
    title = "Internal validation (hold-out): Balanced accuracy by zone and model",
    x = NULL, y = "Balanced accuracy"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

print(p_bacc)

# (2) Confusion matrices (row-normalised) as heatmaps
p_cm <- ggplot(cm_long, aes(x = pred, y = truth, fill = prop)) +
  geom_tile(color = "white", linewidth = 0.4) +
  geom_text(aes(label = paste0(N, "\n", ifelse(is.na(prop), "", percent(prop, accuracy = 1)))), size = 3) +
  facet_grid(zone ~ model) +
  scale_fill_continuous(limits = c(0, 1), labels = percent_format(accuracy = 1), na.value = "grey90") +
  labs(
    title = "Confusion matrices (row-normalised): internal hold-out validation",
    x = "Predicted class", y = "True class", fill = "Row proportion"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    panel.grid = element_blank(),
    strip.text = element_text(face = "bold")
  )

print(p_cm)

# (3) ROC curves per zone (positive = ysd>10)
# Merge AUC for annotation
roc_ann <- merge(roc_tbl, unique(roc_curve_pts[, .(zone, model)]), by = c("zone", "model"), all.y = TRUE)

p_roc <- ggplot(roc_curve_pts, aes(x = fpr, y = tpr, group = model)) +
  geom_line(linewidth = 1) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
  facet_wrap(~ zone, nrow = 1) +
  scale_x_continuous(limits = c(0, 1)) +
  scale_y_continuous(limits = c(0, 1)) +
  labs(
    title = "ROC curves (positive class: ysd>10) — internal hold-out validation",
    x = "False positive rate (1 − specificity)",
    y = "True positive rate (sensitivity)"
  ) +
  theme_minimal(base_size = 12) +
  theme(panel.grid.minor = element_blank())

print(p_roc)

# Also show AUC table in console
message("\nAUC (positive class: ysd>10) by zone × model:")
print(roc_tbl[order(zone, model)])

# (4) Calibration / reliability curves
p_cal <- ggplot(cal_tbl, aes(x = p_mean, y = y_mean)) +
  geom_point(aes(size = n), alpha = 0.85) +
  geom_line() +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
  facet_grid(zone ~ model) +
  scale_x_continuous(limits = c(0, 1)) +
  scale_y_continuous(limits = c(0, 1)) +
  labs(
    title = "Calibration (reliability) curves — internal hold-out validation",
    x = "Mean predicted probability (ysd>10)",
    y = "Observed frequency (ysd>10)",
    size = "n"
  ) +
  theme_minimal(base_size = 12) +
  theme(panel.grid.minor = element_blank())

print(p_cal)

# ======================================================================
# 8.9) Export plots + metrics (optional)
# ======================================================================

val_out_dir <- file.path(out_dir_models, "internal_validation")
dir.create(val_out_dir, showWarnings = FALSE, recursive = TRUE)

ggsave(file.path(val_out_dir, "bacc_by_zone_model.png"), p_bacc, width = 12, height = 4, dpi = 300)
ggsave(file.path(val_out_dir, "confusion_matrices_row_normalised.png"), p_cm, width = 12, height = 6, dpi = 300)
ggsave(file.path(val_out_dir, "roc_by_zone.png"), p_roc, width = 12, height = 4, dpi = 300)
ggsave(file.path(val_out_dir, "calibration_curves.png"), p_cal, width = 12, height = 6, dpi = 300)

data.table::fwrite(metrics_by_zone_model,
                   file.path(val_out_dir, "metrics_by_zone_model.csv"))
data.table::fwrite(metrics_by_zone_model_class,
                   file.path(val_out_dir, "metrics_by_zone_model_class.csv"))
data.table::fwrite(roc_tbl,
                   file.path(val_out_dir, "auc_by_zone_model.csv"))

message("\n[Validation] Saved plots + tables to: ", val_out_dir)



