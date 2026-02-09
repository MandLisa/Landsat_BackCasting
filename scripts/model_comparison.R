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
# 7) RUN: APPLY TO THREE TILES - omit this and ump to the next section!
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
# 7) PREDICTION ON BAP TILES (probability rasters only; no validation)
#     Append this section to the END of 01_train_models_with_split.R
# ======================================================================

# ----------------------------
# 7.0) USER SETTINGS (PRED)
# ----------------------------

# output directory for probability rasters
out_dir_pred <- "/mnt/eo/EO4Backcasting/_preds_Feb_2.0"
dir.create(out_dir_pred, showWarnings = FALSE, recursive = TRUE)

# global forest mask (forest coded as 1; non-forest 0/NA/etc.)
forest_mask_global_file <- "/mnt/eo/EFDA_v211/forest_landuse_aligned.tif"

# forest mask erosion in pixels (0 disables erosion)
erosion_px <- 2

# tiles to predict on
tiles <- list(
  mediterranean = "/mnt/dss_europe/level3_interpolated/X0004_Y0027/19900801_LEVEL3_LNDLG_IBAP.tif",
  temperate     = "/mnt/dss_europe/level3_interpolated/X0011_Y0021/19900801_LEVEL3_LNDLG_IBAP.tif",
  boreal        = "/mnt/dss_europe/level3_interpolated/X0016_Y0006/19900801_LEVEL3_LNDLG_IBAP.tif"
)

# per-algorithm skipping behaviour
skip_if_exists <- TRUE

# ----------------------------
# 7.1) Helper: per-tile forest mask (align + optional erosion)
# ----------------------------

make_forest_mask_for_tile <- function(global_mask, tile_ref,
                                      erosion_px = 2,
                                      save_file = NULL,
                                      overwrite = TRUE) {
  
  # 1) crop to tile extent
  m <- terra::crop(global_mask, terra::ext(tile_ref))
  
  # 2) align CRS/grid
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

# ----------------------------
# 7.2) terra::predict() wrappers
#      Each must return a numeric matrix [n_pixels x 2] in ysd_levels order
# ----------------------------

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

# ----------------------------
# 7.3) (Optional) Load models from disk if not in memory
#      This makes the prediction part re-runnable even if training is skipped.
# ----------------------------

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
    cat("  - MLP not found; skipping\n")
  }
}
cat("All required models are available.\n")

# ----------------------------
# 7.4) Per-tile prediction (mask + prob rasters)
# ----------------------------

predict_probs_on_tile <- function(tile_name, tile_path,
                                  forest_mask_global,
                                  rf_model, svm_bundle, xgb_model, mlp_bundle,
                                  out_dir_pred,
                                  erosion_px = 2,
                                  overwrite = TRUE,
                                  skip_if_exists = TRUE) {
  
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
    
    if (isTRUE(skip_if_exists) && file.exists(out_file)) {
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
  
  write_probs("rf",  rf_model,   rf_fun_probs)
  write_probs("svm", svm_bundle, svm_fun_probs)
  write_probs("xgb", xgb_model,  xgb_fun_probs)
  
  if (!is.null(mlp_bundle)) {
    write_probs("mlp", mlp_bundle, mlp_fun_probs)
  }
  
  rm(bap_forest); gc()
  cat("Done tile:", tile_name, "\n")
}

# ----------------------------
# 7.5) RUN: apply to tiles
# ----------------------------

for (nm in names(tiles)) {
  if (!file.exists(tiles[[nm]])) {
    stop(paste0("Tile file not found for '", nm, "': ", tiles[[nm]]))
  }
}
stopifnot(file.exists(forest_mask_global_file))

forest_mask_global <- terra::rast(forest_mask_global_file)

cat("\n============================================================\n")
cat("PREDICTION PHASE (probability rasters only)\n")
cat("Output directory:\n  ", out_dir_pred, "\n")
cat("============================================================\n")

for (nm in names(tiles)) {
  predict_probs_on_tile(
    tile_name = nm,
    tile_path = tiles[[nm]],
    forest_mask_global = forest_mask_global,
    rf_model   = rf_model,
    svm_bundle = svm_bundle,
    xgb_model  = xgb_model,
    mlp_bundle = mlp_bundle,
    out_dir_pred = out_dir_pred,
    erosion_px = erosion_px,
    overwrite = TRUE,
    skip_if_exists = skip_if_exists
  )
}

cat("\nAll tiles processed. Probability rasters written to:\n", out_dir_pred, "\n")


