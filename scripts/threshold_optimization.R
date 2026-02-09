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
