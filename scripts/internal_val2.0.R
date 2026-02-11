# ======================================================================
# 02_internal_validation_only_3class.R  (FULL SCRIPT)
#
# EO4Backcasting — internal hold-out validation (NO retraining)
# Goal:
#   - Compare algorithms (RF / SVM+Platt / XGB / optional MLP)
#   - Evaluate on VALIDATION split only (ID-based split)
#   - Report metrics overall AND per-zone (Mediterranean / Temperate / Boreal)
#   - Report per-class metrics for ALL 3 classes: stable, ysd1_10, ysd>10
# Decision rule:
#   - Argmax of predicted probabilities
#
# Notes:
#   - Creates split file automatically if missing:
#       train_val_split_ids.csv   (ID-based, stratified by zone × class)
#   - Uses your 3-class model filenames from the TRAIN+PRED script:
#       rf_yclass_3c_prob.rds
#       svm_ovr_liblinear_platt_3c.rds
#       xgb_yclass_3c_softprob.rds
#       mlp_nnet_yclass_3c_scaled.rds   (optional)
#
# Author: Lisa Mandl
# ======================================================================

suppressPackageStartupMessages({
  library(data.table)
  library(terra)
  library(ranger)
  library(xgboost)
  library(nnet)
  library(LiblineaR)
  library(ggplot2)
  library(scales)
  library(gt)
})

# ======================================================================
# 0) USER SETTINGS
# ======================================================================

train_csv      <- "/mnt/eo/EO4Backcasting/_intermediates/training_healthy_disturbed_2711_final.csv"
out_dir_models <- "/mnt/eo/EO4Backcasting/_models_comparison_3c"
out_dir_eval   <- file.path(out_dir_models, "_internal_validation_3class")
dir.create(out_dir_eval, showWarnings = FALSE, recursive = TRUE)

split_file <- file.path(out_dir_models, "train_val_split_ids.csv")

base_pred <- c("blue", "green", "red", "nir", "swir1", "swir2")

# 3-class target
class_levels <- c("stable", "ysd1_10", "ysd>10")
K <- length(class_levels)

zones_levels <- c("mediterranean", "temperate", "boreal")

# Create split: validation fraction at ID-level
val_frac <- 0.20
seed_split <- 42

# If you did not train an MLP, keep TRUE; it will auto-skip if file missing
run_mlp <- TRUE

# Figure export
dpi <- 250
fig_w <- 10
fig_h <- 7

# ======================================================================
# 0.1) Helper: LiblineaR margin extraction (robust)
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
# 1) Create split file if missing (ID-based; stratified by zone × class)
# ======================================================================

if (!file.exists(split_file)) {
  cat("\nSplit file not found. Creating ID-based train/val split (stratified by zone × class):\n",
      split_file, "\n", sep = "")
  
  stopifnot(file.exists(train_csv))
  dt_tmp <- fread(train_csv)
  
  req_cols_tmp <- c("ID", "x", "y", "ysd", "state", base_pred)
  stopifnot(all(req_cols_tmp %in% names(dt_tmp)))
  
  # keep relevant states
  dt_tmp <- dt_tmp[state %in% c("healthy", "disturbed")]
  
  # build y_class
  dt_tmp[, y_class := NA_character_]
  dt_tmp[state == "healthy", y_class := "stable"]
  dt_tmp[state == "disturbed" & ysd >= 1 & ysd <= 10, y_class := "ysd1_10"]
  dt_tmp[state == "disturbed" & ysd > 10,             y_class := "ysd>10"]
  dt_tmp <- dt_tmp[!is.na(y_class)]
  dt_tmp[, y_class := factor(y_class, levels = class_levels)]
  
  # compute zones from EPSG:3035 -> lat
  pts_3035 <- terra::vect(dt_tmp[, .(x, y)], geom = c("x", "y"), crs = "EPSG:3035")
  pts_ll   <- terra::project(pts_3035, "EPSG:4326")
  ll       <- terra::crds(pts_ll)
  dt_tmp[, lat := ll[, 2]]
  
  dt_tmp[, zone := fifelse(
    lat < 45, "mediterranean",
    fifelse(lat < 58, "temperate", "boreal")
  )]
  dt_tmp[, zone := factor(zone, levels = zones_levels)]
  
  # collapse to ONE row per ID for splitting
  # (assumes each ID corresponds to a fixed location -> stable zone; y_class should be consistent per ID)
  id_tbl <- dt_tmp[, .(
    zone = zone[1],
    y_class = y_class[1]
  ), by = ID]
  
  if (id_tbl[, anyDuplicated(ID)] > 0) {
    stop("Internal error: ID table has duplicates (should be unique by ID).")
  }
  
  # stratified ID split by zone × class
  set.seed(seed_split)
  id_tbl[, u := runif(.N)]
  id_tbl[, set := "train"]
  id_tbl[, set := ifelse(u <= val_frac, "val", "train"), by = .(zone, y_class)]
  id_tbl[, u := NULL]
  
  # sanity check (avoid empty strata in val if a stratum is tiny)
  chk <- id_tbl[, .N, by = .(zone, y_class, set)]
  cat("\nSplit counts (IDs) by zone × class × set:\n")
  print(chk[order(zone, y_class, set)])
  
  fwrite(id_tbl[, .(ID, set)], split_file)
  cat("\n✓ Wrote split file: ", split_file, "\n", sep = "")
}

stopifnot(file.exists(split_file))

# ======================================================================
# 2) Load split file (need ID + set)
# ======================================================================

split_dt_raw <- fread(split_file)

if (!all(c("ID", "set") %in% names(split_dt_raw))) {
  stop("Split file must contain at least {ID, set}. Found: ",
       paste(names(split_dt_raw), collapse = ", "))
}

split_dt <- split_dt_raw[, .(ID, set)]
if (split_dt[, anyDuplicated(ID)] > 0) {
  stop("Split file contains duplicate IDs. Recreate split file.")
}

cat("\nSplit file loaded:\n", split_file, "\n")
cat("IDs in split file: ", nrow(split_dt), "\n")

# ======================================================================
# 3) Load data, create 3-class target + zones, attach split
# ======================================================================

stopifnot(file.exists(train_csv))
dt <- fread(train_csv)

req_cols <- c("ID", "x", "y", "ysd", "state", base_pred)
stopifnot(all(req_cols %in% names(dt)))

dt <- dt[state %in% c("healthy", "disturbed")]

# --- build 3-class target y_class ---
dt[, y_class := NA_character_]
dt[state == "healthy", y_class := "stable"]
dt[state == "disturbed" & ysd >= 1 & ysd <= 10, y_class := "ysd1_10"]
dt[state == "disturbed" & ysd > 10,             y_class := "ysd>10"]
dt <- dt[!is.na(y_class)]
dt[, y_class := factor(y_class, levels = class_levels)]

# complete predictors
dt <- dt[complete.cases(dt[, ..base_pred])]

# zones from EPSG:3035 -> lat
pts_3035 <- terra::vect(dt[, .(x, y)], geom = c("x", "y"), crs = "EPSG:3035")
pts_ll   <- terra::project(pts_3035, "EPSG:4326")
ll       <- terra::crds(pts_ll)
dt[, lat := ll[, 2]]

dt[, zone := fifelse(
  lat < 45, "mediterranean",
  fifelse(lat < 58, "temperate", "boreal")
)]
dt[, zone := factor(zone, levels = zones_levels)]

# attach split
dt[, set := NA_character_]
dt[split_dt, on = "ID", set := i.set]
dt <- dt[!is.na(set)]

dt_val <- dt[set == "val"]

cat("\nValidation rows: ", nrow(dt_val), "\n")
cat("Validation IDs:  ", dt_val[, uniqueN(ID)], "\n")
cat("\nValidation counts by zone × class:\n")
print(dt_val[, .N, by = .(zone, y_class)][order(zone, y_class)])

# ======================================================================
# 4) Load trained models (3-class versions from your training script)
# ======================================================================

rf_path  <- file.path(out_dir_models, "rf_yclass_3c_prob.rds")
svm_path <- file.path(out_dir_models, "svm_ovr_liblinear_platt_3c.rds")
xgb_path <- file.path(out_dir_models, "xgb_yclass_3c_softprob.rds")
mlp_path <- file.path(out_dir_models, "mlp_nnet_yclass_3c_scaled.rds")

stopifnot(file.exists(rf_path), file.exists(svm_path), file.exists(xgb_path))

rf_model   <- readRDS(rf_path)
svm_bundle <- readRDS(svm_path)
xgb_model  <- readRDS(xgb_path)

mlp_bundle <- NULL
if (isTRUE(run_mlp) && file.exists(mlp_path)) {
  mlp_bundle <- readRDS(mlp_path)
} else if (isTRUE(run_mlp)) {
  cat("\nNOTE: run_mlp=TRUE but MLP file not found; skipping MLP.\n")
}

# ======================================================================
# 5) Predict on validation set (probabilities + argmax class)
# ======================================================================

Xv <- as.matrix(dt_val[, ..base_pred])

# ---- RF ----
p_rf <- predict(rf_model, data = as.data.frame(Xv))$predictions
p_rf <- p_rf[, class_levels, drop = FALSE]
pred_rf <- factor(colnames(p_rf)[max.col(p_rf)], levels = class_levels)

# ---- SVM OvR + Platt (3-class probabilities) ----
stopifnot(all(c("models", "cal", "mean", "sd", "class_levels") %in% names(svm_bundle)))
stopifnot(identical(as.character(svm_bundle$class_levels), class_levels))

Xs <- scale(Xv, center = svm_bundle$mean, scale = svm_bundle$sd)

p_svm <- matrix(NA_real_, nrow = nrow(Xv), ncol = K)
colnames(p_svm) <- class_levels

for (k in seq_len(K)) {
  pr  <- predict(svm_bundle$models[[k]], Xs, decisionValues = TRUE)
  dec <- extract_margin(pr, n_expected = nrow(Xv))
  if (length(dec) != nrow(Xv)) {
    stop("SVM: decision value length mismatch for class ", class_levels[k])
  }
  p_svm[, k] <- predict(svm_bundle$cal[[k]],
                        newdata = data.frame(dec = dec),
                        type = "response")
}

# normalize rows to sum to 1
rs <- rowSums(p_svm)
rs[rs == 0 | !is.finite(rs)] <- NA_real_
p_svm <- p_svm / rs

pred_svm <- factor(colnames(p_svm)[max.col(p_svm)], levels = class_levels)

# ---- XGB (multiclass softprob) ----
p_xgb_vec <- predict(xgb_model, xgboost::xgb.DMatrix(Xv))
p_xgb <- matrix(p_xgb_vec, ncol = K, byrow = TRUE)
colnames(p_xgb) <- class_levels
pred_xgb <- factor(colnames(p_xgb)[max.col(p_xgb)], levels = class_levels)

# ---- MLP (optional) ----
pred_mlp <- NULL
p_mlp <- NULL
if (!is.null(mlp_bundle)) {
  stopifnot(all(c("model", "mean", "sd", "class_levels") %in% names(mlp_bundle)))
  stopifnot(identical(as.character(mlp_bundle$class_levels), class_levels))
  
  Xs2 <- scale(Xv, center = mlp_bundle$mean, scale = mlp_bundle$sd)
  p_mlp <- predict(mlp_bundle$model, Xs2, type = "raw")
  p_mlp <- as.matrix(p_mlp)
  colnames(p_mlp) <- class_levels
  pred_mlp <- factor(colnames(p_mlp)[max.col(p_mlp)], levels = class_levels)
}

# attach preds
dt_val[, `:=`(
  pred_rf  = pred_rf,
  pred_svm = pred_svm,
  pred_xgb = pred_xgb
)]
if (!is.null(pred_mlp)) dt_val[, pred_mlp := pred_mlp]

# ======================================================================
# 6) Metrics helpers (overall + per-class precision/recall/F1)
# ======================================================================

conf_mat <- function(y_true, y_pred, levels = class_levels) {
  table(factor(y_true, levels = levels), factor(y_pred, levels = levels))
}

metrics_from_cm <- function(cm) {
  # rows=true, cols=pred
  N <- sum(cm)
  acc <- sum(diag(cm)) / N
  
  per_class <- rbindlist(lapply(seq_len(nrow(cm)), function(i) {
    tp <- cm[i, i]
    fn <- sum(cm[i, ]) - tp
    fp <- sum(cm[, i]) - tp
    
    precision <- if ((tp + fp) == 0) NA_real_ else tp / (tp + fp)
    recall    <- if ((tp + fn) == 0) NA_real_ else tp / (tp + fn)
    f1        <- if (is.na(precision) || is.na(recall) || (precision + recall) == 0) NA_real_
    else 2 * precision * recall / (precision + recall)
    
    data.table(
      class = rownames(cm)[i],
      support = sum(cm[i, ]),
      precision = precision,
      recall = recall,
      f1 = f1
    )
  }))
  
  # balanced accuracy = mean recall across classes
  bal_acc <- mean(per_class$recall, na.rm = TRUE)
  
  w <- per_class$support / sum(per_class$support)
  w_precision <- sum(w * per_class$precision, na.rm = TRUE)
  w_recall    <- sum(w * per_class$recall, na.rm = TRUE)
  w_f1        <- sum(w * per_class$f1, na.rm = TRUE)
  
  list(
    overall = data.table(
      accuracy = acc,
      balanced_accuracy = bal_acc,
      weighted_precision = w_precision,
      weighted_recall = w_recall,
      weighted_f1 = w_f1,
      n = N
    ),
    per_class = per_class
  )
}

eval_block <- function(d, pred_col, model_name, zone_label) {
  cm <- conf_mat(d$y_class, d[[pred_col]], levels = class_levels)
  met <- metrics_from_cm(cm)
  list(
    cm = cm,
    overall = cbind(model = model_name, zone = zone_label, met$overall),
    per_class = cbind(model = model_name, zone = zone_label, met$per_class)
  )
}

# ======================================================================
# 7) Evaluate all models (ALL + by-zone)
# ======================================================================

models <- list(
  rf  = list(pred_col = "pred_rf",  name = "RF (ranger)"),
  svm = list(pred_col = "pred_svm", name = "Linear SVM (LiblineaR OvR+Platt)"),
  xgb = list(pred_col = "pred_xgb", name = "XGBoost (gbtree)")
)
if (!is.null(pred_mlp)) {
  models$mlp <- list(pred_col = "pred_mlp", name = "MLP (nnet)")
}

model_order <- sapply(models, `[[`, "name")
zones_all <- c("ALL", zones_levels)

overall_rows <- list()
percls_rows  <- list()
cm_store_all <- list()
cm_store_zone <- list()

for (m in names(models)) {
  info <- models[[m]]
  
  # ALL
  e_all <- eval_block(dt_val, info$pred_col, info$name, "ALL")
  overall_rows[[paste0(m, "_ALL")]] <- e_all$overall
  percls_rows[[paste0(m, "_ALL")]]  <- e_all$per_class
  cm_store_all[[m]] <- e_all$cm
  
  # BY ZONE
  cm_store_zone[[m]] <- list()
  for (z in levels(dt_val$zone)) {
    dz <- dt_val[zone == z]
    if (nrow(dz) == 0) next
    e_z <- eval_block(dz, info$pred_col, info$name, z)
    overall_rows[[paste0(m, "_", z)]] <- e_z$overall
    percls_rows[[paste0(m, "_", z)]]  <- e_z$per_class
    cm_store_zone[[m]][[z]] <- e_z$cm
  }
}

overall_tbl  <- rbindlist(overall_rows, fill = TRUE)
perclass_tbl <- rbindlist(percls_rows, fill = TRUE)

overall_tbl[, model := factor(model, levels = model_order)]
overall_tbl[, zone  := factor(zone,  levels = zones_all)]
perclass_tbl[, model := factor(model, levels = model_order)]
perclass_tbl[, zone  := factor(zone,  levels = zones_all)]
perclass_tbl[, class := factor(class, levels = class_levels)]

# save tables
fwrite(overall_tbl,  file.path(out_dir_eval, "internal_validation_overall.csv"))
fwrite(perclass_tbl, file.path(out_dir_eval, "internal_validation_per_class.csv"))

cat("\nWrote tables to:\n", out_dir_eval, "\n")

# ======================================================================
# 8) ggplot confusion matrices
# ======================================================================

plot_cm_gg <- function(cm, zone_label, model_label, out_png) {
  
  cm_dt <- as.data.table(as.table(cm))
  setnames(cm_dt, c("true", "pred", "n"))
  
  cm_dt[, true := factor(true, levels = class_levels)]
  cm_dt[, pred := factor(pred, levels = class_levels)]
  
  # row-normalized percentages
  cm_dt[, pct := n / sum(n), by = true]
  cm_dt[, label := sprintf("%d\n(%.1f%%)", n, 100 * pct)]
  
  p <- ggplot(cm_dt, aes(x = pred, y = true, fill = pct)) +
    geom_tile(color = "white", linewidth = 0.6) +
    geom_text(aes(label = label), size = 4.5) +
    scale_fill_gradient(
      low = "#f7f7f7",
      high = "#2166ac",
      labels = scales::percent_format(accuracy = 1)
    ) +
    labs(
      title = paste0("Confusion Matrix (Validation, ", zone_label, ") — ", model_label),
      x = "Predicted class",
      y = "Reference class",
      fill = "Row %"
    ) +
    coord_equal() +
    theme_minimal(base_size = 14) +
    theme(
      panel.grid = element_blank(),
      plot.title = element_text(face = "bold"),
      axis.title = element_text(face = "bold")
    )
  
  ggsave(out_png, p, width = fig_w, height = fig_h, dpi = dpi)
}

# ALL
for (m in names(cm_store_all)) {
  plot_cm_gg(
    cm          = cm_store_all[[m]],
    zone_label  = "ALL",
    model_label = models[[m]]$name,
    out_png     = file.path(out_dir_eval, paste0("cm_ALL_", m, ".png"))
  )
}

# BY ZONE
for (m in names(cm_store_zone)) {
  for (z in names(cm_store_zone[[m]])) {
    plot_cm_gg(
      cm          = cm_store_zone[[m]][[z]],
      zone_label  = z,
      model_label = models[[m]]$name,
      out_png     = file.path(out_dir_eval, paste0("cm_", z, "_", m, ".png"))
    )
  }
}

# ======================================================================
# 9) ggplot barplots (accuracy + balanced accuracy)
# ======================================================================

p_acc <- ggplot(overall_tbl, aes(x = model, y = accuracy, fill = zone)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.75) +
  scale_y_continuous(limits = c(0, 1), labels = percent_format(accuracy = 1)) +
  labs(title = "Validation Accuracy by Model and Zone (3-class)",
       x = NULL, y = "Accuracy", fill = "Zone") +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold"),
    axis.text.x = element_text(angle = 25, hjust = 1),
    axis.title = element_text(face = "bold")
  )
ggsave(file.path(out_dir_eval, "accuracy_by_model_and_zone.png"),
       p_acc, width = 12, height = 7, dpi = dpi)

p_bacc <- ggplot(overall_tbl, aes(x = model, y = balanced_accuracy, fill = zone)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.75) +
  scale_y_continuous(limits = c(0, 1), labels = percent_format(accuracy = 1)) +
  labs(title = "Validation Balanced Accuracy by Model and Zone (3-class)",
       x = NULL, y = "Balanced accuracy (mean recall)", fill = "Zone") +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold"),
    axis.text.x = element_text(angle = 25, hjust = 1),
    axis.title = element_text(face = "bold")
  )
ggsave(file.path(out_dir_eval, "balanced_accuracy_by_model_and_zone.png"),
       p_bacc, width = 12, height = 7, dpi = dpi)

# ======================================================================
# 10) ggplot per-class recall and precision (one plot per class)
# ======================================================================

for (cl in class_levels) {
  d <- perclass_tbl[class == cl]
  
  p_recall <- ggplot(d, aes(x = model, y = recall, fill = zone)) +
    geom_col(position = position_dodge(width = 0.8), width = 0.75) +
    scale_y_continuous(limits = c(0, 1), labels = percent_format(accuracy = 1)) +
    labs(title = paste0("Validation Recall (Producer's accuracy) — ", cl),
         x = NULL, y = "Recall", fill = "Zone") +
    theme_minimal(base_size = 14) +
    theme(
      plot.title = element_text(face = "bold"),
      axis.text.x = element_text(angle = 25, hjust = 1),
      axis.title = element_text(face = "bold")
    )
  ggsave(file.path(out_dir_eval, paste0("recall_by_model_and_zone_", cl, ".png")),
         p_recall, width = 12, height = 7, dpi = dpi)
  
  p_prec <- ggplot(d, aes(x = model, y = precision, fill = zone)) +
    geom_col(position = position_dodge(width = 0.8), width = 0.75) +
    scale_y_continuous(limits = c(0, 1), labels = percent_format(accuracy = 1)) +
    labs(title = paste0("Validation Precision (User's accuracy) — ", cl),
         x = NULL, y = "Precision", fill = "Zone") +
    theme_minimal(base_size = 14) +
    theme(
      plot.title = element_text(face = "bold"),
      axis.text.x = element_text(angle = 25, hjust = 1),
      axis.title = element_text(face = "bold")
    )
  ggsave(file.path(out_dir_eval, paste0("precision_by_model_and_zone_", cl, ".png")),
         p_prec, width = 12, height = 7, dpi = dpi)
}

cat("\nFigures written to:\n", out_dir_eval, "\n")

# ======================================================================
# 11) GT metrics table (overall + per-class recall/precision/f1)
#     (sanitize class names for wide columns)
# ======================================================================

perclass_tbl[, class_safe := gsub("[^A-Za-z0-9_]+", "_", as.character(class))]

per_wide <- dcast(
  perclass_tbl,
  model + zone ~ class_safe,
  value.var = c("recall", "precision", "f1")
)

# flatten column names: recall.stable -> recall_stable, etc.
setnames(per_wide, names(per_wide), gsub("\\.", "_", names(per_wide)))

metrics_tbl <- merge(overall_tbl, per_wide, by = c("model", "zone"), all.x = TRUE)
setorder(metrics_tbl, zone, model)

# build expected column names dynamically (robust)
cls_safe <- gsub("[^A-Za-z0-9_]+", "_", class_levels)

rec_cols <- paste0("recall_", cls_safe)
pre_cols <- paste0("precision_", cls_safe)
f1_cols  <- paste0("f1_", cls_safe)

gt_tbl <- gt(metrics_tbl) |>
  fmt_percent(
    columns = c(
      accuracy, balanced_accuracy, weighted_precision, weighted_recall, weighted_f1,
      all_of(rec_cols), all_of(pre_cols), all_of(f1_cols)
    ),
    decimals = 1
  ) |>
  cols_label(
    model = "Model",
    zone = "Zone",
    accuracy = "Overall accuracy",
    balanced_accuracy = "Balanced acc.",
    weighted_f1 = "Weighted F1",
    weighted_precision = "Weighted precision",
    weighted_recall = "Weighted recall",
    n = "N"
  ) |>
  tab_header(
    title = "Internal validation metrics by zone and algorithm (3-class)",
    subtitle = "Argmax decision; recall=Producer's acc., precision=User's acc."
  ) |>
  tab_options(table.font.size = 14)

# Add class spanners with human-readable labels
# stable
if (all(c("recall_stable", "precision_stable", "f1_stable") %in% names(metrics_tbl))) {
  gt_tbl <- gt_tbl |>
    cols_label(
      recall_stable = "Recall",
      precision_stable = "Precision",
      f1_stable = "F1"
    ) |>
    tab_spanner(label = "stable", columns = c(recall_stable, precision_stable, f1_stable))
}

# ysd1_10
if (all(c("recall_ysd1_10", "precision_ysd1_10", "f1_ysd1_10") %in% names(metrics_tbl))) {
  gt_tbl <- gt_tbl |>
    cols_label(
      recall_ysd1_10 = "Recall",
      precision_ysd1_10 = "Precision",
      f1_ysd1_10 = "F1"
    ) |>
    tab_spanner(label = "ysd1_10", columns = c(recall_ysd1_10, precision_ysd1_10, f1_ysd1_10))
}

# ysd>10 (becomes ysd_10 safe)
if (all(c("recall_ysd_10", "precision_ysd_10", "f1_ysd_10") %in% names(metrics_tbl))) {
  gt_tbl <- gt_tbl |>
    cols_label(
      recall_ysd_10 = "Recall",
      precision_ysd_10 = "Precision",
      f1_ysd_10 = "F1"
    ) |>
    tab_spanner(label = "ysd>10", columns = c(recall_ysd_10, precision_ysd_10, f1_ysd_10))
}

gtsave(gt_tbl, file.path(out_dir_eval, "internal_validation_metrics.html"))
fwrite(metrics_tbl, file.path(out_dir_eval, "internal_validation_metrics.csv"))

cat("\nWrote metrics table:\n", file.path(out_dir_eval, "internal_validation_metrics.html"), "\n")

# ======================================================================
# 12) Donut chart: validation sample distribution by zone
# ======================================================================

zone_n <- dt_val[, .N, by = zone][order(-N)]
zone_n[, pct := N / sum(N)]
zone_n[, label := sprintf("%s\n%s (%.1f%%)", zone, format(N, big.mark = ","), 100 * pct)]

p_zone_ring <- ggplot(zone_n, aes(x = 2, y = N, fill = zone)) +
  geom_col(width = 0.9, color = "white") +
  coord_polar(theta = "y") +
  xlim(0.5, 2.5) +
  geom_text(aes(label = label),
            position = position_stack(vjust = 0.5),
            size = 4) +
  labs(
    title = "Validation samples by zone",
    subtitle = sprintf("Total N = %s", format(sum(zone_n$N), big.mark = ",")),
    fill = "Zone"
  ) +
  theme_void(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5),
    legend.position = "none"
  )

ggsave(
  filename = file.path(out_dir_eval, "zone_distribution_ring.png"),
  plot = p_zone_ring,
  width = 10, height = 6, dpi = 300
)

cat("\nDONE.\n")
