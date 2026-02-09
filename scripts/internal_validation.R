# ======================================================================
# 02_internal_validation_only.R  (COMPLETE SCRIPT, ggplot2, NO THRESHOLD TUNING)
#
# EO4Backcasting — internal hold-out validation (NO retraining)
# Goal:
#   - Compare algorithms (RF / SVM+Platt / XGB / optional MLP)
#   - Evaluate on VALIDATION split only (ID-based split from Script 1/2)
#   - Report metrics overall AND per-zone (Mediterranean/Temperate/Boreal)
#   - Report per-class metrics for BOTH classes (ysd1_10, ysd>10)
# Decision rule:
#   - Argmax of predicted probabilities (equivalent to p(ysd>10) >= 0.5 in 2-class)
#
# Outputs (out_dir_eval):
#   - internal_validation_overall.csv
#   - internal_validation_per_class.csv
#   - confusion matrices: cm_ALL_<model>.png and cm_<zone>_<model>.png
#   - barplots:
#       * accuracy_by_model_and_zone.png
#       * balanced_accuracy_by_model_and_zone.png
#       * recall_by_model_and_zone_<class>.png
#       * precision_by_model_and_zone_<class>.png
#
# Assumptions:
#   - x/y in training CSV are EPSG:3035 (as your tiles)
#   - zones derived from latitude with your thresholds:
#       Mediterranean: lat < 45
#       Temperate:     45 <= lat < 58
#       Boreal:        lat >= 58
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
})

# ======================================================================
# 0) USER SETTINGS
# ======================================================================

train_csv      <- "/mnt/eo/EO4Backcasting/_intermediates/training_healthy_disturbed_2711_final.csv"
out_dir_models <- "/mnt/eo/EO4Backcasting/_models_comparison"
out_dir_eval   <- file.path(out_dir_models, "_internal_validation")
dir.create(out_dir_eval, showWarnings = FALSE, recursive = TRUE)

split_file <- file.path(out_dir_models, "train_val_split_ids.csv")

base_pred  <- c("blue", "green", "red", "nir", "swir1", "swir2")
ysd_levels <- c("ysd1_10", "ysd>10")

# If you did not train an MLP, keep TRUE; it will auto-skip if file missing
run_mlp <- TRUE

# Figure export
dpi <- 250
fig_w <- 10
fig_h <- 7

# ======================================================================
# 0.1) Helper: LiblineaR margin extraction
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
  
  if (nrow(dv_mat) == n_expected) return(as.numeric(dv_mat[, 1]))
  if (ncol(dv_mat) == n_expected) return(as.numeric(t(dv_mat)[, 1]))
  
  dv_vec <- as.numeric(dv_mat)
  if (length(dv_vec) == n_expected) return(dv_vec)
  if (length(dv_vec) == 2L * n_expected) return(matrix(dv_vec, ncol = 2)[, 1])
  
  numeric(0)
}

# ======================================================================
# 1) Load split file (support both schemas; we only need ID + set)
# ======================================================================

stopifnot(file.exists(split_file))
split_dt_raw <- fread(split_file)

if (!all(c("ID", "set") %in% names(split_dt_raw))) {
  stop("Split file must contain at least {ID, set}. Found: ",
       paste(names(split_dt_raw), collapse = ", "))
}

split_dt <- split_dt_raw[, .(ID, set)]
if (split_dt[, anyDuplicated(ID)] > 0) {
  stop("Split file contains duplicate IDs. Recreate split file with Script 1/2.")
}

cat("\nSplit file loaded:\n", split_file, "\n")
cat("IDs in split file: ", nrow(split_dt), "\n")

# ======================================================================
# 2) Load data, create target + zones, attach split
# ======================================================================

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

# zones from EPSG:3035 -> lat
pts_3035 <- terra::vect(dt[, .(x, y)], geom = c("x", "y"), crs = "EPSG:3035")
pts_ll   <- terra::project(pts_3035, "EPSG:4326")
ll       <- terra::crds(pts_ll)
dt[, lat := ll[, 2]]

dt[, zone := fifelse(
  lat < 45, "mediterranean",
  fifelse(lat < 58, "temperate", "boreal")
)]
dt[, zone := factor(zone, levels = c("mediterranean", "temperate", "boreal"))]

# attach split
dt[, set := NA_character_]
dt[split_dt, on = "ID", set := i.set]
dt <- dt[!is.na(set)]
dt_val <- dt[set == "val"]

cat("\nValidation rows: ", nrow(dt_val), "\n")
cat("Validation IDs:  ", dt_val[, uniqueN(ID)], "\n")
cat("\nValidation counts by zone × class:\n")
print(dt_val[, .N, by = .(zone, ysd_bin2)][order(zone, ysd_bin2)])

# ======================================================================
# 3) Load trained models
# ======================================================================

rf_path  <- file.path(out_dir_models, "rf_ysd_bin2_prob.rds")
svm_path <- file.path(out_dir_models, "svm_linear_liblinear_platt.rds")
xgb_path <- file.path(out_dir_models, "xgb_ysd_bin2_prob.rds")
mlp_path <- file.path(out_dir_models, "mlp_nnet_ysd_bin2_prob_scaled.rds")

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
# 4) Predict on validation set (probabilities + argmax class)
# ======================================================================

Xv <- as.matrix(dt_val[, ..base_pred])

# RF
p_rf <- predict(rf_model, data = as.data.frame(Xv))$predictions
p_rf <- p_rf[, ysd_levels, drop = FALSE]
pred_rf <- factor(colnames(p_rf)[max.col(p_rf)], levels = ysd_levels)

# SVM + Platt (scaled)
Xs <- scale(Xv, center = svm_bundle$mean, scale = svm_bundle$sd)
pr_svm <- predict(svm_bundle$model, Xs, decisionValues = TRUE)
dec <- extract_margin(pr_svm, n_expected = nrow(Xv))
stopifnot(length(dec) == nrow(Xv))
p1 <- stats::predict(svm_bundle$cal, newdata = data.frame(dec = dec), type = "response")
p_svm <- cbind(`ysd1_10` = 1 - p1, `ysd>10` = p1)
pred_svm <- factor(colnames(p_svm)[max.col(p_svm)], levels = ysd_levels)

# XGB
p1_xgb <- predict(xgb_model, xgboost::xgb.DMatrix(Xv))
p_xgb <- cbind(`ysd1_10` = 1 - p1_xgb, `ysd>10` = p1_xgb)
pred_xgb <- factor(colnames(p_xgb)[max.col(p_xgb)], levels = ysd_levels)

# MLP
pred_mlp <- NULL
if (!is.null(mlp_bundle)) {
  Xs2 <- scale(Xv, center = mlp_bundle$mean, scale = mlp_bundle$sd)
  p_mlp <- predict(mlp_bundle$model, Xs2, type = "raw")
  colnames(p_mlp) <- ysd_levels
  pred_mlp <- factor(colnames(p_mlp)[max.col(p_mlp)], levels = ysd_levels)
}

# attach preds
dt_val[, `:=`(
  pred_rf  = pred_rf,
  pred_svm = pred_svm,
  pred_xgb = pred_xgb
)]
if (!is.null(pred_mlp)) dt_val[, pred_mlp := pred_mlp]

# ======================================================================
# 5) Metrics (accuracy, balanced accuracy, precision/recall per class)
# ======================================================================

conf_mat <- function(y_true, y_pred, levels = ysd_levels) {
  table(factor(y_true, levels = levels), factor(y_pred, levels = levels))
}

metrics_from_cm <- function(cm) {
  # rows=true, cols=pred
  N <- sum(cm)
  acc <- sum(diag(cm)) / N
  
  # per-class precision/recall
  per_class <- rbindlist(lapply(seq_len(nrow(cm)), function(i) {
    tp <- cm[i, i]
    fn <- sum(cm[i, ]) - tp
    fp <- sum(cm[, i]) - tp
    
    precision <- if ((tp + fp) == 0) NA_real_ else tp / (tp + fp)
    recall    <- if ((tp + fn) == 0) NA_real_ else tp / (tp + fn)
    
    data.table(
      class = rownames(cm)[i],
      support = sum(cm[i, ]),
      precision = precision,
      recall = recall
    )
  }))
  
  # balanced accuracy = mean recall across classes
  bal_acc <- mean(per_class$recall, na.rm = TRUE)
  
  # weighted precision/recall
  w <- per_class$support / sum(per_class$support)
  w_precision <- sum(w * per_class$precision, na.rm = TRUE)
  w_recall    <- sum(w * per_class$recall, na.rm = TRUE)
  
  list(
    overall = data.table(
      accuracy = acc,
      balanced_accuracy = bal_acc,
      weighted_precision = w_precision,
      weighted_recall = w_recall,
      n = N
    ),
    per_class = per_class
  )
}

eval_block <- function(d, pred_col, model_name, zone_label) {
  cm <- conf_mat(d$ysd_bin2, d[[pred_col]], levels = ysd_levels)
  met <- metrics_from_cm(cm)
  list(
    cm = cm,
    overall = cbind(model = model_name, zone = zone_label, met$overall),
    per_class = cbind(model = model_name, zone = zone_label, met$per_class)
  )
}

# ======================================================================
# 6) Evaluate all models (ALL + by-zone)
# ======================================================================

models <- list(
  rf  = list(pred_col = "pred_rf",  name = "RF (ranger)"),
  svm = list(pred_col = "pred_svm", name = "Linear SVM (LiblineaR+Platt)"),
  xgb = list(pred_col = "pred_xgb", name = "XGBoost (gbtree)")
)
if (!is.null(pred_mlp)) {
  models$mlp <- list(pred_col = "pred_mlp", name = "MLP (nnet)")
}

model_order <- sapply(models, `[[`, "name")
zones_all <- c("ALL", "mediterranean", "temperate", "boreal")

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

# enforce factor order for plots
overall_tbl[, model := factor(model, levels = model_order)]
overall_tbl[, zone  := factor(zone,  levels = zones_all)]
perclass_tbl[, model := factor(model, levels = model_order)]
perclass_tbl[, zone  := factor(zone,  levels = zones_all)]
perclass_tbl[, class := factor(class, levels = ysd_levels)]

# save tables
fwrite(overall_tbl,  file.path(out_dir_eval, "internal_validation_overall.csv"))
fwrite(perclass_tbl, file.path(out_dir_eval, "internal_validation_per_class.csv"))

cat("\nWrote tables to:\n", out_dir_eval, "\n")

# ======================================================================
# 7) ggplot confusion matrices
# ======================================================================

cm_to_dt <- function(cm, zone_label, model_label) {
  d <- as.data.table(as.table(cm))
  setnames(d, c("true", "pred", "n"))
  d[, true := factor(true, levels = ysd_levels)]
  d[, pred := factor(pred, levels = ysd_levels)]
  d[, zone := factor(zone_label, levels = zones_all)]
  d[, model := factor(model_label, levels = model_order)]
  d
}

plot_cm_gg <- function(cm, zone_label, model_label, out_png) {
  
  cm_dt <- as.data.table(as.table(cm))
  setnames(cm_dt, c("true", "pred", "n"))
  
  cm_dt[, true := factor(true, levels = ysd_levels)]
  cm_dt[, pred := factor(pred, levels = ysd_levels)]
  
  # row-normalized percentages
  cm_dt[, pct := n / sum(n), by = true]
  
  # label with count + percent
  cm_dt[, label := sprintf("%d\n(%.1f%%)", n, 100 * pct)]
  
  p <- ggplot(cm_dt, aes(x = pred, y = true, fill = pct)) +
    geom_tile(color = "white", linewidth = 0.6) +
    geom_text(aes(label = label), size = 5) +
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
  cm <- cm_store_all[[m]]
  plot_cm_gg(
    cm        = cm,
    zone_label  = "ALL",
    model_label = models[[m]]$name,
    out_png   = file.path(out_dir_eval, paste0("cm_ALL_", m, ".png"))
  )
}


# BY ZONE
for (m in names(cm_store_zone)) {
  for (z in names(cm_store_zone[[m]])) {
    cm <- cm_store_zone[[m]][[z]]
    plot_cm_gg(
      cm        = cm,
      zone_label  = z,
      model_label = models[[m]]$name,
      out_png   = file.path(out_dir_eval, paste0("cm_", z, "_", m, ".png"))
    )
  }
}


# ======================================================================
# 8) ggplot barplots (accuracy + balanced accuracy)
# ======================================================================

p_acc <- ggplot(overall_tbl, aes(x = model, y = accuracy, fill = zone)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.75) +
  scale_y_continuous(limits = c(0, 1), labels = percent_format(accuracy = 1)) +
  labs(title = "Validation Accuracy by Model and Zone", x = NULL, y = "Accuracy", fill = "Zone") +
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
  labs(title = "Validation Balanced Accuracy by Model and Zone", x = NULL, y = "Balanced accuracy", fill = "Zone") +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold"),
    axis.text.x = element_text(angle = 25, hjust = 1),
    axis.title = element_text(face = "bold")
  )
ggsave(file.path(out_dir_eval, "balanced_accuracy_by_model_and_zone.png"),
       p_bacc, width = 12, height = 7, dpi = dpi)

# ======================================================================
# 9) ggplot per-class recall and precision (one plot per class)
# ======================================================================

# Recall plots
for (cl in ysd_levels) {
  d <- perclass_tbl[class == cl]
  p <- ggplot(d, aes(x = model, y = recall, fill = zone)) +
    geom_col(position = position_dodge(width = 0.8), width = 0.75) +
    scale_y_continuous(limits = c(0, 1), labels = percent_format(accuracy = 1)) +
    labs(title = paste0("Validation Recall by Model and Zone — ", cl),
         x = NULL, y = "Recall", fill = "Zone") +
    theme_minimal(base_size = 14) +
    theme(
      plot.title = element_text(face = "bold"),
      axis.text.x = element_text(angle = 25, hjust = 1),
      axis.title = element_text(face = "bold")
    )
  ggsave(file.path(out_dir_eval, paste0("recall_by_model_and_zone_", cl, ".png")),
         p, width = 12, height = 7, dpi = dpi)
}

# Precision plots
for (cl in ysd_levels) {
  d <- perclass_tbl[class == cl]
  p <- ggplot(d, aes(x = model, y = precision, fill = zone)) +
    geom_col(position = position_dodge(width = 0.8), width = 0.75) +
    scale_y_continuous(limits = c(0, 1), labels = percent_format(accuracy = 1)) +
    labs(title = paste0("Validation Precision by Model and Zone — ", cl),
         x = NULL, y = "Precision", fill = "Zone") +
    theme_minimal(base_size = 14) +
    theme(
      plot.title = element_text(face = "bold"),
      axis.text.x = element_text(angle = 25, hjust = 1),
      axis.title = element_text(face = "bold")
    )
  ggsave(file.path(out_dir_eval, paste0("precision_by_model_and_zone_", cl, ".png")),
         p, width = 12, height = 7, dpi = dpi)
}

cat("\nFigures written to:\n", out_dir_eval, "\n")
cat("DONE.\n")





# ======================================================================
# Metrics table (NO balanced accuracy) + add per-class accuracy (Producer's accuracy)
# NOTE:
# - "Per-class accuracy" is ambiguous. In classification reporting, the most
#   standard "class accuracy" is Producer's Accuracy = Recall = TP / (TP + FN).
# - To avoid ambiguity, we report:
#     * recall (producer's accuracy) for each class
#     * precision (user's accuracy) for each class
#     * overall accuracy
# ======================================================================

library(data.table)
library(gt)

cm_metrics <- function(cm, zone, model_name) {
  
  # ensure correct ordering
  cm <- cm[ysd_levels, ysd_levels]
  
  N <- sum(cm)
  acc <- sum(diag(cm)) / N
  
  # class 1: ysd1_10
  tp1 <- cm["ysd1_10", "ysd1_10"]
  fn1 <- cm["ysd1_10", "ysd>10"]
  fp1 <- cm["ysd>10",   "ysd1_10"]
  
  recall1 <- if ((tp1 + fn1) == 0) NA_real_ else tp1 / (tp1 + fn1)       # Producer's acc.
  prec1   <- if ((tp1 + fp1) == 0) NA_real_ else tp1 / (tp1 + fp1)       # User's acc.
  
  # class 2: ysd>10
  tp2 <- cm["ysd>10", "ysd>10"]
  fn2 <- cm["ysd>10", "ysd1_10"]
  fp2 <- cm["ysd1_10","ysd>10"]
  
  recall2 <- if ((tp2 + fn2) == 0) NA_real_ else tp2 / (tp2 + fn2)       # Producer's acc.
  prec2   <- if ((tp2 + fp2) == 0) NA_real_ else tp2 / (tp2 + fp2)       # User's acc.
  
  data.table(
    zone = zone,
    model = model_name,
    overall_accuracy = acc,
    
    class_accuracy_ysd1_10 = recall1,
    precision_ysd1_10      = prec1,
    
    class_accuracy_ysd_gt10 = recall2,
    precision_ysd_gt10      = prec2,
    
    n = N
  )
}

# ---- build metrics table from stored confusion matrices ----
metrics_list <- list()

# ALL
for (m in names(cm_store_all)) {
  metrics_list[[length(metrics_list) + 1L]] <-
    cm_metrics(cm_store_all[[m]], "ALL", models[[m]]$name)
}

# ZONES
for (m in names(cm_store_zone)) {
  for (z in names(cm_store_zone[[m]])) {
    metrics_list[[length(metrics_list) + 1L]] <-
      cm_metrics(cm_store_zone[[m]][[z]], z, models[[m]]$name)
  }
}

metrics_tbl <- rbindlist(metrics_list)

# Optional: order rows nicely
zone_order <- c("ALL", "boreal", "temperate", "mediterranean")
metrics_tbl[, zone := factor(zone, levels = zone_order)]

model_order <- sapply(models, `[[`, "name")
metrics_tbl[, model := factor(model, levels = model_order)]
setorder(metrics_tbl, zone, model)

# ---- nice GT table ----
gt_tbl <- gt(metrics_tbl) |>
  cols_move(
    columns = c(class_accuracy_ysd1_10, precision_ysd1_10),
    after = overall_accuracy
  ) |>
  cols_move(
    columns = c(class_accuracy_ysd_gt10, precision_ysd_gt10),
    after = precision_ysd1_10
  ) |>
  fmt_percent(
    columns = c(
      overall_accuracy,
      class_accuracy_ysd1_10, precision_ysd1_10,
      class_accuracy_ysd_gt10, precision_ysd_gt10
    ),
    decimals = 1
  ) |>
  cols_label(
    zone = "Zone",
    model = "Model",
    overall_accuracy = "Overall accuracy",
    class_accuracy_ysd1_10 = "Accuracy (≤10y)",
    precision_ysd1_10 = "Precision (≤10y)",
    class_accuracy_ysd_gt10 = "Accuracy (>10y)",
    precision_ysd_gt10 = "Precision (>10y)",
    n = "N"
  ) |>
  tab_spanner(
    label = "≤10 years class",
    columns = c(class_accuracy_ysd1_10, precision_ysd1_10)
  ) |>
  tab_spanner(
    label = ">10 years class",
    columns = c(class_accuracy_ysd_gt10, precision_ysd_gt10)
  ) |>
  tab_header(
    title = "Internal validation metrics by zone and algorithm",
    subtitle = ""
  )


gt_tbl

# ---- save table ----
gtsave(gt_tbl, file.path(out_dir_eval, "internal_validation_metrics.html"))
fwrite(metrics_tbl, file.path(out_dir_eval, "internal_validation_metrics.csv"))


### subset
xgb_tbl <- metrics_tbl[model == "XGBoost (gbtree)"]
xgb_tbl


library(gt)

gt_xgb <- gt(xgb_tbl) |>
  fmt_percent(
    columns = c(
      overall_accuracy,
      class_accuracy_ysd1_10,
      precision_ysd1_10,
      class_accuracy_ysd_gt10,
      precision_ysd_gt10
    ),
    decimals = 1
  ) |>
  cols_label(
    zone = "Zone",
    overall_accuracy = "Overall accuracy",
    class_accuracy_ysd1_10 = "Accuracy (≤10y)",
    precision_ysd1_10 = "Precision (≤10y)",
    class_accuracy_ysd_gt10 = "Accuracy (>10y)",
    precision_ysd_gt10 = "Precision (>10y)",
    n = "N"
  ) |>
  tab_header(
    title = "Internal validation — XGBoost",
    subtitle = "Accuracy metrics by zone"
  )

gt_xgb

gtsave(gt_xgb, file.path(out_dir_eval, "xgb_internal_validation.html"))


# only one zone
# 1) subset to XGBoost
xgb_tbl <- metrics_tbl[model == "XGBoost (gbtree)"]

# 2) standardize zone labels (prevents duplicates like "ALL" vs "All")
xgb_tbl[, zone := tolower(as.character(zone))]

# 3) keep exactly one row per zone (if duplicates exist, keep the first)
xgb_tbl_unique <- xgb_tbl[!duplicated(zone)]

# optional: order zones nicely
xgb_tbl_unique[, zone := factor(zone, levels = c("all", "mediterranean", "temperate", "boreal"))]
setorder(xgb_tbl_unique, zone)

xgb_tbl_unique[, .N, by = zone]


xgb_tbl_unique


### try again
library(gt)

gt_xgb <- gt(xgb_tbl_unique) |>
  cols_move(
    columns = c(class_accuracy_ysd1_10, class_accuracy_ysd_gt10),
    after = overall_accuracy
  ) |>
  cols_move(
    columns = c(precision_ysd1_10, precision_ysd_gt10),
    after = class_accuracy_ysd_gt10
  ) |>
  fmt_percent(
    columns = c(
      overall_accuracy,
      class_accuracy_ysd1_10,
      class_accuracy_ysd_gt10,
      precision_ysd1_10,
      precision_ysd_gt10
    ),
    decimals = 1
  ) |>
  cols_label(
    zone = "Zone",
    overall_accuracy = "Overall",
    class_accuracy_ysd1_10 = "≤10y",
    class_accuracy_ysd_gt10 = ">10y",
    precision_ysd1_10 = "≤10y",
    precision_ysd_gt10 = ">10y",
    n = "N"
  ) |>
  tab_spanner(
    label = "Class accuracy (recall)",
    columns = c(class_accuracy_ysd1_10, class_accuracy_ysd_gt10)
  ) |>
  tab_spanner(
    label = "Precision",
    columns = c(precision_ysd1_10, precision_ysd_gt10)
  ) |>
  tab_header(title = "XGBoost validation metrics by zone")

gt_xgb


### donut chart
library(data.table)
library(ggplot2)
library(scales)

# --- counts by zone (validation set) ---
zone_n <- dt_val[, .N, by = zone][order(-N)]
zone_n[, pct := N / sum(N)]
zone_n[, label := sprintf("%s\n%s (%.1f%%)", zone, format(N, big.mark = ","), 100 * pct)]

# --- donut (ring) plot ---
p_zone_ring <- ggplot(zone_n, aes(x = 2, y = N, fill = zone)) +
  geom_col(width = 0.9, color = "white") +
  coord_polar(theta = "y") +
  xlim(0.5, 2.5) +  # controls hole size (ring thickness)
  geom_text(
    aes(label = label),
    position = position_stack(vjust = 0.5),
    size = 4
  ) +
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

# --- save ---
ggsave(
  filename = file.path(out_dir_eval, "zone_distribution_ring.png"),
  plot = p_zone_ring,
  width = 10, height = 6, dpi = 300
)



