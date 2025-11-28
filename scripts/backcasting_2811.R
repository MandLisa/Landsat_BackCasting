# ============================================================
# PACKAGES
# ============================================================
suppressPackageStartupMessages({
  library(data.table)
  library(terra)
  library(ranger)
})

# ============================================================
# 1. READ TRAINING DATA AND BUILD 5-YEAR YSD CLASSES
# ============================================================

# training data with one record per pixel × year
# (columns like in your screenshot: blue, green, red, nir, swir1, swir2, NBR, ysd, state, ...)
train_csv <- "/mnt/eo/EO4Backcasting/_intermediates/training_healthy_disturbed_2711_final.csv"
dt <- fread(train_csv)

# according to your description:
# - disturbed pixels only
# - ysd_bin has levels: ysd1_5, ysd6_10, ysd11_15, ysd16_20, ysd20
bin_levels <- c("ysd1_5", "ysd6_10", "ysd11_15", "ysd16_20", "ysd20")

dt <- dt[state == "disturbed" & ysd_bin %in% bin_levels]
dt[, ysd_bin := factor(ysd_bin, levels = bin_levels)]

# ensure required columns exist
stopifnot(all(c("ID", "year", "NBR") %in% names(dt)))

# sort by ID and year
setorder(dt, ID, year)

# base spectral predictors (adapt if needed)
base_pred <- c("blue", "green", "red", "nir", "swir1", "swir2", "NBR")

# ============================================================
# 2. COMPUTE NBR_trend3 (LOCAL 3-YEAR SLOPE: year-1, year, year+1)
# ============================================================

dt[, NBR_trend3 := {
  y <- NBR
  x <- year
  n <- .N
  out <- rep(NA_real_, n)
  if (n >= 2) {
    for (i in seq_len(n)) {
      idx <- which(abs(x - x[i]) <= 1 & !is.na(y))  # window: year-1..year+1
      if (length(idx) >= 2) {
        out[i] <- coef(lm(y[idx] ~ x[idx]))[2]      # slope
      }
    }
  }
  out
}, by = ID]

# ============================================================
# 3. COMPUTE NBR_trend10 (FORWARD 10-YEAR SLOPE: year..year+9)
# ============================================================

dt[, NBR_trend10 := {
  y <- NBR
  x <- year
  n <- .N
  out <- rep(NA_real_, n)
  if (n >= 2) {
    for (i in seq_len(n)) {
      idx <- which(x >= x[i] & x <= x[i] + 9 & !is.na(y))  # window: year..year+9
      if (length(idx) >= 2) {
        out[i] <- coef(lm(y[idx] ~ x[idx]))[2]            # slope
      }
    }
  }
  out
}, by = ID]

# ============================================================
# 4. BUILD TWO MODEL DATASETS (3-year vs 10-year)
# ============================================================

pred3  <- c(base_pred, "NBR_trend3")
pred10 <- c(base_pred, "NBR_trend10")

dt_mod3  <- dt[complete.cases(dt[, ..pred3])]
dt_mod10 <- dt[complete.cases(dt[, ..pred10])]

# For a fair comparison, restrict to IDs that have *both* trends available
ids_common <- intersect(unique(dt_mod3$ID), unique(dt_mod10$ID))

dt_mod3  <- dt_mod3[ID %in% ids_common]
dt_mod10 <- dt_mod10[ID %in% ids_common]

# sanity checks
cat("Number of pixels (IDs) used in both models:", length(ids_common), "\n")
cat("Number of samples (rows) in model3 dataset:", nrow(dt_mod3), "\n")
cat("Number of samples (rows) in model10 dataset:", nrow(dt_mod10), "\n")

# ============================================================
# 5. TRAIN–TEST SPLIT BY ID (SAME SPLIT FOR BOTH MODELS)
# ============================================================

set.seed(42)

ids <- ids_common
n_ids <- length(ids)
train_ids <- sample(ids, size = floor(0.7 * n_ids))
test_ids  <- setdiff(ids, train_ids)

train3 <- dt_mod3[ID %in% train_ids]
test3  <- dt_mod3[ID %in% test_ids]

train10 <- dt_mod10[ID %in% train_ids]
test10  <- dt_mod10[ID %in% test_ids]

cat("Train IDs:", length(train_ids), " Test IDs:", length(test_ids), "\n")

# ============================================================
# 6. TRAIN RANDOM FOREST MODELS
#    Model A: base predictors + NBR_trend3
#    Model B: base predictors + NBR_trend10
# ============================================================

# --- Model with 3-year trend ---
rf3_formula <- as.formula(
  paste("ysd_bin ~", paste(pred3, collapse = " + "))
)

rf3 <- ranger(
  formula        = rf3_formula,
  data           = train3[, c(pred3, "ysd_bin"), with = FALSE],
  num.trees      = 500,
  mtry           = 3,            # tune if desired
  importance     = "impurity",
  probability    = FALSE,
  classification = TRUE
)

# --- Model with 10-year trend ---
rf10_formula <- as.formula(
  paste("ysd_bin ~", paste(pred10, collapse = " + "))
)

rf10 <- ranger(
  formula        = rf10_formula,
  data           = train10[, c(pred10, "ysd_bin"), with = FALSE],
  num.trees      = 500,
  mtry           = 3,
  importance     = "impurity",
  probability    = FALSE,
  classification = TRUE
)

# ============================================================
# 7. PREDICT ON TEST SETS AND COMPARE PERFORMANCE
# ============================================================

# --- predictions ---
pred3_test  <- predict(rf3,  data = test3[,  ..pred3])$predictions
pred10_test <- predict(rf10, data = test10[, ..pred10])$predictions

# --- confusion matrices ---
cm3  <- table(truth = test3$ysd_bin,  pred = pred3_test)
cm10 <- table(truth = test10$ysd_bin, pred = pred10_test)

cat("\nConfusion matrix – Model with 3-year trend:\n")
print(cm3)

cat("\nConfusion matrix – Model with 10-year trend:\n")
print(cm10)

# --- overall accuracy ---
acc3  <- mean(pred3_test  == test3$ysd_bin)
acc10 <- mean(pred10_test == test10$ysd_bin)

cat("\nOverall accuracy – 3-year trend model:  ", round(acc3, 3), "\n")
cat("Overall accuracy – 10-year trend model: ", round(acc10, 3), "\n")

# --- per-class accuracy (user's accuracy) ---
class_acc3  <- diag(prop.table(cm3,  1))  # row-wise proportions
class_acc10 <- diag(prop.table(cm10, 1))

cat("\nPer-class accuracy – 3-year trend model:\n")
print(round(class_acc3, 3))

cat("\nPer-class accuracy – 10-year trend model:\n")
print(round(class_acc10, 3))

# ============================================================
# 8. OPTIONAL: VARIABLE IMPORTANCE
# ============================================================

cat("\nVariable importance – 3-year trend model:\n")
print(sort(rf3$variable.importance, decreasing = TRUE))

cat("\nVariable importance – 10-year trend model:\n")
print(sort(rf10$variable.importance, decreasing = TRUE))