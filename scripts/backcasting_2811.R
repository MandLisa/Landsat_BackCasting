# ============================================================
# PACKAGES
# ============================================================
suppressPackageStartupMessages({
  library(data.table)
  library(ranger)
})

# ============================================================
# 1. READ TRAINING DATA
# ============================================================

train_csv <- "/mnt/eo/EO4Backcasting/_intermediates/training_healthy_disturbed_2711_final.csv"
dt <- fread(train_csv)

# sanity check: required columns
req_cols <- c("ID", "ysd", "state",
              "blue", "green", "red", "nir", "swir1", "swir2", "NBR")
stopifnot(all(req_cols %in% names(dt)))

# ============================================================
# 2. DEFINE NEW 3-CLASS ysd BINS (EARLY / INTERMEDIATE / LATE)
#    early:        ysd 1–5
#    intermediate: ysd 6–10
#    late:         ysd >10
# ============================================================

# keep only disturbed pixels
dt <- dt[state == "disturbed"]

# create new bin variable
dt[, ysd_bin3 := NA_character_]

dt[ysd >=  1 & ysd <=  5, ysd_bin3 := "ysd1_5"]
dt[ysd >=  6 & ysd <= 10, ysd_bin3 := "ysd6_10"]
dt[ysd >  10,              ysd_bin3 := "ysd>10"]

# drop rows without a defined bin (ysd <= 0, missing, etc.)
dt <- dt[!is.na(ysd_bin3)]

# make it an ordered factor
dt[, ysd_bin3 := factor(ysd_bin3,
                        levels = c("ysd1_5", "ysd6_10", "ysd>10"))]

# quick check of class distribution
print(dt[, .N, by = ysd_bin3])

# ============================================================
# 3. BUILD BASELINE DATASET (NO TREND FEATURES)
# ============================================================

base_pred <- c("blue", "green", "red", "nir", "swir1", "swir2", "NBR")

# restrict to rows with complete predictors
dt_base <- dt[complete.cases(dt[, ..base_pred])]

# sort by ID for reproducibility
setorder(dt_base, ID)

# ============================================================
# 4. TRAIN/TEST SPLIT BY PIXEL ID
#    (so the same pixel never appears in both train and test)
# ============================================================

set.seed(42)

ids_all <- unique(dt_base$ID)
n_ids   <- length(ids_all)

train_ids <- sample(ids_all, size = floor(0.7 * n_ids))
test_ids  <- setdiff(ids_all, train_ids)

train_base <- dt_base[ID %in% train_ids]
test_base  <- dt_base[ID %in% test_ids]

cat("Number of IDs – train:", length(train_ids),
    " test:", length(test_ids), "\n")
cat("Number of samples – train:", nrow(train_base),
    " test:", nrow(test_base), "\n")

# ============================================================
# 5. TRAIN BASELINE RANDOM FOREST (NO TREND)
# ============================================================

rf_base_formula <- as.formula(
  paste("ysd_bin3 ~", paste(base_pred, collapse = " + "))
)

rf_base <- ranger(
  formula        = rf_base_formula,
  data           = train_base[, c(base_pred, "ysd_bin3"), with = FALSE],
  num.trees      = 500,
  mtry           = 3,         # can tune; start with something moderate
  importance     = "impurity",
  probability    = FALSE,
  classification = TRUE
)

print(rf_base)

# ============================================================
# 6. EVALUATE ON TEST SET
# ============================================================

# predictions
pred_base_test <- predict(rf_base,
                          data = test_base[, ..base_pred])$predictions

# confusion matrix
cm_base <- table(truth = test_base$ysd_bin3,
                 pred  = pred_base_test)

cat("\nConfusion matrix – baseline model (no trend):\n")
print(cm_base)

# overall accuracy
acc_base <- mean(pred_base_test == test_base$ysd_bin3)
cat("\nOverall accuracy – baseline model: ",
    round(acc_base, 3), "\n")

# per-class (row-wise) accuracy
class_acc_base <- diag(prop.table(cm_base, 1))
cat("\nPer-class accuracy – baseline model:\n")
print(round(class_acc_base, 3))

# ============================================================
# 7. OPTIONAL: VARIABLE IMPORTANCE
# ============================================================

cat("\nVariable importance – baseline model:\n")
print(sort(rf_base$variable.importance, decreasing = TRUE))
