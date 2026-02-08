# ============================================================
# PACKAGES
# ============================================================
suppressPackageStartupMessages({
  library(data.table)
  library(terra)
  library(ranger)
})

# ============================================================
# 1. TRAIN RF CLASSIFIER FOR ysd_bin3 (SPECTRAL BANDS ONLY)
#    (If you already have a model, you can skip this section
#     and just load it with readRDS.)
# ============================================================

# ---------- 1.1 Read training data ----------
train_csv <- "/mnt/eo/EO4Backcasting/_intermediates/training_healthy_disturbed_2711_final.csv"
dt <- fread(train_csv)

# required columns
req_cols <- c("ID", "ysd", "state",
              "blue", "green", "red", "nir", "swir1", "swir2")
stopifnot(all(req_cols %in% names(dt)))

# keep only disturbed pixels
dt <- dt[state == "disturbed"]

# ---------- 1.2 Create 3-class ysd bins ----------
dt[, ysd_bin3 := NA_character_]
dt[ysd >=  1 & ysd <=  5, ysd_bin3 := "ysd1_5"]
dt[ysd >=  6 & ysd <= 10, ysd_bin3 := "ysd6_10"]
dt[ysd >  10,              ysd_bin3 := "ysd>10"]

# drop undefined bins
dt <- dt[!is.na(ysd_bin3)]

# ordered factor
dt[, ysd_bin3 := factor(ysd_bin3,
                        levels = c("ysd1_5", "ysd6_10", "ysd>10"))]

# predictors: ONLY spectral bands (no indices)
base_pred <- c("blue", "green", "red", "nir", "swir1", "swir2")

# restrict to rows with complete predictors
dt_base <- dt[complete.cases(dt[, ..base_pred])]

# sort by ID for reproducibility
setorder(dt_base, ID)

# ---------- 1.3 Train/test split by ID ----------
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

# ---------- 1.4 Class weights (optional but recommended) ----------
freq <- train_base[, .N, by = ysd_bin3][order(ysd_bin3)]
w_vec <- freq$N
names(w_vec) <- as.character(freq$ysd_bin3)
class_weights <- max(w_vec) / w_vec      # inverse frequency weights

print(freq)
print(class_weights)



rf_model <- ranger(
  formula        = rf_formula,
  data           = train_base[, c(base_pred, "ysd_bin3"), with = FALSE],
  num.trees      = 500,
  mtry           = 3,
  min.node.size  = 5,
  importance     = "impurity",
  probability    = TRUE,   # <-- important
  classification = TRUE,
  class.weights  = class_weights,
  num.threads    = 30
)

# optional: save
saveRDS(rf_model, "/mnt/eo/EO4Backcasting/_intermediates/rf_ysd_bin3_prob.rds")

ysd_levels <- levels(train_base$ysd_bin3)  # c("ysd1_5", "ysd6_10", "ysd>10")


rf_fun_probs <- function(model, x, ...) {
  x_df <- as.data.frame(x)
  n    <- nrow(x_df)
  
  # pre-allocate [n rows, 3 cols] with NA
  out  <- matrix(NA_real_, nrow = n, ncol = 3)
  if (n == 0) return(out)
  
  # which rows have complete predictors?
  idx <- stats::complete.cases(x_df)
  if (any(idx)) {
    p <- predict(model, data = x_df[idx, , drop = FALSE])$predictions  # matrix [sum(idx), 3]
    p <- as.matrix(p)  # ensure matrix
    out[idx, ] <- p
  }
  
  out  # terra will turn this into a 3-layer raster
}


library(terra)

# ysd levels for naming
ysd_levels <- levels(train_base$ysd_bin3)  # c("ysd1_5", "ysd6_10", "ysd>10")

prob_file <- "/mnt/eo/EO4Backcasting/_predictions/ysd_probs_tile.tif"

prob_ras <- predict(
  bap_med,        # SpatRaster tile with bands = base_pred
  rf_model,       # ranger model (2nd argument)
  rf_fun_probs,   # function(model, x, ...) (3rd argument)
  filename = prob_file,
  overwrite = TRUE
)

# name layers according to bins
names(prob_ras) <- paste0("prob_", ysd_levels)
prob_ras


# max probability
p_max <- app(prob_ras, fun = max, na.rm = TRUE)

# hard class (1,2,3) from cube
hard_class <- app(prob_ras, fun = function(v) {
  if (all(is.na(v))) return(NA_real_)
  which.max(v)
})

names(p_max)     <- "p_max"
names(hard_class) <- "ysd_class_id"
