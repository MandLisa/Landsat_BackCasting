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

# ---------- 1.5 Train RF classifier (weighted) ----------
rf_formula <- as.formula(
  paste("ysd_bin3 ~", paste(base_pred, collapse = " + "))
)

rf_model <- ranger(
  formula        = rf_formula,
  data           = train_base[, c(base_pred, "ysd_bin3"), with = FALSE],
  num.trees      = 500,
  mtry           = 3,
  importance     = "impurity",
  probability    = FALSE,
  classification = TRUE,
  class.weights  = class_weights,
  num.threads    = 20     
)


print(rf_model)

# quick sanity check
pred_test <- predict(rf_model,
                     data = test_base[, ..base_pred])$predictions
cm_test <- table(truth = test_base$ysd_bin3,
                 pred  = pred_test)
print(cm_test)
cat("Overall test accuracy:", mean(pred_test == test_base$ysd_bin3), "\n")
print("Per-class accuracy:")
print(diag(prop.table(cm_test, 1)))

# ---------- 1.6 Save model ----------
saveRDS(rf_model,
        "/mnt/eo/EO4Backcasting/_intermediates/rf_ysd_bin3_spectral_only.rds")

ysd_levels <- levels(train_base$ysd_bin3)   # c("ysd1_5","ysd6_10","ysd>10")


# ============================================================
# 2. BUILD MEDIAN BAP COMPOSITE 1985–1987 (SPECTRAL BANDS ONLY)
# ============================================================

# Paths and patterns to adapt!
bap_dir   <- "/mnt/eo/eu_mosaics/BAP_comp"  # directory with yearly BAP rasters
years_bap <- 1985:1987
bap_files <- file.path(bap_dir,
                       sprintf("BAP_%d.tif", years_bap))  # adapt to your names

if (!all(file.exists(bap_files))) {
  stop("Some BAP files for 1985–1987 are missing. Check 'bap_files'.")
}

# read all BAP rasters (assumed: same extent, resolution, and band order)
bap_stack <- rast(bap_files)   # this will have (nbands * 3) layers if multiband

# assume each BAP has the same 6 bands in order blue, green, red, nir, swir1, swir2
n_bands <- 6L
if (nlyr(bap_stack) != n_bands * length(years_bap)) {
  stop("Unexpected number of layers in BAP stack, check bands/year layout.")
}

# compute median over years per band
bap_med <- rast()
for (b in 1:n_bands) {
  idx <- seq(b, nlyr(bap_stack), by = n_bands)  # band b across all years
  band_med <- app(bap_stack[[idx]], fun = median, na.rm = TRUE)
  names(band_med) <- names(bap_stack)[b]        # keep band name from first year
  bap_med <- c(bap_med, band_med)
}

# check band names
print(bap_med)

# optional: write median composite to disk
bap_med_file <- "/mnt/eo/EO4Backcasting/_intermediates/BAP_median_1985_1987_spectral.tif"
writeRaster(bap_med, filename = bap_med_file, overwrite = TRUE)


# ============================================================
# 3. APPLY FOREST MASK (WITH 2-PIXEL BORDER REMOVED)
# ============================================================

# Forest mask: 1 = forest, NA = non-forest
forest_mask_file <- "/mnt/eo/EO4Backcasting/_intermediates/forest_mask.tif"
forest <- rast(forest_mask_file)

# remove outer 2-pixel frame to avoid edge effects
nr <- nrow(forest)
nc <- ncol(forest)

forest[1:2, ] <- NA
forest[(nr-1):nr, ] <- NA
forest[, 1:2] <- NA
forest[, (nc-1):nc] <- NA

# optionally save eroded forest mask
forest_eroded_file <- "/mnt/eo/EO4Backcasting/_intermediates/forest_mask_eroded_2px.tif"
writeRaster(forest, forest_eroded_file, overwrite = TRUE)

# align BAP composite to forest mask (if needed)
# (assumes already same grid; if not, use project/align)
bap_med <- rast(bap_med_file)
bap_med <- crop(bap_med, forest)
bap_med <- mask(bap_med, forest)

# ensure band order/names match training predictors
names(bap_med) <- base_pred
print(bap_med)


# ============================================================
# 4. PREDICT ysd_bin3 ON 1985–1987 MEDIAN BAP
# ============================================================

# load model (or reuse rf_model from above)
rf_model <- readRDS("/mnt/eo/EO4Backcasting/_intermediates/rf_ysd_bin3_spectral_only.rds")

# function for terra::predict: returns integer codes 1..3 for factor levels
rf_fun <- function(df, model) {
  df <- as.data.frame(df)
  pred <- predict(model, data = df)$predictions  # factor
  as.integer(pred)                               # 1..length(levels)
}

# predict over BAP median composite (only forest pixels after masking)
pred_ysd_file <- "/mnt/eo/EO4Backcasting/_intermediates/ysd_bin3_pred_BAP_1985_1987.tif"

pred_ysd <- predict(
  bap_med,
  rf_fun,
  model    = rf_model,
  filename = pred_ysd_file,
  overwrite = TRUE
)

# add categorical levels (1: ysd1_5, 2: ysd6_10, 3: ysd>10)
ysd_levels <- c("ysd1_5", "ysd6_10", "ysd>10")

lev_df <- data.frame(
  ID      = 1:length(ysd_levels),
  ysd_bin = ysd_levels
)
levels(pred_ysd) <- lev_df

print(pred_ysd)


# ============================================================
# 5. IDENTIFY POTENTIALLY DISTURBED PIXELS
#    (non-'ysd>10' predictions)
# ============================================================

# read predicted raster (as numeric codes)
pred_ysd <- rast(pred_ysd_file)

# index of the 'ysd>10' class in the levels
idx_late <- which(ysd_levels == "ysd>10")

# mask out 'ysd>10' (late/old forest)
pot_dist <- pred_ysd
pot_dist[pot_dist == idx_late] <- NA   # only early+intermediate remain

# optional: write mask of potentially disturbed pixels
pot_dist_file <- "/mnt/eo/EO4Backcasting/_intermediates/potential_disturbance_mask_1985_1987.tif"
writeRaster(pot_dist, pot_dist_file, overwrite = TRUE)


# ============================================================
# 6. POST-PROCESSING: NBR TREND 1985–2000 FOR POTENTIALLY
#    DISTURBED PIXELS (RECOVERY TRAJECTORY)
# ============================================================

# Paths and patterns to NBR time series
nbr_dir   <- "/mnt/eo/eu_mosaics/NBR_comp"
years_nbr <- 1985:2000
nbr_files <- file.path(nbr_dir, sprintf("NBR_%d.tif", years_nbr))

if (!all(file.exists(nbr_files))) {
  stop("Some NBR files for 1985–2000 are missing. Check 'nbr_files'.")
}

# read NBR stack
nbr_stack <- rast(nbr_files)

# optional: reclassify nodata values (e.g. -10000) to NA
# adapt if your nodata is different
nbr_stack[nbr_stack <= -9999] <- NA

# align NBR stack with BAP / forest grid
nbr_stack <- crop(nbr_stack, bap_med)
nbr_stack <- mask(nbr_stack, forest)  # ensure forest only

# mask NBR stack to potentially disturbed pixels only
# (pot_dist has non-NA where potential disturbance is predicted)
nbr_stack_dist <- mask(nbr_stack, pot_dist)  # keeps only those pixels

# function to compute linear NBR trend (slope) over time
years_vec <- years_nbr

nbr_trend_fun <- function(v) {
  # v is a numeric vector of NBR values over years 1985–2000 for one pixel
  if (all(is.na(v))) return(NA_real_)
  x <- years_vec[!is.na(v)]
  y <- v[!is.na(v)]
  if (length(y) < 2) return(NA_real_)
  coef(lm(y ~ x))[2]   # slope
}

# apply trend function per pixel
nbr_trend <- app(nbr_stack_dist, fun = nbr_trend_fun)

# optional: write NBR trend raster
nbr_trend_file <- "/mnt/eo/EO4Backcasting/_intermediates/NBR_trend_1985_2000_potential_disturbances.tif"
writeRaster(nbr_trend, nbr_trend_file, overwrite = TRUE)

print(nbr_trend)
