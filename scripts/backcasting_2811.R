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
  min.node.size  = 5,
  importance     = "impurity",
  probability    = TRUE,   # <- important
  classification = TRUE,
  class.weights  = class_weights,
  num.threads    = 30
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
        "/mnt/eo/EO4Backcasting/_models/rf_112.rds")

ysd_levels <- levels(train_base$ysd_bin3)   # c("ysd1_5","ysd6_10","ysd>10")


# ============================================================
# 2. BUILD MEDIAN BAP COMPOSITE 1985–1987 (SPECTRAL BANDS ONLY)
# ============================================================
# directory with your LEVEL3 files
in_dir <- "/mnt/dss_europe/level3_interpolated/X0016_Y0020"  

# all IBAP files from 1984–1986
ibap_files <- list.files(
  in_dir,
  pattern = "^198[6-8].*IBAP\\.tif$",   # years 1984–1986, containing "IBAP.tif"
  full.names = TRUE
)

length(ibap_files)
print(ibap_files)

# ------------------------------------------------------------
# 2. Read as one multi-layer SpatRaster
#    (3 rasters × 6 bands = 18 layers)
# ------------------------------------------------------------
ibap_stack <- rast(ibap_files)

n_files <- length(ibap_files)
n_bands <- 6L
stopifnot(nlyr(ibap_stack) == n_files * n_bands)

# keep original band names from the first BAP
orig_names <- names(ibap_stack)[1:n_bands]

# ------------------------------------------------------------
# 3. Median per band across the 3 BAPs
# ------------------------------------------------------------
# index: layers 1,7,13 → band 1; 2,8,14 → band 2; ...
idx <- rep(1:n_bands, times = n_files)

ibap_med <- tapp(
  ibap_stack,
  index = idx,
  fun   = median,
  na.rm = TRUE
)

names(ibap_med) <- orig_names  # e.g. blue, green, red, nir, swir1, swir2

# ------------------------------------------------------------
# 4. Write 6-band median composite
# ------------------------------------------------------------
outdir <- "/mnt/eo/EO4Backcasting/_data"
out_file <- file.path(outdir, "IBAP_median_1986_1988.tif")
writeRaster(ibap_med, out_file, overwrite = TRUE)


# ============================================================
# 3. APPLY FOREST MASK (WITH 2-PIXEL BORDER REMOVED)
# ============================================================

forest <- rast("/mnt/eo/EFDA_v211/forest_landuse_aligned.tif")

# 2. Make it strictly binary 0/1 (no NA), so focal behaves well
#    - forest == 1 -> 1
#    - everything else -> 0
# ensure: 1 = forest, NA = non-forest
forest01 <- ifel(forest == 1, 1, NA)

# optional: write to disk so the next steps run fully on-disk
forest01_file <- "/mnt/eo/EO4Backcasting/_data/forest_mask_binary.tif"
writeRaster(forest01, forest01_file, overwrite = TRUE)
forest01 <- rast(forest01_file)

# non-forest = 1, forest = NA
nonforest <- ifel(is.na(forest01), 1, NA)

nonforest_file <- "/mnt/eo/EO4Backcasting/_data/nonforest_mask.tif"
writeRaster(nonforest, nonforest_file, overwrite = TRUE)
nonforest <- rast(nonforest_file)

# distance to nearest non-forest (i.e. distance to forest edge)
dist_file <- "/mnt/eo/EO4Backcasting/_data/dist_to_nonforest.tif"

dist_to_nf <- distance(
  nonforest,
  filename  = dist_file,
  overwrite = TRUE
)
dist_to_nf



# pixel size (assumes square pixels)
cellsize <- res(forest01)[1]          # e.g. 30
buffer_dist <- 2 * cellsize          # distance corresponding to 2 pixels

# keep only forest pixels that are >= 2 pixels away from non-forest
# step 1: build a mask of "keep" pixels
keep <- ifel(dist_to_nf >= buffer_dist, 1, NA)

# step 2: apply this mask to the original forest mask
forest_eroded <- mask(forest01, keep)

forest_eroded_file <- "/mnt/eo/EO4Backcasting/_data/forest_mask_eroded_2px.tif"
writeRaster(forest_eroded, forest_eroded_file, overwrite = TRUE)

forest_eroded

#--------------------
bap_med_file <- "/mnt/eo/EO4Backcasting/_data/IBAP_median_1986_1988.tif"

# align BAP composite to forest mask (if needed)
# (assumes already same grid; if not, use project/align)
bap_med <- rast(bap_med_file)

crs(bap_med)
crs(forest_eroded)


forest_Crop <- crop(forest_eroded, bap_med)
bap_med <- crop(bap_med, forest_Crop)
bap_med <- mask(bap_med, forest_Crop)

plot(bap_med)

# ensure band order/names match training predictors
names(bap_med) <- base_pred
print(bap_med)

outdir <- "/mnt/eo/EO4Backcasting/_data/bap_med_crop.tif"
writeRaster(bap_med, outdir, overwrite = TRUE)


# ============================================================
# 4. PREDICT ysd_bin3 ON 1985–1987 MEDIAN BAP
# ============================================================

# load model (or reuse rf_model from above)
rf_model <- readRDS("/mnt/eo/EO4Backcasting/_intermediates/rf_3011.rds")

# load cropped bap
bap_med <- rast("/mnt/eo/EO4Backcasting/_data/bap_med_crop.tif")


# wrapper for terra::predict: returns integer codes 1..3 for factor levels
rf_fun_probs <- function(model, x, ...) {
  x_df <- as.data.frame(x)
  n    <- nrow(x_df)
  if (n == 0) return(matrix(NA_real_, nrow = 0, ncol = 3))
  
  out <- matrix(NA_real_, nrow = n, ncol = 3)
  
  idx <- stats::complete.cases(x_df)
  if (any(idx)) {
    p <- predict(model, data = x_df[idx, , drop = FALSE])$predictions  # [sum(idx), 3]
    out[idx, ] <- as.matrix(p)
  }
  
  out  # terra will make 3 layers
}

prob_file <- "/mnt/eo/EO4Backcasting/_predictions/prob_classes_tile.tif"

prob_ras <- predict(
  bap_med,
  rf_model,
  rf_fun_probs,
  filename = prob_file,
  overwrite = TRUE
)

names(prob_ras) <- c("prob_ysd1_5", "prob_ysd6_10", "prob_ysd_gt10")




# predict over BAP median composite (only forest pixels after masking)
pred_ysd_file <- "/mnt/eo/EO4Backcasting/_predictions/pred_112.tif"


pred_brick <- predict(
  bap_med,    # SpatRaster tile with bands = base_pred
  rf_model,   # model (2nd argument)
  rf_fun,     # function(model, x, ...) (3rd argument)
  filename = pred_ysd_file,
  overwrite = TRUE
)

names(pred_brick) <- c("ysd_class", "p_max")


ysd_levels <- c("ysd1_5", "ysd6_10", "ysd>10")
ysd_class  <- pred_brick[[1]]
lev_df     <- data.frame(ID = 1:3, ysd_bin = ysd_levels)
levels(ysd_class) <- lev_df

writeRaster(ysd_class,
            "/mnt/eo/EO4Backcasting/_predictions/ysd_tile.tif",
            overwrite = TRUE)
writeRaster(pred_brick[[2]],
            "/mnt/eo/EO4Backcasting/_predictions/prob_tile.tif",
            overwrite = TRUE)



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
