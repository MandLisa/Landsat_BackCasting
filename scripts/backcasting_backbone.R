# ============================================================
# Reproducing Correia et al. (2024) pre-1985 disturbance mapping
# with annual Landsat BAPs (1985–2024)
# Author: <your name>, Date: <today>
#
# Pipeline:
# 0) Setup & I/O
# 1) Read BAPs, compute annual NBR (1985–2024)
# 2) Compute NBR-based greening mask (1985–2010) via robust OLS per pixel
#    -> classify coefficient & p to greening score [-4..4]
# 3) Build training sets using 1985–2004 disturbed points (+ undisturbed)
#    -> sample post-disturbance recovery signatures 2005–2020
# 4) Train MLPs (keras): this time not stratified by agents (but let's see...)
# 5) Inference for 1965–1984 by temporal shift (t0=1985; recovery window 1985–2000/2010)
#    -> apply greening>2 outside adjustment zones; mask pixels disturbed ≥1985
# 6) Post-processing: remove <1 ha patches; focal median (5 px)
# 7) Optional correction using reference layers (type/year overwrite rules)
#
# Key methodological choices follow Correia et al. (2024):
# - Greening scoring scheme; greening>2 threshold; 1 ha minimum; 5-px focal median
# - MLP with normalization + batchnorm + dropout; softmax + Adam; classification for age
# ============================================================

# ----------------------------
# 0) Setup & I/O
# ----------------------------
library(terra)        # raster operations
library(stars)        # optional, for large rasters
library(data.table)   # fast tables
library(dplyr)        # wrangling
library(ggplot2)      # plots (optional)
library(sf)           # vectors
library(keras)        # MLPs
library(tensorflow)
library(purrr)
library(stringr)
library(RcppRoll)

# ----------------------------
# 0) I/O and options
# ----------------------------
bap_dir  <- "/path/to/BAPs"                       # READ-ONLY (NAS)
out_dir  <- "/mnt/eo/EO4Backcasting/_output"      # WRITES only here
tmp_dir  <- "/mnt/eo/EO4Backcasting/_tmp"         # temp files (local/non-NAS)

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(tmp_dir,  recursive = TRUE, showWarnings = FALSE)

terraOptions(tempdir = tmp_dir, memfrac = 0.7)
Sys.setenv("TMPDIR" = tmp_dir, "TEMP" = tmp_dir, "TMP" = tmp_dir,
           "GDAL_CACHEMAX" = "2048")

# Safety: prevent accidental writes into bap_dir
stopifnot(!grepl(normalizePath(bap_dir, mustWork = FALSE),
                 normalizePath(out_dir, mustWork = FALSE),
                 fixed = TRUE))

wopt <- list(overwrite = TRUE, gdal = c("COMPRESS=LZW","TILED=YES","BIGTIFF=YES"))

# User-config: band mapping (adapt to your BAP)
band_nir_name  <- "NIR"     # or index, e.g. 4
band_swir_name <- "SWIR"    # or index, e.g. 7

# Required inputs
training_points_path  <- "/path/to/training_points_1985_2004.gpkg"  # sf with columns: class, year
undisturbed_mask_path <- "/path/to/undisturbed_mask_1985_2020.tif"  # 1 if NOT disturbed 1985–2020
forest_mask_path      <- "/path/to/forest_mask.tif"                 # 1 for forest

# Optional covariates (can improve age model)
slope_path <- "/path/to/slope.tif"
rad_path   <- "/path/to/annual_radiation.tif"
gdd_path   <- "/path/to/gdd_over5.tif"
npp_path   <- "/path/to/npp.tif"

# Toggle: train/estimate one global AGE model across all agents
ENABLE_AGE_MODEL <- FALSE  # set TRUE when you want the age layer as well

# Time windows
years <- 1985:2024
years_train_disturb   <- 1985:2004     # disturbance years for training
years_recovery_train  <- 2005:2020     # 16 yrs -> t0..t15 (post-2004)
years_recovery_map85  <- 1985:2000     # 16 yrs -> to match training length

# ----------------------------
# 1) Read BAPs and compute annual NBR
# ----------------------------
read_bap <- function(y) {
  f <- file.path(bap_dir, sprintf("BAP_%d.tif", y))
  if (!file.exists(f)) stop("Missing BAP file: ", f)
  rast(f)
}
compute_nbr <- function(r, nir_name, swir_name) {
  nir  <- if (is.character(nir_name)) r[[nir_name]] else r[[nir_name]]
  swir <- if (is.character(swir_name)) r[[swir_name]] else r[[swir_name]]
  (nir - swir) / (nir + swir)
}

message("1) Computing annual NBR …")
nbr_list <- vector("list", length(years))
for (i in seq_along(years)) {
  r <- read_bap(years[i])
  nbr_i <- compute_nbr(r, band_nir_name, band_swir_name)
  names(nbr_i) <- paste0("NBR_", years[i])
  nbr_list[[i]] <- nbr_i
}
nbr_stack <- rast(nbr_list)
writeRaster(nbr_stack, file.path(out_dir, "NBR_1985_2024.tif"), wopt = wopt)

# ----------------------------
# 2) Greening mask (NBR ~ year, 1985–2010) → score [-4..4]
# ----------------------------
message("2) Greening mask 1985–2010 …")
yrs_gmask <- 1985:2010
nbr_g <- nbr_stack[[paste0("NBR_", yrs_gmask)]]

ols_coef_p <- function(v) {
  yrs <- yrs_gmask
  ok  <- !is.na(v)
  if (sum(ok) < 6) return(c(NA_real_, NA_real_))
  y <- v[ok]; x <- yrs[ok]
  m <- lm(y ~ x)
  s <- summary(m)
  slope <- coef(m)[["x"]]
  pval  <- coef(s)["x","Pr(>|t|)"]
  c(slope, pval)
}
coefp <- app(nbr_g, fun = ols_coef_p)
names(coefp) <- c("slope","pval")

greening_score_fun <- function(x) {
  slope <- x[1]; p <- x[2]
  if (is.na(slope) || is.na(p)) return(NA_real_)
  if (p >= 0.05) return(0)
  if (slope < 0) {
    if (p < 0.001) return(-4)
    if (p < 0.01)  return(-3)
    if (p < 0.02)  return(-2)
    if (p < 0.05)  return(-1)
  } else {
    if (p < 0.001) return(4)
    if (p < 0.01)  return(3)
    if (p < 0.02)  return(2)
    if (p < 0.05)  return(1)
  }
  0
}
greening_score <- lapp(coefp, fun = greening_score_fun)
names(greening_score) <- "greening_score"
writeRaster(greening_score, file.path(out_dir, "greening_score_1985_2010.tif"), wopt = wopt)

# ----------------------------
# 3) Training data (binary: disturbed vs undisturbed)
#    Features: NBR t0..t15 (2005–2020); optional covariates
# ----------------------------
message("3) Preparing training data (binary) …")
pts <- st_read(training_points_path, quiet = TRUE)
if (!all(c("class","year") %in% names(pts)))
  stop("training points need columns: 'class' and 'year'")

# Project to raster CRS & thin (~750 m) to reduce spatial autocorrelation
pts <- st_transform(pts, crs(nbr_stack))
cellsize <- 750
gr <- st_make_grid(st_as_sfc(st_bbox(pts)), cellsize = cellsize)
gid <- st_intersects(pts, gr)
keep_idx <- tapply(seq_len(nrow(pts)), sapply(gid, function(i) if (length(i)) i[1] else NA), function(x) x[1])
pts_thin <- pts[na.omit(unlist(keep_idx)),]

# Define disturbed vs undisturbed label
# Disturbed = any of {fire, harvest, insect, wind, …} — adapt label set if needed
disturbed_classes <- c("fire","harvest","insect","wind","storm","other","disturbance")
pts_thin <- pts_thin %>%
  mutate(label = ifelse(class %in% disturbed_classes, "disturbed",
                        ifelse(class %in% c("undisturbed","control","no_disturbance"), "undisturbed", NA))) %>%
  filter(!is.na(label)) %>%
  # For *supervised* age model (optional), keep only 1985–2004 labelled years for disturbed
  mutate(year = ifelse(label=="disturbed", year, NA_integer_),
         year = ifelse(!is.na(year) & year %in% years_train_disturb, year, year))

# Feature extraction (NBR 2005–2020)
nbr_train <- nbr_stack[[paste0("NBR_", years_recovery_train)]]
X_nbr <- terra::extract(nbr_train, vect(pts_thin), ID = FALSE)
colnames(X_nbr) <- paste0("NBR_t", 0:(ncol(X_nbr)-1))

# Optional covariates
cov_list <- list()
if (file.exists(slope_path)) cov_list$slope <- rast(slope_path)
if (file.exists(rad_path))   cov_list$rad   <- rast(rad_path)
if (file.exists(gdd_path))   cov_list$gdd   <- rast(gdd_path)
if (file.exists(npp_path))   cov_list$npp   <- rast(npp_path)
cov_stack <- if (length(cov_list)) rast(cov_list) else NULL
X_cov <- if (!is.null(cov_stack)) terra::extract(cov_stack, vect(pts_thin), ID = FALSE) else NULL

X <- as.data.frame(cbind(X_nbr, X_cov))
df_all <- cbind(X,
                label = factor(pts_thin$label, levels = c("undisturbed","disturbed")),
                year  = pts_thin$year)

# Spatial split by coarse tiles (mitigate spatial overfit)
set.seed(42)
tile_size <- 25000
tiles <- st_make_grid(st_as_sfc(st_bbox(pts_thin)), cellsize = tile_size)
tile_id <- as.integer(st_intersects(pts_thin, tiles))
tile_ids <- unique(tile_id)
test_tiles <- sample(tile_ids, size = max(1, round(0.2*length(tile_ids))))
df_all$set <- ifelse(tile_id %in% test_tiles, "test", "train")

# Remove rows with too many NA predictors
na_ok <- rowSums(is.na(df_all[, grepl("^NBR_t", names(df_all))])) < 4
df_all <- df_all[na_ok,]

# ----------------------------
# 4) Train single binary MLP (disturbed vs undisturbed)
#    Optional: global age model across all disturbed pixels
# ----------------------------
message("4) Training binary MLP …")
to_onehot <- function(f) { to_categorical(as.integer(f) - 1L) }

X_bin <- as.matrix(df_all %>%
                     select(starts_with("NBR_t"), any_of(c("slope","rad","gdd","npp"))) %>%
                     mutate(across(everything(), ~ ifelse(is.na(.), 0, .))))
y_bin <- df_all$label
Y_bin <- to_onehot(y_bin)
idx_tr <- df_all$set=="train"; idx_te <- df_all$set=="test"

build_mlp <- function(input_dim, n_classes=2, hidden=c(256,128,64,32,16), dropout=0.3) {
  model <- keras_model_sequential() |>
    layer_input(shape = input_dim) |>
    layer_layer_normalization()
  for (h in hidden) {
    model <- model |>
      layer_dense(units = h, activation = "relu") |>
      layer_batch_normalization() |>
      layer_dropout(rate = dropout)
  }
  model |>
    layer_dense(units = n_classes, activation = "softmax") |>
    compile(optimizer = optimizer_adam(),
            loss = "categorical_crossentropy",
            metrics = "accuracy")
}
bin_model <- build_mlp(ncol(X_bin), 2)
history_bin <- bin_model %>% fit(
  x = X_bin[idx_tr,], y = Y_bin[idx_tr,],
  validation_data = list(X_bin[idx_te,], Y_bin[idx_te,]),
  epochs = 50, batch_size = 1024, verbose = 2,
  callbacks = list(callback_early_stopping(patience = 5, restore_best_weights = TRUE))
)
save_model_hdf5(bin_model, file.path(out_dir, "mlp_binary_disturbance.h5"))

# Optional: global age model across all disturbed samples (classification on YEAR 1985–2004)
if (ENABLE_AGE_MODEL) {
  message("   Training global AGE model (optional) …")
  df_age <- df_all %>% filter(label=="disturbed", !is.na(year), year %in% years_train_disturb)
  X_age  <- as.matrix(df_age %>%
                        select(starts_with("NBR_t"), any_of(c("slope","rad","gdd","npp"))) %>%
                        mutate(across(everything(), ~ ifelse(is.na(.), 0, .))))
  y_age_fac <- factor(df_age$year, levels = years_train_disturb)
  Y_age <- to_onehot(y_age_fac)
  idx_tra <- df_age$set=="train"; idx_tea <- df_age$set=="test"
  
  age_model <- build_mlp(ncol(X_age), ncol(Y_age))
  history_age <- age_model %>% fit(
    x = X_age[idx_tra,], y = Y_age[idx_tra,],
    validation_data = list(X_age[idx_tea,], Y_age[idx_tea,]),
    epochs = 60, batch_size = 1024, verbose = 2,
    callbacks = list(callback_early_stopping(patience = 6, restore_best_weights = TRUE))
  )
  save_model_hdf5(age_model, file.path(out_dir, "mlp_age_allagents.h5"))
}

# ----------------------------
# 5) Inference (1965–1984) by temporal shift
# ----------------------------
message("5) Predicting 1965–1984 disturbances …")

# Build prediction stack (1985..2000 → 16 years to match training)
nbr_map <- nbr_stack[[paste0("NBR_", years_recovery_map85)]]

# Optional covariates must match training set if used
cov_list <- list()
if (file.exists(slope_path)) cov_list$slope <- rast(slope_path)
if (file.exists(rad_path))   cov_list$rad   <- rast(rad_path)
if (file.exists(gdd_path))   cov_list$gdd   <- rast(gdd_path)
if (file.exists(npp_path))   cov_list$npp   <- rast(npp_path)
cov_stack <- if (length(cov_list)) rast(cov_list) else NULL
pred_stack <- if (!is.null(cov_stack)) c(nbr_map, cov_stack) else nbr_map

# Masks
forest_mask <- rast(forest_mask_path);       if (!compareGeom(forest_mask, nbr_stack, stopOnError=FALSE)) forest_mask <- project(forest_mask, nbr_stack, method="near")
undisturbed_mask <- rast(undisturbed_mask_path); if (!compareGeom(undisturbed_mask, nbr_stack, stopOnError=FALSE)) undisturbed_mask <- project(undisturbed_mask, nbr_stack, method="near")
mask_base <- (forest_mask == 1) & (undisturbed_mask == 1)

vals_mask <- values(mask_base)
idx_mask  <- which(vals_mask == 1)
if (length(idx_mask)==0) stop("Mask selects zero pixels.")

Xpred_all <- as.matrix(values(pred_stack))
Xsel <- Xpred_all[idx_mask, , drop = FALSE]
Xsel[is.na(Xsel)] <- 0

# Binary prediction
probs_bin <- predict(bin_model, Xsel, batch_size = 4096)
# class indices: 1=undisturbed, 2=disturbed (by our Y_bin coding)
cls_idx   <- max.col(probs_bin, ties.method = "first")
is_disturbed <- (cls_idx == 2)

# Allocate output raster (binary)
template <- mask_base
disturb_bin <- rast(template); values(disturb_bin) <- NA_integer_
v <- values(disturb_bin)
v[idx_mask] <- as.integer(is_disturbed)  # 1 disturbed, 0 undisturbed
values(disturb_bin) <- v

# Apply greening constraint: only retain pixels with greening_score > 2
gs <- greening_score
disturb_bin[gs <= 2] <- 0
disturb_bin[mask_base != 1] <- NA

# Optional: global AGE estimation across all agents
if (ENABLE_AGE_MODEL) {
  message("   Inferring age (optional) …")
  age_model <- load_model_hdf5(file.path(out_dir, "mlp_age_allagents.h5"))
  idx_dist <- which(values(disturb_bin)==1)
  year_r <- rast(template); values(year_r) <- NA_integer_
  if (length(idx_dist) > 0) {
    Xsel_dist <- Xpred_all[idx_mask[is_disturbed], , drop = FALSE]
    Xsel_dist[is.na(Xsel_dist)] <- 0
    p_age <- predict(age_model, Xsel_dist, batch_size = 4096)
    year_pred <- years_train_disturb[max.col(p_age, ties.method = "first")]
    vv <- values(year_r)
    vv[idx_mask[is_disturbed]] <- year_pred
    values(year_r) <- vv
  } else {
    year_r <- year_r * NA
  }
}

# ----------------------------
# 6) Post-processing (patch size filter; smoothing binary)
# ----------------------------
message("6) Post-processing …")

# Remove patches < 1 ha (~12 pixels @30m)
if (global(disturb_bin == 1, "sum", na.rm=TRUE)[1] > 0) {
  cc <- patches(disturb_bin == 1, directions = 8)
  cc_freq <- freq(cc)
  small_ids <- cc_freq$value[cc_freq$count < 12]
  disturb_bin[cc %in% small_ids] <- 0
}

# Optional: smooth binary map with focal majority (3x3) to reduce salt-and-pepper
w3 <- matrix(1, 3, 3)
disturb_bin_sm <- focal(disturb_bin, w = w3, fun = modal, na.policy = "omit")

# If age model enabled: smooth years with a small median filter only where disturbed==1
if (ENABLE_AGE_MODEL) {
  disturbed_mask_final <- disturb_bin_sm == 1
  w5 <- matrix(1, 5, 5)
  year_sm <- year_r
  year_sm[!disturbed_mask_final] <- NA
  year_sm <- focal(year_sm, w = w5, fun = median, na.policy="omit", na.rm=TRUE)
}

# ----------------------------
# 7) Save outputs
# ----------------------------
message("7) Writing outputs …")
writeRaster(greening_score, file.path(out_dir, "greening_score_1985_2010.tif"), wopt = wopt)
writeRaster(disturb_bin_sm, file.path(out_dir, "disturbed_1965_1984.tif"), wopt = wopt)

if (ENABLE_AGE_MODEL) {
  writeRaster(year_sm, file.path(out_dir, "disturbed_year_1965_1984.tif"), wopt = wopt)
}

message("Done. Outputs at: ", out_dir)