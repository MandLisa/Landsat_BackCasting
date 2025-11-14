################################################################################
# 0. Environment
################################################################################

library(reticulate)
use_virtualenv("r-tf-mlp", required = TRUE)
py_config()

library(data.table)
library(dplyr)
library(keras)
library(tensorflow)
tf$constant(1)

tf$constant(1)  # sanity check


# -------------------------------------------------------------------
# 1. Load model + metadata
# -------------------------------------------------------------------
model_path <- "/mnt/eo/EO4Backcasting/_models/mlp_bap_only_binary.h5"
meta_path  <- "/mnt/eo/EO4Backcasting/_models/mlp_bap_only_binary_meta.rds"

model <- load_model_hdf5(model_path)
meta  <- readRDS(meta_path)

X_mean    <- meta$mean          # named vector
X_sd      <- meta$sd            # named vector
pred_cols <- meta$pred_cols     # expected: c("b1","b2","b3","b4","b5","b6")

message("Model loaded. Predictors = ", paste(pred_cols, collapse=", "))

# -------------------------------------------------------------------
# 2. Load a BAP tile
# -------------------------------------------------------------------
bap_file <- "/mnt/dss_europe/level3_interpolated/X0016_Y0020/19840801_LEVEL3_LNDLG_IBAP.tif"
bap <- rast(bap_file)

# rename layers to match model inputs
names(bap) <- pred_cols
stopifnot(all(names(bap) == pred_cols))

# -------------------------------------------------------------------
# 3. Define terra prediction function
# -------------------------------------------------------------------
tf_fun <- function(v) {
  # v = matrix of size (n_pixels_block x 6)
  
  # 1) scale
  v_scaled <- scale(v, center = X_mean, scale = X_sd)
  
  # 2) forward pass
  prob <- model %>% predict(v_scaled, verbose = 0)
  
  # 3) ensure numeric vector returned
  as.numeric(prob)
}

# -------------------------------------------------------------------
# 4. Apply block-wise prediction with terra::app()
# -------------------------------------------------------------------
out_prob <- app(
  x       = bap,
  fun     = tf_fun,
  cores   = 4,            # adjust
  filename = "/mnt/eo/EO4Backcasting/_predictions/prob_X0016_Y0020.tif",
  overwrite = TRUE
)

# Convert probability → class
out_class <- out_prob > 0.5
out_class <- classify(out_prob, rbind(c(0,0.5,0), c(0.5,1,1)),
                      include.lowest = TRUE)

writeRaster(
  out_class,
  "/mnt/eo/EO4Backcasting/_predictions/class_X0016_Y0020.tif",
  overwrite = TRUE
)

message("\nDONE ✔\n")
message("Saved probability raster:   prob_X0016_Y0020.tif")
message("Saved binary class raster: class_X0016_Y0020.tif")