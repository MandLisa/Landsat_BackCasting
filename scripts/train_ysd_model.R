# ====================== PREDICTION VIA clusterR (robust with ranger) ======================

library(parallel)

# 1) Build a PSOCK cluster and ensure ranger is loaded on workers
cl <- parallel::makeCluster(TERRA_CORES_PRED, type = "PSOCK")
parallel::clusterEvalQ(cl, {
  suppressPackageStartupMessages(library(terra))
  suppressPackageStartupMessages(library(ranger))
  TRUE
})

# 2) Define a worker-safe prediction function (uses non-exported S3 via asNamespace)
pred_fun_ranger <- function(data_block, model) {
  pr <- get("predict.ranger", envir = asNamespace("ranger"))
  out <- pr(model, data = as.data.frame(data_block), num.threads = 1)
  as.numeric(out$predictions)
}

# 3) BAP prediction (6 bands)
ysd_bap_tile <- terra::clusterR(
  r_bap_masked[[c("b1","b2","b3","b4","b5","b6")]],
  fun = terra::predict,
  args = list(
    model    = model_bap,
    fun      = function(m, d, ...) pred_fun_ranger(d, m),
    filename = file.path(OUT_DIR, "ysd_1985_BAP_tile.tif"),
    overwrite= TRUE,
    wopt     = wopt_flt
  ),
  cl = cl
)

# 4) NBR prediction (single band)
ysd_nbr_tile <- terra::clusterR(
  r_nbr_masked,
  fun = terra::predict,
  args = list(
    model    = model_nbr,
    fun      = function(m, d, ...) pred_fun_ranger(d, m),
    filename = file.path(OUT_DIR, "ysd_1985_NBR_tile.tif"),
    overwrite= TRUE,
    wopt     = wopt_flt
  ),
  cl = cl
)

# 5) Close the cluster
parallel::stopCluster(cl)
# =========================================================================================
