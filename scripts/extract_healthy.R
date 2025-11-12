suppressPackageStartupMessages({
  library(terra)
  library(data.table)
})

# -------- paths --------
UNDIST20_MASK_PATH <- "/mnt/eo/EFDA_v211/undisturbed_forest.tif"  # 1=undisturbed (20y), NA otherwise
BAP2006_PATH       <- "/mnt/dss_europe/mosaics_eu/mosaics_eu_baps/2006_mosaic_eu_cog.tif" # 6 bands
OUT_CSV            <- "/mnt/eo/EO4Backcasting/_intermediates/healthy.csv"
N_SAMPLES          <- 16000

# -------- load & align --------
r_mask <- rast(UNDIST20_MASK_PATH)   # should be 1/NA
r_bap  <- rast(BAP2006_PATH); stopifnot(nlyr(r_bap) == 6)
names(r_bap) <- paste0("b", 1:6)

# ensure same grid (nearest for categorical mask)
if (!same.crs(r_mask, r_bap)) {
  r_mask <- project(r_mask, r_bap, method = "near")
} else if (!all(res(r_mask) == res(r_bap)) || !ext(r_mask) == ext(r_bap)) {
  r_mask <- resample(r_mask, r_bap, method = "near")
}

# normalize mask to 1/NA if needed
r_mask01 <- classify(r_mask, rbind(c(-Inf,0.5,NA), c(0.5,Inf,1)))

# -------- sample points from mask --------
set.seed(42)
pts <- spatSample(r_mask01, size = N_SAMPLES, method = "random",
                  na.rm = TRUE, as.points = TRUE, warn = FALSE)

# -------- extract BAP(2006) at those locations --------
X <- extract(r_bap, pts, ID = FALSE)
dt <- as.data.table(X)
dt[, state := "healthy"]  # your negative class

# (optional) keep coordinates for QC
xy <- crds(pts)
dt[, `:=`(x = xy[,1], y = xy[,2])]

fwrite(dt, OUT_CSV)
message(sprintf("Wrote %d undisturbed samples to %s", nrow(dt), OUT_CSV))
