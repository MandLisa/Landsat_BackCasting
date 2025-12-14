# ======================================================================
# Validation of RF probability predictions against disturbance reference
#
# Author: Lisa
# Project: EO4Backcasting
#
# Purpose:
#   1) Threshold the first probability band at multiple confidence levels
#      (p > 0.5, 0.6, 0.7, 0.8) → binary rasters
#   2) Clip and reclassify a disturbance-year raster to a binary reference
#      (disturbance years 1986–1988 = 1, else NA/0)
#   3) Quantify agreement, commission, and omission errors
#
# Assumptions:
#   - Probability raster is aligned to predictor grid
#   - First band corresponds to the target class of interest
#   - Disturbance raster encodes year-of-disturbance as integer values
# ======================================================================


# ======================================================================
# 0. PACKAGES
# ======================================================================
suppressPackageStartupMessages({
  library(terra)
  library(data.table)
})


# ======================================================================
# 1. LOAD INPUT DATA
# ======================================================================

# probability raster (multi-band)
prob_ras <- rast("/mnt/eo/EO4Backcasting/_predictions/ysd_probs_tile.tif")

# use first probability band only
prob1 <- prob_ras[[1]]

# disturbance raster (year of disturbance)
dist_ras <- rast("/mnt/eo/EFDA_v211/yod_aligned.tif")


# ======================================================================
# 2. CREATE BINARY PROBABILITY THRESHOLD RASTERS
# ======================================================================
thresholds <- c(0.5, 0.6, 0.7, 0.8)

prob_bin <- lapply(thresholds, function(th) {
  r <- prob1 > th
  names(r) <- paste0("prob_pgt_", th)
  r
})

names(prob_bin) <- paste0("pgt_", thresholds)


# ======================================================================
# 3. PREPARE DISTURBANCE REFERENCE RASTER
# ======================================================================

# ---- 3.1 Clip to probability raster extent ----
dist_clip <- crop(dist_ras, prob1)
dist_clip <- mask(dist_clip, prob1)

# ---- 3.2 Reclassify disturbance years (1986–1988) ----
# disturbed = 1, everything else = NA
dist_bin <- dist_clip >= 1986 & dist_clip <= 1988
names(dist_bin) <- "dist_1986_1988"


# ======================================================================
# 4. AGREEMENT / DISAGREEMENT ANALYSIS
# ======================================================================
# Definitions:
#   TP = prob == 1 & dist == 1
#   FP = prob == 1 & dist == 0
#   FN = prob == 0 & dist == 1
#   TN = prob == 0 & dist == 0

results <- rbindlist(lapply(names(prob_bin), function(nm) {
  
  p <- prob_bin[[nm]]
  
  # align grids defensively
  p <- resample(p, dist_bin, method = "near")
  
  # extract paired values
  v <- as.data.table(na.omit(cbind(
    prob = values(p),
    dist = values(dist_bin)
  )))
  
  # confusion components
  TP <- sum(v$prob == 1 & v$dist == 1)
  FP <- sum(v$prob == 1 & v$dist == 0)
  FN <- sum(v$prob == 0 & v$dist == 1)
  TN <- sum(v$prob == 0 & v$dist == 0)
  
  # metrics
  agreement <- TP / (TP + FP + FN)
  commission <- FP / (TP + FP)
  omission   <- FN / (TP + FN)
  
  data.table(
    threshold     = nm,
    TP            = TP,
    FP            = FP,
    FN            = FN,
    TN            = TN,
    agreement     = agreement,
    commission    = commission,
    omission      = omission
  )
}))


# ======================================================================
# 5. OUTPUT
# ======================================================================

print(results)

fwrite(
  results,
  "/mnt/eo/EO4Backcasting/_validation/ysd_prob_validation_summary.csv"
)


# ======================================================================
# OPTIONAL: WRITE BINARY RASTERS
# ======================================================================
for (nm in names(prob_bin)) {
  writeRaster(
    prob_bin[[nm]],
    file.path(
      "/mnt/eo/EO4Backcasting/_validation",
      paste0(nm, ".tif")
    ),
    overwrite = TRUE
  )
}

writeRaster(
  dist_bin,
  "/mnt/eo/EO4Backcasting/_validation/dist_1986_1988_binary.tif",
  overwrite = TRUE
)


# ============================================================
# ACCURACY ASSESSMENT & CONFUSION MATRICES
# ============================================================

library(terra)
library(data.table)
library(ggplot2)

thresholds <- c(0.5, 0.6, 0.7, 0.8)

acc_tbl <- rbindlist(lapply(thresholds, function(th) {
  
  # binary probability raster
  p <- prob1 > th
  p <- resample(p, dist_bin, method = "near")
  
  # extract paired values
  v <- na.omit(data.table(
    prob = values(p),
    dist = values(dist_bin)
  ))
  
  # confusion matrix elements
  TP <- sum(v$prob == 1 & v$dist == 1)
  FP <- sum(v$prob == 1 & v$dist == 0)
  FN <- sum(v$prob == 0 & v$dist == 1)
  TN <- sum(v$prob == 0 & v$dist == 0)
  
  # metrics
  OA        <- (TP + TN) / (TP + FP + FN + TN)
  precision <- TP / (TP + FP)
  recall    <- TP / (TP + FN)
  f1        <- 2 * precision * recall / (precision + recall)
  
  data.table(
    threshold  = th,
    TP = TP, FP = FP, FN = FN, TN = TN,
    overall_accuracy = OA,
    precision = precision,
    recall    = recall,
    f1_score  = f1
  )
}))

print(acc_tbl)


# ============================================================
# CONFUSION MATRIX HEATMAP (FACETTED)
# ============================================================

cm_long <- rbindlist(lapply(1:nrow(acc_tbl), function(i) {
  
  row <- acc_tbl[i]
  
  data.table(
    threshold = factor(row$threshold),
    truth = rep(c("Disturbed", "Undisturbed"), each = 2),
    prediction = rep(c("Disturbed", "Undisturbed"), times = 2),
    count = c(row$TP, row$FN, row$FP, row$TN)
  )
}))

# convert counts to percentages per threshold
cm_long_perc <- cm_long[, {
  
  total <- sum(count)
  
  .(
    prediction = prediction,
    truth      = truth,
    percent    = 100 * count / total
  )
  
}, by = threshold]


ggplot(cm_long_perc, aes(prediction, truth, fill = percent)) +
  geom_tile(color = "grey70", linewidth = 0.4) +
  geom_text(
    aes(label = sprintf("%.1f%%", percent)),
    size = 3
  ) +
  scale_fill_gradient(
    low  = "#deebf7",
    high = "#08519c",
    name = "Percent of pixels"
  ) +
  facet_wrap(~ threshold) +
  labs(
    title = "Confusion matrices (percent of evaluated pixels)",
    x = "Prediction",
    y = "Reference (disturbance 1986–1988)"
  ) +
  theme_classic() +
  theme(
    strip.background = element_rect(fill = "grey90", color = NA),
    strip.text       = element_text(face = "bold"),
    axis.text        = element_text(color = "black")
  )


### try lower threshold + MMU
# lower threshold than before
prob_thresh <- 0.5

prob_bin <- classify(
  prob1,
  rcl = matrix(
    c(-Inf, prob_thresh, NA,
      prob_thresh, Inf, 1),
    ncol = 3,
    byrow = TRUE
  )
)

names(prob_bin) <- "prob_bin"

clumps <- patches(prob_bin, directions = 4)

freq_tbl <- as.data.table(freq(clumps))
freq_tbl <- freq_tbl[, .(patch_id = value, n_pixels = count)]

# MMU in pixels
mmu_px <- 6

keep_ids <- freq_tbl[n_pixels >= mmu_px, patch_id]

prob_mmu <- clumps %in% keep_ids
names(prob_mmu) <- "prob_p04_mmu4px"


prob_mmu <- classify(
  prob_mmu,
  rcl = matrix(c(0, 0, NA,
                 1, 1, 1),
               ncol = 3, byrow = TRUE)
)


out_file <- "/mnt/eo/EO4Backcasting/_predictions/ysd_prob_p05_mmu8px.tif"

writeRaster(
  prob_mmu,
  filename  = out_file,
  overwrite = TRUE,
  datatype  = "INT1U",
  gdal      = c("COMPRESS=DEFLATE", "ZLEVEL=9")
)



