#!/usr/bin/env R

# ===============================================================
# 0. Libraries
# ===============================================================
library(data.table)
library(ranger)
library(caret)

# ===============================================================
# 1. Load long-format BAP dataset
#    Structure expected:
#    ID, year, blue, green, red, nir, swir1, swir2, yod, state, ...
# ===============================================================

DT <- fread("/mnt/eo/EO4Backcasting/_intermediates/training_healthy_disturbed_1911_final.csv")

# Ensure ordering
setorder(DT, ID, year)

# ==============================================================
# 1. DEFINE INPUT- AND TARGET-YEAR OFFSETS
# ==============================================================

# Input-Offsets: t0, t1, t2, t3, t4, t5
input_lags <- 0:5

# Target-Offsets: t-1, t-2, t-3, t-4, t-5
target_lags <- 1:5

band_cols <- c("blue","green","red","nir","swir1","swir2")


# ==============================================================
# 2. FUNCTION TO CREATE SEQUENCES FOR ONE PIXEL
# ==============================================================

make_sequences <- function(df_pixel) {
  
  # df_pixel: nur ein Pixel, mehrere Jahre
  
  years <- df_pixel$year
  out <- list()
  idx <- 1
  
  for (t0 in years) {
    
    # Input years t0..t5
    yrs_in <- t0 + input_lags
    if (!all(yrs_in %in% years))
      next
    
    # Target years t-1..t-5
    yrs_tar <- t0 - target_lags
    if (!all(yrs_tar %in% years))
      next
    
    # Input-Matrix zusammenstellen
    input_vals <- df_pixel[year %in% yrs_in][order(year), ..band_cols]
    input_vals <- as.numeric(as.matrix(input_vals))
    
    # Targets (eine pro lag)
    targets <- sapply(target_lags, function(k) {
      df_pixel[year == (t0 - k), dist]  # oder falls du Targets ≠ dist willst, hier anpassen
    })
    
    out[[idx]] <- data.table(
      ID = df_pixel$ID[1],
      t0_year = t0,
      matrix(input_vals, nrow = 1,
             dimnames = list(NULL, paste0(rep(band_cols, each=6), "_t", input_lags))),
      matrix(targets, nrow = 1,
             dimnames = list(NULL, paste0("dist_t", target_lags)))
    )
    
    idx <- idx + 1
  }
  
  if (length(out) == 0) return(NULL)
  
  return(rbindlist(out))
}


# ==============================================================
# 3. APPLY FUNCTION TO ALL PIXELS
# ==============================================================
DT[, dist := as.integer(year == yod)]
DT[is.na(yod), dist := 0]   # undisturbed pixels


message("Building sequences for each pixel...")

SEQ <- DT[, make_sequences(.SD), by = ID]

# Entferne leere Gruppen
SEQ <- SEQ[!is.na(ID)]

message("Done. Total sequences: ", nrow(SEQ))


# ==============================================================
# 4. SAVE OUTPUT
# ==============================================================

OUT <- "/mnt/eo/EO4Backcasting/_intermediates/sequence_data_t0_t5_to_tminus.csv"
fwrite(SEQ, OUT)

message("🎉 Sequences saved to: ", OUT)


