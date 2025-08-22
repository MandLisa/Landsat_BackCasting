# Packages
# install.packages("gdalUtilities")  # only if missing
library(gdalUtilities)
suppressPackageStartupMessages({
  have_terra <- requireNamespace("terra", quietly = TRUE)
  if (have_terra) library(terra)
})


yrs        <- 1987:2024
input_dir  <- "/mnt/dss_europe/level3_interpolated"
out_dir    <- "/mnt/dss_europe/mosaic/mosaics_BAPs_vrt"
nodata_val <- -9999          # set to 0 if your tiles use NoData=0
product    <- "IBAP"
stamp_mmdd <- "0801"

pb <- txtProgressBar(min = 0, max = length(yrs), style = 3)
res <- vector("list", length(yrs))

for (i in seq_along(yrs)) {
  y  <- yrs[i]
  t0 <- Sys.time()
  res[[i]] <- tryCatch({
    vrt <- build_ibap_vrt_year_R(
      year       = y,
      input_dir  = input_dir,
      out_dir    = out_dir,
      nodata_val = nodata_val,
      product    = product,
      stamp_mmdd = stamp_mmdd
    )
    list(year = y, ok = TRUE,  vrt = vrt,                      msg = "",
         elapsed_min = as.numeric(difftime(Sys.time(), t0, units = "mins")))
  },
  error = function(e) {
    list(year = y, ok = FALSE, vrt = NA_character_,            msg = conditionMessage(e),
         elapsed_min = as.numeric(difftime(Sys.time(), t0, units = "mins")))
  }
  )
  setTxtProgressBar(pb, i)
}
close(pb)

summary_df <- do.call(rbind, lapply(res, as.data.frame))
print(summary_df, row.names = FALSE)

# Persist a small build log next to your VRTs
logfile <- file.path(out_dir, "build_ibap_vrts_1987_2024_summary.csv")
write.csv(summary_df, logfile, row.names = FALSE)
cat("\nSummary written to:", logfile, "\n")