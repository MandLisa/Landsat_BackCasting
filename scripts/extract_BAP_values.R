library(terra)
library(data.table)

# inputs
bap_dir <- "/mnt/dss_europe/mosaic/mosaics_BAPs_vrt"
pts     <- vect("/mnt/eo/EO4Backcasting/_intermediates/sample_points_core_1985_2005_yod.gpkg")
stopifnot(inherits(pts, "SpatVector"))
if (is.null(pts$ID)) pts$ID <- seq_len(nrow(pts))           # make sure you have an ID
stopifnot("yod" %in% names(pts))                             # you said YOD is present

# discover VRTs (IBAP!)
bap_files <- list.files(bap_dir,
                        pattern = "^[0-9]{8}_LEVEL3_LNDLG_IBAP\\.vrt$",
                        full.names = TRUE)
stopifnot(length(bap_files) > 0)
years <- as.integer(substr(basename(bap_files), 1, 4))
ord   <- order(years)
bap_files <- bap_files[ord]; years <- years[ord]
years <- years[years <= 2024]                                # cap if desired
bap_files <- bap_files[years <= 2024]

# ensure CRS (EPSG:3035)
tmpl <- rast(bap_files[1])
if (!same.crs(tmpl, pts)) pts <- project(pts, tmpl)

# coords once (handy for joins)
xy <- as.data.table(crds(pts, df = TRUE)); setnames(xy, c("x","y"))

# band names (assume usual order)
bn <- c("blue","green","red","nir","swir1","swir2")

# output file (append as we go)
out_csv <- "/mnt/eo/EO4Backcasting/_intermediates/ibap_samples_1985_2024.csv"
if (file.exists(out_csv)) file.remove(out_csv)

# extraction loop (per year → low memory)
for (i in seq_along(bap_files)) {
  yy <- years[i]
  r  <- rast(bap_files[i])
  NAflag(r) <- -9999                 # set to 0 if you built VRTs with NoData=0
  if (nlyr(r) != 6) stop("Unexpected number of bands in ", bap_files[i])
  names(r) <- bn
  
  # keep only points with yod ≤ yy (or all points if you prefer)
  sel <- which(!is.na(pts$yod) & pts$yod <= yy)
  if (!length(sel)) { message("Year ", yy, ": 0 points (YOD filter)"); next }
  
  # extract; chunk if you have very many points (adjust chunk size)
  chunk_size <- 150000L
  idx_seq <- split(sel, ceiling(seq_along(sel)/chunk_size))
  
  for (idx in idx_seq) {
    vals <- extract(r, pts[idx, ], ID = FALSE)
    # scale reflectance to [0,1] if your IBAP are 0..10000
    vals <- vals / 10000
    
    dt <- data.table(
      ID   = pts$ID[idx],
      x    = xy$x[idx],
      y    = xy$y[idx],
      year = yy
    )
    dt <- cbind(dt, as.data.table(vals))
    # append row block
    fwrite(dt, out_csv, append = file.exists(out_csv))
  }
  message("Year ", yy, ": wrote values for ", length(sel), " points")
}

message("Done. Output: ", out_csv)
