# ============================================================
# One-value backcasting (NBR at anchor year)
#  - TRAIN: learn from anchors A ∈ {1990, 1995, 2000, 2005}
#           using single NBR@A (robustly normalized per tile & year)
#           Models:
#             (i) binary: pre-A disturbance in last 20 yrs?   (0/1)
#             (ii) 5-year bin among {A-20..A-16, A-15..A-11, A-10..A-6, A-5..A-1} (0=none)
#  - MAP:   apply at A = 1985 ⇒ outputs for 1965–1984:
#             binary (0/1) and 5-year bins (0..4)
#
# Inputs:
#   - YoD raster (single band, cell value = disturbance year; NA/0=no known event)
#   - Tile folders with annual NBR rasters: /mnt/dss_europe/level3_interpolated/X####_Y####/*_NBR.tif
#     (year is first 4 digits of filename)
# Optional but recommended gate:
#   - Greening (1985–2010) via NBR trend; threshold > 2
#
# Outputs (to /mnt/eo/EO4Backcasting/_output):
#   - models: mlp_onevalue_binary.h5, mlp_onevalue_bin5.h5
#   - per-tile rasters at A=1985:
#       <tile>_pre1985_disturbed_binary.tif
#       <tile>_pre1985_disturbance_bin5.tif
#       <tile>_greening_score_1985_2010.tif
#   - a small codebook CSV for bins
# ============================================================

suppressPackageStartupMessages({
  library(terra)
  library(sf)
  library(dplyr)
  library(stringr)
  library(keras)
})

# ----------------------------
# 0) CONFIG
# ----------------------------
# Tile root with annual NBR
level3_dir  <- "/mnt/dss_europe/level3_interpolated"

# Year-of-Disturbance raster (single band)
yod_path    <- "/path/to/panEU_yod.tif"      # <-- set

# Output & temp (non-NAS only)
out_dir <- "/mnt/eo/EO4Backcasting/_output"
tmp_dir <- "/mnt/eo/EO4Backcasting/_tmp"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(tmp_dir,  recursive = TRUE, showWarnings = FALSE)
terraOptions(tempdir = tmp_dir, memfrac = 0.7, progress = 1)
Sys.setenv("TMPDIR" = tmp_dir, "TEMP" = tmp_dir, "TMP" = tmp_dir, "GDAL_CACHEMAX" = "2048")
wopt <- list(overwrite = TRUE, gdal = c("COMPRESS=LZW","TILED=YES","BIGTIFF=YES"))

# Toggles
DO_TRAIN <- TRUE
DO_MAP   <- TRUE

# Training anchors (years at which we read one NBR and learn)
anchors_train <- c(1990L, 1995L, 2000L, 2005L)

# Mapping anchor (backcasting anchor)
anchor_map <- 1985L

# Class definition relative to an anchor A
bin_edges <- list(
  c(20,16),  # bin 1: A-20..A-16
  c(15,11),  # bin 2: A-15..A-11
  c(10,6),   # bin 3: A-10..A-6
  c(5,1)     # bin 4: A-5..A-1
)

# Greening gate
DO_GREENING   <- TRUE
years_greening <- 1985:2010
GREEN_THRESH   <- 2

# Sampling (for training)
hex_km         <- 50       # spatial balance
target_total   <- 200000   # total samples across all anchors (after balancing)
min_spacing_m  <- 750
set.seed(42)

# ------------------------------------------------------------
# 1) Helpers
# ------------------------------------------------------------
list_tiles <- function(base_dir = level3_dir) {
  d <- list.dirs(base_dir, full.names = FALSE, recursive = FALSE)
  d[grepl("^X\\d{4}_Y\\d{4}$", d)]
}

find_tile_nbr <- function(tile_id, year, base_dir = level3_dir) {
  td <- file.path(base_dir, tile_id)
  fl <- list.files(td, pattern = "_LEVEL3_LNDLG_NBR\\.tif$", full.names = TRUE)
  if (!length(fl)) return(NA_character_)
  yrs <- as.integer(substr(basename(fl), 1, 4))
  f <- fl[yrs == year]
  if (length(f) == 0) return(NA_character_) else f[1]
}

build_tile_nbr_stack <- function(tile_id, years_target, base_dir = level3_dir) {
  td <- file.path(base_dir, tile_id)
  fl <- list.files(td, pattern = "_LEVEL3_LNDLG_NBR\\.tif$", full.names = TRUE)
  if (!length(fl)) stop("No *_NBR.tif in ", td)
  yrs <- as.integer(substr(basename(fl), 1, 4))
  o <- order(yrs); fl <- fl[o]; yrs <- yrs[o]
  tmpl <- rast(fl[1])
  lst <- vector("list", length(years_target))
  for (i in seq_along(years_target)) {
    y <- years_target[i]
    f <- fl[yrs == y]
    r <- if (length(f) == 1) rast(f) else tmpl*NA
    names(r) <- paste0("NBR_", y)
    lst[[i]] <- r
  }
  rast(lst)
}

# Greening score from slope/p over 1985–2010
ols_coef_p <- function(v) {
  yrs <- years_greening
  ok  <- !is.na(v)
  if (sum(ok) < 6) return(c(NA_real_, NA_real_))
  y <- v[ok]; x <- yrs[ok]
  m <- lm(y ~ x); s <- summary(m)
  c(coef(m)[["x"]], coef(s)["x","Pr(>|t|)"])
}
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

# Class logic at anchor A from YoD
yod_to_bin <- function(yod_val, A) {
  if (is.na(yod_val) || yod_val == 0L || yod_val >= A) return(0L)
  d <- A - yod_val
  if (d >= bin_edges[[1]][1] && d <= bin_edges[[1]][2]) return(1L)
  if (d >= bin_edges[[2]][1] && d <= bin_edges[[2]][2]) return(2L)
  if (d >= bin_edges[[3]][1] && d <= bin_edges[[3]][2]) return(3L)
  if (d >= bin_edges[[4]][1] && d <= bin_edges[[4]][2]) return(4L)
  # older than A-20 → also 0 for this task
  return(0L)
}

# Coarse thinning for spacing
thin_by_grid <- function(sf_pts, spacing_m = 750) {
  if (nrow(sf_pts) <= 1) return(sf_pts)
  bb <- st_as_sfc(st_bbox(sf_pts))
  grd <- st_make_grid(bb, cellsize = spacing_m, what = "polygons")
  idx <- st_intersects(sf_pts, grd)
  keep <- tapply(seq_len(nrow(sf_pts)), sapply(idx, function(i) if (length(i)) i[1] else NA),
                 function(ix) ix[1])
  sf_pts[na.omit(unlist(keep)), ]
}

# ------------------------------------------------------------
# 2) Load static YoD and set up hex grid (for sampling in TRAIN)
# ------------------------------------------------------------
yod <- rast(yod_path)
tiles <- list_tiles(level3_dir)

# Build hex grid over yod extent (no forest mask)
hex_km_m <- hex_km * 1000
bb <- st_as_sfc(st_bbox(yod)) |> st_set_crs(crs(yod, proj=TRUE))
hex <- st_make_grid(bb, cellsize = hex_km_m, square = FALSE)
hex <- st_sf(hex_id = seq_along(hex), geometry = hex)

# ------------------------------------------------------------
# 3) TRAIN: assemble dataset with one value per anchor and fit models
# ------------------------------------------------------------
if (DO_TRAIN) {
  message("TRAIN: assembling samples and fitting models …")
  
  # Determine per-hex quota across all anchors (balanced total)
  n_hex <- nrow(hex)
  k_per_hex_total <- ceiling(target_total / n_hex)
  # Split approx half/half between class 0 vs (1..4) by sampling strategy; we’ll rebalance later
  
  # For speed: we’ll sample candidates per hex once (positions), then fetch NBR@A & labels per anchor.
  # Strategy:
  #  - sample up to k_per_hex_total points per hex, stratified by YoD ∈ [min(anchors)-20, max(anchors)-1] vs others
  #  - later, per anchor, many of these will fall into bin 0 (no prior disturbance window) vs 1..4
  #  - we rebalance per anchor before training
  
  yod_min_needed <- min(anchors_train) - 20L
  yod_max_needed <- max(anchors_train) - 1L
  
  # Masks for candidate pools
  cand_in_window <- (yod >= yod_min_needed & yod <= yod_max_needed)
  cand_else      <- is.na(yod) | (yod == 0L) | (yod < yod_min_needed) | (yod >= (max(anchors_train)+1L))
  
  sample_one_hex <- function(hx) {
    # crop YoD once
    y_h <- mask(crop(yod, vect(hx)), vect(hx))
    vy <- values(y_h)
    if (all(is.na(vy))) return(NULL)
    nc <- length(vy)
    ix_window <- which(vy >= yod_min_needed & vy <= yod_max_needed)
    ix_else   <- which(is.na(vy) | vy == 0L | vy < yod_min_needed | vy >= (max(anchors_train)+1L))
    
    quota <- k_per_hex_total
    take_w <- if (length(ix_window) > 0) sample(ix_window, min(length(ix_window), round(0.6*quota))) else integer(0)
    take_e <- if (length(ix_else)   > 0) sample(ix_else,   min(length(ix_else),   quota - length(take_w))) else integer(0)
    take <- c(take_w, take_e)
    if (!length(take)) return(NULL)
    xy <- xyFromCell(y_h, take)
    out <- st_as_sf(data.frame(x = xy[,1], y = xy[,2]), coords = c("x","y"), crs = crs(y_h, proj=TRUE))
    out$yod <- as.integer(vy[take])
    out
  }
  
  message("Sampling candidate points per hex …")
  cand_list <- vector("list", length=nrow(hex))
  for (i in seq_len(nrow(hex))) {
    if (i %% 50 == 0) message("  … hex ", i, "/", nrow(hex))
    cand_list[[i]] <- sample_one_hex(hex[i,])
  }
  pts <- do.call(rbind, cand_list)
  if (is.null(pts) || nrow(pts)==0) stop("No candidates sampled. Check YoD raster.")
  
  # Attach tile_id to each point (by intersecting with tiles’ extents)
  # We’ll build a quick tile bbox index from one NBR file per tile
  tile_bbox_sf <- function(tid) {
    f_any <- find_tile_nbr(tid, year=anchors_train[1])
    if (is.na(f_any)) {
      # fallback: try a different year that likely exists
      f_any <- find_tile_nbr(tid, year=2000L)
      if (is.na(f_any)) return(NULL)
    }
    r <- rast(f_any)
    st_as_sfc(st_bbox(r)) |> st_set_crs(crs(r, proj=TRUE)) |> st_transform(st_crs(pts)) |> st_sf(tile=tid, geometry = _)
  }
  tile_polys <- do.call(rbind, Filter(Negate(is.null), lapply(tiles, tile_bbox_sf)))
  pts <- st_transform(pts, st_crs(tile_polys))
  ov <- st_intersects(pts, tile_polys)
  pts$tile_id <- vapply(ov, function(ix) if (length(ix)) tile_polys$tile[ix[1]] else NA_character_, character(1))
  pts <- pts[!is.na(pts$tile_id), ]
  pts$pt_id <- seq_len(nrow(pts))
  
  # ---- Build training table per anchor (feature = robust z of NBR@A per tile-year) ----
  make_train_for_anchor <- function(A) {
    message("  Anchor ", A, ": extracting NBR and labels …")
    # split by tile to read a single raster per tile
    split(pts, pts$tile_id) |>
      lapply(function(g) {
        f_nbr <- find_tile_nbr(unique(g$tile_id), A)
        if (is.na(f_nbr)) return(NULL)
        r <- rast(f_nbr)
        g_r <- st_transform(g, crs(r))
        v <- terra::extract(r, vect(g_r), ID=FALSE)[,1]
        # robust tile-year normalization
        # (use all pixels of the tile-year to compute med/MAD if cheap; otherwise compute from sampled v)
        # Here we compute from the raster quickly via terra::global on a coarse aggregate for stability:
        med <- global(aggregate(r, fact=4, fun=median, na.rm=TRUE), "median", na.rm=TRUE)[1]
        mad <- global(aggregate(r, fact=4, fun=function(x) mad(x, center = median(x, na.rm=TRUE), na.rm=TRUE)),
                      "max", na.rm=TRUE)[1]
        if (is.na(mad) || mad == 0) mad <- 1e-6
        znbr <- (v - med) / mad
        # labels
        bin5 <- vapply(g$yod, yod_to_bin, integer(1), A = A)        # 0..4
        biny <- as.integer(bin5 > 0)                                 # 0/1
        data.frame(pt_id = g$pt_id, tile_id = g$tile_id, anchor = A,
                   znbr = znbr, biny = biny, bin5 = factor(bin5, levels=0:4))
      }) |>
      (\(lst) do.call(rbind, lst))()
  }
  
  train_tbl <- do.call(rbind, lapply(anchors_train, make_train_for_anchor))
  train_tbl <- train_tbl[complete.cases(train_tbl), ]
  if (nrow(train_tbl) == 0) stop("Empty training table.")
  
  # Balance per anchor & class (avoid dominance of class 0)
  balance_anchor <- function(df, max_per_class = 20000) {
    out <- lapply(split(df, df$anchor), function(d) {
      # for binary
      d_pos <- d[d$biny==1, ]
      d_neg <- d[d$biny==0, ]
      n <- min(nrow(d_pos), nrow(d_neg), max_per_class)
      d_bal <- rbind(d_pos[sample(nrow(d_pos), n), ],
                     d_neg[sample(nrow(d_neg), n), ])
      # also create a balanced multi-class (0..4)
      d_bal$bin5 <- factor(d_bal$bin5, levels=0:4)
      d_bal
    })
    do.call(rbind, out)
  }
  train_bal <- balance_anchor(train_tbl)
  
  # Prepare matrices
  X_bin  <- as.matrix(train_bal[, c("znbr","anchor")])
  y_bin  <- to_categorical(as.integer(train_bal$biny), num_classes = 2)
  
  X_bin5 <- as.matrix(train_bal[, c("znbr","anchor")])
  y_bin5 <- to_categorical(as.integer(train_bal$bin5), num_classes = 5)
  
  # Small MLPs (one hidden layer is enough with two inputs, but we keep a tiny stack)
  build_mlp <- function(input_dim, n_classes) {
    keras_model_sequential() |>
      layer_input(shape = input_dim) |>
      layer_layer_normalization() |>
      layer_dense(64, activation="relu") |>
      layer_batch_normalization() |>
      layer_dropout(0.2) |>
      layer_dense(32, activation="relu") |>
      layer_batch_normalization() |>
      layer_dropout(0.2) |>
      layer_dense(n_classes, activation="softmax") |>
      compile(optimizer = optimizer_adam(),
              loss = "categorical_crossentropy",
              metrics = "accuracy")
  }
  
  # Train/val split (by tile to reduce leakage)
  set.seed(42)
  tile_ids <- unique(train_bal$tile_id)
  val_tiles <- sample(tile_ids, max(1, round(0.2*length(tile_ids))))
  is_val <- train_bal$tile_id %in% val_tiles
  
  # Binary
  model_bin <- build_mlp(ncol(X_bin), 2)
  hist_bin <- model_bin |> fit(
    x = X_bin[!is_val,], y = y_bin[!is_val,],
    validation_data = list(X_bin[is_val,], y_bin[is_val,]),
    epochs = 30, batch_size = 4096, verbose = 2,
    callbacks = list(callback_early_stopping(patience=4, restore_best_weights=TRUE))
  )
  save_model_hdf5(model_bin, file.path(out_dir, "mlp_onevalue_binary.h5"))
  
  # 5-class
  model_bin5 <- build_mlp(ncol(X_bin5), 5)
  hist_bin5 <- model_bin5 |> fit(
    x = X_bin5[!is_val,], y = y_bin5[!is_val,],
    validation_data = list(X_bin5[is_val,], y_bin5[is_val,]),
    epochs = 35, batch_size = 4096, verbose = 2,
    callbacks = list(callback_early_stopping(patience=5, restore_best_weights=TRUE))
  )
  save_model_hdf5(model_bin5, file.path(out_dir, "mlp_onevalue_bin5.h5"))
  
  message("TRAIN done. Models saved to: ", out_dir)
}

# ------------------------------------------------------------
# 4) MAP at A = 1985 (binary + 5-year bins), no forest mask
# ------------------------------------------------------------
if (DO_MAP) {
  message("MAP: applying at anchor = ", anchor_map)
  
  # Load models (either from this run or existing)
  model_bin  <- load_model_hdf5(file.path(out_dir, "mlp_onevalue_binary.h5"))
  model_bin5 <- load_model_hdf5(file.path(out_dir, "mlp_onevalue_bin5.h5"))
  
  # Bin codebook (written once)
  codebook_csv <- file.path(out_dir, "pre1985_bin5_codebook.csv")
  if (!file.exists(codebook_csv)) {
    write.csv(data.frame(
      value = 0:4,
      label = c("none/undisturbed_or_outside","1965–1969","1970–1974","1975–1979","1980–1984")
    ), codebook_csv, row.names = FALSE)
  }
  
  # Tile loop
  tiles_to_process <- tiles
  # tiles_to_process <- tiles_to_process[1:2]  # subset for pilots
  
  for (tid in tiles_to_process) {
    message("Tile ", tid, " …")
    
    # 4.1 NBR at anchor A, plus greening (optional)
    fA <- find_tile_nbr(tid, anchor_map)
    if (is.na(fA)) { warning("No NBR file for ", anchor_map, " in tile ", tid, "; skipping."); next }
    rA <- rast(fA)
    tmpl <- rA
    
    gs <- NULL
    if (DO_GREENING) {
      nbr_g <- build_tile_nbr_stack(tid, years_target = years_greening)
      coefp <- app(nbr_g, fun = ols_coef_p); names(coefp) <- c("slope","pval")
      gs <- lapp(coefp, fun = greening_score_fun); names(gs) <- "greening_score"
    }
    
    # 4.2 Robust tile-year normalization for rA
    # Use coarse aggregate to stabilize med/MAD
    med <- global(aggregate(rA, fact=4, fun=median, na.rm=TRUE), "median", na.rm=TRUE)[1]
    mad <- global(aggregate(rA, fact=4, fun=function(x) mad(x, center=median(x, na.rm=TRUE), na.rm=TRUE)),
                  "max", na.rm=TRUE)[1]
    if (is.na(mad) || mad == 0) mad <- 1e-6
    
    # 4.3 Build features for all pixels
    vA <- values(rA)
    zn <- (vA - med) / mad
    X <- cbind(znbr = ifelse(is.na(zn), 0, zn), anchor = rep(anchor_map, length(zn)))
    
    # 4.4 Exclude pixels with YoD in [1985, 2024] (we only want pre-1985)
    y_t <- if (!compareGeom(yod, tmpl, stopOnError=FALSE)) project(yod, tmpl, method="near") else crop(yod, tmpl, snap="out")
    keep <- is.na(values(y_t)) | values(y_t) <= 1984 | values(y_t) > 2024
    keep_idx <- which(keep == 1)
    if (!length(keep_idx)) { warning("No eligible pixels in tile ", tid); next }
    
    # Prepare outputs
    r_bin  <- rast(tmpl); values(r_bin)  <- NA_integer_
    r_bin5 <- rast(tmpl); values(r_bin5) <- NA_integer_
    
    # 4.5 Predict only eligible pixels
    Xsel <- X[keep_idx, , drop = FALSE]
    p_bin <- predict(model_bin,  Xsel, batch_size = 4096)
    c_bin <- max.col(p_bin, ties.method = "first")   # 1=class0, 2=class1
    is_pre <- (c_bin == 2)
    
    # Bin5 only where binary==1
    v_bin  <- rep(NA_integer_, ncell(r_bin))
    v_bin[keep_idx] <- as.integer(is_pre)  # 0/1
    
    v_bin5 <- rep(NA_integer_, ncell(r_bin5))
    if (any(is_pre)) {
      p_bin5 <- predict(model_bin5, Xsel[is_pre, , drop=FALSE], batch_size = 4096)
      c5 <- max.col(p_bin5, ties.method = "first") - 1L  # 0..4
      # ensure only 1..4 for positives; 0 for negatives
      tmp <- integer(length(is_pre)); tmp[is_pre] <- c5
      v_bin5[keep_idx] <- tmp
    } else {
      v_bin5[keep_idx] <- 0L
    }
    
    values(r_bin)  <- v_bin
    values(r_bin5) <- v_bin5
    
    # 4.6 Apply greening gate (optional)
    if (DO_GREENING) {
      r_bin [gs <= GREEN_THRESH]  <- 0
      r_bin5[gs <= GREEN_THRESH]  <- 0
    }
    
    # 4.7 Remove tiny patches on binary (≥ ~1 ha) and smooth
    MIN_PATCH_PIXELS <- 12
    if (global(r_bin == 1, "sum", na.rm=TRUE)[1] > 0) {
      cc <- patches(r_bin == 1, directions = 8)
      fq <- freq(cc)
      small_ids <- fq$value[fq$count < MIN_PATCH_PIXELS]
      r_bin[cc %in% small_ids] <- 0
    }
    r_bin_sm <- focal(r_bin,  w = matrix(1,3,3), fun = modal, na.policy="omit")
    
    # Constrain bins by final binary and smooth bins a bit
    r_bin5[r_bin_sm != 1] <- 0
    r_bin5_sm <- focal(r_bin5, w = matrix(1,3,3), fun = modal, na.policy="omit")
    
    # 4.8 Write
    if (DO_GREENING) writeRaster(gs,        file.path(out_dir, paste0(tid, "_greening_score_1985_2010.tif")), wopt=wopt)
    writeRaster(r_bin_sm,  file.path(out_dir, paste0(tid, "_pre1985_disturbed_binary.tif")), wopt=wopt)
    writeRaster(r_bin5_sm, file.path(out_dir, paste0(tid, "_pre1985_disturbance_bin5.tif")), wopt=wopt)
    
    message("  → saved: ", tid)
  }
  
  message("MAP done.")
}
