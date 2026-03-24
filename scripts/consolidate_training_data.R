suppressPackageStartupMessages({
  library(data.table)
  library(terra)
})

# ============================================================
# SETTINGS
# ============================================================

# Tabelle mit NUR den Trainingspunkten
# Muss mindestens enthalten: point_id, tile, x, y
train_points_file <- "/mnt/eo/EO4Backcasting/model_input/training_biomes.csv"

# Root mit Tile-Ordnern, z. B. /mnt/dss_europe/level3_interpolated
ibap_root <- "/mnt/dss_europe/level3_interpolated"

# Output-Datei
out_file <- "/mnt/eo/EO4Backcasting/model_input/training_expanded.csv"

# CRS der Punktkoordinaten
# Anpassen, falls deine Punkte nicht im selben CRS wie die Raster sind
points_crs <- "EPSG:3035"

# NoData-Werte
nodata_values <- c(-9999, -10000)

# Skalierungsfaktor der Rasterwerte
# Falls keine Skalierung: 1
scale_factor <- 1

# Nur bestimmte Jahre verwenden; NULL = alle gefundenen Jahre
years_keep <- NULL
# years_keep <- 1984:2023

# Erwartete Anzahl IBAP-Bänder
expected_nbands <- 6

# Ob nur Dateien vom 1. August berücksichtigt werden sollen
# Bei deinen Dateinamen wie 19840801_LEVEL3_LNDLG_IBAP.tif ist das sinnvoll
keep_only_aug01 <- TRUE

# ============================================================
# LOAD TRAINING POINTS
# ============================================================

pts <- fread(train_points_file)

required_cols <- c("point_id", "tile", "x", "y")
missing_cols <- setdiff(required_cols, names(pts))
if (length(missing_cols) > 0) {
  stop(sprintf(
    "Missing required columns in training table: %s",
    paste(missing_cols, collapse = ", ")
  ))
}

cat("Training points loaded:", nrow(pts), "\n")
cat("Unique training points:", uniqueN(pts$point_id), "\n")
cat("Unique tiles:", uniqueN(pts$tile), "\n")

# ============================================================
# HELPER FUNCTIONS
# ============================================================

extract_date <- function(x) {
  x <- basename(x)
  d <- substr(x, 1, 8)
  ok <- grepl("^\\d{8}$", d)
  d[!ok] <- NA_character_
  d
}

extract_year <- function(x) {
  d <- extract_date(x)
  y <- substr(d, 1, 4)
  y[is.na(d)] <- NA_character_
  as.integer(y)
}

extract_monthday <- function(x) {
  d <- extract_date(x)
  md <- substr(d, 5, 8)
  md[is.na(d)] <- NA_character_
  md
}

safe_med <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) == 0) return(NA_real_)
  median(x)
}

safe_iqr <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) == 0) return(NA_real_)
  IQR(x, na.rm = TRUE, type = 7)
}

safe_mad <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) == 0) return(NA_real_)
  mad(x, center = median(x), constant = 1, na.rm = TRUE)
}

safe_q10 <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) == 0) return(NA_real_)
  as.numeric(quantile(x, probs = 0.10, names = FALSE, type = 7, na.rm = TRUE))
}

safe_q90 <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) == 0) return(NA_real_)
  as.numeric(quantile(x, probs = 0.90, names = FALSE, type = 7, na.rm = TRUE))
}

safe_n <- function(x) {
  sum(is.finite(x))
}

# ============================================================
# PROCESS ONE TILE
# ============================================================

process_tile <- function(tile_id, pts_tile, ibap_root, years_keep = NULL) {
  
  tile_dir <- file.path(ibap_root, tile_id)
  
  if (!dir.exists(tile_dir)) {
    warning(sprintf("Tile directory not found: %s", tile_dir))
    return(NULL)
  }
  
  files <- list.files(
    tile_dir,
    pattern = "(?i)ibap.*\\.tif$",
    full.names = TRUE
  )
  
  if (length(files) == 0) {
    warning(sprintf("No IBAP tif files found for tile %s", tile_id))
    return(NULL)
  }
  
  file_dt <- data.table(
    file = files,
    date = extract_date(files),
    year = extract_year(files),
    monthday = extract_monthday(files)
  )
  
  # Nur sauber datierte Dateien behalten
  file_dt <- file_dt[!is.na(date)]
  
  # Optional: nur 1. August
  if (keep_only_aug01) {
    file_dt <- file_dt[monthday == "0801"]
  }
  
  # Optional: Jahre filtern
  if (!is.null(years_keep)) {
    file_dt <- file_dt[year %in% years_keep]
  }
  
  # Doppelte Jahre abfangen: erste Datei pro Jahr behalten
  # Falls das nie vorkommt, schadet es nicht
  setorder(file_dt, year, date, file)
  file_dt <- file_dt[!duplicated(year)]
  
  if (nrow(file_dt) == 0) {
    warning(sprintf("No usable IBAP files left for tile %s", tile_id))
    return(NULL)
  }
  
  cat("\n--------------------------------------------------\n")
  cat("Tile:", tile_id, "\n")
  cat("Points:", nrow(pts_tile), "\n")
  cat("Files used:", nrow(file_dt), "\n")
  cat("Years:", paste(range(file_dt$year, na.rm = TRUE), collapse = " - "), "\n")
  
  # Punkte als terra-Vektor
  v <- vect(
    pts_tile[, .(x, y)],
    geom = c("x", "y"),
    crs = points_crs
  )
  
  yearly_list <- vector("list", nrow(file_dt))
  
  for (i in seq_len(nrow(file_dt))) {
    f <- file_dt$file[i]
    yr <- file_dt$year[i]
    
    r <- rast(f)
    
    if (nlyr(r) < expected_nbands) {
      warning(sprintf(
        "Tile %s, file %s has only %d bands",
        tile_id, basename(f), nlyr(r)
      ))
    }
    
    vals <- terra::extract(r, v, ID = FALSE)
    vals <- as.data.table(vals)
    
    # auf erwartete Bänder begrenzen
    vals <- vals[, seq_len(min(expected_nbands, ncol(vals))), with = FALSE]
    
    # konsistente Bandnamen setzen
    setnames(vals, paste0("ibap_B", seq_len(ncol(vals))))
    
    # Falls einzelne Raster weniger Bänder haben: fehlende ergänzen
    for (b in paste0("ibap_B", 1:expected_nbands)) {
      if (!b %in% names(vals)) vals[, (b) := NA_real_]
    }
    
    # Reihenfolge sichern
    setcolorder(vals, paste0("ibap_B", 1:expected_nbands))
    
    # NoData / Skalierung
    for (cc in names(vals)) {
      vals[[cc]][vals[[cc]] %in% nodata_values] <- NA_real_
      vals[[cc]] <- as.numeric(vals[[cc]]) / scale_factor
    }
    
    vals[, year := yr]
    vals[, point_id := pts_tile$point_id]
    
    yearly_list[[i]] <- vals
  }
  
  dt_long <- rbindlist(yearly_list, use.names = TRUE, fill = TRUE)
  
  band_cols <- paste0("ibap_B", 1:expected_nbands)
  
  # Feature-Berechnung pro Punkt
  feat_dt <- dt_long[, {
    out <- list()
    
    for (b in band_cols) {
      x <- get(b)
      
      out[[paste0(b, "_med")]] <- safe_med(x)
      out[[paste0(b, "_iqr")]] <- safe_iqr(x)
      out[[paste0(b, "_mad")]] <- safe_mad(x)
      out[[paste0(b, "_p10")]] <- safe_q10(x)
      out[[paste0(b, "_p90")]] <- safe_q90(x)
      
      # optional zur Qualitätskontrolle
      out[[paste0(b, "_n")]] <- safe_n(x)
    }
    
    out
  }, by = point_id]
  
  # Meta wieder anhängen
  meta_cols <- setdiff(names(pts_tile), c("x", "y"))
  meta_dt <- unique(pts_tile[, ..meta_cols], by = "point_id")
  
  out_tile <- merge(meta_dt, feat_dt, by = "point_id", all.y = TRUE)
  
  out_tile[]
}

# ============================================================
# RUN ALL TILES
# ============================================================

tile_ids <- unique(pts$tile)
res_list <- vector("list", length(tile_ids))

for (i in seq_along(tile_ids)) {
  tile_id <- tile_ids[i]
  pts_tile <- pts[tile == tile_id]
  
  res_list[[i]] <- process_tile(
    tile_id = tile_id,
    pts_tile = pts_tile,
    ibap_root = ibap_root,
    years_keep = years_keep
  )
}

out_dt <- rbindlist(res_list, use.names = TRUE, fill = TRUE)

# ============================================================
# CLEAN / SAVE
# ============================================================

feature_cols <- grep(
  "^ibap_B[1-6]_(med|iqr|mad|p10|p90)$",
  names(out_dt),
  value = TRUE
)

# Punkte entfernen, bei denen alle Set-B-Features NA sind
if (length(feature_cols) > 0) {
  out_dt <- out_dt[
    rowSums(is.na(out_dt[, ..feature_cols])) < length(feature_cols)
  ]
}

# Optional: _n-Spalten entfernen, wenn du sie nicht behalten willst
# n_cols <- grep("_n$", names(out_dt), value = TRUE)
# out_dt[, (n_cols) := NULL]

fwrite(out_dt, out_file)

cat("\n==================================================\n")
cat("Done.\n")
cat("Rows written:", nrow(out_dt), "\n")
cat("Columns written:", ncol(out_dt), "\n")
cat("Output:", out_file, "\n")
cat("Feature columns:\n")
print(feature_cols)



# ---------------------- APPEND NNH FROM GPKG --------------------------

suppressPackageStartupMessages({
  library(sf)
})

gpkg_path  <- "/mnt/eo/EO4Backcasting/_data/biomes_join.gpkg"
gpkg_layer <- NULL   # NULL = ersten Layer nehmen

out_training_NNH <- file.path(out_dir, "training_selected_ibap_nbr_NNH.csv")
out_gpkg_NNH     <- file.path(out_dir, "training_selected_ibap_nbr_NNH.gpkg")

# Read GPKG
if (is.null(gpkg_layer)) {
  gpkg_layers <- st_layers(gpkg_path)$name
  if (length(gpkg_layers) == 0) stop("No layers found in GPKG: ", gpkg_path)
  gpkg_layer <- gpkg_layers[1]
}

gpkg <- st_read(gpkg_path, layer = gpkg_layer, quiet = TRUE)


# Checks
if (!"point_id" %in% names(gpkg)) {
  stop("Column 'point_id' not found in GPKG. A join by point_id is therefore not possible.")
}
if (!"NNH" %in% names(gpkg)) {
  stop("Column 'NNH' not found in GPKG.")
}

# Keep only what is needed
gpkg_dt <- as.data.table(st_drop_geometry(gpkg))[, .(point_id, NNH)]

# Handle duplicate point_id values if necessary
if (anyDuplicated(gpkg_dt$point_id) > 0) {
  warning("Duplicate point_id values found in GPKG. Keeping the first record per point_id.")
  gpkg_dt <- gpkg_dt[, .SD[1], by = point_id]
}

# Join NNH to merged training table
dt_all_NNH <- merge(
  dt_all,
  gpkg_dt,
  by = "point_id",
  all.x = TRUE,
  sort = FALSE
)

# Reorder columns: put NNH after label
cols_final <- c(
  "point_id", "x", "y", "tile", "label_undisturbed_20y", "NNH",
  setdiff(names(dt_all_NNH), c("point_id", "x", "y", "tile", "label_undisturbed_20y", "NNH"))
)
setcolorder(dt_all_NNH, cols_final)

# Add biome label from NHH
dt_all_NNH[, biome := fifelse(
  NNH == 1, "Boreal",
  fifelse(
    NNH == 2, "Temperate",
    fifelse(
      NNH == 3, "Mediterranean",
      NA_character_
    )
  )
)]

# Reorder columns: put NHH and biome after label
cols_final <- c(
  "point_id", "x", "y", "tile", "label_undisturbed_20y", "NNH", "biome",
  setdiff(names(dt_all_NNH), c("point_id", "x", "y", "tile", "label_undisturbed_20y", "NHH", "biome"))
)
setcolorder(dt_all_NNH, cols_final)

# Quick check
cat("\nNNH join summary:\n")
cat("Rows in merged dataset:", nrow(dt_all), "\n")
cat("Rows after NNH join:", nrow(dt_all_NNH), "\n")
cat("Rows with missing NNH:", sum(is.na(dt_all_NNH$NNH)), "\n")

# Write CSV
fwrite(dt_all_NNH, out_training_NNH)
cat("Wrote training table with NNH:", out_training_NNH, "\n")

# Optional: also write as GPKG using x/y coordinates
training_sf <- st_as_sf(dt_all_NNH, coords = c("x", "y"), crs = st_crs(gpkg), remove = FALSE)
st_write(training_sf, out_gpkg_NNH, delete_dsn = TRUE, quiet = TRUE)
cat("Wrote GPKG with NNH:", out_gpkg_NNH, "\n")


#--- visualise

suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
  library(patchwork)
  library(sf)
  library(scales)
})

# ---------------------------- SETTINGS ---------------------------------

infile  <- "/mnt/eo/EO4Backcasting/model_input/2303/training_selected_ibap_nbr_NNH.csv"
out_dir <- "/mnt/eo/EO4Backcasting/_figs"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# ---------------------------- READ DATA --------------------------------

dt <- fread(infile)

# Labels schön benennen
dt[, class := fifelse(label_undisturbed_20y == 1, "Undisturbed", "Disturbed")]

# Reihenfolge für Plots
dt[, class := factor(class, levels = c("Disturbed", "Undisturbed"))]

if ("biome" %in% names(dt)) {
  dt[, biome := factor(biome, levels = c("Boreal", "Temperate", "Mediterranean"))]
}

# Modellvariablen
ibap_cols <- paste0("ibap_B", 1:6, "_med")
nbr_cols  <- c("nbr_t0", "nbr_mean", "nbr_min", "nbr_max", "nbr_sd", "nbr_mad", "nbr_range", "nbr_sen_slope")
model_cols <- c(ibap_cols, nbr_cols)

# Nur vorhandene Spalten verwenden
ibap_cols  <- ibap_cols[ibap_cols %in% names(dt)]
nbr_cols   <- nbr_cols[nbr_cols %in% names(dt)]
model_cols <- model_cols[model_cols %in% names(dt)]

# ---------------------- 1) SPATIAL DISTRIBUTION ------------------------

p_map_class <- ggplot(dt, aes(x = x, y = y, color = class)) +
  geom_point(alpha = 0.45, size = 0.05) +
  coord_equal() +
  theme_minimal(base_size = 12) +
  labs(
    title = "",
    x = "",
    y = "",
    color = "Class"
  )

print(p_map_class)

ggsave(
  filename = file.path(out_dir, "training_map_class.png"),
  plot = p_map_class,
  width = 8.5, height = 6, dpi = 300
)

if ("biome" %in% names(dt)) {
  p_map_biome <- ggplot(dt, aes(x = x, y = y, color = biome)) +
    geom_point(alpha = 0.45, size = 0.01) +
    coord_equal() +
    theme_minimal(base_size = 12) +
    labs(
      title = "",
      x = "",
      y = "",
      color = "Biome"
    )
  
  ggsave(
    filename = file.path(out_dir, "training_map_biome.png"),
    plot = p_map_biome,
    width = 8.5, height = 6, dpi = 300
  )
}

print(p_map_biome)
# ---------------------- 2) CLASS BALANCE -------------------------------

p_class <- ggplot(dt[, .N, by = class], aes(x = class, y = N, fill = class)) +
  geom_col(width = 0.7) +
  theme_minimal(base_size = 12) +
  scale_y_continuous(labels = comma) +
  labs(
    title = "",
    x = NULL,
    y = "Number of points"
  ) +
  guides(fill = "none")

print(p_class)

ggsave(
  filename = file.path(out_dir, "class_balance_overall.png"),
  plot = p_class,
  width = 5.5, height = 4.5, dpi = 300
)

if ("biome" %in% names(dt)) {
  p_class_biome <- ggplot(dt[, .N, by = .(biome, class)], aes(x = biome, y = N, fill = class)) +
    geom_col(position = "dodge") +
    theme_minimal(base_size = 12) +
    scale_y_continuous(labels = comma) +
    labs(
      title = "",
      x = NULL,
      y = "Number of points",
      fill = "Class"
    )
  
  ggsave(
    filename = file.path(out_dir, "class_balance_by_biome.png"),
    plot = p_class_biome,
    width = 7, height = 4.8, dpi = 300
  )
}

print(p_class_biome)
# ---------------------- 3) TILE SUMMARY -------------------------------

tile_sum <- dt[, .(
  n = .N,
  n_disturbed = sum(class == "Disturbed"),
  n_undisturbed = sum(class == "Undisturbed")
), by = tile][order(-n)]

tile_sum_long <- melt(
  tile_sum,
  id.vars = "tile",
  measure.vars = c("n_disturbed", "n_undisturbed"),
  variable.name = "class",
  value.name = "n"
)

tile_sum_long[, class := fifelse(class == "n_disturbed", "Disturbed", "Undisturbed")]
tile_sum_long[, tile := factor(tile, levels = tile_sum[order(n)]$tile)]

p_tile <- ggplot(tile_sum_long, aes(x = tile, y = n, fill = class)) +
  geom_col() +
  coord_flip() +
  theme_minimal(base_size = 11) +
  scale_y_continuous(labels = comma) +
  labs(
    title = "Training points per tile",
    x = "Tile",
    y = "Number of points",
    fill = "Class"
  )

print(p_tile)


ggsave(
  filename = file.path(out_dir, "tile_summary.png"),
  plot = p_tile,
  width = 8.5, height = 10, dpi = 300
)

# ---------------------- 4) IBAP DISTRIBUTIONS --------------------------

if (length(ibap_cols) > 0) {
  dt_ibap <- melt(
    dt,
    id.vars = c("point_id", "class", intersect("biome", names(dt))),
    measure.vars = ibap_cols,
    variable.name = "feature",
    value.name = "value"
  )
  
  p_ibap_density <- ggplot(dt_ibap, aes(x = value, color = class, fill = class)) +
    geom_density(alpha = 0.15) +
    facet_wrap(~ feature, scales = "free", ncol = 2) +
    theme_minimal(base_size = 12) +
    labs(
      title = "Distribution of BAP values",
      x = "Wavelength",
      y = "Density",
      color = "Class",
      fill = "Class"
    )
  
  ggsave(
    filename = file.path(out_dir, "ibap_density_by_class.png"),
    plot = p_ibap_density,
    width = 10, height = 9, dpi = 300
  )
}

print(p_ibap_density)

# ---------------------- 5) NBR DISTRIBUTIONS ---------------------------

if (length(nbr_cols) > 0) {
  dt_nbr <- melt(
    dt,
    id.vars = c("point_id", "class", intersect("biome", names(dt))),
    measure.vars = nbr_cols,
    variable.name = "feature",
    value.name = "value"
  )
  
  p_nbr_box <- ggplot(dt_nbr, aes(x = class, y = value, fill = class)) +
    geom_boxplot(outlier.size = 0.2, width = 0.7) +
    facet_wrap(~ feature, scales = "free", ncol = 2) +
    theme_minimal(base_size = 12) +
    labs(
      title = "Distribution of NBR features by class",
      x = NULL,
      y = "Feature value",
      fill = "Class"
    ) +
    guides(fill = "none")
  
  ggsave(
    filename = file.path(out_dir, "nbr_boxplots_by_class.png"),
    plot = p_nbr_box,
    width = 10, height = 11, dpi = 300
  )
}

print(p_nbr_box)
# ---------------------- 6) BIOME-SPECIFIC NBR SLOPE --------------------

if (all(c("biome", "nbr_sen_slope") %in% names(dt))) {
  p_slope_biome <- ggplot(dt, aes(x = biome, y = nbr_sen_slope, fill = class)) +
    geom_violin(trim = TRUE, alpha = 0.7) +
    geom_boxplot(width = 0.15, outlier.size = 0.2, alpha = 0.8) +
    theme_minimal(base_size = 12) +
    labs(
      title = "NBR Theil–Sen slope by biome and class",
      x = NULL,
      y = "NBR Theil–Sen slope",
      fill = "Class"
    )
  
  ggsave(
    filename = file.path(out_dir, "nbr_sen_slope_biome_class.png"),
    plot = p_slope_biome,
    width = 8, height = 5.5, dpi = 300
  )
}

print(p_slope_biome)
# ---------------------- 7) CORRELATION HEATMAP -------------------------

if (length(model_cols) > 1) {
  cor_dt <- copy(dt[, ..model_cols])
  
  for (j in names(cor_dt)) {
    cor_dt[[j]] <- as.numeric(cor_dt[[j]])
  }
  
  cor_mat <- cor(cor_dt, use = "pairwise.complete.obs", method = "spearman")
  
  cor_long <- as.data.table(as.table(cor_mat))
  setnames(cor_long, c("Var1", "Var2", "Freq"), c("feature_x", "feature_y", "correlation"))
  
  p_cor <- ggplot(cor_long, aes(feature_x, feature_y, fill = correlation)) +
    geom_tile() +
    coord_equal() +
    theme_minimal(base_size = 11) +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),
      axis.title = element_blank()
    ) +
    labs(
      title = "Spearman correlation among selected predictors",
      fill = "rho"
    )
  
  ggsave(
    filename = file.path(out_dir, "correlation_heatmap.png"),
    plot = p_cor,
    width = 9, height = 8, dpi = 300
  )
}

print(p_cor)
# ---------------------- 8) PCA PLOT ------------------------------------

if (length(model_cols) > 1) {
  pca_dt <- copy(dt[, c("class", intersect("biome", names(dt)), model_cols), with = FALSE])
  pca_complete <- pca_dt[complete.cases(pca_dt[, ..model_cols])]
  
  if (nrow(pca_complete) > 10) {
    xmat <- scale(as.matrix(pca_complete[, ..model_cols]))
    pca <- prcomp(xmat, center = FALSE, scale. = FALSE)
    
    pca_scores <- as.data.table(pca$x[, 1:2])
    pca_scores[, class := pca_complete$class]
    if ("biome" %in% names(pca_complete)) {
      pca_scores[, biome := pca_complete$biome]
    }
    
    p_pca <- ggplot(pca_scores, aes(x = PC1, y = PC2, color = class)) +
      geom_point(alpha = 0.35, size = 0.5) +
      theme_minimal(base_size = 12) +
      labs(
        title = "PCA of selected training predictors",
        x = "PC1",
        y = "PC2",
        color = "Class"
      )
    
    ggsave(
      filename = file.path(out_dir, "pca_training_features.png"),
      plot = p_pca,
      width = 7, height = 5.5, dpi = 300
    )
    
    if ("biome" %in% names(pca_scores)) {
      p_pca_biome <- ggplot(pca_scores, aes(x = PC1, y = PC2, color = biome)) +
        geom_point(alpha = 0.35, size = 0.5) +
        theme_minimal(base_size = 12) +
        labs(
          title = "PCA of selected training predictors by biome",
          x = "PC1",
          y = "PC2",
          color = "Biome"
        )
      
      ggsave(
        filename = file.path(out_dir, "pca_training_features_biome.png"),
        plot = p_pca_biome,
        width = 7, height = 5.5, dpi = 300
      )
    }
  }
}

cat("Finished writing figures to:", out_dir, "\n")

print(p_pca_biome)


