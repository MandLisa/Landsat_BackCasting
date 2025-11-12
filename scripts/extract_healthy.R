suppressPackageStartupMessages({
  library(terra)
  library(data.table)
})

# -------- paths --------
UNDIST20_MASK_PATH <- "/mnt/eo/EFDA_v211/undisturbed_forest.tif"  # 1=undisturbed (20y), NA otherwise
BAP2006_PATH       <- "/mnt/dss_europe/mosaics_eu/mosaics_eu_baps/2006_mosaic_eu_cog.tif" # 6 bands
OUT_CSV            <- "/mnt/eo/EO4Backcasting/_intermediates/training_healthy_disturbed.csv"
N_SAMPLES          <- 32000

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

# 1) Add an integer id to the healthy df (continue after disturbed ids if present)
next_id <- if ("id" %in% names(pts)) max(pts$id, na.rm=TRUE) + 1L else 1L
dt[, id := seq_len(.N) + (if (is.finite(next_id)) next_id else 0L)]


# 2) Ensure healthy has all columns that exist in disturbed; fill with NA of proper type
add_missing_cols <- function(dst, ref) {
  miss <- setdiff(names(ref), names(dst))
  if (length(miss)) {
    for (nm in miss) {
      # create type-consistent NA based on ref column class
      cls <- class(ref[[nm]])[1]
      na_val <- switch(cls,
                       "integer"   = as.integer(NA),
                       "numeric"   = as.numeric(NA),
                       "logical"   = as.logical(NA),
                       "character" = NA_character_,
                       "factor"    = factor(NA, levels = levels(ref[[nm]])),
                       "Date"      = as.Date(NA),
                       "POSIXct"   = as.POSIXct(NA),
                       as.numeric(NA) # default
      )
      dst[, (nm) := na_val]
    }
  }
  # also ensure ref has any columns present only in dst (rare, but for completeness)
  miss_ref <- setdiff(names(dst), names(ref))
  if (length(miss_ref)) {
    for (nm in miss_ref) ref[, (nm) := NA]
  }
  list(dst = dst, ref = ref)
}


tmp <- add_missing_cols(dt, pts)
dt_fixed <- tmp$dst
pts_fixed <- tmp$ref


# 3) Harmonize some common types explicitly (optional but robust)
int_cols <- intersect(c("yod","year","ysd","ysd_bin","id"), names(dt_fixed))
num_cols <- intersect(c("EVI","NBR", paste0("b",1:6)), names(dt_fixed))
log_cols <- intersect(c("bap_available"), names(dt_fixed))
chr_cols <- intersect(c("class_label","state"), names(dt_fixed))

for (nm in int_cols) if (!is.integer(dt_fixed[[nm]])) dt_fixed[, (nm) := as.integer(get(nm))]
for (nm in num_cols) if (!is.numeric(dt_fixed[[nm]])) dt_fixed[, (nm) := as.numeric(get(nm))]
for (nm in log_cols) if (!is.logical(dt_fixed[[nm]])) dt_fixed[, (nm) := as.logical(get(nm))]
for (nm in chr_cols) if (!is.character(dt_fixed[[nm]])) dt_fixed[, (nm) := as.character(get(nm))]

# 4) Bind (rows) with fill
both <- rbindlist(list(pts_fixed, dt_fixed), use.names = TRUE, fill = TRUE)

fwrite(both, OUT_CSV)



### ploot
# ----------------- assumptions -----------------
# 'both' has at least: state ∈ {"healthy","disturbed"}, b1..b6, and (for disturbed) ysd
stopifnot(exists("both"))
band_cols <- paste0("b", 1:6)
stopifnot(all(band_cols %in% names(both)), "state" %in% names(both))

# If BAP are scaled integers (e.g., ×10000), set SCALE accordingly
SCALE <- 1  # set to 10000 if you want reflectance in [0,1]


DT <- as.data.table(both)
band_cols <- paste0("b", 1:6)
SCALE <- 1          # set to 10000 if BAP are scaled integers

# Ensure grouping exists (healthy vs YSD bins)
if (!"group" %in% names(DT)) {
  DT[, group := fifelse(
    state == "healthy", "healthy",
    as.character(cut(
      ysd, breaks=c(0,5,10,15,20,Inf), right=TRUE,
      labels=c("ysd 1–5","ysd >5–10","ysd >10–15","ysd >15–20","ysd >20")
    ))
  )]
}
DT <- DT[!is.na(group)]

# Melt correctly: specify measure.vars, then scale
long <- melt(
  DT,
  measure.vars = band_cols,
  variable.name = "band",
  value.name = "val"
)[, val := val / SCALE]

# Aggregate mean spectra
spec_means <- long[, .(mean_val = mean(val, na.rm = TRUE)), by = .(group, band)]
spec_means[, band  := factor(band, levels = band_cols)]
spec_means[, group := factor(group, levels = c("healthy","ysd 1–5","ysd >5–10","ysd >10–15","ysd >15–20","ysd >20"))]

# Plot: healthy thick, YSD bins coloured
spec_means[, group := factor(group,
                             levels = c("healthy","ysd 1–5","ysd >5–10","ysd >10–15","ysd >15–20","ysd >20"))]


ggplot(spec_means, aes(band, mean_val, group = group,
                       color = group, size = group, linetype = group)) +
  geom_line() +
  geom_point() +
  scale_color_manual(values = c(
    "healthy"   = "black",
    "ysd 1–5"   = "#d62728",
    "ysd >5–10" = "#ff7f0e",
    "ysd >10–15"= "#2ca02c",
    "ysd >15–20"= "#9467bd",
    "ysd >20"   = "#8c564b"
  )) +
  scale_size_manual(values = c(
    "healthy"   = 1.6,
    "ysd 1–5"   = 0.7,
    "ysd >5–10" = 0.7,
    "ysd >10–15"= 0.7,
    "ysd >15–20"= 0.7,
    "ysd >20"   = 0.7
  ), guide = "none") +                 # hide size legend if you want
  scale_linetype_manual(values = c(
    "healthy"   = "solid",
    "ysd 1–5"   = "dashed",
    "ysd >5–10" = "dashed",
    "ysd >10–15"= "dashed",
    "ysd >15–20"= "dashed",
    "ysd >20"   = "dashed"
  )) +
  labs(x="Band", y="Mean BAP value", color="Group", linetype="Group",
       title="") +
  theme_minimal(base_size = 12) +
  theme(legend.position = "right")

ggsave(
  filename = "/mnt/eo/EO4Backcasting/_figs/spectral_profile.jpg",
  width = 8,         # in inches
  height = 4,
  units = "in",
  dpi = 300          # high-quality for publication
)



### boxplot
pal <- c("healthy"   = "black",
             "ysd 1–5"   = "#d62728",
             "ysd >5–10" = "#ff7f0e",
             "ysd >10–15"= "#2ca02c",
             "ysd >15–20"= "#9467bd",
             "ysd >20"   = "#8c564b")

# Plot: healthy thick, YSD bins coloured
# 1) Define desired order once
lvl <- c("healthy","ysd 1–5","ysd >5–10","ysd >10–15","ysd >15–20","ysd >20")

# 2) Apply to the plotted data
long_plot[, group := factor(group, levels = lvl)]   # data.table syntax

# boxplot
ggplot(long_plot, aes(x = band, y = val, fill = group)) +
  geom_boxplot(outlier.alpha = 0.2, width = 0.75,
               position = position_dodge2(preserve = "single")) +
  scale_fill_manual(values = pal, breaks = lvl) +
  labs(x = "Band", y = "Mean BAP value", fill = "Group",
       title = "") +
  theme_minimal(base_size = 12) +
  theme(legend.position = "right")

ggsave(
  filename = "/mnt/eo/EO4Backcasting/_figs/spectral_profile_boxplots.jpg",
  width = 8,         # in inches
  height = 4,
  units = "in",
  dpi = 300          # high-quality for publication
)










