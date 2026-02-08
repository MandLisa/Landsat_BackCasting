#!/usr/bin/env R

# ============================================================
# 0. Load libraries
# ============================================================
library(data.table)
library(ggplot2)

# ============================================================
# 1. Load data
# ============================================================
DT <- fread("/mnt/eo/EO4Backcasting/_intermediates/training_healthy_disturbed_1911_final.csv")

# ============================================================
# 2. Define bands and levels
# ============================================================
band_cols <- c("blue","green","red","nir","swir1","swir2")

# --- ysd_bin groups ---
lvl <- c("undisturbed","ysd1_5","ysd6_10","ysd11_15","ysd16_20","ysd_20")
pal <- c(
  "undisturbed" = "black",
  "ysd1_5"      = "#d62728",
  "ysd6_10"     = "#ff7f0e",
  "ysd11_15"    = "#2ca02c",
  "ysd16_20"    = "#9467bd",
  "ysd_20"      = "#8c564b"
)

# --- ysd15_bin groups ---
lvl15 <- c("undisturbed","ysd_lt15","ysd_ge15")
pal15 <- c(
  "undisturbed" = "black",
  "ysd_lt15"    = "#1f77b4",
  "ysd_ge15"    = "#d62728"
)

# ============================================================
# 3. Create group column (based on ysd_bin)
# ============================================================
DT[, group := factor(ysd_bin, levels = lvl)]

# ============================================================
# 4. Melt for plotting (ysd_bin)
# ============================================================
long_plot <- melt(
  DT,
  measure.vars = band_cols,
  variable.name = "band",
  value.name = "val"
)

setDT(long_plot)

long_plot[, band  := factor(band,  levels = band_cols)]
long_plot[, group := factor(group, levels = lvl)]

# ============================================================
# 5. Plot for ysd_bin
# ============================================================
p1 <- ggplot(long_plot, aes(x = band, y = val, fill = group)) +
  geom_boxplot(outlier.alpha = 0.2, width = 0.75,
               position = position_dodge2(preserve = "single")) +
  scale_fill_manual(values = pal, breaks = lvl) +
  labs(
    x = "Band",
    y = "Reflectance",
    fill = "Group",
    title = "Spectral BAP Distributions – YSD Bins"
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "right")

# Save
ggsave(
  filename = "/mnt/eo/EO4Backcasting/_figs/spectral_profile_boxplots_1911.jpg",
  plot = p1,
  width = 8, height = 4, units = "in", dpi = 300
)

# ============================================================
# 6. Melt & plot for ysd15_bin
# ============================================================
long15 <- melt(
  DT,
  measure.vars = band_cols,
  variable.name = "band",
  value.name = "val"
)

setDT(long15)

long15[, band := factor(band, levels = band_cols)]

# Replace NA by undisturbed
long15[, group15 := ysd15_bin]
long15[is.na(group15), group15 := "undisturbed"]
long15[, group15 := factor(group15, levels = lvl15)]

# Plot
p2 <- ggplot(long15, aes(x = band, y = val, fill = group15)) +
  geom_boxplot(outlier.alpha = 0.2, width = 0.75,
               position = position_dodge2(preserve = "single")) +
  scale_fill_manual(values = pal15, breaks = lvl15) +
  labs(
    x = "Band",
    y = "Reflectance",
    fill = "Group",
    title = "Spectral BAP Distributions – <15 vs. ≥15 Years"
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "right")

# Save
ggsave(
  filename = "/mnt/eo/EO4Backcasting/_figs/spectral_profile_boxplots_ysd15_1911.jpg",
  plot = p2,
  width = 8, height = 4, units = "in", dpi = 300
)

# ============================================================
# Done
# ============================================================
cat("Plots created and saved successfully.\n")
