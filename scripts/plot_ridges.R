# libs
library(readr)
library(dplyr)
library(ggplot2)
library(ggridges)
library(forcats)

# Paths
fp_area  <- "/mnt/eo/EO4Backcasting/_data/country_probability_samples_area_based.csv"
fp_fixed <- "/mnt/eo/EO4Backcasting/_data/country_probability_samples_2000.csv"

df_area  <- read_csv(fp_area, show_col_types = FALSE)
df_fixed <- read_csv(fp_fixed, show_col_types = FALSE)

# define function
plot_ridge_gradient_reds <- function(df, title = NULL) {
  
  mean_df <- df %>%
    group_by(country) %>%
    summarise(
      mean_prob = mean(probability, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    arrange(desc(mean_prob))
  
  df_plot <- df %>%
    mutate(country = factor(country, levels = mean_df$country))
  
  mean_df <- mean_df %>%
    mutate(country = factor(country, levels = mean_df$country))
  
  ggplot(df_plot, aes(x = probability, y = country, fill = after_stat(x))) +
    geom_density_ridges_gradient(
      scale = 1.2,
      rel_min_height = 0.001,
      color = "black",
      linewidth = 0.3,
      alpha = 1
    ) +
    geom_point(
      data = mean_df,
      aes(x = mean_prob, y = country),
      inherit.aes = FALSE,
      color = "black",
      size = 1.8
    ) +
    scale_fill_gradientn(
      colours = c("#F7F7F7", "#F3CDBF", "#EE9C7B", "#F95C3C", "#E31A1C", "#7F0000"),
      values = scales::rescale(c(0.15, 0.250, 0.5, 0.60, 0.7, 0.8)),
      limits = c(0, 1),
      name = "Probability"
    ) +
    scale_x_continuous(
      limits = c(0, 1),
      expand = c(0.01, 0)
    ) +
    labs(
      x = "Predicted probability",
      y = NULL,
      title = title
    ) +
    theme_ridges() +
    theme(
      panel.grid = element_blank(),
      axis.text.y = element_text(size = 18),
      axis.text.x = element_text(size = 14),
      axis.title.x = element_text(size = 16, hjust = 0.5),
      axis.title.y = element_text(size = 16),
      plot.title = element_text(size = 18, face = "bold"),
      legend.title = element_text(size = 15),
      legend.text = element_text(size = 13),
      legend.position = "right"
    )
}


# apply
p_fixed <- plot_ridge_gradient_reds(
  df_fixed,
  title = ""
)

#p_area
p_fixed


# save
ggsave(
  filename = "/mnt/eo/EO4Backcasting/_figs/ridgeplot_fixed_2000.png",
  plot = p_fixed,
  width = 9.5,
  height = 13,
  dpi = 300,
  bg = "white"
)


