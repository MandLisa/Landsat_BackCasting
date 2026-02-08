---
title: "Backcasting"
output: 
  html_document:
    self_contained: false
---

## Probability map (RF)

```{r, echo=FALSE}
library(terra)
library(leaflet)

# Probability raster
r <- rast("/mnt/eo/EO4Backcasting/_predictions/bap1990_prob_crop.tif")

if (crs(r) != "EPSG:4326") {
  r_wgs <- project(r, "EPSG:4326")
} else {
  r_wgs <- r
}

pal <- colorNumeric("viridis", values(r_wgs), na.color = "transparent")

leaflet() |> 
  addProviderTiles("Esri.WorldImagery") |>
  addRasterImage(r_wgs, colors = pal, opacity = 0.8) |>
  addLegend(pal = pal, values = values(r_wgs), title = "Probability")


