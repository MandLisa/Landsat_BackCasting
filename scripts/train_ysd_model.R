library(terra)
library(data.table)
library(stringr)


points_csv   <- "/mnt/eo/EO4Backcasting/_intermediates/training_data.csv"

pts <- fread(points_csv)[, .(id, x, y, yod, year, ysd, ysd_bin, class_label, EVI, NBR)]


