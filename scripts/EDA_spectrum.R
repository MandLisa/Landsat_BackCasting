suppressPackageStartupMessages(library(data.table))

# ---- set your path ----
TRAIN_CSV <- "/mnt/eo/EO4Backcasting/_intermediates/training_data.csv"

# ---- read ----
pts <- fread(TRAIN_CSV)

# ---- add/overwrite column 'state' ----
pts[, state := "disturbed"]              # character column with constant value
# If you prefer a factor:
# pts[, state := factor("disturbed", levels = c("healthy", "disturbed"))]


# ---- write back (overwrite original) ----
fwrite(pts, TRAIN_CSV)

message("Done. Added column 'state' = 'disturbed' for all rows.")
message(sprintf("Backup written to: %s", bk))
