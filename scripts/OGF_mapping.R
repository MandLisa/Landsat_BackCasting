library(terra)
library(data.table)
library(xgboost)

# -------------------- PATHS --------------------
comp_dir   <- "/path/to/your_5y_composites/"        # already computed 5y composites (6 bands)
nbr_dir    <- "/path/to/annual_NBR/"                # yearly NBR rasters
dist_file  <- "/path/to/disturbance_year_1985_2023.tif"
forest_file<- "/path/to/forest_mask.tif"
out_dir    <- "/path/to/output_longstable/"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

tmp_dir <- file.path(out_dir, "_tmp_terra")
dir.create(tmp_dir, showWarnings = FALSE, recursive = TRUE)
terraOptions(tempdir = tmp_dir, memfrac = 0.6, progress = 3)

t0_years <- 2005:2014
n_per_class_per_t0 <- 20000
set.seed(42)

# -------------------- HELPERS --------------------
# adapt this to your filenames:
comp_file <- function(t0) file.path(comp_dir, sprintf("bap_med5_t0_%d.tif", t0))

get_year <- function(x) as.integer(sub(".*(19|20\\d{2}).*", "\\1", basename(x)))

nbr_files <- list.files(nbr_dir, pattern="\\.tif$", full.names=TRUE)
nbr_dt <- data.table(file=nbr_files)
nbr_dt[, year := get_year(file)]
setkey(nbr_dt, year)

dist  <- rast(dist_file)
fmask <- resample(rast(forest_file), dist, method="near")  # align

# 1) label raster: y=1 if no disturbance in [t0-20, t0] AND forest=1
make_label <- function(t0, dist, fmask, out_file){
  lo <- t0 - 20
  hi <- t0
  
  y <- app(c(dist, fmask), fun = function(x){
    d <- x[1]; fm <- x[2]
    if (is.na(fm) || fm != 1) return(NA)
    # disturbed in prev20?
    disturbed <- (!is.na(d)) && (d >= lo) && (d <= hi)
    as.integer(!disturbed)
  }, filename=out_file, overwrite=TRUE)
  names(y) <- "y_long_stable"
  y
}

# 2) Sen slope of NBR in [t0..t0+10]
make_sen_slope <- function(t0, nbr_dt, out_file){
  yrs <- t0:(t0+10)
  files <- nbr_dt[J(yrs), file]
  if (any(is.na(files))) stop("Missing NBR years: ", paste(yrs[is.na(files)], collapse=", "))
  
  s <- rast(files)  # 11 layers
  tvec <- yrs
  
  sen_fun <- function(v){
    if (all(is.na(v))) return(NA_real_)
    ok <- which(!is.na(v))
    if (length(ok) < 6) return(NA_real_)  # require enough points
    x <- v[ok]; tt <- tvec[ok]
    n <- length(x)
    slopes <- numeric(0)
    for (i in 1:(n-1)) for (j in (i+1):n) slopes <- c(slopes, (x[j]-x[i])/(tt[j]-tt[i]))
    median(slopes)
  }
  
  slope <- app(s, sen_fun, filename=out_file, overwrite=TRUE)
  names(slope) <- "nbr_sen_slope10y"
  slope
}

# 3) balanced sampling from label raster
sample_balanced <- function(y_r, n_per_class, seed){
  set.seed(seed)
  pts <- spatSample(y_r, size = n_per_class*2, method="stratified",
                    strata=TRUE, na.rm=TRUE, as.points=TRUE, values=TRUE)
  pts
}

# 4) extract training table for one t0
build_train_t0 <- function(t0){
  message("\n=== t0 = ", t0, " ===")
  
  comp_path <- comp_file(t0)
  if (!file.exists(comp_path)) stop("Missing composite: ", comp_path)
  
  comp <- rast(comp_path)             # 6 bands
  comp <- resample(comp, dist, method="bilinear")  # ensure aligned (cheap if already aligned)
  
  # label
  y_path <- file.path(out_dir, sprintf("label_prev20_t0_%d.tif", t0))
  y_r <- make_label(t0, dist, fmask, y_path)
  
  # slope
  slope_path <- file.path(out_dir, sprintf("nbr_sen_slope10y_t0_%d.tif", t0))
  slope <- make_sen_slope(t0, nbr_dt, slope_path)
  
  # feature stack
  feats <- c(comp, slope)
  feats <- resample(feats, y_r, method="bilinear")
  
  # sample
  pts <- sample_balanced(y_r, n_per_class_per_t0, seed = 100 + t0)
  if (nrow(pts) == 0) {
    warning("No points for t0=", t0)
    return(NULL)
  }
  
  # extract
  vals <- terra::extract(feats, pts, ID=FALSE)
  dt <- as.data.table(vals)
  yv <- pts[[1]]  # sampled label values
  xy <- crds(pts)
  dt[, `:=`(y = as.integer(yv), x = xy[,1], ycoord = xy[,2], t0 = t0)]
  
  out_csv <- file.path(out_dir, sprintf("train_t0_%d.csv", t0))
  fwrite(dt, out_csv)
  rm(comp, y_r, slope, feats, pts); gc()
  out_csv
}

# -------------------- BUILD ALL TRAINING FILES --------------------
train_csvs <- Filter(Negate(is.null), lapply(t0_years, build_train_t0))

# -------------------- TRAIN XGBOOST --------------------
dt <- rbindlist(lapply(train_csvs, fread), use.names=TRUE, fill=TRUE)

feat_cols <- setdiff(names(dt), c("y","x","ycoord","t0"))

# simple spatial block split
block_size <- 10000
dt[, bx := floor(x / block_size)]
dt[, by := floor(ycoord / block_size)]
dt[, block_id := as.factor(paste(bx, by, sep="_"))]

set.seed(42)
blocks <- unique(dt$block_id)
test_blocks <- sample(blocks, size = floor(0.2 * length(blocks)))
dt[, is_test := block_id %in% test_blocks]

train_dt <- dt[!is_test]
test_dt  <- dt[ is_test]

dtrain <- xgb.DMatrix(as.matrix(train_dt[, ..feat_cols]), label=train_dt$y)
dtest  <- xgb.DMatrix(as.matrix(test_dt[, ..feat_cols]),  label=test_dt$y)

params <- list(
  booster="gbtree",
  objective="binary:logistic",
  eval_metric="auc",
  eta=0.05,
  max_depth=6,
  subsample=0.8,
  colsample_bytree=0.8
)

set.seed(42)
model <- xgb.train(
  params=params, data=dtrain,
  nrounds=2000,
  watchlist=list(train=dtrain, test=dtest),
  early_stopping_rounds=50,
  verbose=1
)

saveRDS(model, file.path(out_dir, "xgb_longstable.rds"))
writeLines(feat_cols, file.path(out_dir, "feature_columns.txt"))

# -------------------- APPLY TO 1985-1990 COMPOSITE --------------------
t0_target <- 1985
comp_target <- rast(comp_file(t0_target))
comp_target <- resample(comp_target, dist, method="bilinear")

slope_target <- make_sen_slope(t0_target, nbr_dt,
                               file.path(out_dir, "nbr_sen_slope10y_t0_1985.tif"))

feat_target <- c(comp_target, slope_target)
feat_target <- resample(feat_target, fmask, method="bilinear")
feat_target <- mask(feat_target, fmask)  # only forest

feat_cols <- readLines(file.path(out_dir, "feature_columns.txt"))
pred_fun <- function(df){
  m <- as.matrix(df)
  if (nrow(m) == 0) return(numeric(0))
  m <- m[, feat_cols, drop=FALSE]
  as.numeric(predict(model, m))
}

prob_file <- file.path(out_dir, "prob_longstable_backcast_1985.tif")
prob <- predict(feat_target, fun=pred_fun, filename=prob_file,
                overwrite=TRUE, wopt=list(datatype="FLT4S", gdal="COMPRESS=ZSTD"))

message("Done: ", prob_file)
