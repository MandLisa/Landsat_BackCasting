SRC="/mnt/eo/EO4Alps/level2/"
#DST_LND="/mnt/dss_alps/fdrebes/alps/level2_LS"
DST_SEN="/mnt/dss_alps/fdrebes/alps/level2_S2"

BW=70000   # KiB/s 



# 2) Sentinel-2
rsync -rltD --no-perms --no-owner --no-group --omit-dir-times \
  --size-only --ignore-existing \
  --bwlimit="$BW" \
  --info=progress2 \
  --partial --inplace \
  --prune-empty-dirs \
  --include='*/' \
  --include='*SEN2A*' --include='*SEN2B*' \
  --include='*.tif' --include='*.tiff' \
  --exclude='*' \
  "$SRC" "$DST_SEN"
  

# 1) Landsat
rsync -rltD --no-perms --no-owner --no-group --omit-dir-times \
  --size-only --ignore-existing \
  --bwlimit="$BW" \
  --info=progress2 \
  --partial --inplace \
  --prune-empty-dirs \
  --include='*/' \
  --include='*LND04*' --include='*LND05*' --include='*LND06*' \
  --include='*LND07*' --include='*LND08*' --include='*LND09*' \
  --include='*.tif' --include='*.tiff' \
  --exclude='*' \
  "$SRC" "$DST_LND"
  