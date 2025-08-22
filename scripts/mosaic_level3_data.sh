#!/usr/bin/env bash
set -euo pipefail

IN="/mnt/dss_europe/level3_interpolated"
OUT="/mnt/eo/eu_mosaics"
mkdir -p "$OUT"

# collect available years, then keep only 2000–2024
mapfile -t YEARS < <(
  find "$IN" -type f -name "*NDVI*.tif" -printf "%f\n" \
  | awk '{print substr($0,1,4)}' \
  | awk '$1>=2000 && $1<=2024' \
  | sort -u
)

mosaic_one_year () {
  local year="$1" in="$2" out="$3"
  # build VRT over all tiles for this year
  tmpvrt="$(mktemp --suffix=".vrt")"
  find "$in" -type f -name "${year}*NDVI*.tif" -print0 \
    | xargs -0 gdalbuildvrt \
        -srcnodata -10000 -vrtnodata -10000 \
        -resolution highest \
        "$tmpvrt"

  # write compressed GeoTIFF
  gdal_translate "$tmpvrt" "${out}/NDVI_${year}.tif" \
      -co TILED=YES -co COMPRESS=LZW -co BIGTIFF=YES \
      -co NUM_THREADS=ALL_CPUS -co SPARSE_OK=TRUE -a_nodata -10000
  rm -f "$tmpvrt"
}
export -f mosaic_one_year

if command -v parallel >/dev/null 2>&1; then
  parallel --jobs "$(nproc)" mosaic_one_year {} "$IN" "$OUT" ::: "${YEARS[@]}"
else
  for y in "${YEARS[@]}"; do mosaic_one_year "$y" "$IN" "$OUT"; done
fi
