#!/usr/bin/env bash
# Yearly mosaics (VRT) for YYYY0801_LEVEL3_LNDLG_IBAP.tif
# Inputs are READ-ONLY. We create per-tile proxy VRTs + one mosaic VRT per year.
set -uo pipefail

# -------- CONFIG --------
input_dir="/mnt/dss_europe/level3_interpolated"           # X####_Y#### folders
out_dir="/mnt/dss_europe/mosaic/mosaics_BAPs_vrt"         # <-- only change requested
proxy_root="$out_dir/_proxies"                            # keep proxies; VRTs depend on them
mkdir -p "$out_dir" "$proxy_root"

START_YEAR="${START_YEAR:-1984}"
END_YEAR="${END_YEAR:-2024}"
NODATA_VAL="${NODATA_VAL:--9999}"
export GDAL_NUM_THREADS=ALL_CPUS

# -------- FUNCTIONS --------
make_proxy () {
  local tif="$1" year="$2"
  local base tile outdir proxy
  base="$(basename "$tif")"                    # e.g., 19840801_LEVEL3_LNDLG_IBAP.tif
  tile="$(basename "$(dirname "$tif")")"       # e.g., X0001_Y0024
  outdir="$proxy_root/$year/$tile"
  proxy="$outdir/${base%.tif}.vrt"
  [[ -f "$proxy" ]] && return 0
  mkdir -p "$outdir"
  gdal_translate -q -of VRT -a_srs EPSG:3035 "$tif" "$proxy"   # wrapper VRT; input untouched
}

build_year () {
  local year="$1"
  echo "==> Year $year"

  # collect exactly YYYY0801 IBAP tiles
  mapfile -t tiles < <(find "$input_dir" -type f -name "${year}0801_LEVEL3_LNDLG_IBAP.tif" -print | sort)
  local n_tiles=${#tiles[@]}
  if (( n_tiles == 0 )); then
    echo "   [WARN] no tiles — skipping."
    return 0
  fi
  echo "   Tiles found: $n_tiles"

  # proxies
  local failed="$proxy_root/$year/failed_tiles.txt"
  : > "$failed"
  local built=0 failc=0
  for tif in "${tiles[@]}"; do
    if make_proxy "$tif" "$year"; then
      ((built++)); (( built % 25 == 0 )) && printf "."
    else
      echo "$tif" >> "$failed"; ((failc++))
    fi
  done
  printf "\n"
  echo "   Proxies attempted: $built   failed: $failc"

  mapfile -t proxies < <(find "$proxy_root/$year" -type f -name '*.vrt' -print | sort)
  local n_proxies=${#proxies[@]}
  echo "   Proxies present:  $n_proxies"
  if (( n_proxies != n_tiles )); then
    echo "   [ERROR] proxies ($n_proxies) != tiles ($n_tiles). See $failed"
    return 2
  fi

  # mosaic proxies -> yearly VRT
  local proxies_list="$proxy_root/$year/proxies_${year}.txt"
  printf "%s\n" "${proxies[@]}" > "$proxies_list"

  local vrtfile="$out_dir/${year}0801_LEVEL3_LNDLG_IBAP.vrt"
  if ! gdalbuildvrt -q -overwrite \
       -input_file_list "$proxies_list" \
       -resolution highest \
       -srcnodata "$NODATA_VAL" -vrtnodata "$NODATA_VAL" \
       "$vrtfile"; then
    echo "   [ERROR] gdalbuildvrt failed (see $proxies_list)"; return 3
  fi

  local n_sources
  n_sources=$(grep -c '<SourceFilename' "$vrtfile" || echo 0)
  echo "   VRT sources:      $n_sources"
  if (( n_sources != n_tiles )); then
    echo "   [ERROR] VRT sources ($n_sources) != tiles ($n_tiles). Inspect $vrtfile"
    return 4
  fi

  echo "   OK → $vrtfile"
}

# -------- MAIN --------
for (( y=START_YEAR; y<=END_YEAR; y++ )); do
  build_year "$y" || { echo "[STOP] at year $y"; exit 1; }
done

echo "All done."
echo "Yearly VRTs: $out_dir"
echo "Proxies kept: $proxy_root"
