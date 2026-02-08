# =====================================================================
# 0. Libraries
# =====================================================================
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import rioxarray as rxr
import xarray as xr
import rasterio
from rasterio.enums import Resampling
import os

np.random.seed(1234)

# =====================================================================
# 1. Load data
# =====================================================================

TRAIN_CSV = "/mnt/eo/EO4Backcasting/_intermediates/training_data_topo.csv"
df = pd.read_csv(TRAIN_CSV)

# Only keep rows with valid BAP
df = df[df["bap_available"] == True]

# =====================================================================
# 2. Create target variable (binary)
# =====================================================================

def classify_target(row):
    if row["state"] == "undisturbed":
        return 0
    if row["state"] == "disturbed" and row["ysd"] <= 15:
        return 1
    if row["state"] == "disturbed" and row["ysd"] > 15:
        return 0
    return np.nan

df["target"] = df.apply(classify_target, axis=1)
df = df.dropna(subset=["target"])

pred_cols = ["b1","b2","b3","b4","b5","b6"]
df = df.dropna(subset=pred_cols)

print("Training rows:", len(df))
print(df["target"].value_counts())

# =====================================================================
# 3. Train/validation split
# =====================================================================

train_df, val_df = train_test_split(df, test_size=0.2, random_state=1234)

X_train = train_df[pred_cols].values
y_train = train_df["target"].values

X_val = val_df[pred_cols].values
y_val = val_df["target"].values

# =====================================================================
# 4. Random Forest (ranger equivalent)
# =====================================================================

rf = RandomForestClassifier(
    n_estimators=500,
    max_features=3,
    random_state=1234,
    class_weight=None,
    n_jobs=-1
)

rf.fit(X_train, y_train)

print(rf)

# =====================================================================
# 5. Validation
# =====================================================================

val_prob = rf.predict_proba(X_val)[:, 1]

threshold = 0.75
val_pred = (val_prob > threshold).astype(int)

acc = (val_pred == y_val).mean()
print("\nValidation accuracy:", acc)

print("\nConfusion matrix:")
print(confusion_matrix(y_val, val_pred))

auc = roc_auc_score(y_val, val_prob)
print("\nAUC:", auc)

# =====================================================================
# 6. Save model + metadata
# =====================================================================

joblib.dump(rf, "/mnt/eo/EO4Backcasting/_models/rf_bap_binary.joblib")
joblib.dump({"pred_cols": pred_cols},
            "/mnt/eo/EO4Backcasting/_models/rf_bap_recentdist_binary_meta.joblib")

print("\nModel + metadata saved.")

# =====================================================================
# 7. Threshold evaluation
# =====================================================================

thresholds = np.arange(0.01, 1.0, 0.01)

def compute_metrics(t, y, p):
    pred = (p >= t).astype(int)
    TP = np.sum((pred == 1) & (y == 1))
    FP = np.sum((pred == 1) & (y == 0))
    FN = np.sum((pred == 0) & (y == 1))
    TN = np.sum((pred == 0) & (y == 0))

    precision = TP / (TP + FP) if (TP + FP) > 0 else np.nan
    recall = TP / (TP + FN) if (TP + FN) > 0 else np.nan
    CE = FP / (TP + FP) if (TP + FP) > 0 else np.nan
    OE = FN / (TP + FN) if (TP + FN) > 0 else np.nan
    accuracy = (TP + TN) / (TP + FP + FN + TN)

    return {
        "threshold": t,
        "precision": precision,
        "recall": recall,
        "CE": CE,
        "OE": OE,
        "accuracy": accuracy
    }

metric_list = pd.DataFrame([compute_metrics(t, y_val, val_prob) for t in thresholds])

# ---- Plot ----
metric_long = metric_list.melt(
    id_vars="threshold",
    value_vars=["precision","recall","CE","OE"],
    var_name="metric",
    value_name="value"
)

plt.figure(figsize=(10,6))
sns.lineplot(data=metric_long, x="threshold", y="value", hue="metric")
plt.title("Threshold curves")
plt.xlabel("Threshold")
plt.ylabel("Value")
plt.legend(loc="lower center", ncol=4)
plt.tight_layout()
plt.show()

# =====================================================================
# 8. Raster Prediction (terra::predict equivalent)
# =====================================================================

INFILE = "/mnt/eo/EO4Backcasting/_bap_local/19900801_LEVEL3_LNDLG_IBAP.tif"
prob_file = "/mnt/eo/EO4Backcasting/_predictions/bap1990_prob.tif"

# Load raster as xarray
r_bap = rxr.open_rasterio(INFILE).squeeze("band")
r_bap = r_bap.assign_coords({"band": pred_cols})

# Convert to (rows*cols, bands)
arr = np.stack([r_bap.sel(band=b).values for b in pred_cols], axis=-1)

nan_mask = np.any(np.isnan(arr), axis=-1)
flat_pixels = arr.reshape(-1, arr.shape[-1])
flat_pixels[np.isnan(flat_pixels)] = -9999  # or masked

# Predict
flat_proba = rf.predict_proba(flat_pixels)[:,1]
flat_proba[nan_mask.reshape(-1)] = np.nan

# Write probability raster
proba_map = flat_proba.reshape(r_bap.shape[1], r_bap.shape[2])

with rasterio.open(
    prob_file,
    "w",
    driver="GTiff",
    height=r_bap.shape[1],
    width=r_bap.shape[2],
    count=1,
    dtype="float32",
    crs=r_bap.rio.crs,
    transform=r_bap.rio.transform(),
) as dst:
    dst.write(proba_map.astype("float32"), 1)

print("Probability raster written:", prob_file)

# =====================================================================
# 9. Classification with multiple thresholds
# =====================================================================

thresholds = [0.5, 0.75, 0.9]
outdir = "/mnt/eo/EO4Backcasting/_predictions/"
os.makedirs(outdir, exist_ok=True)

for thr in thresholds:
    class_map = (proba_map >= thr).astype("uint8")
    outfile = os.path.join(outdir, f"class_thr_{thr:.2f}.tif")

    with rasterio.open(
        outfile,
        "w",
        driver="GTiff",
        height=r_bap.shape[1],
        width=r_bap.shape[2],
        count=1,
        dtype="uint8",
        crs=r_bap.rio.crs,
        transform=r_bap.rio.transform(),
    ) as dst:
        dst.write(class_map, 1)

    print("Written:", outfile)

