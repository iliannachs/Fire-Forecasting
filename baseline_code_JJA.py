import os
from glob import glob
import numpy as np
import pandas as pd
import xarray as xr

from scipy import ndimage
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)
from sklearn.inspection import permutation_importance
from sklearn.model_selection import ParameterSampler
from xgboost import XGBRegressor

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature


cluster_to_model = 0  # 0, 1, 2, or 3

era_path = "/projects/0/ctdas/ilianna/1.Data/raw/era5_merged.nc"
modis1 = "/projects/0/ctdas/ilianna/1.Data/raw/MODIS_2000_2010.nc"
modis2 = "/projects/0/ctdas/ilianna/1.Data/raw/MODIS_2010_2025.nc"

outdir = f"/projects/0/ctdas/ilianna/xgb_cluster_{cluster_to_model}"
os.makedirs(outdir, exist_ok=True)

years_target = np.arange(2000, 2026)
train_years = np.arange(2000, 2020)
test_years = np.arange(2020, 2025)
evaluation_year = 2025

target_fire_months = [6, 7, 8]
res05 = 0.5

iberia_lon_min, iberia_lon_max = -10.5, 4.5
iberia_lat_min, iberia_lat_max = 36.5, 44.5

# KMeans settings
k = 3
target_cluster = 1
keep_largest_component = True

random_state = 42

# Meteorology ONLY
era5_met_vars = ["t2m", "d2m", "tp", "windspeed"]

# Search 40 randomly sampled hyperparameter combinations.
n_parameter_candidates = 40

# Non-overlapping, chronological validation blocks.
validation_folds = [
    (np.arange(2000, 2014), np.arange(2014, 2016)),
    (np.arange(2000, 2016), np.arange(2016, 2018)),
    (np.arange(2000, 2018), np.arange(2018, 2020)),
]

selected_sib_vars = []
lagged_sib_predictors = []
needed_sib_vars = []


def mask_fill_values(da):
    fill = da.attrs.get("_FillValue", None)
    if fill is not None:
        da = da.where(da != fill)
    missing_value = da.attrs.get("missing_value", None)
    if missing_value is not None:
        da = da.where(da != missing_value)
    return da


def get_time_name(da):
    if "valid_time" in da.coords:
        return "valid_time"
    if "time" in da.coords:
        return "time"
    raise ValueError("No time coordinate found.")


def seasonal_agg(da, months, agg="mean"):
    tname = get_time_name(da)
    da_s = da.sel({tname: da[tname].dt.month.isin(months)})
    if agg == "sum":
        out = da_s.groupby(da_s[tname].dt.year).sum(tname)
    else:
        out = da_s.groupby(da_s[tname].dt.year).mean(tname)
    out = out.rename({"year": "time"})
    out = out.assign_coords(time=pd.to_datetime(out["time"].values, format="%Y"))
    return out


def parse_label(label):
    parts = label.split("_")
    return "_".join(parts[:-1]), parts[-1]


def build_pft_field_from_nlu(ds, varname, target_refnum):
    da = mask_fill_values(ds[varname])
    mask = ds["lu_pref"] == target_refnum
    return (da * ds["lu_area"]).where(mask).sum("nlu", skipna=True)


def to_latlon_grid(da_land, ds_sib):
    lat = np.asarray(ds_sib["latsib"].data)
    lon = np.asarray(ds_sib["lonsib"].data)

    pairs = np.column_stack([lat, lon])
    _, unique_idx = np.unique(pairs, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)

    da_land = da_land.isel(landpoints=unique_idx)
    da_land = da_land.assign_coords(
        lat=("landpoints", lat[unique_idx]),
        lon=("landpoints", lon[unique_idx]),
    )

    return (
        da_land.set_index(landpoints=["lat", "lon"])
        .unstack("landpoints")
        .sortby("lat")
        .sortby("lon")
    )


def align_to_burn_grid(da):
    da2 = da.sel(lat=burn_05.lat, lon=burn_05.lon, method="nearest")
    da2 = da2.assign_coords(lat=burn_05.lat.values, lon=burn_05.lon.values)
    if "time" in da2.dims:
        return da2.transpose("time", "lat", "lon")
    return da2.transpose("lat", "lon")


def safe_2d(da):
    return (
        da.squeeze(drop=True)
        .assign_coords(lat=burn_05.lat.values, lon=burn_05.lon.values)
        .transpose("lat", "lon")
    )


def series_to_2d(ser, lat_vals, lon_vals, name):
    da = ser.to_xarray()
    da = da.reindex(lat=lat_vals, lon=lon_vals)
    return da.rename(name).transpose("lat", "lon")


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def setup_map(ax):
    proj = ccrs.PlateCarree()
    ax.set_extent(
        [iberia_lon_min, iberia_lon_max, iberia_lat_min, iberia_lat_max],
        crs=proj,
    )
    ax.add_feature(cfeature.LAND, facecolor="0.95")
    ax.add_feature(cfeature.OCEAN, facecolor="white")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.6)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.4)
    gl.top_labels = False
    gl.right_labels = False
    return proj


# ============================================================
# 1. MODIS BURNED AREA
# ============================================================

ds1 = xr.open_dataset(modis1)
ds2 = xr.open_dataset(modis2)

ds_modis = xr.concat([ds1, ds2], dim="time").sortby("time")
ds_modis = ds_modis.sel(time=~ds_modis.get_index("time").duplicated())

burn_da = ds_modis["BurnDate"].sel(
    lon=slice(iberia_lon_min, iberia_lon_max),
    lat=slice(iberia_lat_max, iberia_lat_min),
)

fill = burn_da.attrs.get("_FillValue", None)
if fill is not None:
    burn_da = burn_da.where(burn_da != fill)

dlat = float(abs(burn_da.lat.diff("lat").median()))
dlon = float(abs(burn_da.lon.diff("lon").median()))

R = 6371000.0
deg2rad = np.pi / 180.0
dy = R * deg2rad * dlat
dx = R * deg2rad * dlon * np.cos(np.deg2rad(burn_da.lat))
pixel_area_ha = (dx * dy) / 1e4

lat_edges = np.arange(iberia_lat_min, iberia_lat_max + res05, res05)
lon_edges = np.arange(iberia_lon_min, iberia_lon_max + res05, res05)

lat_centers = lat_edges[:-1] + res05 / 2
lon_centers = lon_edges[:-1] + res05 / 2

burn_maps_05 = []

for year in years_target:
    jja = burn_da.sel(
        time=(
            (burn_da.time.dt.year == year)
            & (burn_da.time.dt.month.isin(target_fire_months))
        )
    )

    if jja.time.size == 0:
        continue

    burned_any = (jja > 0).any("time")
    burned_area_native_ha = burned_any * pixel_area_ha

    ba_lon = burned_area_native_ha.groupby_bins(
        "lon", lon_edges, labels=lon_centers
    ).sum()

    ba05 = ba_lon.groupby_bins(
        "lat", lat_edges, labels=lat_centers
    ).sum()

    ba05 = ba05.rename({"lat_bins": "lat", "lon_bins": "lon"}).transpose("lat", "lon")

    burn_maps_05.append(
        ba05.assign_coords(time=pd.Timestamp(f"{year}-01-01")).expand_dims("time")
    )

burn_05 = xr.concat(burn_maps_05, dim="time").sortby("lat").sortby("lon")


# ============================================================
# 2. FIRE-REGIME CLUSTERS (Clustered on 2000-2019)
# ============================================================

burn_log = np.log1p(burn_05)

# Slice burn_log to train_years (2000-2019) for KMeans clustering
burn_log_train = burn_log.sel(time=burn_log.time.dt.year.isin(train_years))

X_cluster = burn_log_train.stack(cell=("lat", "lon")).transpose("cell", "time")
valid = X_cluster.sum("time") > 0
X_valid = X_cluster[valid]

data_std = StandardScaler().fit_transform(X_valid.values)

labels = KMeans(
    n_clusters=k,
    random_state=0,
    n_init=20,
).fit_predict(data_std)

cluster_1d = xr.DataArray(
    labels,
    dims=("cell",),
    coords={"cell": X_valid.cell},
    name="cluster",
)

cluster_raw = cluster_1d.unstack("cell").sortby("lat").sortby("lon")

cluster_full = xr.full_like(burn_05.isel(time=0), np.nan, dtype=float)
cluster_full.loc[dict(lat=cluster_raw.lat, lon=cluster_raw.lon)] = cluster_raw.astype(float)

cluster_vals = cluster_full.values.copy()

target_mask = np.isfinite(cluster_vals) & (cluster_vals == target_cluster)

labeled, ncomp = ndimage.label(target_mask, structure=np.ones((3, 3), dtype=int))
remove_mask = np.zeros_like(target_mask, dtype=bool)

if ncomp > 1 and keep_largest_component:
    sizes = ndimage.sum(target_mask, labeled, index=np.arange(1, ncomp + 1))
    keep_label = int(np.argmax(sizes) + 1)
    remove_mask = (labeled > 0) & (labeled != keep_label)

new_cluster = cluster_vals.copy()
ny, nx = new_cluster.shape

for i, j in zip(*np.where(remove_mask)):
    neighbors = []
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue
            ii, jj = i + di, j + dj
            if 0 <= ii < ny and 0 <= jj < nx:
                val = new_cluster[ii, jj]
                if np.isfinite(val) and val != target_cluster:
                    neighbors.append(int(val))

    if neighbors:
        counts = np.bincount(neighbors, minlength=k)
        new_cluster[i, j] = np.argmax(counts)
    else:
        new_cluster[i, j] = np.nan

cluster = xr.DataArray(
    new_cluster,
    dims=("lat", "lon"),
    coords={"lat": burn_05.lat.values, "lon": burn_05.lon.values},
    name="cluster",
)


# ============================================================
# 3. ERA5 METEOROLOGY ONLY
# ============================================================

ds_era = xr.open_dataset(era_path)

if (
    "longitude" in ds_era.coords
    and float(ds_era.longitude.max()) > 180
):
    ds_era = ds_era.assign_coords(
        longitude=((ds_era.longitude + 180) % 360) - 180
    ).sortby("longitude")

ds_era = ds_era.sel(
    longitude=slice(iberia_lon_min, iberia_lon_max),
    latitude=slice(iberia_lat_max, iberia_lat_min),
)

ds_era["windspeed"] = np.sqrt(
    ds_era["u10"] ** 2 + ds_era["v10"] ** 2
)

met_yearly = {}

for met_var in era5_met_vars:
    met = ds_era[met_var]
    agg = "sum" if met_var == "tp" else "mean"

    met_year = seasonal_agg(
        met,
        months=target_fire_months,
        agg=agg,
    )

    met_year = met_year.rename(
        {"latitude": "lat", "longitude": "lon"}
    ).sortby("lat").sortby("lon")

    met_year = met_year.sel(
        time=met_year.time.dt.year.isin(years_target)
    )

    met_yearly[met_var] = align_to_burn_grid(met_year)

available_met_vars = list(met_yearly.keys())
print("Meteorological variables:", available_met_vars)


# ============================================================
# 4. BUILD PIXEL TABLE: METEOROLOGY ONLY
# ============================================================

def build_pixel_table(only_years):
    rows = []

    for t in burn_05.time.values:
        year = pd.Timestamp(t).year

        if year not in only_years:
            continue

        data_vars = {
            "burned_area": burn_05.sel(time=t).squeeze(drop=True),
            "cluster": cluster,
        }

        for mv, mda in met_yearly.items():
            matching_years = mda.time.dt.year.values

            if year in matching_years:
                data_vars[mv] = safe_2d(
                    mda.sel(time=mda.time.dt.year == year).isel(time=0)
                )

        missing = [
            v for v in available_met_vars
            if v not in data_vars
        ]

        if missing:
            print(f"Skipping {year}: missing {missing}")
            continue

        df_year = (
            xr.Dataset(data_vars)
            .stack(cell=("lat", "lon"))
            .to_dataframe()
            .reset_index()
        )

        df_year["year"] = year
        rows.append(df_year)

    if not rows:
        raise ValueError(
            f"No pixel data found for years {sorted(only_years)}."
        )

    df = pd.concat(rows, ignore_index=True)

    df = df[
        ["year", "lat", "lon", "cluster", "burned_area"]
        + available_met_vars
    ]

    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    df = df[np.isfinite(df["cluster"])].copy()
    df["cluster"] = df["cluster"].astype(int)

    df = df[df["cluster"] == cluster_to_model].copy()

    if df.empty:
        raise ValueError(
            f"No valid pixels found for cluster {cluster_to_model}. "
            "Check k, cluster_to_model and the spatial mask."
        )

    return df


# ============================================================
# 5. BUILD TRAIN / TEST / 2025 DATASETS
# ============================================================

df_train = build_pixel_table(set(train_years))
df_test = build_pixel_table(set(test_years))
df_eval_2025 = build_pixel_table({evaluation_year})

feature_cols = available_met_vars.copy()

print("\nMeteorological features:")
for f in feature_cols:
    print(" ", f)

print("\nNumber of rows:")
print("Train:", len(df_train))
print("Test:", len(df_test))
print("2025:", len(df_eval_2025))

X_train = df_train[feature_cols]
y_train = df_train["burned_area"]

X_test = df_test[feature_cols]
y_test = df_test["burned_area"]

X_2025 = df_eval_2025[feature_cols]
y_2025 = df_eval_2025["burned_area"]


# ============================================================
# 6. XGBOOST HYPERPARAMETER TUNING (Best Annual R^2 Selection)
# ============================================================

param_distributions = {
    "n_estimators": [200, 400, 600, 800, 1000],
    "max_depth": [2, 3, 4, 5, 6, 8],
    "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08, 0.1],
    "min_child_weight": [1, 3, 5, 7, 10],
    "subsample": [0.6, 0.75, 0.85, 1.0],
    "colsample_bytree": [0.6, 0.75, 0.85, 1.0],
    "gamma": [0, 0.05, 0.1, 0.2, 0.5],
    "reg_alpha": [0, 0.001, 0.01, 0.1, 1.0],
    "reg_lambda": [1, 2, 5, 10, 20],
}

candidates = list(
    ParameterSampler(
        param_distributions,
        n_iter=n_parameter_candidates,
        random_state=random_state,
    )
)

max_rows_per_fold = 150_000


def sample_rows_by_year(frame, max_rows, seed):
    if len(frame) <= max_rows:
        return frame

    years = np.sort(frame["year"].unique())
    quota = max(1, max_rows // len(years))
    parts = []

    for year in years:
        part = frame[frame["year"] == year]
        n = min(len(part), quota)
        parts.append(part.sample(n=n, random_state=seed))

    sampled = pd.concat(parts, ignore_index=True)

    if len(sampled) > max_rows:
        sampled = sampled.sample(n=max_rows, random_state=seed)

    return sampled


def make_xgb(params):
    return XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        random_state=random_state,
        n_jobs=1,
        verbosity=0,
        **params,
    )


search_results = []

for candidate_idx, params in enumerate(candidates, start=1):
    fold_scores = []

    print(f"\nCandidate {candidate_idx}/{len(candidates)}: {params}")

    for fold_idx, (fit_years, valid_years) in enumerate(validation_folds, start=1):
        fit_df = df_train[df_train["year"].isin(fit_years)]
        valid_df = df_train[df_train["year"].isin(valid_years)].copy()

        if fit_df.empty or valid_df.empty:
            raise ValueError(f"Empty data in validation fold {fold_idx}.")

        fit_sample = sample_rows_by_year(
            fit_df,
            max_rows_per_fold,
            random_state + fold_idx,
        )

        model = make_xgb(params)
        model.fit(
            fit_sample[feature_cols],
            fit_sample["burned_area"],
            verbose=False,
        )

        valid_df["pred"] = np.clip(
            model.predict(valid_df[feature_cols]), 0, None
        )

        # Evaluate performance based on annual total R^2
        annual_obs = valid_df.groupby("year")["burned_area"].sum()
        annual_pred = valid_df.groupby("year")["pred"].sum()
        score = r2_score(annual_obs, annual_pred)

        fold_scores.append(score)

        print(
            f"   Fold {fold_idx}: validation years {valid_years[0]}-{valid_years[-1]}, Annual R2={score:.5f}"
        )

    mean_score = float(np.mean(fold_scores))
    std_score = float(np.std(fold_scores))

    search_results.append({
        "candidate": candidate_idx,
        "mean_validation_annual_r2": mean_score,
        "std_validation_annual_r2": std_score,
        "fold_annual_r2": fold_scores,
        "params": params,
    })

    print(f"   Mean validation annual R2 = {mean_score:.5f}")


# Sort to select the candidate with highest mean annual validation R2
search_results.sort(
    key=lambda x: x["mean_validation_annual_r2"],
    reverse=True,
)

best_result = search_results[0]
best_params = best_result["params"]

print("\n============================================")
print("BEST XGBOOST HYPERPARAMETERS (BY ANNUAL R^2)")
print("============================================")
print(best_params)
print("Best mean validation annual R2:", best_result["mean_validation_annual_r2"])
print("Validation fold standard deviation:", best_result["std_validation_annual_r2"])

tuning_df = pd.DataFrame([
    {
        "candidate": r["candidate"],
        "mean_validation_annual_r2": r["mean_validation_annual_r2"],
        "std_validation_annual_r2": r["std_validation_annual_r2"],
        **r["params"],
    }
    for r in search_results
])

tuning_df.to_csv(
    os.path.join(outdir, "xgboost_hyperparameter_search.csv"),
    index=False,
)

pd.DataFrame([{
    **best_params,
    "best_mean_validation_annual_r2": best_result["mean_validation_annual_r2"],
    "validation_annual_r2_std": best_result["std_validation_annual_r2"],
}]).to_csv(
    os.path.join(outdir, "best_xgboost_parameters.csv"),
    index=False,
)


# ============================================================
# 7. FIT FINAL MODEL ON ALL YEARS 2000-2019
# ============================================================

xgb_model = make_xgb(best_params)

xgb_model.fit(
    X_train,
    y_train,
    verbose=False,
)

df_train["predicted"] = np.clip(xgb_model.predict(X_train), 0, None)
df_test["predicted"] = np.clip(xgb_model.predict(X_test), 0, None)
df_eval_2025["predicted"] = np.clip(xgb_model.predict(X_2025), 0, None)

for frame in [df_train, df_test, df_eval_2025]:
    frame["difference"] = frame["predicted"] - frame["burned_area"]


# ============================================================
# 8. EVALUATION METRICS
# ============================================================

df_all = pd.concat([df_train, df_test, df_eval_2025], ignore_index=True)
annual_totals = df_all.groupby("year")[["burned_area", "predicted"]].sum()

train_ann = annual_totals.loc[annual_totals.index.isin(train_years)]
test_ann = annual_totals.loc[annual_totals.index.isin(test_years)]

metrics = {
    "cluster": cluster_to_model,
    "features": ", ".join(feature_cols),

    "pixel_train_r2": r2_score(df_train["burned_area"], df_train["predicted"]),
    "pixel_test_r2": r2_score(df_test["burned_area"], df_test["predicted"]),
    "pixel_eval_2025_r2": r2_score(df_eval_2025["burned_area"], df_eval_2025["predicted"]),

    "annual_train_r2": r2_score(train_ann["burned_area"], train_ann["predicted"]),
    "annual_test_r2": r2_score(test_ann["burned_area"], test_ann["predicted"]),
    "annual_overall_r2": r2_score(annual_totals["burned_area"], annual_totals["predicted"]),

    "train_rmse": rmse(df_train["burned_area"], df_train["predicted"]),
    "test_rmse": rmse(df_test["burned_area"], df_test["predicted"]),
    "eval_2025_rmse": rmse(df_eval_2025["burned_area"], df_eval_2025["predicted"]),

    "observed_total_2025": float(df_eval_2025["burned_area"].sum()),
    "predicted_total_2025": float(df_eval_2025["predicted"].sum()),
    "best_mean_validation_annual_r2": best_result["mean_validation_annual_r2"],
}

metrics_df = pd.DataFrame([metrics])
metrics_df.to_csv(os.path.join(outdir, "metrics.csv"), index=False)

print("\nFINAL METRICS")
print(metrics_df.T)


# ============================================================
# 9. XGBOOST FEATURE IMPORTANCE
# ============================================================

fi = pd.DataFrame({
    "feature": feature_cols,
    "importance": xgb_model.feature_importances_,
}).sort_values("importance", ascending=False)

fi.to_csv(os.path.join(outdir, "feature_importance.csv"), index=False)

plt.figure(figsize=(8, 5))
plot_fi = fi.sort_values("importance")
plt.barh(plot_fi["feature"], plot_fi["importance"])
plt.xlabel("XGBoost feature importance")
plt.title(f"Cluster {cluster_to_model}: meteorology-only XGBoost")
plt.tight_layout()
plt.savefig(os.path.join(outdir, "feature_importance.png"), dpi=200)
plt.close()


# ============================================================
# 10. PERMUTATION IMPORTANCE ON TEST DATA
# ============================================================

perm = permutation_importance(
    xgb_model,
    X_test,
    y_test,
    n_repeats=5,
    random_state=random_state,
    n_jobs=1,
    scoring="r2",
)

perm_df = pd.DataFrame({
    "feature": feature_cols,
    "importance_mean": perm.importances_mean,
    "importance_std": perm.importances_std,
}).sort_values("importance_mean", ascending=False)

perm_df.to_csv(os.path.join(outdir, "permutation_importance_test.csv"), index=False)

plt.figure(figsize=(8, 5))
plot_perm = perm_df.sort_values("importance_mean")
plt.barh(
    plot_perm["feature"],
    plot_perm["importance_mean"],
    xerr=plot_perm["importance_std"],
)
plt.xlabel("Decrease in test R2 after permutation")
plt.title(f"Cluster {cluster_to_model}: test permutation importance")
plt.tight_layout()
plt.savefig(os.path.join(outdir, "permutation_importance.png"), dpi=200)
plt.close()


# ============================================================
# 11. SAVE PREDICTION TABLES
# ============================================================

df_train.to_csv(os.path.join(outdir, "training_table_2000_2019.csv"), index=False)
df_test.to_csv(os.path.join(outdir, "test_table_2020_2024.csv"), index=False)
df_eval_2025.to_csv(os.path.join(outdir, "evaluation_2025.csv"), index=False)


# ============================================================
# 12. ANNUAL TOTAL BURNED AREA RECONSTRUCTION PLOT
# ============================================================

fig, ax = plt.subplots(figsize=(10, 5))

# Plot time series
ax.plot(
    annual_totals.index,
    annual_totals["burned_area"],
    marker="o",
    color="#1f77b4",
    linewidth=1.5,
    markersize=4,
    label="Observed total",
)
ax.plot(
    annual_totals.index,
    annual_totals["predicted"],
    marker="s",
    color="#ff7f0e",
    linewidth=1.5,
    markersize=4,
    label="Predicted total",
)

# Highlight Test period (2020-2024) and 2025 evaluation
ax.axvspan(2019.5, 2024.5, color="#dce8f5", alpha=0.6, label="Test period")
ax.axvspan(2024.5, 2025.5, color="#b8d3ec", alpha=0.8, label="2025 evaluation")

ax.set_title(
    f"Cluster {cluster_to_model}: total burned area reconstruction",
    fontsize=13,
)
ax.set_xlabel("Year", fontsize=11)
ax.set_ylabel("Total burned area (ha)", fontsize=11)

ax.set_xlim(1999.5, 2025.5)
ax.set_xticks(np.arange(2000, 2026, 5))
ax.set_ylim(bottom=0)

ax.grid(True, linestyle="-", linewidth=0.5, alpha=0.3)
ax.legend(loc="upper left", frameon=True)

plt.tight_layout()
plt.savefig(
    os.path.join(outdir, f"annual_reconstruction_cluster_{cluster_to_model}.png"),
    dpi=300,
)
plt.close()

print("\nDone.")
print("Output directory:", outdir)