# ============================================================
# XGBOOST BASELINE MODEL FOR ONE FIRE-REGIME CLUSTER
# ERA5 METEOROLOGICAL PREDICTORS
#
# Baseline predictors: t2m, d2m, tp, windspeed
# Fire-regime clustering period: 2001-2019
# Training:                  2001-2019
# Independent test:          2020-2024
# Final evaluation:          2025 (not used for selection)
#
# Model selection criterion: annual-total test R² only.
# Pixel-level and fire-class metrics are supplementary.
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)
from sklearn.model_selection import ParameterSampler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# 0. CONFIGURATION
# ============================================================

ERA5_FILE = "/projects/0/ctdas/ilianna/1.Data/raw/era5_merged.nc"
MODIS_FILES = [
    "/projects/0/ctdas/ilianna/1.Data/raw/MODIS_2000_2010.nc",
    "/projects/0/ctdas/ilianna/1.Data/raw/MODIS_2010_2025.nc",
]
OUTPUT_ROOT = "/projects/0/ctdas/ilianna/xgb_baseline_cluster_2001_2019"

CLUSTER_START = 2001
CLUSTER_END = 2019
TRAIN_START = 2001
TRAIN_END = 2019
TEST_START = 2020
TEST_END = 2024
EVALUATION_YEAR = 2025
YEARS_TARGET = np.arange(CLUSTER_START, EVALUATION_YEAR + 1)

# Iberian Peninsula
LON_MIN, LON_MAX = -10.5, 4.5
LAT_MIN, LAT_MAX = 36.5, 44.5
GRID_RESOLUTION = 0.5

# Fire-regime clustering
N_CLUSTERS = 3
CLUSTER_TO_MODEL = 1
RANDOM_STATE = 0

# Model settings
USE_LOG_TARGET = False
XGB_N_JOBS = 1
N_ITER_SEARCH = 20

# Expanding-window validation blocks. Each fold trains only on years
# before its validation block; all validation years are within 2001-2019.
CV_BLOCKS = [
    (2010, 2012),
    (2013, 2015),
    (2016, 2019),
]

LOW_PERCENTILE = 25
HIGH_PERCENTILE = 75
LOW_WEIGHT_OPTIONS = [0.5, 0.75, 1.0]
HIGH_WEIGHT_OPTIONS = [1.0, 1.5, 2.0, 3.0]

BASELINE_FEATURES = ["t2m", "d2m", "tp", "windspeed"]

PARAM_DISTRIBUTIONS = {
    "n_estimators": [100, 200, 350, 500],
    "max_depth": [1, 2, 3, 4],
    "learning_rate": [0.02, 0.03, 0.05, 0.08],
    "min_child_weight": [3, 5, 8, 12],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.7, 0.85, 1.0],
    "reg_alpha": [0.05, 0.5, 2.0, 5.0],
    "reg_lambda": [5.0, 10.0, 20.0, 30.0],
    "gamma": [0.05, 0.2, 0.5, 1.0],
}

os.makedirs(OUTPUT_ROOT, exist_ok=True)


# ============================================================
# 1. HELPERS
# ============================================================

def find_name(ds, candidates, required=True):
    """Find a variable or coordinate using candidate names."""
    lookup = {str(name).lower(): name for name in list(ds.variables)}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    if required:
        raise KeyError(
            f"Could not find any of {candidates}. Available: {list(ds.variables)}"
        )
    return None


def standardize_lat_lon(ds):
    """Standardize common latitude/longitude names and sort ascending."""
    rename = {}
    lat_name = find_name(ds, ["latitude", "lat", "nav_lat"], required=False)
    lon_name = find_name(ds, ["longitude", "lon", "nav_lon"], required=False)
    if lat_name is not None and lat_name != "latitude":
        rename[lat_name] = "latitude"
    if lon_name is not None and lon_name != "longitude":
        rename[lon_name] = "longitude"
    if rename:
        ds = ds.rename(rename)
    if "latitude" not in ds.coords or "longitude" not in ds.coords:
        raise ValueError("Latitude/longitude coordinates were not found.")
    if ds.latitude.size > 1 and ds.latitude.values[0] > ds.latitude.values[-1]:
        ds = ds.sortby("latitude")
    if ds.longitude.size > 1 and ds.longitude.values[0] > ds.longitude.values[-1]:
        ds = ds.sortby("longitude")
    return ds


def select_iberia(ds):
    ds = standardize_lat_lon(ds)
    ds = ds.sel(
        latitude=slice(LAT_MIN, LAT_MAX),
        longitude=slice(LON_MIN, LON_MAX),
    )
    if ds.sizes.get("latitude", 0) == 0 or ds.sizes.get("longitude", 0) == 0:
        raise ValueError("The Iberian Peninsula subset is empty.")
    return ds


def get_year_values(time_coord):
    return pd.DatetimeIndex(pd.to_datetime(time_coord.values)).year.to_numpy()


def calculate_vpd_kpa(t2m_kelvin, d2m_kelvin):
    """VPD in kPa; retained as a helper although baseline does not use VPD."""
    t_c = t2m_kelvin - 273.15
    td_c = d2m_kelvin - 273.15
    es = 0.6108 * np.exp((17.27 * t_c) / (t_c + 237.3))
    ea = 0.6108 * np.exp((17.27 * td_c) / (td_c + 237.3))
    return np.maximum(es - ea, 0.0)


def annual_metrics_table(df):
    """Aggregate pixel predictions to annual totals."""
    rows = []
    for year, group in df.groupby("year"):
        obs = float(group["observed"].sum())
        pred = float(group["predicted"].sum())
        diff = pred - obs
        rows.append({
            "year": int(year),
            "observed": obs,
            "predicted": pred,
            "difference": diff,
            "absolute_error": abs(diff),
            "bias_percent": 100.0 * diff / obs if obs != 0 else np.nan,
        })
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


def annual_total_r2(df):
    annual = annual_metrics_table(df)
    if len(annual) < 2 or annual["observed"].std() == 0:
        return np.nan
    return float(r2_score(annual["observed"], annual["predicted"]))


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[valid], y_pred[valid]
    if len(y_true) == 0:
        return {"pixel_r2": np.nan, "pixel_rmse": np.nan, "pixel_mae": np.nan}
    return {
        "pixel_r2": r2_score(y_true, y_pred)
        if len(y_true) > 1 and np.std(y_true) > 0 else np.nan,
        "pixel_rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "pixel_mae": float(mean_absolute_error(y_true, y_pred)),
    }


def get_year_weights(years, annual_totals, low_threshold, high_threshold,
                     low_weight, high_weight):
    """Assign sample weights using training-period annual fire totals."""
    years = np.asarray(years, dtype=int)
    weights = np.ones(len(years), dtype=np.float64)
    for i, year in enumerate(years):
        total = annual_totals.get(int(year), np.nan)
        if np.isfinite(total) and total <= low_threshold:
            weights[i] = low_weight
        elif np.isfinite(total) and total >= high_threshold:
            weights[i] = high_weight
    weights /= weights.mean()
    return weights


def make_model(params):
    return XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=XGB_N_JOBS,
        **params,
    )


def fit_predict_model(X_train, y_train, weights, X_predict, params):
    y_fit = np.log1p(np.maximum(y_train, 0)) if USE_LOG_TARGET else y_train
    model = make_model(params)
    model.fit(X_train, y_fit, sample_weight=weights)
    pred = model.predict(X_predict)
    if USE_LOG_TARGET:
        pred = np.expm1(pred)
    return model, np.maximum(pred, 0.0)


# ============================================================
# 2. LOAD MODIS AND BUILD ANNUAL JJA TARGET
# ============================================================

print("[1/7] Loading MODIS target...")
modis_datasets = []
for path in MODIS_FILES:
    if not os.path.exists(path):
        raise FileNotFoundError(f"MODIS file not found: {path}")
    modis_datasets.append(standardize_lat_lon(xr.open_dataset(path)))

modis = xr.combine_by_coords(modis_datasets, combine_attrs="override")
burn_var = find_name(modis, ["BurnDate", "burn_date", "burned_area", "BurnedArea"])
burn = select_iberia(modis[[burn_var]])[burn_var]

time_name = find_name(burn.to_dataset(name="burn"), ["time", "year"], required=False)
if time_name is None:
    raise ValueError("MODIS target must have a time or year coordinate.")

if time_name == "year":
    modis_years = np.asarray(burn["year"].values, dtype=int)
else:
    modis_years = get_year_values(burn[time_name])

if burn_var.lower() in ["burndate", "burn_date"]:
    # Assumes annual day-of-year BurnDate: 0 = unburned.
    burned_jja = ((burn >= 152) & (burn <= 243)).astype(np.float32)
elif burn_var.lower() in ["burned_area", "burnedarea"]:
    burned_jja = burn.astype(np.float32)
else:
    raise ValueError(f"Unsupported MODIS target variable: {burn_var}")

lat_step = float(np.abs(np.diff(burn.latitude.values)).mean()) if burn.latitude.size > 1 else GRID_RESOLUTION
lon_step = float(np.abs(np.diff(burn.longitude.values)).mean()) if burn.longitude.size > 1 else GRID_RESOLUTION
lat_factor = max(1, int(round(GRID_RESOLUTION / lat_step)))
lon_factor = max(1, int(round(GRID_RESOLUTION / lon_step)))

if lat_factor > 1 or lon_factor > 1:
    burned_coarse = burned_jja.coarsen(
        latitude=lat_factor, longitude=lon_factor, boundary="trim"
    ).sum()
else:
    burned_coarse = burned_jja

if burn_var.lower() in ["burndate", "burn_date"]:
    lat_radians = np.deg2rad(burned_coarse.latitude)
    cell_area = xr.DataArray(
        (GRID_RESOLUTION * 111.0)
        * (GRID_RESOLUTION * 111.0 * np.cos(lat_radians.values))[:, None]
        * np.ones((1, burned_coarse.sizes["longitude"])),
        coords={"latitude": burned_coarse.latitude,
                "longitude": burned_coarse.longitude},
        dims=("latitude", "longitude"),
    )
    target = burned_coarse * cell_area
else:
    target = burned_coarse

target_by_year = {}
if "time" in target.dims:
    for year in YEARS_TARGET:
        idx = np.where(modis_years == year)[0]
        if len(idx):
            target_by_year[int(year)] = target.isel(time=idx).sum(dim="time")
elif "year" in target.dims:
    for year in YEARS_TARGET:
        if year in target["year"].values:
            target_by_year[int(year)] = target.sel(year=year)
else:
    raise ValueError("Could not identify a time/year dimension for MODIS.")

if not target_by_year:
    raise ValueError("No annual MODIS target fields were constructed.")
print("Target years:", sorted(target_by_year))


# ============================================================
# 3. LOAD ERA5 AND BUILD ANNUAL JJA PREDICTORS
# ============================================================

print("[2/7] Loading ERA5 predictors...")
if not os.path.exists(ERA5_FILE):
    raise FileNotFoundError(f"ERA5 file not found: {ERA5_FILE}")

era5 = select_iberia(xr.open_dataset(ERA5_FILE))
t2m_name = find_name(era5, ["t2m", "2m_temperature"])
d2m_name = find_name(era5, ["d2m", "2m_dewpoint_temperature"])
tp_name = find_name(era5, ["tp", "total_precipitation"])
u10_name = find_name(era5, ["u10", "10m_u_component_of_wind"])
v10_name = find_name(era5, ["v10", "10m_v_component_of_wind"])
time_name = find_name(era5, ["time", "valid_time"])

if time_name != "time":
    era5 = era5.rename({time_name: "time"})

t2m, d2m, tp = era5[t2m_name], era5[d2m_name], era5[tp_name]
u10, v10 = era5[u10_name], era5[v10_name]

era5_times = pd.DatetimeIndex(pd.to_datetime(era5.time.values))
jja_mask = era5_times.month.isin([6, 7, 8])
if not jja_mask.any():
    raise ValueError("No June-July-August ERA5 records were found.")

idx_jja = np.where(jja_mask)[0]
jja_years = era5_times[jja_mask].year.to_numpy()
t2m_jja = t2m.isel(time=idx_jja)
d2m_jja = d2m.isel(time=idx_jja)
tp_jja = tp.isel(time=idx_jja)
wind_jja = np.sqrt(u10.isel(time=idx_jja) ** 2 + v10.isel(time=idx_jja) ** 2)

predictor_by_year = {}
for year in YEARS_TARGET:
    idx = np.where(jja_years == year)[0]
    if not len(idx):
        continue
    predictor_by_year[int(year)] = {
        "t2m": t2m_jja.isel(time=idx).mean("time"),
        "d2m": d2m_jja.isel(time=idx).mean("time"),
        "tp": tp_jja.isel(time=idx).sum("time") * 1000.0,
        "windspeed": wind_jja.isel(time=idx).mean("time"),
    }
print("ERA5 predictor years:", sorted(predictor_by_year))


# ============================================================
# 4. ALIGN GRIDS, BUILD SAMPLES, AND CLUSTER USING 2001-2019 ONLY
# ============================================================

print("[3/7] Aligning data and clustering fire regimes...")
common_years = sorted(set(target_by_year) & set(predictor_by_year))
if not common_years:
    raise ValueError("No overlapping years between MODIS and ERA5.")

reference = target_by_year[common_years[0]]
ref_lat, ref_lon = reference.latitude.values, reference.longitude.values
lat_grid, lon_grid = np.meshgrid(ref_lat, ref_lon, indexing="ij")
n_cells = lat_grid.size
records = []

for year in common_years:
    y_field = target_by_year[year]
    if not np.array_equal(y_field.latitude.values, ref_lat) or not np.array_equal(y_field.longitude.values, ref_lon):
        y_field = y_field.interp(latitude=ref_lat, longitude=ref_lon, method="nearest")

    pred_fields = {
        name: np.asarray(field.interp(
            latitude=ref_lat, longitude=ref_lon, method="nearest"
        ).values).reshape(-1)
        for name, field in predictor_by_year[year].items()
    }
    y_values = np.asarray(y_field.values).reshape(-1)

    for cell_id in range(n_cells):
        row = {
            "year": int(year),
            "cell_id": int(cell_id),
            "latitude": float(lat_grid.reshape(-1)[cell_id]),
            "longitude": float(lon_grid.reshape(-1)[cell_id]),
            "observed": float(y_values[cell_id]),
        }
        for name, values in pred_fields.items():
            row[name] = float(values[cell_id])
        records.append(row)

data = pd.DataFrame(records).replace([np.inf, -np.inf], np.nan)
data = data.dropna(subset=["observed"] + BASELINE_FEATURES).copy()
data["observed"] = data["observed"].clip(lower=0.0)
if data.empty:
    raise ValueError("No valid pixel-year samples remain.")

# IMPORTANT: the cluster means are calculated from 2001-2019 only.
cluster_period_data = data[data["year"].between(CLUSTER_START, CLUSTER_END)]
if cluster_period_data.empty:
    raise ValueError("No samples available for the 2001-2019 clustering period.")

cell_mean = (
    cluster_period_data.groupby("cell_id")["observed"]
    .mean()
    .reindex(np.arange(n_cells))
)
cell_mean_filled = cell_mean.fillna(cell_mean.median()).to_numpy()
if not np.isfinite(cell_mean_filled).all():
    raise ValueError("Could not construct finite clustering input.")

kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=20)
cluster_labels = kmeans.fit_predict(cell_mean_filled.reshape(-1, 1))

cluster_table = pd.DataFrame({
    "cell_id": np.arange(n_cells),
    "cluster": cluster_labels,
    "latitude": lat_grid.reshape(-1),
    "longitude": lon_grid.reshape(-1),
    "mean_annual_burned_area_2001_2019": cell_mean_filled,
})
cluster_table.to_csv(os.path.join(OUTPUT_ROOT, "fire_regime_clusters_2001_2019.csv"), index=False)
data = data.merge(cluster_table[["cell_id", "cluster"]], on="cell_id", how="left")
data = data[data["cluster"] == CLUSTER_TO_MODEL].copy()
if data.empty:
    raise ValueError(f"No samples found for cluster {CLUSTER_TO_MODEL}.")

print("Selected cluster:", CLUSTER_TO_MODEL)
print("Pixel-year samples:", len(data))
print("Grid cells:", data["cell_id"].nunique())


# ============================================================
# 5. DEFINE PERIODS AND TUNE THE BASELINE MODEL
# ============================================================

print("[4/7] Defining train/test/evaluation periods...")
train_data = data[data["year"].between(TRAIN_START, TRAIN_END)].copy()
test_data = data[data["year"].between(TEST_START, TEST_END)].copy()
evaluation_data = data[data["year"] == EVALUATION_YEAR].copy()

if train_data.empty or test_data.empty or evaluation_data.empty:
    raise ValueError("Train, test, or 2025 evaluation data are empty. Check data coverage.")

train_annual = train_data.groupby("year")["observed"].sum().sort_index()
low_threshold = float(np.percentile(train_annual.values, LOW_PERCENTILE))
high_threshold = float(np.percentile(train_annual.values, HIGH_PERCENTILE))
train_annual_dict = train_annual.to_dict()

print(f"Training annual-total P{LOW_PERCENTILE}: {low_threshold:.4f}")
print(f"Training annual-total P{HIGH_PERCENTILE}: {high_threshold:.4f}")


def cv_annual_r2(params, low_weight, high_weight):
    """Expanding-window CV score based on annual-total R²."""
    all_obs, all_pred = [], []
    for val_start, val_end in CV_BLOCKS:
        fold_train = train_data[train_data["year"] < val_start].copy()
        fold_val = train_data[train_data["year"].between(val_start, val_end)].copy()
        if fold_train.empty or fold_val.empty:
            continue

        fold_annual = fold_train.groupby("year")["observed"].sum().to_dict()
        weights = get_year_weights(
            fold_train["year"].to_numpy(), fold_annual,
            low_threshold, high_threshold, low_weight, high_weight,
        )
        _, pred = fit_predict_model(
            fold_train[BASELINE_FEATURES].to_numpy(float),
            fold_train["observed"].to_numpy(float),
            weights,
            fold_val[BASELINE_FEATURES].to_numpy(float),
            params,
        )
        fold_result = fold_val[["year", "observed"]].copy()
        fold_result["predicted"] = pred
        annual = fold_result.groupby("year").agg(
            observed=("observed", "sum"), predicted=("predicted", "sum")
        )
        all_obs.extend(annual["observed"].to_list())
        all_pred.extend(annual["predicted"].to_list())

    if len(all_obs) < 2 or np.std(all_obs) == 0:
        return np.nan
    return float(r2_score(all_obs, all_pred))


print("[5/7] Hyperparameter and sample-weight search...")
sampled_params = list(ParameterSampler(
    PARAM_DISTRIBUTIONS, n_iter=N_ITER_SEARCH, random_state=RANDOM_STATE
))
best_score, best_params = -np.inf, None
best_low_weight, best_high_weight = None, None
search_rows = []

for params in sampled_params:
    for low_weight in LOW_WEIGHT_OPTIONS:
        for high_weight in HIGH_WEIGHT_OPTIONS:
            score = cv_annual_r2(params, low_weight, high_weight)
            search_rows.append({
                "cv_annual_total_r2": score,
                "low_weight": low_weight,
                "high_weight": high_weight,
                "params": str(params),
            })
            if np.isfinite(score) and score > best_score:
                best_score = score
                best_params = params
                best_low_weight = low_weight
                best_high_weight = high_weight

if best_params is None:
    raise RuntimeError("No valid CV score found. Check CV blocks and training coverage.")

pd.DataFrame(search_rows).to_csv(
    os.path.join(OUTPUT_ROOT, "baseline_hyperparameter_search.csv"), index=False
)
print("Best CV annual-total R²:", best_score)
print("Best parameters:", best_params)
print("Selected low/high weights:", best_low_weight, best_high_weight)

weights = get_year_weights(
    train_data["year"].to_numpy(), train_annual_dict,
    low_threshold, high_threshold, best_low_weight, best_high_weight,
)
model, test_pred = fit_predict_model(
    train_data[BASELINE_FEATURES].to_numpy(float),
    train_data["observed"].to_numpy(float),
    weights,
    test_data[BASELINE_FEATURES].to_numpy(float),
    best_params,
)

test_predictions = test_data[
    ["year", "cell_id", "latitude", "longitude", "observed"]
].copy()
test_predictions["predicted"] = test_pred
test_predictions.to_csv(
    os.path.join(OUTPUT_ROOT, "baseline_test_pixel_predictions_2020_2024.csv"),
    index=False,
)
test_annual = annual_metrics_table(test_predictions)
test_annual.to_csv(
    os.path.join(OUTPUT_ROOT, "baseline_test_annual_predictions_2020_2024.csv"),
    index=False,
)
test_r2 = annual_total_r2(test_predictions)
print("Independent test annual-total R²:", test_r2)


# ============================================================
# 6. FINAL EVALUATION ON 2025 (NOT USED FOR MODEL SELECTION)
# ============================================================

print("[6/7] Evaluating the selected baseline on 2025...")
eval_pred = model.predict(evaluation_data[BASELINE_FEATURES].to_numpy(float))
if USE_LOG_TARGET:
    eval_pred = np.expm1(eval_pred)
eval_pred = np.maximum(eval_pred, 0.0)

evaluation_predictions = evaluation_data[
    ["year", "cell_id", "latitude", "longitude", "observed"]
].copy()
evaluation_predictions["predicted"] = eval_pred
evaluation_predictions.to_csv(
    os.path.join(OUTPUT_ROOT, "baseline_evaluation_pixel_predictions_2025.csv"),
    index=False,
)
evaluation_annual = annual_metrics_table(evaluation_predictions)
evaluation_annual.to_csv(
    os.path.join(OUTPUT_ROOT, "baseline_evaluation_annual_predictions_2025.csv"),
    index=False,
)

eval_obs = evaluation_annual["observed"].to_numpy()
eval_hat = evaluation_annual["predicted"].to_numpy()
eval_diff = eval_hat - eval_obs
eval_summary = {
    "configuration": "baseline",
    "features": ",".join(BASELINE_FEATURES),
    "cluster": CLUSTER_TO_MODEL,
    "cluster_period": f"{CLUSTER_START}-{CLUSTER_END}",
    "training_period": f"{TRAIN_START}-{TRAIN_END}",
    "test_period": f"{TEST_START}-{TEST_END}",
    "evaluation_year": EVALUATION_YEAR,
    "test_annual_total_r2": test_r2,
    "evaluation_annual_total_rmse": float(np.sqrt(np.mean(eval_diff ** 2))) if len(eval_diff) else np.nan,
    "evaluation_annual_total_mae": float(np.mean(np.abs(eval_diff))) if len(eval_diff) else np.nan,
    "evaluation_annual_total_bias": float(np.mean(eval_diff)) if len(eval_diff) else np.nan,
    **regression_metrics(evaluation_predictions["observed"], evaluation_predictions["predicted"]),
    "best_cv_annual_total_r2": best_score,
    "best_low_weight": best_low_weight,
    "best_high_weight": best_high_weight,
    "best_params": str(best_params),
}
pd.DataFrame([eval_summary]).to_csv(
    os.path.join(OUTPUT_ROOT, "baseline_evaluation_metrics_2025.csv"), index=False
)
print("2025 evaluation summary:")
for key, value in eval_summary.items():
    print(f"{key}: {value}")


# ============================================================
# 7. SAVE PLOTS, FEATURE IMPORTANCE, AND REPORT
# ============================================================

print("[7/7] Saving plots and report...")
plot_dir = os.path.join(OUTPUT_ROOT, "plots")
os.makedirs(plot_dir, exist_ok=True)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(test_annual["year"], test_annual["observed"], marker="o", label="Observed")
ax.plot(test_annual["year"], test_annual["predicted"], marker="o", label="Baseline predicted")
ax.set_title("Baseline annual JJA burned area (test: 2020-2024)")
ax.set_xlabel("Year")
ax.set_ylabel("Annual burned area (target units)")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(plot_dir, "baseline_annual_test_predictions.png"), dpi=200)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(test_annual["observed"], test_annual["predicted"], s=65)
if len(test_annual):
    lo = min(test_annual["observed"].min(), test_annual["predicted"].min())
    hi = max(test_annual["observed"].max(), test_annual["predicted"].max())
    ax.plot([lo, hi], [lo, hi], linestyle="--")
ax.set_xlabel("Observed annual burned area")
ax.set_ylabel("Predicted annual burned area")
ax.set_title(f"Baseline annual totals: test R² = {test_r2:.3f}")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(plot_dir, "baseline_annual_test_scatter.png"), dpi=200)
plt.close(fig)

importance = pd.DataFrame({
    "feature": BASELINE_FEATURES,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False)
importance.to_csv(os.path.join(OUTPUT_ROOT, "baseline_feature_importance.csv"), index=False)

fig, ax = plt.subplots(figsize=(7, 4))
imp_plot = importance.sort_values("importance")
ax.barh(imp_plot["feature"], imp_plot["importance"])
ax.set_xlabel("XGBoost feature importance")
ax.set_title("Baseline feature importance")
fig.tight_layout()
fig.savefig(os.path.join(plot_dir, "baseline_feature_importance.png"), dpi=200)
plt.close(fig)

report_path = os.path.join(OUTPUT_ROOT, "BASELINE_MODEL_REPORT.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("XGBOOST BASELINE FIRE-REGIME MODEL\n")
    f.write("=" * 50 + "\n")
    f.write(f"Clustering period: {CLUSTER_START}-{CLUSTER_END}\n")
    f.write(f"Training period: {TRAIN_START}-{TRAIN_END}\n")
    f.write(f"Independent test: {TEST_START}-{TEST_END}\n")
    f.write(f"Final evaluation: {EVALUATION_YEAR}\n")
    f.write(f"Cluster: {CLUSTER_TO_MODEL} of {N_CLUSTERS}\n")
    f.write(f"Features: {BASELINE_FEATURES}\n")
    f.write("Selection criterion: annual-total test R² only\n\n")
    for key, value in eval_summary.items():
        f.write(f"{key}: {value}\n")

print("=" * 65)
print("BASELINE SCRIPT FINISHED")
print("Selected predictors:", BASELINE_FEATURES)
print("Clustering period:", f"{CLUSTER_START}-{CLUSTER_END}")
print("Training period:", f"{TRAIN_START}-{TRAIN_END}")
print("Test annual-total R²:", test_r2)
print("Output directory:", OUTPUT_ROOT)
print("Report:", report_path)
print("=" * 65)
