#!/usr/bin/env python3
"""
XGBoost SEAS5 Burned-Area Forecast – June 2025 Initialisation
==============================================================
Adapts baseline_code_JJA_sib.py to produce SEAS5 ensemble forecasts
initialised in June 2025.

Lagged SiB4 pools (Apr-May): ERA5-driven historical run (same for all ensembles)
In-season SiB4 (JJA):        June-2025-initialised SiB4 forecast
SEAS5 meteorology (JJA):     June-2025-initialised SEAS5 forecast

SiB4 forecast files  : qsib_2025{MM}_ens{NN}.lu.qp3.nc
                       in .../ElNino/sib4_seas_202506_202508_22343142/all/

SEAS5 met files (June init, lead times 1/2/3):
  June   → .../2025/06/lt1/det_EQM/seas_0.5deg_lt1_ens{NN}_202506.nc
  July   → .../2025/07/lt2/det_EQM/seas_0.5deg_lt2_ens{NN}_202507.nc
  August → .../2025/08/lt3/det_EQM/seas_0.5deg_lt3_ens{NN}_202508.nc
"""

import os
import gc
import traceback
from glob import glob

import numpy as np
import pandas as pd
import xarray as xr

from scipy import ndimage
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.inspection import permutation_importance
from sklearn.model_selection import ParameterSampler
from xgboost import XGBRegressor

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

xr.set_options(use_new_combine_kwarg_defaults=True)


# ============================================================
# CONFIGURATION
# ============================================================

cluster_to_model = 0   # 0, 1, 2, …
use_sib4         = False  # True  → ERA5 met + SiB4 predictors
                          # False → ERA5 met only (no SiB4)

# ---- Paths ----
era_path      = "/projects/0/ctdas/ilianna/1.Data/raw/era5_merged.nc"
modis1        = "/projects/0/ctdas/ilianna/1.Data/raw/MODIS_2000_2010.nc"
modis2        = "/projects/0/ctdas/ilianna/1.Data/raw/MODIS_2010_2025.nc"

sib_hist_path = (
    "/projects/0/ctdas/ruben/forecast/sib/run_global/run/output/EU/"
    "ERA_IC_ERA_driver/sib_era_2000_2025_fires"
)

# SiB4 forecast: June-2025 initialisation, same qsib_*.lu.qp3.nc format.
# Only JJA months (06, 07, 08) are needed for in-season predictors.
SIB_FC_DIR = (
    "/projects/0/ctdas/ruben/forecast/sib/run_global/run/output/"
    "ElNino/sib4_seas_202506_202508_22343142/all"
)
# Filename pattern per month+ensemble:  qsib_2025{MM}_ens{NN}.lu.qp3.nc
SIB_FC_PATTERN = "qsib_2025{month:02d}_ens{ens}.lu.qp3.nc"

# SEAS5 root; month sub-directories follow the structure shown above
SEAS5_ROOT = "/projects/0/ctdas/ruben/forecast/data/seas_processed/eu/2025"
# lead time per forecast month for JUNE initialisation (lt1/lt2/lt3)
SEAS5_LEAD = {6: 1, 7: 2, 8: 3}
# Filename pattern per month:  seas_0.5deg_lt{lead}_ens{NN}_2025{MM}.nc
SEAS5_PATTERN = "seas_0.5deg_lt{lead}_ens{ens}_2025{month:02d}.nc"

_mode  = "meteo_sib4" if use_sib4 else "meteo_only"
outdir = f"/projects/0/ctdas/ilianna/xgb_seas5_jun2025_{_mode}_cluster_{cluster_to_model}"
os.makedirs(outdir, exist_ok=True)

# ---- Time ranges ----
years_target    = np.arange(2000, 2026)
train_years     = np.arange(2000, 2020)
test_years      = np.arange(2020, 2025)
evaluation_year = 2025
clim_years      = np.arange(2000, 2025)
PREDICTION_YEAR = 2025
ENSEMBLES       = [f"{i:02d}" for i in range(51)]

# ---- Domain ----
iberia_lon_min, iberia_lon_max = -10.5, 4.5
iberia_lat_min, iberia_lat_max = 36.5, 44.5
res05 = 0.5

# ---- Fire season ----
target_fire_months      = [6, 7, 8]
lagged_predictor_months = [4, 5]      # Apr-May for biomass pools

# ---- Clustering ----
k                      = 3
target_cluster         = 1
keep_largest_component = True
random_state           = 42

# ---- Meteorological predictors ----
era5_met_vars = ["t2m", "d2m", "tmax", "tp", "windspeed", "vpd"]

# Which SEAS5 raw variable names to try for each predictor
SEAS5_CANDIDATES = {
    "t2m":       ["tm", "t2m"],
    "d2m":       ["sh", "d2m"],
    "tp":        ["tp"],
    "windspeed": ["spdm", "windspeed"],
}

# ---- Hyperparameter search ----
n_parameter_candidates = 40
validation_folds = [
    (np.arange(2000, 2014), np.arange(2014, 2016)),
    (np.arange(2000, 2016), np.arange(2016, 2018)),
    (np.arange(2000, 2018), np.arange(2018, 2020)),
]

# ---- Percentile bins ----
PERCENTILE_BINS   = [0, 10, 25, 50, 75, 90, 100]
PERCENTILE_LABELS = ["0-10", "10-25", "25-50", "50-75", "75-90", "90-100"]

# ---- PFT lookup (for SiB4 forecast files) ----
PFT_LOOKUP = {
    "enf": 1, "dbf": 2, "dnf": 3, "ebf":  4,
    "c3g": 5, "c4g": 6, "shb": 7, "dbg":  8, "crop": 9,
}

# ============================================================
# SiB4 PREDICTORS PER CLUSTER
# ============================================================

predictors_by_cluster = {
    0: {
        "biomass": ["cwt_stem_shb", "cwt_leaf_enf", "pool_leaf_enf"],
        "soil":    ["rsoil_c3g", "rsoil_enf", "rsoil_shb"],
        "hydro":   ["ect_dbg", "ecmass_dbg", "egsmax_enf"],
    },
    1: {
        "biomass": ["cwt_stem_enf", "cwt_stem_shb", "cwt_agb_enf"],
        "soil":    ["rhsoil_c3g", "pawfrw_sha", "pawftop_sha"],
        "hydro":   ["egsmax_sha", "ecmass_dbg", "ect_dbg"],
    },
    2: {
        "biomass": ["cwt_brwn_dbf", "cwt_brwn_wwt"],
        "soil":    ["rhsoil_c4g", "hc_c4g", "rsoil_c4g"],
        "hydro":   ["eci_c4g", "egsmax_c4g", "ecmass_c4g"],
    },
}

lagged_sib_predictors   = predictors_by_cluster[cluster_to_model]["biomass"]
inseason_sib_predictors = (
    predictors_by_cluster[cluster_to_model]["soil"]
    + predictors_by_cluster[cluster_to_model]["hydro"]
)
selected_sib_vars = lagged_sib_predictors + inseason_sib_predictors
needed_sib_vars   = sorted(set(v.rsplit("_", 1)[0] for v in selected_sib_vars))

if use_sib4:
    print("Selected SiB4 predictors:", selected_sib_vars)
    print("  Lagged (Apr-May)  :", lagged_sib_predictors)
    print("  In-season (JJA)   :", inseason_sib_predictors)
else:
    print("Mode: meteorology only – SiB4 predictors disabled.")


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def mask_fill_values(da):
    for key in ("_FillValue", "missing_value"):
        val = da.attrs.get(key)
        if val is not None:
            da = da.where(da != val)
    return da


def get_time_name(da):
    if "valid_time" in da.coords:
        return "valid_time"
    if "time" in da.coords:
        return "time"
    raise ValueError("No time coordinate found.")


def ensure_celsius(da):
    if float(da.mean(skipna=True).values) > 100.0:
        return da - 273.15
    return da


def compute_vpd(t2m, d2m):
    tc  = ensure_celsius(t2m)
    tdc = ensure_celsius(d2m)
    es  = 0.6108 * np.exp(17.27 * tc  / (tc  + 237.3))
    ea  = 0.6108 * np.exp(17.27 * tdc / (tdc + 237.3))
    return (es - ea).clip(min=0).rename("vpd")


def seasonal_agg(da, months, agg="mean"):
    tname = get_time_name(da)
    sub   = da.sel({tname: da[tname].dt.month.isin(months)})
    if agg == "sum":
        out = sub.groupby(sub[tname].dt.year).sum(tname)
    else:
        out = sub.groupby(sub[tname].dt.year).mean(tname)
    out = out.rename({"year": "time"})
    # Drop sentinel years
    valid = (out.time >= 1900) & (out.time <= 2100)
    out   = out.sel(time=valid)
    years      = [int(y) for y in out.time.values]
    out = out.assign_coords(
        time=[pd.Timestamp(year=y, month=1, day=1) for y in years]
    )
    return out


def rename_latlon(da):
    rn = {}
    if "latitude"  in da.dims or "latitude"  in da.coords: rn["latitude"]  = "lat"
    if "longitude" in da.dims or "longitude" in da.coords: rn["longitude"] = "lon"
    return da.rename(rn) if rn else da


def parse_label(label):
    parts = label.split("_")
    return "_".join(parts[:-1]), parts[-1]


def build_pft_field_from_nlu(ds, varname, target_refnum):
    da   = mask_fill_values(ds[varname])
    mask = ds["lu_pref"] == target_refnum
    return (da * ds["lu_area"]).where(mask).sum("nlu", skipna=True)


def to_latlon_grid(da_land, ds_sib):
    lat = np.asarray(ds_sib["latsib"].data)
    lon = np.asarray(ds_sib["lonsib"].data)
    pairs = np.column_stack([lat, lon])
    _, unique_idx = np.unique(pairs, axis=0, return_index=True)
    unique_idx    = np.sort(unique_idx)
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
    da = rename_latlon(da)
    if "lon" in da.coords and float(da.lon.max()) > 180:
        da = da.assign_coords(lon=((da.lon + 180) % 360) - 180).sortby("lon")
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


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def percentile_rank(value, reference):
    ref = np.asarray(reference, dtype=float)
    ref = ref[np.isfinite(ref)]
    return 100.0 * np.mean(ref < value) if ref.size else np.nan


def first_present(ds, names):
    return next((n for n in names if n in ds.data_vars), None)


def setup_map(ax):
    proj = ccrs.PlateCarree()
    ax.set_extent(
        [iberia_lon_min, iberia_lon_max, iberia_lat_min, iberia_lat_max], crs=proj
    )
    ax.add_feature(cfeature.LAND,      facecolor="0.95")
    ax.add_feature(cfeature.OCEAN,     facecolor="white")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.6)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.4)
    gl.top_labels = False; gl.right_labels = False
    return proj


# ============================================================
# SiB4 GRID CONVERSION (generalised)
# ============================================================

def convert_nsib(da, ds_ref=None):
    """Convert SiB4 land-point DataArray → 2-D lat/lon grid."""
    da = rename_latlon(da)
    point_dim = next((d for d in ("nsib", "landpoints") if d in da.dims), None)
    if point_dim is None:
        return da

    if ds_ref is None:
        raise ValueError(f"'{point_dim}' dim present but no reference Dataset supplied.")
    if "latsib" not in ds_ref or "lonsib" not in ds_ref:
        raise KeyError(f"Reference Dataset must have 'latsib'/'lonsib'. Got: {list(ds_ref.variables)}")

    lat = ds_ref["latsib"]
    lon = ds_ref["lonsib"]
    if point_dim in lat.dims:
        lat = lat.isel({d: 0 for d in lat.dims if d != point_dim})
        lon = lon.isel({d: 0 for d in lon.dims if d != point_dim})

    lat_v = np.asarray(lat.values).reshape(-1)
    lon_v = np.asarray(lon.values).reshape(-1)

    if len(lat_v) != da.sizes[point_dim]:
        raise ValueError(
            f"SiB coordinate length mismatch: variable has {da.sizes[point_dim]} pts, "
            f"latsib={len(lat_v)}."
        )

    da = da.assign_coords(lat=(point_dim, lat_v), lon=(point_dim, lon_v))
    return (
        da.set_index({point_dim: ["lat", "lon"]})
        .unstack(point_dim)
        .sortby("lat")
        .sortby("lon")
    )


def extract_sib_variable(ds, var_name):
    """Extract a variable (possibly PFT-specific composite) from a SiB Dataset."""
    if var_name in ds.data_vars:
        return ds[var_name]
    if "_" in var_name:
        base_var, pft_suffix = var_name.rsplit("_", 1)
        if base_var in ds.data_vars:
            da = mask_fill_values(ds[base_var])
            if "nlu" in da.dims:
                tgt = PFT_LOOKUP.get(pft_suffix.lower())
                if tgt is not None and "lu_pref" in ds:
                    mask = ds["lu_pref"] == tgt
                    da   = (
                        (da * ds["lu_area"]).where(mask).sum("nlu", skipna=True)
                        if "lu_area" in ds
                        else da.where(mask).sum("nlu", skipna=True)
                    )
                else:
                    da = da.mean("nlu", skipna=True)
            return da.rename(var_name)
    raise KeyError(
        f"Cannot resolve SiB variable '{var_name}'. "
        f"Available: {list(ds.data_vars)}"
    )


# ============================================================
# 1. MODIS BURNED AREA
# ============================================================

print("Loading MODIS burned area …")

ds1 = xr.open_dataset(modis1)
ds2 = xr.open_dataset(modis2)
ds_modis = xr.concat([ds1, ds2], dim="time").sortby("time")
ds_modis = ds_modis.sel(time=~ds_modis.get_index("time").duplicated())

burn_da = ds_modis["BurnDate"].sel(
    lon=slice(iberia_lon_min, iberia_lon_max),
    lat=slice(iberia_lat_max, iberia_lat_min),
)
fill = burn_da.attrs.get("_FillValue")
if fill is not None:
    burn_da = burn_da.where(burn_da != fill)

dlat = float(abs(burn_da.lat.diff("lat").median()))
dlon = float(abs(burn_da.lon.diff("lon").median()))
R    = 6371000.0
dy   = R * (np.pi / 180) * dlat
dx   = R * (np.pi / 180) * dlon * np.cos(np.deg2rad(burn_da.lat))
pixel_area_ha = dx * dy / 1e4

lat_edges   = np.arange(iberia_lat_min, iberia_lat_max + res05, res05)
lon_edges   = np.arange(iberia_lon_min, iberia_lon_max + res05, res05)
lat_centers = lat_edges[:-1] + res05 / 2
lon_centers = lon_edges[:-1] + res05 / 2

burn_maps_05 = []
for year in years_target:
    jja = burn_da.sel(
        time=(burn_da.time.dt.year == year) & burn_da.time.dt.month.isin(target_fire_months)
    )
    if jja.time.size == 0:
        continue
    ba = (jja > 0).any("time") * pixel_area_ha
    ba = ba.groupby_bins("lon", lon_edges, labels=lon_centers).sum()
    ba = ba.groupby_bins("lat", lat_edges, labels=lat_centers).sum()
    ba = ba.rename({"lat_bins": "lat", "lon_bins": "lon"}).transpose("lat", "lon")
    burn_maps_05.append(
        ba.assign_coords(time=pd.Timestamp(f"{year}-01-01")).expand_dims("time")
    )

burn_05 = xr.concat(burn_maps_05, dim="time").sortby("lat").sortby("lon")
ds1.close(); ds2.close()
del ds1, ds2, ds_modis, burn_da
gc.collect()
print("Burn grid:", dict(burn_05.sizes))


# ============================================================
# 2. FIRE-REGIME CLUSTERS  (trained on 2000-2019)
# ============================================================

print("Computing fire-regime clusters …")

burn_log_train = np.log1p(
    burn_05.sel(time=burn_05.time.dt.year.isin(train_years))
)
X_cluster = burn_log_train.stack(cell=("lat", "lon")).transpose("cell", "time")
valid     = X_cluster.sum("time") > 0
X_valid   = X_cluster[valid]

data_std = StandardScaler().fit_transform(X_valid.values)
labels   = KMeans(n_clusters=k, random_state=0, n_init=20).fit_predict(data_std)

cluster_1d  = xr.DataArray(labels, dims="cell", coords={"cell": X_valid.cell})
cluster_raw = cluster_1d.unstack("cell").sortby("lat").sortby("lon")

cluster_full = xr.full_like(burn_05.isel(time=0), np.nan, dtype=float)
cluster_full.loc[dict(lat=cluster_raw.lat, lon=cluster_raw.lon)] = cluster_raw.astype(float)

cvals       = cluster_full.values.copy()
target_mask = np.isfinite(cvals) & (cvals == target_cluster)
labeled, ncomp = ndimage.label(target_mask, structure=np.ones((3, 3), dtype=int))
remove_mask    = np.zeros_like(target_mask, dtype=bool)

if ncomp > 1 and keep_largest_component:
    sizes       = ndimage.sum(target_mask, labeled, index=np.arange(1, ncomp + 1))
    keep_label  = int(np.argmax(sizes) + 1)
    remove_mask = (labeled > 0) & (labeled != keep_label)

new_cluster = cvals.copy()
ny, nx = new_cluster.shape
for i, j in zip(*np.where(remove_mask)):
    nbrs = []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == dj == 0: continue
            ii, jj = i + di, j + dj
            if 0 <= ii < ny and 0 <= jj < nx:
                v = new_cluster[ii, jj]
                if np.isfinite(v) and v != target_cluster:
                    nbrs.append(int(v))
    new_cluster[i, j] = np.argmax(np.bincount(nbrs, minlength=k)) if nbrs else np.nan

cluster = xr.DataArray(
    new_cluster,
    dims=("lat", "lon"),
    coords={"lat": burn_05.lat.values, "lon": burn_05.lon.values},
    name="cluster",
)
cluster.to_netcdf(os.path.join(outdir, "fire_regime_clusters.nc"))
print(f"Cluster {cluster_to_model} selected.")


# ============================================================
# 3. OBSERVED CLIMATOLOGY (2000-2024) FOR PERCENTILE RANKING
# ============================================================

clim_rows = []
for t in burn_05.time.values:
    year = pd.Timestamp(t).year
    if year in clim_years:
        total = float(burn_05.sel(time=t).where(cluster == cluster_to_model).sum().values)
        clim_rows.append({"year": year, "burned_area": total})

clim_yearly    = pd.DataFrame(clim_rows)
clim_reference = clim_yearly["burned_area"].to_numpy()

clim_yearly["percentile"]     = [percentile_rank(v, clim_reference) for v in clim_reference]
clim_yearly["percentile_bin"] = pd.cut(
    clim_yearly["percentile"],
    bins=PERCENTILE_BINS, labels=PERCENTILE_LABELS,
    include_lowest=True, right=True,
)
clim_yearly.to_csv(
    os.path.join(outdir, "historical_climatology_percentiles_2000_2024.csv"),
    index=False,
)


# ============================================================
# 4. ERA5 METEOROLOGY
# ============================================================

print("Loading ERA5 meteorology …")

ds_era = xr.open_dataset(era_path)
if "longitude" in ds_era.coords and float(ds_era.longitude.max()) > 180:
    ds_era = ds_era.assign_coords(
        longitude=((ds_era.longitude + 180) % 360) - 180
    ).sortby("longitude")
ds_era = ds_era.sortby("latitude").sortby("longitude")
ds_era = ds_era.sel(
    longitude=slice(iberia_lon_min, iberia_lon_max),
    latitude=slice(iberia_lat_min, iberia_lat_max),
)

ds_era["t2m"]       = ensure_celsius(ds_era["t2m"])
ds_era["d2m"]       = ensure_celsius(ds_era["d2m"])
ds_era["vpd"]       = compute_vpd(ds_era["t2m"], ds_era["d2m"])
ds_era["windspeed"] = np.sqrt(ds_era["u10"] ** 2 + ds_era["v10"] ** 2)
tn = get_time_name(ds_era["t2m"])
ds_era["tmax"] = ds_era["t2m"].resample({tn: "1D"}).max()
print("Daily tmax computed.")

met_yearly_era = {}
for met_var in era5_met_vars:
    agg    = "sum" if met_var == "tp" else "mean"
    met_yr = seasonal_agg(ds_era[met_var], months=target_fire_months, agg=agg)
    met_yr = rename_latlon(met_yr).sortby("lat").sortby("lon")
    met_yr = met_yr.sel(time=met_yr.time.dt.year.isin(years_target))
    met_yearly_era[met_var] = align_to_burn_grid(met_yr)

ds_era.close()
del ds_era
gc.collect()
print("ERA5 variables loaded:", list(met_yearly_era.keys()))


# ============================================================
# 5. HISTORICAL SiB4-ERA5 PREDICTORS  (skipped when use_sib4=False)
# ============================================================

if use_sib4:
    print("Loading historical SiB4 predictors …")

    files = sorted(glob(os.path.join(sib_hist_path, "qsib_*.lu.qp3.nc")))
    if not files:
        raise FileNotFoundError(f"No SiB4 files in {sib_hist_path}")

    keep_vars = ["latsib", "lonsib", "lu_area", "lu_pref",
                 "pft_names", "pft_refnums"] + needed_sib_vars

    with xr.open_dataset(files[0], decode_times=True) as ds_check:
        missing = [v for v in keep_vars if v not in ds_check.variables]
    if missing:
        raise KeyError(f"First SiB4 file missing: {missing}")

    ds_sib_hist = xr.open_mfdataset(
        files, decode_times=True, engine="netcdf4",
        combine="by_coords", parallel=False,
        data_vars="minimal", coords="minimal", compat="override",
    )[keep_vars]

    ds_sib_hist["pft_names"] = ds_sib_hist["pft_names"].astype(str)
    pft_lookup_hist = dict(
        zip(ds_sib_hist["pft_names"].values.astype(str),
            ds_sib_hist["pft_refnums"].values.astype(int))
    )

    sib_yearly = {}
    for lbl in selected_sib_vars:
        sib_var, pft_code = parse_label(lbl)
        if sib_var not in ds_sib_hist or pft_code not in pft_lookup_hist:
            print(f"  Skipping historical SiB4 predictor '{lbl}': unavailable")
            continue
        tgt_ref = pft_lookup_hist[pft_code]
        da_land = build_pft_field_from_nlu(ds_sib_hist, sib_var, tgt_ref)
        da_grid = to_latlon_grid(da_land, ds_sib_hist)
        da_grid = da_grid.sel(
            lon=slice(iberia_lon_min, iberia_lon_max),
            lat=slice(iberia_lat_min, iberia_lat_max),
        ).sortby("lat").sortby("lon")
        months = lagged_predictor_months if lbl in lagged_sib_predictors else target_fire_months
        da_yr  = seasonal_agg(da_grid, months, agg="mean")
        da_yr  = da_yr.sel(time=da_yr.time.dt.year.isin(years_target))
        sib_yearly[lbl] = align_to_burn_grid(da_yr)

    available_sib_vars = [v for v in selected_sib_vars if v in sib_yearly]
    missing_sib        = [v for v in selected_sib_vars if v not in sib_yearly]
    if missing_sib:
        raise ValueError(f"Could not construct SiB4 predictors: {missing_sib}")
    print("Available SiB4 predictors:", available_sib_vars)

else:
    sib_yearly         = {}
    available_sib_vars = []


# ============================================================
# 6. BUILD PIXEL TABLES (ERA5-driven)
# ============================================================

def build_pixel_table(only_years):
    rows         = []
    required_vars = era5_met_vars + available_sib_vars

    for t in burn_05.time.values:
        year = pd.Timestamp(t).year
        if year not in only_years:
            continue
        data_vars = {
            "burned_area": burn_05.sel(time=t).squeeze(drop=True),
            "cluster":     cluster,
        }
        for mv in era5_met_vars:
            idx = np.where(met_yearly_era[mv].time.dt.year.values == year)[0]
            if idx.size:
                data_vars[mv] = safe_2d(met_yearly_era[mv].isel(time=int(idx[0])))
        for sv in available_sib_vars:
            idx = np.where(sib_yearly[sv].time.dt.year.values == year)[0]
            if idx.size:
                data_vars[sv] = safe_2d(sib_yearly[sv].isel(time=int(idx[0])))

        missing = [v for v in required_vars if v not in data_vars]
        if missing:
            print(f"  Skipping {year}: missing {missing}")
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
        raise ValueError(f"No pixel data for years {sorted(only_years)}.")

    df = pd.concat(rows, ignore_index=True)
    df = df[["year", "lat", "lon", "cluster", "burned_area"] + required_vars]
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df[np.isfinite(df["cluster"])].copy()
    df["cluster"] = df["cluster"].astype(int)
    df = df[df["cluster"] == cluster_to_model].copy()
    if df.empty:
        raise ValueError(f"No valid pixels for cluster {cluster_to_model}.")
    return df


df_train     = build_pixel_table(set(train_years))
df_test      = build_pixel_table(set(test_years))
df_eval_2025 = build_pixel_table({evaluation_year})

feature_cols = [
    v for v in era5_met_vars + available_sib_vars
    if v in df_train.columns and v in df_test.columns and v in df_eval_2025.columns
]
print("\nFeatures:", feature_cols)
print(f"Train rows: {len(df_train)}, Test: {len(df_test)}, ERA5-2025: {len(df_eval_2025)}")

X_train = df_train[feature_cols];  y_train = df_train["burned_area"]
X_test  = df_test[feature_cols];   y_test  = df_test["burned_area"]
X_2025  = df_eval_2025[feature_cols]; y_2025 = df_eval_2025["burned_area"]


# ============================================================
# 7. HYPERPARAMETER TUNING
# ============================================================

param_distributions = {
    "n_estimators":    [200, 400, 600, 800, 1000],
    "max_depth":       [2, 3, 4, 5, 6, 8],
    "learning_rate":   [0.01, 0.02, 0.03, 0.05, 0.08, 0.1],
    "min_child_weight":[1, 3, 5, 7, 10],
    "subsample":       [0.6, 0.75, 0.85, 1.0],
    "colsample_bytree":[0.6, 0.75, 0.85, 1.0],
    "gamma":           [0, 0.05, 0.1, 0.2, 0.5],
    "reg_alpha":       [0, 0.001, 0.01, 0.1, 1.0],
    "reg_lambda":      [1, 2, 5, 10, 20],
}
candidates = list(
    ParameterSampler(param_distributions, n_iter=n_parameter_candidates, random_state=random_state)
)
max_rows_per_fold = 150_000


def sample_rows_by_year(frame, max_rows, seed):
    if len(frame) <= max_rows:
        return frame
    years = np.sort(frame["year"].unique())
    quota = max(1, max_rows // len(years))
    parts = [
        frame[frame["year"] == y].sample(
            n=min(len(frame[frame["year"] == y]), quota), random_state=seed
        )
        for y in years
    ]
    sampled = pd.concat(parts, ignore_index=True)
    return sampled.sample(n=max_rows, random_state=seed) if len(sampled) > max_rows else sampled


def make_xgb(params):
    return XGBRegressor(
        objective="reg:squarederror", eval_metric="rmse",
        tree_method="hist", random_state=random_state,
        n_jobs=1, verbosity=0, **params,
    )


search_results = []
for candidate_idx, params in enumerate(candidates, start=1):
    fold_scores = []
    print(f"\nCandidate {candidate_idx}/{len(candidates)}: {params}")
    for fold_idx, (fit_years, valid_years) in enumerate(validation_folds, start=1):
        fit_df   = df_train[df_train["year"].isin(fit_years)]
        valid_df = df_train[df_train["year"].isin(valid_years)].copy()
        if fit_df.empty or valid_df.empty:
            raise ValueError(f"Empty data in fold {fold_idx}.")
        fit_sample = sample_rows_by_year(fit_df, max_rows_per_fold, random_state + fold_idx)
        mdl = make_xgb(params)
        mdl.fit(fit_sample[feature_cols], fit_sample["burned_area"], verbose=False)
        valid_df["pred"] = np.clip(mdl.predict(valid_df[feature_cols]), 0, None)
        annual_obs  = valid_df.groupby("year")["burned_area"].sum()
        annual_pred = valid_df.groupby("year")["pred"].sum()
        score = r2_score(annual_obs, annual_pred)
        fold_scores.append(score)
        print(f"   Fold {fold_idx}: {valid_years[0]}-{valid_years[-1]}, R2={score:.5f}")
    mean_score = float(np.mean(fold_scores))
    search_results.append({
        "candidate": candidate_idx,
        "mean_validation_annual_r2": mean_score,
        "std_validation_annual_r2":  float(np.std(fold_scores)),
        "fold_annual_r2": fold_scores,
        "params": params,
    })
    print(f"   Mean R2 = {mean_score:.5f}")

search_results.sort(key=lambda x: x["mean_validation_annual_r2"], reverse=True)
best_result = search_results[0]
best_params = best_result["params"]
print("\n=== BEST HYPERPARAMETERS ===")
print(best_params)
print("Best mean validation R2:", best_result["mean_validation_annual_r2"])

tuning_df = pd.DataFrame([
    {"candidate": r["candidate"],
     "mean_validation_annual_r2": r["mean_validation_annual_r2"],
     "std_validation_annual_r2":  r["std_validation_annual_r2"],
     **r["params"]}
    for r in search_results
])
tuning_df.to_csv(os.path.join(outdir, "xgboost_hyperparameter_search.csv"), index=False)
pd.DataFrame([{**best_params,
               "best_mean_validation_annual_r2": best_result["mean_validation_annual_r2"],
               "validation_annual_r2_std": best_result["std_validation_annual_r2"]}
              ]).to_csv(os.path.join(outdir, "best_xgboost_parameters.csv"), index=False)


# ============================================================
# 8. FIT FINAL MODEL  (ERA5, 2000-2019)
# ============================================================

xgb_model = make_xgb(best_params)
xgb_model.fit(X_train, y_train, verbose=False)
xgb_model.save_model(os.path.join(outdir, "xgb_model.json"))

df_train["predicted"]     = np.clip(xgb_model.predict(X_train), 0, None)
df_test["predicted"]      = np.clip(xgb_model.predict(X_test),  0, None)
df_eval_2025["predicted"] = np.clip(xgb_model.predict(X_2025),  0, None)
for f in [df_train, df_test, df_eval_2025]:
    f["difference"] = f["predicted"] - f["burned_area"]


# ============================================================
# 9. ERA5-DRIVEN METRICS
# ============================================================

df_all        = pd.concat([df_train, df_test, df_eval_2025], ignore_index=True)
annual_totals = df_all.groupby("year")[["burned_area", "predicted"]].sum()
train_ann     = annual_totals.loc[annual_totals.index.isin(train_years)]
test_ann      = annual_totals.loc[annual_totals.index.isin(test_years)]

metrics = {
    "cluster":  cluster_to_model,
    "features": ", ".join(feature_cols),
    "pixel_train_r2":            r2_score(df_train["burned_area"],     df_train["predicted"]),
    "pixel_test_r2":             r2_score(df_test["burned_area"],      df_test["predicted"]),
    "pixel_eval_2025_r2":        r2_score(df_eval_2025["burned_area"], df_eval_2025["predicted"]),
    "annual_train_r2":           r2_score(train_ann["burned_area"],    train_ann["predicted"]),
    "annual_test_r2":            r2_score(test_ann["burned_area"],     test_ann["predicted"]),
    "annual_overall_r2":         r2_score(annual_totals["burned_area"],annual_totals["predicted"]),
    "train_rmse":                rmse(df_train["burned_area"],          df_train["predicted"]),
    "test_rmse":                 rmse(df_test["burned_area"],           df_test["predicted"]),
    "eval_2025_rmse":            rmse(df_eval_2025["burned_area"],      df_eval_2025["predicted"]),
    "observed_total_2025":       float(df_eval_2025["burned_area"].sum()),
    "predicted_total_2025_era5": float(df_eval_2025["predicted"].sum()),
    "best_mean_validation_r2":   best_result["mean_validation_annual_r2"],
}
pd.DataFrame([metrics]).to_csv(os.path.join(outdir, "metrics_era5_baseline.csv"), index=False)
print("\nFINAL ERA5-BASELINE METRICS")
print(pd.DataFrame([metrics]).T)


# ============================================================
# 10. FEATURE IMPORTANCE
# ============================================================

fi = pd.DataFrame({
    "feature":    feature_cols,
    "importance": xgb_model.feature_importances_,
}).sort_values("importance", ascending=False)
fi.to_csv(os.path.join(outdir, "feature_importance.csv"), index=False)

plt.figure(figsize=(8, 5))
pp = fi.sort_values("importance")
plt.barh(pp["feature"], pp["importance"])
plt.xlabel("XGBoost feature importance")
plt.title(f"Cluster {cluster_to_model}: feature importance")
plt.tight_layout()
plt.savefig(os.path.join(outdir, "feature_importance.png"), dpi=200)
plt.close()

perm = permutation_importance(
    xgb_model, X_test, y_test,
    n_repeats=5, random_state=random_state, n_jobs=1, scoring="r2",
)
perm_df = pd.DataFrame({
    "feature":         feature_cols,
    "importance_mean": perm.importances_mean,
    "importance_std":  perm.importances_std,
}).sort_values("importance_mean", ascending=False)
perm_df.to_csv(os.path.join(outdir, "permutation_importance_test.csv"), index=False)

plt.figure(figsize=(8, 5))
pp = perm_df.sort_values("importance_mean")
plt.barh(pp["feature"], pp["importance_mean"], xerr=pp["importance_std"])
plt.xlabel("Decrease in test R² after permutation")
plt.title(f"Cluster {cluster_to_model}: permutation importance (2020-2024)")
plt.tight_layout()
plt.savefig(os.path.join(outdir, "permutation_importance.png"), dpi=200)
plt.close()


# ============================================================
# 11. SAVE ERA5-DRIVEN TABLES & RECONSTRUCTION PLOT
# ============================================================

df_train.to_csv(    os.path.join(outdir, "training_table_2000_2019.csv"),  index=False)
df_test.to_csv(     os.path.join(outdir, "test_table_2020_2024.csv"),       index=False)
df_eval_2025.to_csv(os.path.join(outdir, "evaluation_era5_2025.csv"),       index=False)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(annual_totals.index, annual_totals["burned_area"],
        marker="o", color="#1f77b4", linewidth=1.5, markersize=4, label="Observed")
ax.plot(annual_totals.index, annual_totals["predicted"],
        marker="s", color="#ff7f0e", linewidth=1.5, markersize=4, label="ERA5-predicted")
ax.axvspan(2019.5, 2024.5, color="#dce8f5", alpha=0.6, label="Test period")
ax.axvspan(2024.5, 2025.5, color="#b8d3ec", alpha=0.8, label="ERA5 2025 (reference)")
ax.set_title(f"Cluster {cluster_to_model}: annual burned area (ERA5-driven)", fontsize=13)
ax.set_xlabel("Year"); ax.set_ylabel("Burned area (ha)")
ax.set_xlim(1999.5, 2025.5)
ax.set_xticks(np.arange(2000, 2026, 5))
ax.set_ylim(bottom=0)
ax.grid(True, linestyle="-", linewidth=0.5, alpha=0.3)
ax.legend(loc="upper left")
plt.tight_layout()
plt.savefig(
    os.path.join(outdir, f"annual_reconstruction_cluster_{cluster_to_model}.png"),
    dpi=300,
)
plt.close()


# ============================================================
# 12. SEAS5 LOADER  (June 2025 initialisation)
# ============================================================

def seas5_path(month, ensemble):
    """
    Build the full path to a SEAS5 monthly file.
    Pattern: {SEAS5_ROOT}/{MM}/lt{lead}/det_EQM/
             seas_0.5deg_lt{lead}_ens{NN}_2025{MM}.nc
    """
    lead = SEAS5_LEAD[month]
    fname = SEAS5_PATTERN.format(lead=lead, ens=ensemble, month=month)
    return os.path.join(SEAS5_ROOT, f"{month:02d}", f"lt{lead}", "det_EQM", fname)


def load_month_hourly(path, month):
    d = xr.open_dataset(path, decode_times=False)
    if "valid_time" in d.coords and "time" not in d.coords:
        d = d.rename({"valid_time": "time"})
    if "time" not in d.coords:
        d.close()
        raise ValueError(f"No time coordinate in {path}")
    n = d.sizes.get("time", 0)
    if n == 0:
        d.close()
        raise ValueError(f"Empty time dimension: {path}")
    d = d.assign_coords(
        time=pd.date_range(start=f"2025-{month:02d}-01 00:30:00", periods=n, freq="1h")
    )
    return d


def load_seas5_met_ensemble(ensemble):
    """Load SEAS5 JJA meteorology for one ensemble member and return seasonal means."""
    monthly = []
    for month in target_fire_months:
        path = seas5_path(month, ensemble)
        if not os.path.exists(path):
            print(f"  [ens{ensemble}] Missing: {path}")
            for d in monthly: d.close()
            return None
        monthly.append(load_month_hourly(path, month))

    try:
        dsf    = xr.concat(monthly, dim="time").sortby("time")
        fields = {}

        for model_var, candidates in SEAS5_CANDIDATES.items():
            src = first_present(dsf, candidates)
            if src is None and model_var == "tp":
                if "lspr_scaled" in dsf and "cupr_scaled" in dsf:
                    dsf["tp_combined"] = dsf["lspr_scaled"] + dsf["cupr_scaled"]; src = "tp_combined"
                elif "lspr" in dsf and "cupr" in dsf:
                    dsf["tp_combined"] = dsf["lspr"] + dsf["cupr"]; src = "tp_combined"
            if src is None:
                continue
            da = convert_nsib(dsf[src], dsf)
            if float(da.lon.max()) > 180:
                da = da.assign_coords(lon=((da.lon + 180) % 360) - 180)
            da = da.sortby("lat").sortby("lon").sel(
                lon=slice(iberia_lon_min, iberia_lon_max),
                lat=slice(iberia_lat_min, iberia_lat_max),
            )
            fields[model_var] = da.load()

        if "t2m" in fields: fields["t2m"] = ensure_celsius(fields["t2m"])
        if "d2m" in fields: fields["d2m"] = ensure_celsius(fields["d2m"])
        if "t2m" in fields and "d2m" in fields:
            fields["vpd"] = compute_vpd(fields["t2m"], fields["d2m"]).load()
        if "t2m" in fields:
            tn2 = get_time_name(fields["t2m"])
            fields["tmax"] = fields["t2m"].resample({tn2: "1D"}).max(skipna=True).load()

        missing_met = [v for v in era5_met_vars if v not in fields]
        if missing_met:
            print(f"  [ens{ensemble}] Missing met vars: {missing_met}")
            return None

        result = {}
        for var in era5_met_vars:
            agg    = "sum" if var == "tp" else "mean"
            annual = seasonal_agg(fields[var], target_fire_months, agg=agg)
            result[var] = align_to_burn_grid(annual).load()
        return result

    finally:
        for d in monthly: d.close()
        gc.collect()


# ============================================================
# 13. SiB4 FORECAST LOADER
#
# June initialisation: forecast covers only Jun-Aug, so Apr-May
# lagged pools are NOT available from forecast files.
# SIB_FC_HAS_LAGGED will be False → ERA5 fallback used automatically.
# ============================================================

_first_lagged_file = os.path.join(
    SIB_FC_DIR,
    SIB_FC_PATTERN.format(month=lagged_predictor_months[0], ens="00"),
)
SIB_FC_HAS_LAGGED = os.path.exists(_first_lagged_file)
print(f"SiB4 forecast covers lagged months (Apr-May): {SIB_FC_HAS_LAGGED}")


def load_sib_fc_ensemble(ensemble):
    """
    Load SiB4 forecast variables for one ensemble member.
    Returns {"lagged": {var: DataArray}, "inseason": {var: DataArray}}.
    "lagged" is empty for June init (SIB_FC_HAS_LAGGED=False).
    """
    months_needed = target_fire_months[:]
    if SIB_FC_HAS_LAGGED:
        months_needed = lagged_predictor_months + months_needed

    monthly_files = []
    for month in months_needed:
        fname = SIB_FC_PATTERN.format(month=month, ens=ensemble)
        path  = os.path.join(SIB_FC_DIR, fname)
        if not os.path.exists(path):
            print(f"  [ens{ensemble}] Missing SiB4 file: {path}")
            return None
        monthly_files.append(path)

    needed_base = sorted(set(
        v.rsplit("_", 1)[0]
        for v in lagged_sib_predictors + inseason_sib_predictors
    ))
    keep = ["latsib", "lonsib", "lu_area", "lu_pref",
            "pft_names", "pft_refnums"] + needed_base

    with xr.open_dataset(monthly_files[0], decode_times=True) as ds_chk:
        missing_vars = [v for v in keep if v not in ds_chk.variables]
    if missing_vars:
        print(f"  [ens{ensemble}] SiB4 forecast missing vars: {missing_vars}")
        return None

    ds_fc = xr.open_mfdataset(
        monthly_files,
        decode_times=True,
        engine="netcdf4",
        combine="by_coords",
        parallel=False,
        data_vars="minimal",
        coords="minimal",
        compat="override",
    )[keep]

    ds_fc["pft_names"] = ds_fc["pft_names"].astype(str)
    pft_lookup_fc = dict(
        zip(ds_fc["pft_names"].values.astype(str),
            ds_fc["pft_refnums"].values.astype(int))
    )

    def _extract(predictors, months):
        out = {}
        for lbl in predictors:
            sib_var, pft_code = parse_label(lbl)
            if sib_var not in ds_fc or pft_code not in pft_lookup_fc:
                print(f"  [ens{ensemble}] Cannot resolve '{lbl}'")
                ds_fc.close()
                return None
            tgt_ref = pft_lookup_fc[pft_code]
            da_land = build_pft_field_from_nlu(ds_fc, sib_var, tgt_ref)
            da_grid = to_latlon_grid(da_land, ds_fc).sel(
                lon=slice(iberia_lon_min, iberia_lon_max),
                lat=slice(iberia_lat_min, iberia_lat_max),
            ).sortby("lat").sortby("lon")
            da_yr = seasonal_agg(da_grid, months, agg="mean")
            out[lbl] = align_to_burn_grid(da_yr).load()
        return out

    lagged_res   = _extract(lagged_sib_predictors,   lagged_predictor_months) \
                   if SIB_FC_HAS_LAGGED else {}
    inseason_res = _extract(inseason_sib_predictors, target_fire_months)

    ds_fc.close()
    if inseason_res is None or len(inseason_res) != len(inseason_sib_predictors):
        return None
    return {"lagged": lagged_res, "inseason": inseason_res}


# ============================================================
# 14. ERA5-DRIVEN LAGGED POOLS – FALLBACK
#     June init forecast doesn't cover Apr-May, so always use ERA5 here.
# ============================================================

era5_lagged_pools_2025 = {}
if use_sib4 and not SIB_FC_HAS_LAGGED:
    print("SiB4 forecast lacks Apr-May → using ERA5-driven lagged pools as fallback.")
    for var in lagged_sib_predictors:
        da_2025 = sib_yearly[var].sel(time=sib_yearly[var].time.dt.year == PREDICTION_YEAR)
        if da_2025.sizes.get("time", 0) == 0:
            raise RuntimeError(
                f"Historical SiB4 has no {PREDICTION_YEAR} Apr-May data for '{var}'."
            )
        era5_lagged_pools_2025[var] = safe_2d(da_2025.isel(time=0))
    print("ERA5 fallback lagged pools loaded:", list(era5_lagged_pools_2025.keys()))
elif use_sib4:
    print("SiB4 forecast covers Apr-May → lagged pools come from forecast per ensemble.")
else:
    print("Mode: meteorology only – lagged SiB4 pools skipped.")


# ============================================================
# 15. PREDICT ALL SEAS5 ENSEMBLE MEMBERS
# ============================================================

print(f"\n{'='*55}")
print(f"SEAS5 ensemble predictions – June 2025 initialisation")
print(f"{'='*55}\n")

ensemble_rows = []

for ensemble in ENSEMBLES:
    print(f"--- Ensemble {ensemble} ---")
    try:
        # (a) SEAS5 meteorology
        met_2025 = load_seas5_met_ensemble(ensemble)
        if met_2025 is None:
            print(f"  [ens{ensemble}] Meteorology missing – skipping.")
            continue

        # (b) SiB4 forecast (in-season; lagged from ERA5 fallback for June init)
        if use_sib4:
            sib_fc = load_sib_fc_ensemble(ensemble)
            if sib_fc is None:
                print(f"  [ens{ensemble}] SiB4 forecast missing – skipping.")
                continue
        else:
            sib_fc = {"lagged": {}, "inseason": {}}

        # (c) Assemble predictor fields
        pred_fields = {"cluster": cluster}
        for var in era5_met_vars:
            pred_fields[var] = safe_2d(met_2025[var])
        if use_sib4:
            if SIB_FC_HAS_LAGGED:
                for var in lagged_sib_predictors:
                    pred_fields[var] = safe_2d(sib_fc["lagged"][var])
            else:
                for var in lagged_sib_predictors:
                    pred_fields[var] = era5_lagged_pools_2025[var]
            for var in inseason_sib_predictors:
                pred_fields[var] = safe_2d(sib_fc["inseason"][var])

        # (d) Build pixel frame
        ds_pred = xr.Dataset(pred_fields)
        frame   = ds_pred.stack(cell=("lat", "lon")).to_dataframe().reset_index()
        frame   = (
            frame[["lat", "lon", "cluster"] + feature_cols]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        frame["cluster"] = frame["cluster"].astype(int)
        frame = frame[frame["cluster"] == cluster_to_model].copy()
        del ds_pred

        if frame.empty:
            print(f"  [ens{ensemble}] No valid prediction pixels.")
            continue

        # (e) Predict
        prediction = np.clip(
            xgb_model.predict(frame[feature_cols].to_numpy(dtype=np.float32)),
            0, None,
        )
        total   = float(prediction.sum())
        pct     = percentile_rank(total, clim_reference)
        pct_bin = pd.cut(
            [pct], bins=PERCENTILE_BINS, labels=PERCENTILE_LABELS,
            include_lowest=True, right=True,
        )[0]

        ensemble_rows.append({
            "year":                PREDICTION_YEAR,
            "initialisation":      "202506",
            "forecast_period":     "JJA",
            "ensemble":            ensemble,
            "n_prediction_pixels": len(frame),
            "predicted_total_ha":  total,
            "percentile":          pct,
            "percentile_bin":      pct_bin,
        })
        print(f"  ens{ensemble}: {total:,.1f} ha | pct={pct:.1f} | bin={pct_bin}")

        del met_2025, sib_fc, frame, prediction
        gc.collect()

    except Exception as exc:
        print(f"  [FAILED ens{ensemble}] {exc!r}")
        traceback.print_exc()
        gc.collect()

if not ensemble_rows:
    raise RuntimeError("No ensemble predictions completed successfully.")

ensemble_df = pd.DataFrame(ensemble_rows)
ensemble_df.to_csv(os.path.join(outdir, "ensemble_predictions_2025.csv"), index=False)


# ============================================================
# 16. SUMMARY STATISTICS
# ============================================================

chance = (
    ensemble_df.groupby(["year", "percentile_bin"], observed=False)
    .size()
    .reset_index(name="n_ensembles")
)
chance["chance_percent"] = (
    100 * chance["n_ensembles"]
    / chance.groupby("year")["n_ensembles"].transform("sum")
)
chance.to_csv(os.path.join(outdir, "ensemble_percentile_bin_chances_2025.csv"), index=False)

summary = pd.DataFrame([{
    "cluster":                   cluster_to_model,
    "initialisation":            "June 2025",
    "successful_ensembles":      len(ensemble_df),
    "mean_predicted_total_ha":   ensemble_df["predicted_total_ha"].mean(),
    "median_predicted_total_ha": ensemble_df["predicted_total_ha"].median(),
    "min_predicted_total_ha":    ensemble_df["predicted_total_ha"].min(),
    "max_predicted_total_ha":    ensemble_df["predicted_total_ha"].max(),
    "mean_percentile":           ensemble_df["percentile"].mean(),
    "era5_observed_2025_ha":     float(df_eval_2025["burned_area"].sum()),
    "era5_predicted_2025_ha":    float(df_eval_2025["predicted"].sum()),
}])
summary.to_csv(os.path.join(outdir, "ensemble_summary_2025.csv"), index=False)
print("\nEnsemble Summary:")
print(summary.to_string(index=False))


# ============================================================
# 17. ENSEMBLE VISUALISATIONS
#
#  (a) Spaghetti time-series  – observed history + all 51 ensemble
#      lines fanning out at 2025, ERA5 reference marked
#  (b) Box / violin plot      – distribution of ensemble totals
#  (c) Percentile-bin bar     – chance (%) per severity category
#  (d) Spatial map            – ensemble-mean predicted burned area
#                               per 0.5° pixel
# ============================================================

ens_vals       = ensemble_df["predicted_total_ha"].values
mean_2025      = float(ens_vals.mean())
median_2025    = float(np.median(ens_vals))
p10, p90       = float(np.percentile(ens_vals, 10)), float(np.percentile(ens_vals, 90))
era5_2025_total = float(df_eval_2025["predicted"].sum())
last_obs        = float(clim_yearly.loc[clim_yearly["year"] == 2024, "burned_area"].iloc[0])

ens_colors = plt.cm.cool(
    (ens_vals - ens_vals.min()) / max(ens_vals.max() - ens_vals.min(), 1)
)

# ------------------------------------------------------------------
# (a) SPAGHETTI TIME-SERIES
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(15, 6), constrained_layout=True)

ax.fill_between([2024, 2025], [last_obs, p10], [last_obs, p90],
                color="steelblue", alpha=0.15, label="10–90th pct range")

for color, (_, row) in zip(ens_colors, ensemble_df.iterrows()):
    ax.plot([2024, 2025], [last_obs, row["predicted_total_ha"]],
            color=color, alpha=0.4, linewidth=0.8)
    ax.scatter(2025, row["predicted_total_ha"],
               color=color, s=25, alpha=0.7, zorder=3)

# Observed 2000-2024
ax.plot(clim_yearly["year"], clim_yearly["burned_area"],
        "-o", color="#1f77b4", linewidth=2, markersize=5,
        label="Observed (2000-2024)", zorder=5)

# Observed 2025 (from MODIS burn_05, cluster-masked)
obs_2025_rows = clim_yearly.loc[clim_yearly["year"] == 2025, "burned_area"]
if obs_2025_rows.empty:
    # clim_years only goes to 2024; compute directly from burn_05
    _t2025 = [t for t in burn_05.time.values if pd.Timestamp(t).year == 2025]
    if _t2025:
        obs_2025 = float(burn_05.sel(time=_t2025[0]).where(cluster == cluster_to_model).sum().values)
        obs_2025_rows = pd.Series([obs_2025])
if not obs_2025_rows.empty:
    obs_2025 = float(obs_2025_rows.iloc[0])
    ax.scatter(2025, obs_2025, color="darkorange", marker="*",
               s=250, zorder=9, label=f"Observed 2025  ({obs_2025:,.0f} ha)")
ax.plot(annual_totals.index, annual_totals["predicted"],
        "--", color="gray", linewidth=1.4, alpha=0.8,
        label="ERA5-driven model (2000-2025)", zorder=4)

ax.plot([2024, 2025], [last_obs, mean_2025],
        color="black", linewidth=2.5, zorder=6, label="SEAS5 ensemble mean")
ax.scatter(2025, mean_2025, color="black", marker="D", s=100, zorder=7)

ax.scatter(2025, era5_2025_total, color="crimson", marker="x",
           s=140, linewidths=2.5, zorder=8, label="ERA5 2025 reference")

ax.axvspan(2019.5, 2024.5, color="#dce8f5", alpha=0.4, label="Test period")
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("JJA Burned Area (ha)", fontsize=12)
ax.set_title(
    f"Cluster {cluster_to_model}: Observed 2000-2024 + "
    f"SEAS5 Jun-2025 Forecast ({len(ensemble_df)} members)",
    fontsize=13,
)
ax.set_xlim(1999.5, 2025.5)
ax.set_xticks(np.arange(2000, 2026, 5))
ax.set_ylim(bottom=0)
ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

plt.savefig(
    os.path.join(outdir, f"cluster{cluster_to_model}_spaghetti_seas5_2025.png"),
    dpi=200,
)
plt.close(fig)
print("Saved: spaghetti plot")

# ------------------------------------------------------------------
# (b) BOX / VIOLIN PLOT
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)

parts = ax.violinplot(ens_vals, positions=[1], widths=0.6,
                      showmedians=False, showextrema=False)
for pc in parts["bodies"]:
    pc.set_facecolor("steelblue")
    pc.set_alpha(0.5)

bp = ax.boxplot(ens_vals, positions=[1], widths=0.25,
                patch_artist=True, notch=False,
                boxprops=dict(facecolor="steelblue", alpha=0.6),
                medianprops=dict(color="white", linewidth=2),
                whiskerprops=dict(linewidth=1.2),
                capprops=dict(linewidth=1.2),
                flierprops=dict(marker="o", markersize=4,
                                markerfacecolor="steelblue", alpha=0.6))

ax.scatter([1] * len(ens_vals), ens_vals,
           color="steelblue", alpha=0.5, s=20, zorder=3)

ax.axhline(mean_2025, color="black", linewidth=1.8,
           linestyle="--", label=f"Ensemble mean  {mean_2025:,.0f} ha")
ax.axhline(era5_2025_total, color="crimson", linewidth=1.8,
           linestyle=":", label=f"ERA5 reference  {era5_2025_total:,.0f} ha")

# Observed 2025 (from MODIS, cluster-masked)
_obs_2025_rows = clim_yearly.loc[clim_yearly["year"] == 2025, "burned_area"]
if _obs_2025_rows.empty:
    _t2025 = [t for t in burn_05.time.values if pd.Timestamp(t).year == 2025]
    if _t2025:
        _obs_2025_rows = pd.Series([
            float(burn_05.sel(time=_t2025[0]).where(cluster == cluster_to_model).sum().values)
        ])
if not _obs_2025_rows.empty:
    obs_2025_val = float(_obs_2025_rows.iloc[0])
    ax.axhline(obs_2025_val, color="darkorange", linewidth=1.8,
               linestyle="-.", label=f"Observed 2025  {obs_2025_val:,.0f} ha")

ax.set_xticks([1])
ax.set_xticklabels(["SEAS5\nJun-2025"])
ax.set_ylabel("JJA Burned Area (ha)", fontsize=12)
ax.set_title(
    f"Cluster {cluster_to_model}: Ensemble distribution 2025\n"
    f"n={len(ensemble_df)}, mean={mean_2025:,.0f} ha, "
    f"median={median_2025:,.0f} ha",
    fontsize=11,
)
ax.legend(fontsize=9, framealpha=0.9)
ax.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=0.5)
ax.set_ylim(bottom=0)

plt.savefig(
    os.path.join(outdir, f"cluster{cluster_to_model}_violin_box_seas5_2025.png"),
    dpi=200,
)
plt.close(fig)
print("Saved: violin/box plot")

# ------------------------------------------------------------------
# (c) PERCENTILE-BIN BAR CHART
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

bin_counts = (
    ensemble_df.groupby("percentile_bin", observed=False)["ensemble"]
    .count()
    .reindex(PERCENTILE_LABELS, fill_value=0)
)
bar_colors = plt.cm.RdYlBu_r(np.linspace(0.1, 0.9, len(PERCENTILE_LABELS)))

bars = ax.bar(PERCENTILE_LABELS, bin_counts.values, color=bar_colors, edgecolor="white")

for bar, count in zip(bars, bin_counts.values):
    if count > 0:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{count}\n({100*count/len(ensemble_df):.0f}%)",
            ha="center", va="bottom", fontsize=9,
        )

ax.set_xlabel("Percentile bin (relative to 2000-2024)", fontsize=11)
ax.set_ylabel("Number of ensemble members", fontsize=11)
ax.set_title(
    f"Cluster {cluster_to_model}: SEAS5 Jun-2025 – severity distribution\n"
    f"({len(ensemble_df)} members)",
    fontsize=12,
)
ax.set_ylim(0, max(bin_counts.values) * 1.25 + 1)
ax.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=0.5)

plt.savefig(
    os.path.join(outdir, f"cluster{cluster_to_model}_percentile_bins_seas5_2025.png"),
    dpi=200,
)
plt.close(fig)
print("Saved: percentile-bin bar chart")

# ------------------------------------------------------------------
# (d) SPATIAL MAP – ensemble-mean predicted burned area per pixel
# ------------------------------------------------------------------

# Re-run predictions for all successful ensemble members, keeping pixel-level
# output this time (lightweight: we already have the trained model in memory).
pixel_sum  = None
pixel_count = 0

for ensemble in ENSEMBLES:
    try:
        met_2025 = load_seas5_met_ensemble(ensemble)
        if met_2025 is None:
            continue
        if use_sib4:
            sib_fc = load_sib_fc_ensemble(ensemble)
            if sib_fc is None:
                continue
        else:
            sib_fc = {"lagged": {}, "inseason": {}}

        pred_fields = {"cluster": cluster}
        for var in era5_met_vars:
            pred_fields[var] = safe_2d(met_2025[var])
        if use_sib4:
            if SIB_FC_HAS_LAGGED:
                for var in lagged_sib_predictors:
                    pred_fields[var] = safe_2d(sib_fc["lagged"][var])
            else:
                for var in lagged_sib_predictors:
                    pred_fields[var] = era5_lagged_pools_2025[var]
            for var in inseason_sib_predictors:
                pred_fields[var] = safe_2d(sib_fc["inseason"][var])

        ds_pred = xr.Dataset(pred_fields)
        frame   = ds_pred.stack(cell=("lat", "lon")).to_dataframe().reset_index()
        frame   = (
            frame[["lat", "lon", "cluster"] + feature_cols]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        frame["cluster"] = frame["cluster"].astype(int)
        frame = frame[frame["cluster"] == cluster_to_model].copy()
        del ds_pred

        if frame.empty:
            continue

        frame["predicted"] = np.clip(
            xgb_model.predict(frame[feature_cols].to_numpy(dtype=np.float32)),
            0, None,
        )

        # Accumulate into a lat/lon grid
        piv = frame.pivot_table(index="lat", columns="lon",
                                values="predicted", aggfunc="sum")
        da_ens = xr.DataArray(piv.values,
                              dims=("lat", "lon"),
                              coords={"lat": piv.index.values,
                                      "lon": piv.columns.values})
        if pixel_sum is None:
            pixel_sum = da_ens.fillna(0)
        else:
            pixel_sum = pixel_sum + da_ens.reindex_like(pixel_sum,
                                                         method="nearest",
                                                         fill_value=0)
        pixel_count += 1

        del met_2025, sib_fc, frame
        gc.collect()

    except Exception:
        gc.collect()

if pixel_sum is not None and pixel_count > 0:
    ens_mean_map = pixel_sum / pixel_count

    fig = plt.figure(figsize=(10, 6))
    ax  = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    proj = setup_map(ax)

    vmax = float(np.nanpercentile(ens_mean_map.values, 98))
    vmax = max(vmax, 1.0)

    im = ax.pcolormesh(
        ens_mean_map.lon.values,
        ens_mean_map.lat.values,
        ens_mean_map.values,
        transform=proj,
        cmap="YlOrRd",
        vmin=0,
        vmax=vmax,
    )
    cbar = plt.colorbar(im, ax=ax, orientation="vertical",
                        pad=0.03, shrink=0.85)
    cbar.set_label("Ensemble-mean predicted burned area (ha)", fontsize=10)

    ax.set_title(
        f"Cluster {cluster_to_model}: SEAS5 Jun-2025 ensemble-mean "
        f"JJA burned area\n({pixel_count} members)",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(outdir, f"cluster{cluster_to_model}_spatial_mean_seas5_2025.png"),
        dpi=200,
    )
    plt.close(fig)
    print("Saved: spatial ensemble-mean map")
else:
    print("Spatial map skipped: no valid ensemble pixel data.")

print("\nDone.")
print("Output directory:", outdir)
