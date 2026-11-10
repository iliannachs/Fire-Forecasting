#!/usr/bin/env python3
"""
XGBoost Monthly Burned-Area Model – Aggregated to Seasonal Total
=================================================================
Adapts baseline_code_JJA_sib.py to predict burned area per pixel per
MONTH (June, July, August separately) and then sum to a JJA seasonal total.

Key differences from the seasonal baseline:
  - MODIS target: burned area per pixel per calendar month (not "burned
    any time in JJA"). A pixel can burn in June and again in August —
    each month contributes independently.
  - ERA5 meteorology: monthly means/sums (one value per month per pixel).
  - SiB4 in-season predictors: monthly means.
  - Lagged SiB4 pools (Apr-May): same Apr-May mean for every fire month
    of a given year (the Apr-May state does not change within the season).
  - Training rows: pixel × month × year  (3× more rows than seasonal).
  - Month is added as a categorical feature so the model can learn
    month-specific relationships.
  - Validation/evaluation: monthly pixel R², plus seasonal totals obtained
    by summing June+July+August predictions per year.

NOTE on daily predictions:
  MODIS BurnDate gives the day-of-year a pixel first burned, but each pixel
  is only observed once per ~16-day overpass cycle, so a single annual value
  is recorded per pixel. There is no daily burned-area product at this
  resolution suitable for training a pixel-level daily model. Daily
  predictions would require a different target (e.g. FIRMS active fire
  radiative power) and are therefore not implemented here.
"""

import os
from glob import glob
import gc

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
use_sib4         = True

sib_hist_path = (
    "/projects/0/ctdas/ruben/forecast/sib/run_global/run/output/EU/"
    "ERA_IC_ERA_driver/sib_era_2000_2025_fires"
)
era_path = "/projects/0/ctdas/ilianna/1.Data/raw/era5_merged.nc"
modis1   = "/projects/0/ctdas/ilianna/1.Data/raw/MODIS_2000_2010.nc"
modis2   = "/projects/0/ctdas/ilianna/1.Data/raw/MODIS_2010_2025.nc"

_mode  = "meteo_sib4" if use_sib4 else "meteo_only"
outdir = f"/projects/0/ctdas/ilianna/xgb_monthly_{_mode}_cluster_{cluster_to_model}"
os.makedirs(outdir, exist_ok=True)

years_target    = np.arange(2000, 2026)
train_years     = np.arange(2000, 2020)
test_years      = np.arange(2020, 2025)
evaluation_year = 2025

target_fire_months      = [6, 7, 8]   # June, July, August
lagged_predictor_months = [4, 5]      # Apr-May for biomass pools

iberia_lon_min, iberia_lon_max = -10.5, 4.5
iberia_lat_min, iberia_lat_max = 36.5, 44.5
res05 = 0.5

k                      = 3
target_cluster         = 1
keep_largest_component = True
random_state           = 42

era5_met_vars = ["t2m", "d2m", "tmax", "tp", "windspeed", "vpd"]

n_parameter_candidates = 40
validation_folds = [
    (np.arange(2000, 2014), np.arange(2014, 2016)),
    (np.arange(2000, 2016), np.arange(2016, 2018)),
    (np.arange(2000, 2018), np.arange(2018, 2020)),
]

PFT_LOOKUP = {
    "enf": 1, "dbf": 2, "dnf": 3, "ebf": 4,
    "c3g": 5, "c4g": 6, "shb": 7, "dbg": 8, "crop": 9,
}

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
    if "valid_time" in da.coords: return "valid_time"
    if "time"       in da.coords: return "time"
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


def monthly_agg(da, agg="mean"):
    """Aggregate sub-daily data to monthly means or sums.
    Returns a DataArray with a monthly DatetimeIndex."""
    tname = get_time_name(da)
    if agg == "sum":
        out = da.resample({tname: "1MS"}).sum(skipna=True)
    else:
        out = da.resample({tname: "1MS"}).mean(skipna=True)
    return out.rename({tname: "time"}) if tname != "time" else out


def seasonal_agg_from_monthly(da_monthly, months, agg="mean"):
    """Given already-monthly data, return one value per year per pixel
    for the given months (mean or sum across those months)."""
    tname = get_time_name(da_monthly)
    sub   = da_monthly.sel({tname: da_monthly[tname].dt.month.isin(months)})
    if agg == "sum":
        out = sub.groupby(sub[tname].dt.year).sum(tname)
    else:
        out = sub.groupby(sub[tname].dt.year).mean(tname)
    out = out.rename({"year": "time"})
    valid = (out.time >= 1900) & (out.time <= 2100)
    out   = out.sel(time=valid)
    years = [int(y) for y in out.time.values]
    return out.assign_coords(
        time=[pd.Timestamp(year=y, month=1, day=1) for y in years]
    )


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
        .sortby("lat").sortby("lon")
    )


def align_to_burn_grid(da):
    da = rename_latlon(da)
    if "lon" in da.coords and float(da.lon.max()) > 180:
        da = da.assign_coords(lon=((da.lon + 180) % 360) - 180).sortby("lon")
    da2 = da.sel(lat=burn_05_seasonal.lat, lon=burn_05_seasonal.lon, method="nearest")
    da2 = da2.assign_coords(
        lat=burn_05_seasonal.lat.values,
        lon=burn_05_seasonal.lon.values,
    )
    if "time" in da2.dims:
        return da2.transpose("time", "lat", "lon")
    return da2.transpose("lat", "lon")


def align_to_burn_monthly(da):
    """Like align_to_burn_grid but for monthly DataArrays."""
    da = rename_latlon(da)
    if "lon" in da.coords and float(da.lon.max()) > 180:
        da = da.assign_coords(lon=((da.lon + 180) % 360) - 180).sortby("lon")
    da2 = da.sel(lat=burn_05_monthly.lat, lon=burn_05_monthly.lon, method="nearest")
    da2 = da2.assign_coords(
        lat=burn_05_monthly.lat.values,
        lon=burn_05_monthly.lon.values,
    )
    if "time" in da2.dims:
        return da2.transpose("time", "lat", "lon")
    return da2.transpose("lat", "lon")


def safe_2d(da, ref_da):
    out = (
        da.squeeze(drop=True)
        .assign_coords(lat=ref_da.lat.values, lon=ref_da.lon.values)
        .transpose("lat", "lon")
    )
    # Drop any residual scalar time coordinate so all fields in a Dataset
    # share the same (absent) time dimension and don't conflict.
    if "time" in out.coords:
        out = out.drop_vars("time")
    return out


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


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
# 1. MODIS BURNED AREA – MONTHLY AND SEASONAL GRIDS
# ============================================================
# Monthly target: for each calendar month, count pixels that burned
# that month (BurnDate falls within that month's day-of-year range).
# Seasonal target: pixels that burned at any point in JJA (for clustering
# and climatology — the same as the seasonal baseline).

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

# Day-of-year ranges for each fire month in a non-leap year.
# MODIS BurnDate stores day-of-year (1-366).
MONTH_DOY = {
    6: (152, 181),   # June
    7: (182, 212),   # July
    8: (213, 243),   # August
}
# Leap-year offsets (+1 from March onwards):
MONTH_DOY_LEAP = {
    6: (153, 182),
    7: (183, 213),
    8: (214, 244),
}

burn_maps_monthly  = []   # one entry per (year, month)
burn_maps_seasonal = []   # one entry per year (JJA union)

for year in years_target:
    # Select the annual MODIS layer for this year
    annual_burn = burn_da.sel(
        time=(burn_da.time.dt.year == year)
    )
    if annual_burn.time.size == 0:
        continue

    is_leap = (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0))
    doy_map  = MONTH_DOY_LEAP if is_leap else MONTH_DOY

    for month in target_fire_months:
        doy_lo, doy_hi = doy_map[month]
        # Select MODIS time steps that fall within this month
        monthly_burn = annual_burn.sel(
            time=annual_burn.time.dt.month == month
        )
        if monthly_burn.time.size == 0:
            # Try using BurnDate DOY directly (some MODIS files have one
            # time step per year with day-of-year values)
            if annual_burn.time.size > 0:
                burned_m = (
                    (annual_burn > doy_lo) & (annual_burn <= doy_hi)
                ).any("time")
            else:
                continue
        else:
            burned_m = (monthly_burn > 0).any("time")

        ba_m = burned_m * pixel_area_ha
        ba_m = ba_m.groupby_bins("lon", lon_edges, labels=lon_centers).sum()
        ba_m = ba_m.groupby_bins("lat", lat_edges, labels=lat_centers).sum()
        ba_m = ba_m.rename({"lat_bins": "lat", "lon_bins": "lon"}).transpose("lat", "lon")

        burn_maps_monthly.append(
            ba_m.assign_coords(
                time=pd.Timestamp(year=year, month=month, day=1)
            ).expand_dims("time")
        )

    # Seasonal: burned at any point in JJA
    jja = annual_burn.sel(time=annual_burn.time.dt.month.isin(target_fire_months))
    if jja.time.size == 0:
        # Fallback: use DOY range for entire JJA
        doy_lo_jja = doy_map[6][0]
        doy_hi_jja = doy_map[8][1]
        jja_burned = (
            (annual_burn > doy_lo_jja) & (annual_burn <= doy_hi_jja)
        ).any("time")
    else:
        jja_burned = (jja > 0).any("time")

    ba_s = jja_burned * pixel_area_ha
    ba_s = ba_s.groupby_bins("lon", lon_edges, labels=lon_centers).sum()
    ba_s = ba_s.groupby_bins("lat", lat_edges, labels=lat_centers).sum()
    ba_s = ba_s.rename({"lat_bins": "lat", "lon_bins": "lon"}).transpose("lat", "lon")
    burn_maps_seasonal.append(
        ba_s.assign_coords(time=pd.Timestamp(f"{year}-01-01")).expand_dims("time")
    )

burn_05_monthly  = xr.concat(burn_maps_monthly,  dim="time").sortby("lat").sortby("lon").sortby("time")
burn_05_seasonal = xr.concat(burn_maps_seasonal, dim="time").sortby("lat").sortby("lon")

ds1.close(); ds2.close()
del ds1, ds2, ds_modis, burn_da
gc.collect()
print(f"Monthly burn grid: {dict(burn_05_monthly.sizes)}")
print(f"Seasonal burn grid: {dict(burn_05_seasonal.sizes)}")


# ============================================================
# 2. FIRE-REGIME CLUSTERS  (trained on seasonal 2000-2019)
# ============================================================

print("Computing fire-regime clusters …")

burn_log_train = np.log1p(
    burn_05_seasonal.sel(time=burn_05_seasonal.time.dt.year.isin(train_years))
)
X_cluster = burn_log_train.stack(cell=("lat", "lon")).transpose("cell", "time")
valid     = X_cluster.sum("time") > 0
X_valid   = X_cluster[valid]

data_std = StandardScaler().fit_transform(X_valid.values)
labels   = KMeans(n_clusters=k, random_state=0, n_init=20).fit_predict(data_std)

cluster_1d  = xr.DataArray(labels, dims="cell", coords={"cell": X_valid.cell})
cluster_raw = cluster_1d.unstack("cell").sortby("lat").sortby("lon")

cluster_full = xr.full_like(burn_05_seasonal.isel(time=0), np.nan, dtype=float)
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
    coords={"lat": burn_05_seasonal.lat.values, "lon": burn_05_seasonal.lon.values},
    name="cluster",
)
cluster.to_netcdf(os.path.join(outdir, "fire_regime_clusters.nc"))
print(f"Cluster {cluster_to_model} selected.")


# ============================================================
# 3. ERA5 METEOROLOGY – MONTHLY MEANS
# ============================================================

print("Loading ERA5 meteorology (monthly aggregation) …")

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

# Monthly aggregation for each met variable
met_monthly = {}
for met_var in era5_met_vars:
    agg    = "sum" if met_var == "tp" else "mean"
    da_mon = monthly_agg(ds_era[met_var], agg=agg)
    da_mon = rename_latlon(da_mon).sortby("lat").sortby("lon")
    # Keep only JJA months of target years
    da_mon = da_mon.sel(
        time=(da_mon.time.dt.month.isin(target_fire_months))
              & da_mon.time.dt.year.isin(years_target)
    )
    met_monthly[met_var] = align_to_burn_monthly(da_mon)

ds_era.close(); del ds_era; gc.collect()
print("ERA5 monthly variables loaded:", list(met_monthly.keys()))


# ============================================================
# 4. HISTORICAL SiB4-ERA5 PREDICTORS – MONTHLY
# ============================================================

if use_sib4:
    print("Loading historical SiB4 predictors (monthly) …")

    files = sorted(glob(os.path.join(sib_hist_path, "qsib_*.lu.qp3.nc")))
    if not files:
        raise FileNotFoundError(f"No SiB4 files in {sib_hist_path}")

    keep_vars = ["latsib", "lonsib", "lu_area", "lu_pref",
                 "pft_names", "pft_refnums"] + needed_sib_vars

    with xr.open_dataset(files[0], decode_times=True) as ds_chk:
        missing = [v for v in keep_vars if v not in ds_chk.variables]
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

    # Lagged predictors: Apr-May mean (one value per year, repeated for
    # each fire month within that year)
    sib_lagged_yearly = {}
    for lbl in lagged_sib_predictors:
        sib_var, pft_code = parse_label(lbl)
        if sib_var not in ds_sib_hist or pft_code not in pft_lookup_hist:
            print(f"  Skipping '{lbl}': unavailable"); continue
        tgt_ref = pft_lookup_hist[pft_code]
        da_land = build_pft_field_from_nlu(ds_sib_hist, sib_var, tgt_ref)
        da_grid = to_latlon_grid(da_land, ds_sib_hist).sel(
            lon=slice(iberia_lon_min, iberia_lon_max),
            lat=slice(iberia_lat_min, iberia_lat_max),
        ).sortby("lat").sortby("lon")
        # Apr-May mean → one value per year
        da_yr = seasonal_agg_from_monthly(da_grid, lagged_predictor_months, agg="mean")
        da_yr = da_yr.sel(time=da_yr.time.dt.year.isin(years_target))
        sib_lagged_yearly[lbl] = align_to_burn_grid(da_yr)

    # In-season predictors: monthly mean for each fire month
    sib_inseason_monthly = {}
    for lbl in inseason_sib_predictors:
        sib_var, pft_code = parse_label(lbl)
        if sib_var not in ds_sib_hist or pft_code not in pft_lookup_hist:
            print(f"  Skipping '{lbl}': unavailable"); continue
        tgt_ref = pft_lookup_hist[pft_code]
        da_land = build_pft_field_from_nlu(ds_sib_hist, sib_var, tgt_ref)
        da_grid = to_latlon_grid(da_land, ds_sib_hist).sel(
            lon=slice(iberia_lon_min, iberia_lon_max),
            lat=slice(iberia_lat_min, iberia_lat_max),
        ).sortby("lat").sortby("lon")
        # Keep only JJA months of target years (SiB4 is already monthly)
        da_mon = da_grid.sel(
            time=(da_grid.time.dt.month.isin(target_fire_months))
                  & da_grid.time.dt.year.isin(years_target)
        )
        sib_inseason_monthly[lbl] = align_to_burn_monthly(da_mon)

    available_lagged    = [v for v in lagged_sib_predictors   if v in sib_lagged_yearly]
    available_inseason  = [v for v in inseason_sib_predictors if v in sib_inseason_monthly]
    available_sib_vars  = available_lagged + available_inseason
    missing_sib         = [v for v in selected_sib_vars if v not in available_sib_vars]
    if missing_sib:
        raise ValueError(f"Could not construct SiB4 predictors: {missing_sib}")
    print("SiB4 predictors loaded:", available_sib_vars)
else:
    sib_lagged_yearly   = {}
    sib_inseason_monthly= {}
    available_lagged    = []
    available_inseason  = []
    available_sib_vars  = []


# ============================================================
# 5. BUILD MONTHLY PIXEL TABLE
# ============================================================

def build_monthly_pixel_table(only_years):
    """
    Returns a DataFrame with one row per pixel × month × year.
    Columns: year, month, lat, lon, cluster, burned_area, <features>
    """
    rows = []
    all_sib_vars = available_lagged + available_inseason

    for t in burn_05_monthly.time.values:
        ts    = pd.Timestamp(t)
        year  = ts.year
        month = ts.month
        if year not in only_years:
            continue

        data_vars = {
            "burned_area": burn_05_monthly.sel(time=t).squeeze(drop=True),
            "cluster":     cluster,
        }

        # Monthly met predictors
        for mv in era5_met_vars:
            da_m = met_monthly[mv]
            idx  = np.where(
                (da_m.time.dt.year.values == year) & (da_m.time.dt.month.values == month)
            )[0]
            if idx.size:
                data_vars[mv] = safe_2d(da_m.isel(time=int(idx[0])), burn_05_monthly)

        if use_sib4:
            # Lagged (Apr-May) pools: look up by year only
            for lbl in available_lagged:
                da_y = sib_lagged_yearly[lbl]
                idx  = np.where(da_y.time.dt.year.values == year)[0]
                if idx.size:
                    data_vars[lbl] = safe_2d(da_y.isel(time=int(idx[0])), burn_05_seasonal)

            # In-season: look up by year AND month
            for lbl in available_inseason:
                da_m = sib_inseason_monthly[lbl]
                idx  = np.where(
                    (da_m.time.dt.year.values == year) & (da_m.time.dt.month.values == month)
                )[0]
                if idx.size:
                    data_vars[lbl] = safe_2d(da_m.isel(time=int(idx[0])), burn_05_monthly)

        required = era5_met_vars + all_sib_vars
        missing  = [v for v in required if v not in data_vars]
        if missing:
            print(f"  Skipping {year}-{month:02d}: missing {missing}")
            continue

        df_m = (
            xr.Dataset(data_vars)
            .stack(cell=("lat", "lon"))
            .to_dataframe()
            .reset_index()
        )
        df_m["year"]  = year
        df_m["month"] = month
        rows.append(df_m)

    if not rows:
        raise ValueError(f"No monthly pixel data for years {sorted(only_years)}.")

    df = pd.concat(rows, ignore_index=True)
    df = df[["year", "month", "lat", "lon", "cluster", "burned_area"]
            + era5_met_vars + all_sib_vars]
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df[np.isfinite(df["cluster"])].copy()
    df["cluster"] = df["cluster"].astype(int)
    df = df[df["cluster"] == cluster_to_model].copy()
    if df.empty:
        raise ValueError(f"No valid pixels for cluster {cluster_to_model}.")
    return df


df_train     = build_monthly_pixel_table(set(train_years))
df_test      = build_monthly_pixel_table(set(test_years))
df_eval_2025 = build_monthly_pixel_table({evaluation_year})

# Month is added as a feature so the model can distinguish June/July/August
feature_cols = (
    ["month"]
    + [v for v in era5_met_vars + available_sib_vars
       if v in df_train.columns and v in df_test.columns and v in df_eval_2025.columns]
)

print("\nFeatures:", feature_cols)
print(f"Train rows: {len(df_train)}, Test: {len(df_test)}, ERA5-2025: {len(df_eval_2025)}")

X_train = df_train[feature_cols];  y_train = df_train["burned_area"]
X_test  = df_test[feature_cols];   y_test  = df_test["burned_area"]
X_2025  = df_eval_2025[feature_cols]; y_2025 = df_eval_2025["burned_area"]


# ============================================================
# 6. HYPERPARAMETER TUNING  (annual seasonal R² as objective)
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
    ParameterSampler(param_distributions, n_iter=n_parameter_candidates,
                     random_state=random_state)
)
max_rows_per_fold = 150_000


def sample_rows(frame, max_rows, seed):
    if len(frame) <= max_rows:
        return frame
    # Sample proportionally across year-month combinations
    groups = frame.groupby(["year", "month"])
    quota  = max(1, max_rows // len(groups))
    parts  = [g.sample(n=min(len(g), quota), random_state=seed) for _, g in groups]
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
        fit_sample = sample_rows(fit_df, max_rows_per_fold, random_state + fold_idx)
        mdl = make_xgb(params)
        mdl.fit(fit_sample[feature_cols], fit_sample["burned_area"], verbose=False)
        valid_df["pred"] = np.clip(mdl.predict(valid_df[feature_cols]), 0, None)
        # Score on annual SEASONAL totals (sum across all months per year)
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

pd.DataFrame([
    {"candidate": r["candidate"],
     "mean_validation_annual_r2": r["mean_validation_annual_r2"],
     "std_validation_annual_r2":  r["std_validation_annual_r2"],
     **r["params"]}
    for r in search_results
]).to_csv(os.path.join(outdir, "xgboost_hyperparameter_search.csv"), index=False)

pd.DataFrame([{**best_params,
               "best_mean_validation_annual_r2": best_result["mean_validation_annual_r2"],
               "validation_annual_r2_std": best_result["std_validation_annual_r2"]}
              ]).to_csv(os.path.join(outdir, "best_xgboost_parameters.csv"), index=False)


# ============================================================
# 7. FIT FINAL MODEL
# ============================================================

xgb_model = make_xgb(best_params)
xgb_model.fit(X_train, y_train, verbose=False)
xgb_model.save_model(os.path.join(outdir, "xgb_model.json"))

for frame, X in [(df_train, X_train), (df_test, X_test), (df_eval_2025, X_2025)]:
    frame["predicted"]  = np.clip(xgb_model.predict(X), 0, None)
    frame["difference"] = frame["predicted"] - frame["burned_area"]


# ============================================================
# 8. EVALUATION METRICS
# ============================================================

df_all = pd.concat([df_train, df_test, df_eval_2025], ignore_index=True)

# Monthly totals
monthly_totals = df_all.groupby(["year", "month"])[["burned_area", "predicted"]].sum()

# Seasonal totals (sum of June+July+August per year)
seasonal_totals = df_all.groupby("year")[["burned_area", "predicted"]].sum()
train_seas = seasonal_totals.loc[seasonal_totals.index.isin(train_years)]
test_seas  = seasonal_totals.loc[seasonal_totals.index.isin(test_years)]

metrics = {
    "cluster":  cluster_to_model,
    "features": ", ".join(feature_cols),
    # Pixel-level
    "pixel_train_r2":     r2_score(df_train["burned_area"],     df_train["predicted"]),
    "pixel_test_r2":      r2_score(df_test["burned_area"],      df_test["predicted"]),
    "pixel_eval_2025_r2": r2_score(df_eval_2025["burned_area"], df_eval_2025["predicted"]),
    # Seasonal totals
    "seasonal_train_r2":    r2_score(train_seas["burned_area"],    train_seas["predicted"]),
    "seasonal_test_r2":     r2_score(test_seas["burned_area"],     test_seas["predicted"]),
    "seasonal_overall_r2":  r2_score(seasonal_totals["burned_area"], seasonal_totals["predicted"]),
    # RMSE
    "train_rmse":    rmse(df_train["burned_area"],     df_train["predicted"]),
    "test_rmse":     rmse(df_test["burned_area"],      df_test["predicted"]),
    "eval_2025_rmse":rmse(df_eval_2025["burned_area"], df_eval_2025["predicted"]),
    # 2025 totals
    "observed_seasonal_2025":  float(df_eval_2025["burned_area"].sum()),
    "predicted_seasonal_2025": float(df_eval_2025["predicted"].sum()),
    "best_mean_validation_r2": best_result["mean_validation_annual_r2"],
}
pd.DataFrame([metrics]).to_csv(os.path.join(outdir, "metrics.csv"), index=False)
print("\nFINAL METRICS")
print(pd.DataFrame([metrics]).T)


# ============================================================
# 9. FEATURE IMPORTANCE
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
plt.title(f"Cluster {cluster_to_model}: feature importance (monthly model)")
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
# 10. SAVE TABLES
# ============================================================

df_train.to_csv(    os.path.join(outdir, "training_table_monthly_2000_2019.csv"),  index=False)
df_test.to_csv(     os.path.join(outdir, "test_table_monthly_2020_2024.csv"),       index=False)
df_eval_2025.to_csv(os.path.join(outdir, "evaluation_monthly_era5_2025.csv"),       index=False)
seasonal_totals.to_csv(os.path.join(outdir, "seasonal_totals_all_years.csv"))
monthly_totals.to_csv( os.path.join(outdir, "monthly_totals_all_years.csv"))


# ============================================================
# 11. PLOTS
# ============================================================

# (a) Seasonal reconstruction from summed monthly predictions
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(seasonal_totals.index, seasonal_totals["burned_area"],
        marker="o", color="#1f77b4", linewidth=1.5, markersize=4, label="Observed")
ax.plot(seasonal_totals.index, seasonal_totals["predicted"],
        marker="s", color="#ff7f0e", linewidth=1.5, markersize=4,
        label="Predicted (sum of monthly)")
ax.axvspan(2019.5, 2024.5, color="#dce8f5", alpha=0.6, label="Test period")
ax.axvspan(2024.5, 2025.5, color="#b8d3ec", alpha=0.8, label="ERA5 2025 (reference)")
ax.set_title(f"Cluster {cluster_to_model}: JJA seasonal total (from monthly predictions)",
             fontsize=13)
ax.set_xlabel("Year"); ax.set_ylabel("Burned area (ha)")
ax.set_xlim(1999.5, 2025.5)
ax.set_xticks(np.arange(2000, 2026, 5))
ax.set_ylim(bottom=0)
ax.grid(True, linestyle="-", linewidth=0.5, alpha=0.3)
ax.legend(loc="upper left")
plt.tight_layout()
plt.savefig(
    os.path.join(outdir, f"seasonal_reconstruction_cluster_{cluster_to_model}.png"),
    dpi=300,
)
plt.close()

# (b) Monthly breakdown: observed vs predicted per month across years
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True, constrained_layout=True)
month_names = {6: "June", 7: "July", 8: "August"}
for ax, month in zip(axes, target_fire_months):
    df_m = monthly_totals.xs(month, level="month")
    ax.plot(df_m.index, df_m["burned_area"],
            marker="o", linewidth=1.5, markersize=3, label="Observed")
    ax.plot(df_m.index, df_m["predicted"],
            marker="s", linewidth=1.5, markersize=3, label="Predicted")
    ax.axvspan(2019.5, 2024.5, color="#dce8f5", alpha=0.5)
    ax.set_title(month_names[month], fontsize=12)
    ax.set_xlabel("Year")
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.4)
    if ax == axes[0]:
        ax.set_ylabel("Burned area (ha)")
        ax.legend(fontsize=8)
fig.suptitle(f"Cluster {cluster_to_model}: monthly burned area", fontsize=13)
plt.savefig(
    os.path.join(outdir, f"monthly_breakdown_cluster_{cluster_to_model}.png"),
    dpi=200,
)
plt.close()

print("\nDone.")
print("Output directory:", outdir)
