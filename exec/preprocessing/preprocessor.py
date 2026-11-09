"""
Preprocessing: target burned-area grid, fire-regime clusters, yearly predictor
fields and the pixel tables used for training/testing.

Expensive steps are cached (see exec/cache.py) and only recomputed when the
cache file is missing or OVERWRITE_CACHE = True.
"""
import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from exec.data.reader import get_time_name, mask_fill_values


def seasonal_agg(da, months, agg="mean"):
    """Aggregate the given calendar months of every year (mean or sum)."""
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
    """'cwt_stem_enf' -> ('cwt_stem', 'enf')."""
    parts = label.split("_")
    return "_".join(parts[:-1]), parts[-1]


def build_pft_field_from_nlu(ds, varname, target_refnum):
    """Area-weighted sum of a land-unit variable over all land units of one PFT."""
    da = mask_fill_values(ds[varname])
    mask = ds["lu_pref"] == target_refnum
    return (da * ds["lu_area"]).where(mask).sum("nlu", skipna=True)


def to_latlon_grid(da_land, ds_sib):
    """Convert SiB4 land points to a regular lat/lon grid."""
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

class Preprocessor:
    """Builds everything that is needed before a model can be trained."""

    def __init__(self, settings, reader, cache):
        self.s = settings
        self.reader = reader
        self.cache = cache
        self.burn_05 = None   # burned area on the 0.5 degree grid (target)
        self.cluster = None   # fire-regime cluster map

    # ------------------------------------------------------------
    # helpers (grid alignment)
    # ------------------------------------------------------------
    def _align_to_burn_grid(self, da):
        """Nearest-neighbour regridding of a predictor onto the burned-area grid."""
        burn = self.burn_05
        da2 = da.sel(lat=burn.lat, lon=burn.lon, method="nearest")
        da2 = da2.assign_coords(lat=burn.lat.values, lon=burn.lon.values)
        if "time" in da2.dims:
            return da2.transpose("time", "lat", "lon")
        return da2.transpose("lat", "lon")

    def _safe_2d(self, da):
        return (
            da.squeeze(drop=True)
            .assign_coords(lat=self.burn_05.lat.values, lon=self.burn_05.lon.values)
            .transpose("lat", "lon")
        )

    def _domain_key(self):
        s = self.s
        return {
            "lon": [s.IBERIA_LON_MIN, s.IBERIA_LON_MAX],
            "lat": [s.IBERIA_LAT_MIN, s.IBERIA_LAT_MAX],
            "years": list(s.YEARS_TARGET),
        }

    # ------------------------------------------------------------
    # 1. MODIS burned area on the 0.5 degree grid
    # ------------------------------------------------------------
    def burned_area_grid(self):
        key = {
            **self._domain_key(),
            "modis": [str(p) for p in self.s.MODIS_PATHS],
            "res": self.s.GRID_RESOLUTION,
            "months": list(self.s.TARGET_FIRE_MONTHS),
        }
        self.burn_05 = self.cache.get_or_compute(
            "burned_area_0.5deg", key, self._compute_burned_area_grid
        )
        return self.burn_05

    def _compute_burned_area_grid(self):
        s = self.s
        burn_da = self.reader.read_modis_burn_date()

        dlat = float(abs(burn_da.lat.diff("lat").median()))
        dlon = float(abs(burn_da.lon.diff("lon").median()))

        # Pixel area (ha) from spherical geometry: dy = R dlat, dx = R dlon cos(lat)
        R = 6371000.0
        deg2rad = np.pi / 180.0
        dy = R * deg2rad * dlat
        dx = R * deg2rad * dlon * np.cos(np.deg2rad(burn_da.lat))
        pixel_area_ha = (dx * dy) / 1e4

        res = s.GRID_RESOLUTION
        lat_edges = np.arange(s.IBERIA_LAT_MIN, s.IBERIA_LAT_MAX + res, res)
        lon_edges = np.arange(s.IBERIA_LON_MIN, s.IBERIA_LON_MAX + res, res)
        lat_centers = lat_edges[:-1] + res / 2
        lon_centers = lon_edges[:-1] + res / 2

        burn_maps_05 = []
        for year in s.YEARS_TARGET:
            jja = burn_da.sel(
                time=(
                    (burn_da.time.dt.year == year)
                    & (burn_da.time.dt.month.isin(s.TARGET_FIRE_MONTHS))
                )
            )
            if jja.time.size == 0:
                continue

            # A pixel counts as burned once if it burned on any day of the season.
            burned_any = (jja > 0).any("time")
            burned_area_native_ha = burned_any * pixel_area_ha

            ba_lon = burned_area_native_ha.groupby_bins(
                "lon", lon_edges, labels=lon_centers
            ).sum()
            ba05 = ba_lon.groupby_bins("lat", lat_edges, labels=lat_centers).sum()
            ba05 = ba05.rename({"lat_bins": "lat", "lon_bins": "lon"}).transpose("lat", "lon")

            burn_maps_05.append(
                ba05.assign_coords(time=pd.Timestamp(f"{year}-01-01")).expand_dims("time")
            )

        burn_05 = xr.concat(burn_maps_05, dim="time").sortby("lat").sortby("lon")
        return burn_05.load()

    # ------------------------------------------------------------
    # 2. Fire-regime clusters (K-means on training years only)
    # ------------------------------------------------------------
    def fire_clusters(self):
        s = self.s
        key = {
            **self._domain_key(),
            "modis": [str(p) for p in s.MODIS_PATHS],
            "res": s.GRID_RESOLUTION,
            "months": list(s.TARGET_FIRE_MONTHS),
            "train_years": list(s.TRAIN_YEARS),
            "k": s.K,
            "target_cluster": s.TARGET_CLUSTER,
            "keep_largest": s.KEEP_LARGEST_COMPONENT,
            "kmeans_random_state": s.KMEANS_RANDOM_STATE,
            "kmeans_n_init": s.KMEANS_N_INIT,
        }
        self.cluster = self.cache.get_or_compute(
            "fire_clusters", key, self._compute_fire_clusters
        )
        return self.cluster

    def _compute_fire_clusters(self):
        s = self.s
        burn_05 = self.burn_05

        # Log transform stabilises the skewed burned-area totals.
        burn_log = np.log1p(burn_05)

        # Only the training years are used so no future information enters the clusters.
        burn_log_train = burn_log.sel(time=burn_log.time.dt.year.isin(s.TRAIN_YEARS))

        X_cluster = burn_log_train.stack(cell=("lat", "lon")).transpose("cell", "time")
        valid = X_cluster.sum("time") > 0
        X_valid = X_cluster[valid]

        data_std = StandardScaler().fit_transform(X_valid.values)

        labels = KMeans(
            n_clusters=s.K,
            random_state=s.KMEANS_RANDOM_STATE,
            n_init=s.KMEANS_N_INIT,
        ).fit_predict(data_std)

        cluster_1d = xr.DataArray(
            labels, dims=("cell",), coords={"cell": X_valid.cell}, name="cluster"
        )
        cluster_raw = cluster_1d.unstack("cell").sortby("lat").sortby("lon")

        cluster_full = xr.full_like(burn_05.isel(time=0), np.nan, dtype=float)
        cluster_full.loc[dict(lat=cluster_raw.lat, lon=cluster_raw.lon)] = cluster_raw.astype(float)
        cluster_vals = cluster_full.values.copy()

        # Spatial cleanup: keep only the largest connected (8-neighbour) body of
        # TARGET_CLUSTER; isolated cells get the majority cluster of their neighbours.
        target_mask = np.isfinite(cluster_vals) & (cluster_vals == s.TARGET_CLUSTER)
        labeled, ncomp = ndimage.label(target_mask, structure=np.ones((3, 3), dtype=int))
        remove_mask = np.zeros_like(target_mask, dtype=bool)

        if ncomp > 1 and s.KEEP_LARGEST_COMPONENT:
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
                        if np.isfinite(val) and val != s.TARGET_CLUSTER:
                            neighbors.append(int(val))

            if neighbors:
                new_cluster[i, j] = np.argmax(np.bincount(neighbors, minlength=s.K))
            else:
                new_cluster[i, j] = np.nan

        return xr.DataArray(
            new_cluster,
            dims=("lat", "lon"),
            coords={"lat": burn_05.lat.values, "lon": burn_05.lon.values},
            name="cluster",
        )

    # ------------------------------------------------------------
    # 3. ERA5 predictors (yearly fields on the burned-area grid)
    # ------------------------------------------------------------
    def meteorology(self):
        """Dict {feature name: DataArray(time, lat, lon)} for the ERA5 predictors."""
        s = self.s
        specs = s.ERA5_PREDICTORS
        met_yearly = {}
        ds_era = None   # opened lazily, only if something is not cached yet

        for name, spec in specs.items():
            var = spec.get("variable", name)
            months, agg = list(spec["months"]), spec.get("agg", "mean")

            key = {
                **self._domain_key(),
                "era5": str(s.ERA5_PATH), "variable": var,
                "months": months, "agg": agg,
                "grid": [s.GRID_RESOLUTION, list(map(float, self.burn_05.lat.values)),
                         list(map(float, self.burn_05.lon.values))],
            }

            def compute(var=var, months=months, agg=agg):
                nonlocal ds_era
                if ds_era is None:
                    ds_era = self.reader.read_era5(
                        {spec.get("variable", n) for n, spec in specs.items()}
                    )
                met_year = seasonal_agg(ds_era[var], months=months, agg=agg)
                met_year = met_year.rename(
                    {"latitude": "lat", "longitude": "lon"}
                ).sortby("lat").sortby("lon")
                met_year = met_year.sel(time=met_year.time.dt.year.isin(s.YEARS_TARGET))
                return self._align_to_burn_grid(met_year).load()

            met_yearly[name] = self.cache.get_or_compute(f"era5_{name}", key, compute)

        print("Meteorological variables:", list(met_yearly.keys()))
        return met_yearly

    # ------------------------------------------------------------
    # 4. SiB4 predictors (yearly fields on the burned-area grid)
    # ------------------------------------------------------------
    def sib4(self):
        """Dict {feature name: DataArray(time, lat, lon)} for the SiB4 predictors."""
        s = self.s
        if not s.USE_SIB4:
            return {}

        specs = s.SIB4_PREDICTORS_BY_CLUSTER.get(s.CLUSTER_TO_MODEL)
        if not specs:
            raise ValueError(
                f"USE_SIB4 = True but no SiB4 predictors defined for cluster {s.CLUSTER_TO_MODEL}."
            )

        sib_yearly = {}
        opened = None   # (dataset, PFT lookup), opened lazily

        for name, spec in specs.items():
            label = spec.get("variable", name)
            months, agg = list(spec["months"]), spec.get("agg", "mean")

            key = {
                **self._domain_key(),
                "sib4_dir": str(s.SIB4_DIR), "label": label,
                "months": months, "agg": agg,
                "grid": [s.GRID_RESOLUTION, list(map(float, self.burn_05.lat.values)),
                         list(map(float, self.burn_05.lon.values))],
            }

            def compute(label=label, months=months, agg=agg):
                nonlocal opened
                if opened is None:
                    needed = {parse_label(spec.get("variable", n))[0] for n, spec in specs.items()}
                    opened = self.reader.read_sib4(needed)
                ds_sib, pft_lookup = opened

                sib_var, pft_code = parse_label(label)
                if sib_var not in ds_sib or pft_code not in pft_lookup:
                    raise ValueError(
                        f"SiB4 predictor {label}: variable or PFT code unavailable"
                    )

                da_land = build_pft_field_from_nlu(ds_sib, sib_var, pft_lookup[pft_code])
                da_grid = to_latlon_grid(da_land, ds_sib)
                da_grid = da_grid.sel(
                    lon=slice(s.IBERIA_LON_MIN, s.IBERIA_LON_MAX),
                    lat=slice(s.IBERIA_LAT_MIN, s.IBERIA_LAT_MAX),
                ).sortby("lat").sortby("lon")

                da_year = seasonal_agg(da_grid, months, agg=agg)
                da_year = da_year.sel(time=da_year.time.dt.year.isin(s.YEARS_TARGET))
                return self._align_to_burn_grid(da_year).load()

            sib_yearly[name] = self.cache.get_or_compute(f"sib4_{name}", key, compute)

        print("SiB4 predictors:", list(sib_yearly.keys()))
        return sib_yearly

    # ------------------------------------------------------------
    # 5. Pixel tables
    # ------------------------------------------------------------
    def _build_pixel_table(self, only_years, predictors, feature_cols):
        """One row per grid cell and year of the modelled cluster."""
        rows = []

        for t in self.burn_05.time.values:
            year = pd.Timestamp(t).year
            if year not in only_years:
                continue

            data_vars = {
                "burned_area": self.burn_05.sel(time=t).squeeze(drop=True),
                "cluster": self.cluster,
            }

            for name, da in predictors.items():
                matching = da.time.dt.year.values == year
                if matching.any():
                    data_vars[name] = self._safe_2d(da.isel(time=np.where(matching)[0][0]))

            missing = [v for v in feature_cols if v not in data_vars]
            if missing:
                print(f"Skipping {year}: missing {missing}")
                continue

            # (lat, lon) row order is kept identical to the original stack(cell=("lat", "lon")).
            df_year = (
                xr.Dataset(data_vars)
                .to_dataframe(dim_order=["lat", "lon"])
                .reset_index()
            )
            df_year["year"] = year
            rows.append(df_year)

        if not rows:
            raise ValueError(f"No pixel data found for years {sorted(only_years)}.")

        df = pd.concat(rows, ignore_index=True)
        df = df[["year", "lat", "lon", "cluster", "burned_area"] + feature_cols]
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        df = df[np.isfinite(df["cluster"])].copy()
        df["cluster"] = df["cluster"].astype(int)
        df = df[df["cluster"] == self.s.CLUSTER_TO_MODEL].copy()

        if df.empty:
            raise ValueError(
                f"No valid pixels found for cluster {self.s.CLUSTER_TO_MODEL}. "
                "Check K, CLUSTER_TO_MODEL and the spatial mask."
            )
        return df

    def prepare(self):
        """
        Runs all preprocessing steps and returns
        (df_train, df_test, df_eval, feature_cols).
        """
        s = self.s
        self.burned_area_grid()
        self.fire_clusters()

        predictors = {}
        predictors.update(self.meteorology())
        sib = self.sib4()
        duplicate = set(predictors) & set(sib)
        if duplicate:
            raise ValueError(f"Predictor names used by both ERA5 and SiB4: {duplicate}")
        predictors.update(sib)

        feature_cols = list(predictors.keys())
        if not feature_cols:
            raise ValueError("No predictors selected in settings.py.")

        df_train = self._build_pixel_table(set(s.TRAIN_YEARS), predictors, feature_cols)
        df_test = self._build_pixel_table(set(s.TEST_YEARS), predictors, feature_cols)
        df_eval = self._build_pixel_table({s.EVALUATION_YEAR}, predictors, feature_cols)

        print("\nFeatures used:")
        for f in feature_cols:
            print(" ", f)
        print("\nNumber of rows:")
        print("Train:", len(df_train))
        print("Test:", len(df_test))
        print("Evaluation:", len(df_eval))

        return df_train, df_test, df_eval, feature_cols
