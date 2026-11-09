"""Reading of the raw input data (MODIS burned area, ERA5, SiB4)."""
import os
from glob import glob

import numpy as np
import xarray as xr


def mask_fill_values(da):
    """Replace _FillValue / missing_value entries by NaN."""
    fill = da.attrs.get("_FillValue", None)
    if fill is not None:
        da = da.where(da != fill)
    missing_value = da.attrs.get("missing_value", None)
    if missing_value is not None:
        da = da.where(da != missing_value)
    return da


def get_time_name(da):
    """ERA5 files use 'valid_time', other files 'time'."""
    if "valid_time" in da.coords:
        return "valid_time"
    if "time" in da.coords:
        return "time"
    raise ValueError("No time coordinate found.")


class DataReader:
    """Opens the raw datasets and applies the spatial subsetting of the original script."""

    def __init__(self, settings):
        self.s = settings

    # ------------------------------------------------------------
    # MODIS
    # ------------------------------------------------------------
    def read_modis_burn_date(self):
        """Combined MODIS BurnDate over Iberia (duplicate timestamps removed)."""
        s = self.s
        datasets = [xr.open_dataset(p) for p in s.MODIS_PATHS]

        ds_modis = xr.concat(datasets, dim="time").sortby("time")
        ds_modis = ds_modis.sel(time=~ds_modis.get_index("time").duplicated())

        burn_da = ds_modis["BurnDate"].sel(
            lon=slice(s.IBERIA_LON_MIN, s.IBERIA_LON_MAX),
            lat=slice(s.IBERIA_LAT_MAX, s.IBERIA_LAT_MIN),
        )

        fill = burn_da.attrs.get("_FillValue", None)
        if fill is not None:
            burn_da = burn_da.where(burn_da != fill)

        return burn_da

    # ------------------------------------------------------------
    # ERA5
    # ------------------------------------------------------------
    def read_era5(self, needed_variables):
        """
        ERA5 dataset over Iberia containing the requested variables.
        Derived variables (windspeed, vpd, tmax) are computed when requested.
        """
        s = self.s
        ds_era = xr.open_dataset(s.ERA5_PATH)

        # Longitudes in 0..360 are converted to -180..180.
        if "longitude" in ds_era.coords and float(ds_era.longitude.max()) > 180:
            ds_era = ds_era.assign_coords(
                longitude=((ds_era.longitude + 180) % 360) - 180
            ).sortby("longitude")

        ds_era = ds_era.sel(
            longitude=slice(s.IBERIA_LON_MIN, s.IBERIA_LON_MAX),
            latitude=slice(s.IBERIA_LAT_MAX, s.IBERIA_LAT_MIN),
        )

        needed = set(needed_variables)

        if "vpd" in needed:
            # Temperatures from Kelvin to Celsius
            t2m_c = ds_era["t2m"] - 273.15
            d2m_c = ds_era["d2m"] - 273.15
            # Saturation vapour pressure (kPa)
            es_t2m = 0.6108 * np.exp((17.27 * t2m_c) / (t2m_c + 237.3))
            # Actual vapour pressure (kPa), estimated from dewpoint
            ea_d2m = 0.6108 * np.exp((17.27 * d2m_c) / (d2m_c + 237.3))
            # Vapour pressure deficit (kPa)
            ds_era["vpd"] = (es_t2m - ea_d2m).clip(min=0)

        if "windspeed" in needed:
            ds_era["windspeed"] = np.sqrt(ds_era["u10"] ** 2 + ds_era["v10"] ** 2)

        if "tmax" in needed:
            tname = get_time_name(ds_era["t2m"])
            ds_era["tmax"] = ds_era["t2m"].resample({tname: "1D"}).max()

        missing = [v for v in needed if v not in ds_era]
        if missing:
            raise KeyError(f"ERA5 variables not available: {missing}")

        return ds_era

    # ------------------------------------------------------------
    # SiB4
    # ------------------------------------------------------------
    def read_sib4(self, needed_variables):
        """
        Multi-file SiB4 dataset (monthly *.lu.qp3.nc) restricted to the needed
        variables, plus a lookup {PFT code: PFT reference number}.
        """
        s = self.s
        files = sorted(glob(os.path.join(str(s.SIB4_DIR), s.SIB4_FILE_PATTERN)))
        if not files:
            raise FileNotFoundError(
                f"No SiB4 files matching {s.SIB4_FILE_PATTERN} were found in {s.SIB4_DIR}"
            )

        keep_vars = [
            "latsib", "lonsib", "lu_area", "lu_pref", "pft_names", "pft_refnums",
        ] + sorted(set(needed_variables))

        # Check requested variables before opening the full multi-file dataset.
        with xr.open_dataset(files[0], decode_times=True, engine="netcdf4") as ds_check:
            missing = [v for v in keep_vars if v not in ds_check.variables]
        if missing:
            raise KeyError(
                "The first SiB4 file is missing required fields: " + ", ".join(missing)
            )

        ds_sib = xr.open_mfdataset(
            files,
            decode_times=True,
            engine="netcdf4",
            combine="by_coords",
            parallel=False,
            data_vars="minimal",
            coords="minimal",
            compat="override",
        )[keep_vars]

        ds_sib["pft_names"] = ds_sib["pft_names"].astype(str)

        pft_codes = ds_sib["pft_names"].values.astype(str)
        pft_refs = ds_sib["pft_refnums"].values.astype(int)
        return ds_sib, dict(zip(pft_codes, pft_refs))
