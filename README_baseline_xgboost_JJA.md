# XGBoost Baseline Fire-Regime Model

This repository contains a baseline XGBoost regression workflow for estimating annual June–August (JJA) burned area over the Iberian Peninsula using ERA5 meteorological predictors and a MODIS burned-area target.

## Baseline predictors

The baseline model uses exactly four predictors:

- `t2m`: mean JJA 2 m air temperature
- `d2m`: mean JJA 2 m dew-point temperature
- `tp`: total JJA precipitation, converted from metres to millimetres
- `windspeed`: mean JJA wind speed calculated from 10 m `u` and `v` wind components

## Time periods

| Purpose | Years |
|---|---|
| Fire-regime clustering | 2001–2019 |
| Model training | 2001–2019 |
| Independent test | 2020–2024 |
| Final evaluation | 2025 |

The spatial fire-regime clusters are calculated from each grid cell's mean annual observed burned area during **2001–2019 only**. The 2020–2025 observations are not used to define the cluster labels. The model is tuned using expanding-window validation blocks within the training period.

**Model-selection criterion:** annual-total test R² over 2020–2024. Pixel-level regression metrics and the 2025 evaluation are not used to select model settings.

## Files

- `baseline_xgboost_cluster_2001_2019.py` — complete script; can also be copied into a notebook cell.
- The script writes cluster labels, hyperparameter-search results, predictions, feature importance, plots, and a text report to `OUTPUT_ROOT`.

## Requirements

Python packages:

```bash
pip install numpy pandas xarray matplotlib scikit-learn xgboost netCDF4
```

Depending on the NetCDF files, an additional xarray backend such as `h5netcdf` may be needed.

## Configure input/output paths

Before running, update `ERA5_FILE`, `MODIS_FILES`, and `OUTPUT_ROOT` near the top of the script to match your environment. The supplied paths are specific to the original computing environment and will not work on another machine unless the same directory structure exists.

## Run

```bash
python baseline_xgboost_cluster_2001_2019.py
```

For a Jupyter notebook, paste the script into a single code cell and run it after confirming that the input files and Python dependencies are available.

## Important data assumptions

1. For `BurnDate` inputs, the script assumes annual day-of-year values, with `0` meaning unburned, and identifies JJA burns using day-of-year values 152–243.
2. For variables named `burned_area` or `BurnedArea`, values are treated as already representing burned area and are retained as supplied. Confirm the units and temporal meaning of this variable before interpreting results.
3. The script assumes ERA5 total precipitation (`tp`) is stored in metres and multiplies the JJA sum by 1,000 to obtain millimetres.
4. The burned-area cell-area conversion is approximate and depends on the target grid resolution. Check the MODIS grid resolution and units before interpreting absolute area totals.
5. Cluster IDs from KMeans are arbitrary labels. `CLUSTER_TO_MODEL = 1` refers to cluster label 1 for this run, not necessarily a consistently ordered fire-regime category across different datasets or clustering runs.

## Outputs

The script writes its results under `OUTPUT_ROOT`, including:

- `fire_regime_clusters_2001_2019.csv`
- `baseline_hyperparameter_search.csv`
- `baseline_test_pixel_predictions_2020_2024.csv`
- `baseline_test_annual_predictions_2020_2024.csv`
- `baseline_evaluation_pixel_predictions_2025.csv`
- `baseline_evaluation_annual_predictions_2025.csv`
- `baseline_evaluation_metrics_2025.csv`
- `baseline_feature_importance.csv`
- `plots/`
- `BASELINE_MODEL_REPORT.txt`

## Reproducibility note

The script fixes the random seed for KMeans and XGBoost. Hyperparameter search is randomized but reproducible under the fixed seed and package versions. Record the Python and package versions when preparing final results.
