# Fire-Forecasting-modular

Modular XGBoost pipeline to predict **June–August burned area on the Iberian Peninsula** (MODIS, 0.5° grid) from ERA5 meteorology and, optionally, SiB4 biosphere variables. Its purpose is to quantify the **added value of SiB4 information** by comparing runs with and without biosphere predictors.

The processing is a restructured version of the original `baseline_code_JJA*.py` scripts (the ERA5-only run reproduces the original outputs exactly). Everything that used to be hard-coded is now in a settings file.

## Quick start

```bash
pip install numpy pandas xarray scipy scikit-learn xgboost matplotlib netcdf4 dask shap

# 1. edit paths and predictors in settings.py (or a copy, see below)
# 2a. run on terminal
python exec/run_pipeline.py                    # uses settings.py
python exec/run_pipeline.py [settings_file]    # uses settings file

# 2b. run as job
sbatch submit_pipeline [settings_file]         # takes the settings file that is given as argument
```


The first run is slow: it builds the cached intermediate data and runs the hyperparameter search. Later runs reuse the cache and stored model.

## Repository structure

```text
Fire-Forecasting-modular/
├── README.md
├── settings.py              # ALL paths and settings (default)
├── settings_baseline.py     # example: ERA5-only experiment
├── settings_sib4.py         # example: ERA5 + SiB4 experiment
├── .gitignore               # outputs/ is not tracked
├── exec/
│   ├── run_pipeline.py      # entry point; connects the stages
│   ├── cache.py             # DiskCache: store/reuse expensive results
│   ├── experiment.py        # experiment folder + settings snapshot
│   ├── data/reader.py                  # DataReader
│   ├── preprocessing/preprocessor.py   # Preprocessor
│   ├── model/trainer.py                # ModelTrainer
│   ├── model/predictor.py              # Predictor
│   ├── evaluation/evaluator.py         # Evaluator
│   └── visualization/plotter.py        # Plotter
└── outputs/                 # generated, ignored by Git
    ├── cache/
    └── experiments/<EXPERIMENT_NAME>/cluster_<N>/
```

## Pipeline stages

| Stage | Class | What it does |
|---|---|---|
| 1 | `DataReader` | Opens MODIS BurnDate, ERA5 (derives `windspeed`, `vpd`, `tmax` when requested) and SiB4 land-unit files. |
| 2 | `Preprocessor` | Pixel area → 0.5° JJA burned-area grid (ha); K-means fire regimes on training years only (log1p + standardisation, largest-component cleanup of `TARGET_CLUSTER`); yearly predictor fields; pixel-year tables for train / test / evaluation. |
| 3 | `ModelTrainer` | Random search over `PARAM_DISTRIBUTIONS` with chronological folds, ranked by annual-total R²; final fit on all training years. |
| 4 | `Predictor` | Predictions, clipped at 0. |
| 5 | `Evaluator` | Pixel and annual R², RMSE, XGBoost and test-set permutation importance. |
| 6 | `Plotter` | Importance plots and observed-vs-predicted annual totals. |
| 7 | `Explainer` | Optional SHAP analysis of the final model on the test years (`RUN_SHAP`). |

Periods (editable): train 2000–2019, test 2020–2024, evaluation 2025.

## Settings (`settings.py`)

Only UPPERCASE names are used. Main groups:

- **Experiment and paths**: `EXPERIMENT_NAME`, `OUTPUT_DIR`, `CACHE_DIR`; input data in `DATA_DIR`, `ERA5_PATH`, `MODIS_PATHS`, `SIB4_DIR`. Output paths are relative to the repository; input paths must be set for your system. There are no command-line path overrides.
- **Domain, years, clustering**: `IBERIA_*`, `YEARS_TARGET`, `TRAIN_YEARS`, `TEST_YEARS`, `EVALUATION_YEAR`, `K`, `TARGET_CLUSTER`, `CLUSTER_TO_MODEL`, …
- **Predictors and lags**:
  - `ERA5_PREDICTORS = {name: {"months": [...], "agg": "mean"|"sum"}}`
  - `USE_SIB4` switches SiB4 predictors on or off.
  - `SIB4_PREDICTORS_BY_CLUSTER = {cluster: {label: {...}}}`, where the label is `<sib4 variable>_<PFT>`, e.g. `cwt_stem_enf`.
  - `months` is the lag window (e.g. `[6, 7, 8]` = JJA, `[4, 5]` = April–May). The same variable can be used several times under different names with `"variable": "t2m"`.
  - The available ERA5 and SiB4 variables and PFT codes are listed as comments in the file.
- **Diagnostics**: `RUN_SHAP` switches the SHAP analysis on or off.
- **Model**: `RANDOM_STATE`, `N_PARAMETER_CANDIDATES`, `PARAM_DISTRIBUTIONS`, `VALIDATION_FOLDS`, `MAX_ROWS_PER_FOLD`, `PERMUTATION_N_REPEATS`.
- **Tuning scheme**: `TUNING_SCHEME`, `TUNING_METRIC`, `TUNING_FIRST_VALIDATION_YEAR`, `TUNING_BUFFER_YEARS` (see "Hyperparameter tuning schemes").

To run several experiments, copy `settings.py` (e.g. `settings_myexp.py`), give it a new `EXPERIMENT_NAME`, and run `python exec/run_pipeline.py settings_myexp`.

Known fix versus the original SiB4 script: the missing comma in the cluster-1 soil list (`"rhsoil_c3g" "pawfrw_sha"`) is corrected.

## Caching and reruns

| What | Where | Recomputed when |
|---|---|---|
| Burned-area grid, clusters, yearly predictor fields (one file per predictor) | `outputs/cache/<name>__<hash>.pkl` | file missing, or `OVERWRITE_CACHE = True` |
| Hyperparameter search and trained model | experiment folder (`xgb_model.json`, `best_params.json`) | files missing, or `OVERWRITE_MODEL = True` |

The hash in the cache file name is derived from the settings that influence that result (domain, years, variable, months, aggregation, …). Changing such a setting creates a new cache file instead of silently reusing a stale one. A new user starts with an empty `outputs/`, so the first run recreates everything.

If an experiment folder already exists and its scientific settings differ from the current settings, the run stops with an error. Choose a new `EXPERIMENT_NAME` or set `OVERWRITE_MODEL = True`.

## Experiment outputs

`outputs/experiments/<EXPERIMENT_NAME>/cluster_<N>/` contains:

- `settings_snapshot.json`: all settings used, with a fingerprint. For SiB4 runs this includes the predictors actually used.
- `settings_used.py`: literal copy of the settings file.
- `xgb_model.json`, `best_params.json`, `best_xgboost_parameters.csv`, `xgboost_hyperparameter_search.csv`
- `metrics.csv`
- `feature_importance.csv` / `.png`, `permutation_importance_test.csv`, `permutation_importance.png`
- `training_table_2000_2019.csv`, `test_table_2020_2024.csv`, `evaluation_2025.csv` (per pixel and year, with `predicted` and `difference`)
- `shap_summary_beeswarm.png`, `shap_importance_bar.png`, `shap_values.csv` (only if `RUN_SHAP = True`): SHAP on the test years 2020–2024. `shap_values.csv` has one row per pixel and year with a `shap_<feature>` column per predictor, and is reused on the next run unless `OVERWRITE_MODEL = True`.
- `annual_reconstruction_cluster_<N>.png`

## Notes

- **ERA5 `tmax`:** if the ERA5 file holds monthly data, the daily maximum in the original script is effectively monthly `t2m`.
- **Cluster 3:** with `K = 3`, valid clusters are 0, 1 and 2.
- **Tuning metric:** the default (`TUNING_SCHEME = "original"`) is the original, noisy scheme; see "Hyperparameter tuning schemes" below.
- **Reproducibility:** results are tied to the library versions used (xgboost, scikit-learn, xarray). Tuning with the same seed and settings gives the same results in the same environment.

## Hyperparameter tuning schemes

All schemes judge candidates on the **annual regional total** of burned area in held-out validation years. They differ in how the folds are built and how the score is aggregated. Choose with `TUNING_SCHEME` and `TUNING_METRIC` in the settings file.

| `TUNING_SCHEME` | Folds | Score |
|---|---|---|
| `"original"` (default) | `VALIDATION_FOLDS` (2 validation years each) | R² per fold, mean over folds (as in the original scripts) |
| `"pooled"` | `VALIDATION_FOLDS` | predictions of all folds pooled, one score over all 6 validation years |
| `"yearly_forward"` | train on years before Y, validate on Y, for Y = `TUNING_FIRST_VALIDATION_YEAR` (default 2010) to 2019 | pooled over the 10 validation years; about 3× more model fits per candidate |
| `"leave_one_year_out"` | every training year Y is validated once; fit on all other training years except the `TUNING_BUFFER_YEARS` (default 1) neighbours on each side of Y | the out-of-fold annual totals of all years are pooled into one R²; one fit per training year per candidate (about 19 with the current data, roughly 6× the cost of `"original"`) |

`TUNING_METRIC` is `"annual_r2"` (higher is better) or `"annual_rmse"` (lower is better; no variance denominator).

**Why the original scheme is problematic.** Each fold's R² rests on only **two data points**. R² is relative to the variance of the observed totals, so if the two years burned similar amounts, even a small error gives a very large negative R². Validation scores are therefore extreme (about -22 for ERA5 and -8 for SiB4 in the stored original runs), while test-period R² is much better. The candidate ranking is driven by a few lucky or unlucky years, so the chosen hyperparameters may not generalise, and with-versus-without-SiB4 differences may partly reflect this noise.

**Extra diagnostics.** The search CSV (`xgboost_hyperparameter_search.csv`) always contains, for every candidate:
- `mean_validation_annual_r2` and `std_validation_annual_r2`
- `validation_annual_rmse`
- `pooled_validation_annual_r2`
- `pooled_validation_skill_vs_mean`: 1 − SSE / SSE of a reference forecast (the mean annual total of the fit years). A value above 0 means better than that reference.
- `selection_score`: the value actually used for ranking

`metrics.csv` also records the scheme, metric and these best-candidate values.

**Notes.**
- Each experiment stores its scheme in `settings_snapshot.json`. Choose a new `EXPERIMENT_NAME` when you switch scheme, otherwise the run stops (see "Caching and reruns").
- `"yearly_forward"` keeps the time order, so there is no leakage from the future.
- `"leave_one_year_out"` does train on later years, which is acceptable for ranking candidates but not for estimating operational skill. The buffer (`TUNING_BUFFER_YEARS`) drops neighbouring years to limit leakage through year-to-year memory (fuel, soil moisture); `0` gives plain leave-one-year-out. Check the annual totals for a strong trend before relying on it. The test years (2020–2024) and 2025 are never used in tuning. The K-means clusters use all training years, which is a mild leak in every scheme.
- Use identical folds and seeds for the ERA5 and ERA5+SiB4 experiments so tuning noise is not attributed to SiB4.
- With about 20 annual points per cluster, differences between candidates, and between experiments, can still be within noise. For the with-versus-without-SiB4 comparison, consider a bootstrap or repeated-seed spread of the test metrics.
- Folds with a single validation year have no R². Those entries are NaN in `fold_annual_r2`, and only pooled scores are used.
