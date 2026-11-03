"""
Central settings for the Fire-Forecasting-modular pipeline.

ALL paths, predictor choices and numerical/model settings live in this file.
Only UPPERCASE names are read by the pipeline and stored in the experiment
folder as a settings snapshot. Values are copied 1:1 from the original
baseline_code_JJA*.py scripts unless stated otherwise.

Run with:   python exec/run_pipeline.py
"""
from pathlib import Path

# ============================================================
# 1. EXPERIMENT NAME AND OUTPUT LOCATIONS
# ============================================================

# Name of this experiment. Results (model, tables, plots, settings snapshot)
# are stored in  outputs/experiments/<EXPERIMENT_NAME>/cluster_<N>/.
# Use a descriptive name, e.g. "era5_only" or "era5_plus_sib4_biomass".
EXPERIMENT_NAME = "era5_plus_sib4"

# Everything is relative to the folder containing this file, so the repository
# can be cloned anywhere.
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CACHE_DIR = OUTPUT_DIR / "cache"              # reusable intermediate results
EXPERIMENTS_DIR = OUTPUT_DIR / "experiments"  # one sub-folder per experiment

# Cached intermediate results (burned area grid, clusters, yearly predictor
# fields) are only recomputed when missing. Set True to recompute and overwrite.
OVERWRITE_CACHE = False

# A stored trained model / hyperparameter search of the experiment is reused
# when present. Set True to retune and retrain and overwrite them.
OVERWRITE_MODEL = False

# ============================================================
# 2. INPUT DATA (edit these for your own system)
# ============================================================

DATA_DIR = Path("/projects/0/ctdas/ilianna/1.Data/raw")
ERA5_PATH = DATA_DIR / "era5_merged.nc"
MODIS_PATHS = [DATA_DIR / "MODIS_2000_2010.nc", DATA_DIR / "MODIS_2010_2025.nc"]

# Historical SiB4-ERA5 output (monthly *.lu.qp3.nc files); only read when
# USE_SIB4 = True.
SIB4_DIR = Path(
    "/projects/0/ctdas/ruben/forecast/sib/run_global/run/output/"
    "ERA_IC_ERA_driver/sib_era_2000_2025_fires"
)
SIB4_FILE_PATTERN = "qsib_*.lu.qp3.nc"

# ============================================================
# 3. STUDY DOMAIN, PERIODS AND TARGET
# ============================================================

IBERIA_LON_MIN, IBERIA_LON_MAX = -10.5, 4.5
IBERIA_LAT_MIN, IBERIA_LAT_MAX = 36.5, 44.5
GRID_RESOLUTION = 0.5                      # degrees (burned-area target grid)

YEARS_TARGET = list(range(2000, 2026))
TRAIN_YEARS = list(range(2000, 2020))
TEST_YEARS = list(range(2020, 2025))
EVALUATION_YEAR = 2025

# Months over which the MODIS burned area (the target) is accumulated.
TARGET_FIRE_MONTHS = [6, 7, 8]

# ============================================================
# 4. FIRE-REGIME CLUSTERING (K-means on 2000-2019 burned area only)
# ============================================================

K = 3
TARGET_CLUSTER = 1                         # cluster whose isolated cells are cleaned up
KEEP_LARGEST_COMPONENT = True
KMEANS_RANDOM_STATE = 0
KMEANS_N_INIT = 20

# Cluster that is modelled in this run (0, 1 or 2 for K = 3).
CLUSTER_TO_MODEL = 0

# ============================================================
# 5. PREDICTORS
# ============================================================
# Every predictor is defined as   feature_name: {settings}.
# 'months' = calendar months of each year that are aggregated, i.e. the
# "lag window" of the predictor. E.g. [6, 7, 8] = June-August (same as fire
# season), [4, 5] = April-May (lead of 1-2 months before the fire season).
# 'agg' = 'mean' or 'sum' over those months.
# Optional key 'variable' lets you use one variable under several names
# (e.g. "t2m_spring": {"variable": "t2m", "months": [4, 5], "agg": "mean"}).

JJA = [6, 7, 8]

# ------------------------------------------------------------
# 5a. ERA5 meteorology
# ------------------------------------------------------------
# Available variables:
#   raw in era5_merged.nc : t2m, d2m, tp, u10, v10
#   derived by the pipeline: windspeed = sqrt(u10^2 + v10^2)   (m/s)
#                            vpd   = es(t2m) - ea(d2m), clipped >= 0  (kPa)
#                            tmax  = daily maximum of t2m
# Original baseline_code_JJA.py selection:
ERA5_PREDICTORS = {
    "t2m": {"months": JJA, "agg": "mean"},
    "d2m": {"months": JJA, "agg": "mean"},
    "tp": {"months": JJA, "agg": "sum"},
    "windspeed": {"months": JJA, "agg": "mean"},
    "tmax": {"months": JJA, "agg": "mean"},
    "vpd": {"months": JJA, "agg": "mean"},
}

# ------------------------------------------------------------
# 5b. SiB4 biosphere predictors
# ------------------------------------------------------------
# Set True to add the SiB4 predictors of the modelled cluster below to the
# ERA5 predictors. False = ERA5-only run (baseline) for the added-value test.
USE_SIB4 = True

# Labels are "<sib4 variable>_<PFT code>", aggregated per PFT from the land-unit
# (nlu) dimension. Original selection of the SiB4 script (per cluster) with the
# missing comma in cluster 1 ("rhsoil_c3g" "pawfrw_sha") fixed.
# Original lags: biomass -> April-May [4, 5]; soil and hydro -> JJA.
SIB4_PREDICTORS_BY_CLUSTER = {
    0: {
        # biomass
        "cwt_stem_shb": {"months": [4, 5], "agg": "mean"},
        "cwt_leaf_enf": {"months": [4, 5], "agg": "mean"},
        "pool_leaf_enf": {"months": [4, 5], "agg": "mean"},
        
        # soil
        "rsoil_c3g": {"months": JJA, "agg": "mean"},
        "rsoil_enf": {"months": JJA, "agg": "mean"},
        "rsoil_shb": {"months": JJA, "agg": "mean"},
        
        # hydrology
        "ect_dbg": {"months": JJA, "agg": "mean"},
        "ecmass_dbg": {"months": JJA, "agg": "mean"},
        "egsmax_enf": {"months": JJA, "agg": "mean"},
    },
    1: {
        "cwt_stem_enf": {"months": [4, 5], "agg": "mean"},
        "cwt_stem_shb": {"months": [4, 5], "agg": "mean"},
        "cwt_agb_enf": {"months": [4, 5], "agg": "mean"},
        "rhsoil_c3g": {"months": JJA, "agg": "mean"},
        "pawfrw_sha": {"months": JJA, "agg": "mean"},
        "pawftop_sha": {"months": JJA, "agg": "mean"},
        "egsmax_sha": {"months": JJA, "agg": "mean"},
        "ecmass_dbg": {"months": JJA, "agg": "mean"},
        "ect_dbg": {"months": JJA, "agg": "mean"},
    },
    2: {
        "cwt_brwn_dbf": {"months": [4, 5], "agg": "mean"},
        "cwt_brwn_wwt": {"months": [4, 5], "agg": "mean"},
        "rhsoil_c4g": {"months": JJA, "agg": "mean"},
        "hc_c4g": {"months": JJA, "agg": "mean"},
        "rsoil_c4g": {"months": JJA, "agg": "mean"},
        "eci_c4g": {"months": JJA, "agg": "mean"},
        "egsmax_c4g": {"months": JJA, "agg": "mean"},
        "ecmass_c4g": {"months": JJA, "agg": "mean"},
    },
}

# CATALOG of everything that can be used above (read from the SiB4 files):
#   Variables (any with a land-unit dimension):
#     capacc_liq, capacg, cwt_agb, cwt_brwn, cwt_froot, cwt_leaf, cwt_prod,
#     cwt_seed, cwt_stem, eci, ecmass, ect, egmass, egs, egsmax, hc, lai,
#     pawfrw, pawftop, pool_leaf, poros, rhsoil, roff, roffo, rsoil, satfrac,
#     vol_liq_l01..l10, www_liq_l01..l10, wetfracc, www_tot
#   PFT codes:
#     dbg, enf, dnf, ebf, dbf, shb, sha, c3a, c3g, c4g, c3c, c4c, mze, soy, wwt
#   Label = "<variable>_<PFT>", e.g. "cwt_stem_enf", "rsoil_c3g", "lai_dbf".
#   (Not every variable/PFT combination necessarily carries data.)

# ============================================================
# 6. XGBOOST HYPERPARAMETER SEARCH
# ============================================================

RANDOM_STATE = 42
N_PARAMETER_CANDIDATES = 40               # randomly sampled combinations

PARAM_DISTRIBUTIONS = {
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

# Non-overlapping, chronological validation blocks: (fit years, validation years).
# Candidates are ranked on the annual-total R2 of the validation years.
VALIDATION_FOLDS = [
    (list(range(2000, 2014)), list(range(2014, 2016))),
    (list(range(2000, 2016)), list(range(2016, 2018))),
    (list(range(2000, 2018)), list(range(2018, 2020))),
]

# Maximum number of training rows per validation fit (sampled evenly per year).
MAX_ROWS_PER_FOLD = 150_000

# How hyperparameter candidates are validated and ranked (always on the ANNUAL
# regional total of burned area, see README "Open point: noisy tuning metric"):
#   "original"       : VALIDATION_FOLDS above; R2 per fold, mean over folds
#                      (original behaviour; each fold R2 rests on only 2 years).
#   "pooled"         : VALIDATION_FOLDS above; predictions of all folds are pooled
#                      and ONE score is computed over all validation years.
#   "yearly_forward" : train on all years before Y, validate on Y, for every Y from
#                      TUNING_FIRST_VALIDATION_YEAR to the last training year;
#                      scores are pooled (VALIDATION_FOLDS is not used).
#   "leave_one_year_out" : every training year is validated once; the model is fit
#                      on all other training years except the TUNING_BUFFER_YEARS
#                      neighbours on either side. The 20 out-of-fold annual totals
#                      are pooled into ONE R2 (VALIDATION_FOLDS is not used).
#                      Costs one fit per training year per candidate.
TUNING_SCHEME = "leave_one_year_out"
# Metric used to rank candidates: "annual_r2" (higher is better) or
# "annual_rmse" (lower is better, no variance denominator).
TUNING_METRIC = "annual_r2"
# Only used by "yearly_forward".
TUNING_FIRST_VALIDATION_YEAR = 2010
# Only used by "leave_one_year_out": years excluded from the fit on each side of
# the validation year (1 = also drop Y-1 and Y+1; 0 = plain leave-one-year-out).
TUNING_BUFFER_YEARS = 1
# The search CSV always also lists RMSE and the skill relative to predicting the
# mean annual total of the fit years (skill > 0 = better than that reference).

# Number of shuffles for the permutation importance on the test years.
PERMUTATION_N_REPEATS = 5

# ============================================================
# 7. DIAGNOSTICS
# ============================================================

# SHAP analysis of the final model on the TEST years: beeswarm summary plot,
# mean |SHAP| bar plot and shap_values.csv in the experiment folder.
RUN_SHAP = True
