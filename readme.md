# XGBoost Burned Area Analysis & Reconstruction Pipeline

## Overview

This repository contains an end-to-end Python pipeline for modeling, predicting, and reconstructing summer burned areas across the Iberian Peninsula ($10.5^\circ\text{W}$ to $4.5^\circ\text{E}$, $36.5^\circ\text{N}$ to $44.5^\circ\text{N}$) using **ERA5 meteorological predictors** and **XGBoost regression**. 

The workflow integrates spatial fire-regime clustering restricted to historical training years ($2000\text{--}2019$), non-overlapping chronological cross-validation tuned specifically for macro-level **annual total $R^2$ performance**, and out-of-sample temporal evaluation ($2020\text{--}2024$ test period and $2025$ evaluation year).

---

## Technical Pipeline Architecture

```
[ MODIS Datasets (2000-2025) ]       [ ERA5 Meteorology (2000-2025) ]
              │                                      │
              ▼                                      ▼
   Burn Date Masking &                   JJA Seasonal Aggregation
   0.5° Grid Resampling                   (t2m, d2m, tp, windspeed)
              │                                      │
              ▼                                      │
   K-Means Fire Clustering                           │
    (Strictly 2000-2019)                             │
              │                                      │
              └──────────────────┬───────────────────┘
                                 ▼
                     Tabular Dataset Assembly
               (Train: 2000-2019 | Test: 2020-2024)
                                 │
                                 ▼
                    Chronological Validation &
                 Annual R² Hyperparameter Tuning
                                 │
                                 ▼
                     Final Model Training & 
                  Annual Reconstruction Plot
```

---

## Detailed Section Breakdown

### 1. Data Ingestion & Spatial Preprocessing (`MODIS`)
* **File Combination**: Merges MODIS products across time horizons (`2000–2010` and `2010–2025`), ensuring duplicate timestamps are filtered out.
* **Geospatial Area Calculation**: Converts pixel dimensions from spherical degree coordinates into physical area in hectares ($1 \text{ ha} = 10,000 \text{ m}^2$):
  $$\Delta y = R \cdot \Delta \text{lat} \cdot \frac{\pi}{180}, \quad \Delta x = R \cdot \Delta \text{lon} \cdot \frac{\pi}{180} \cdot \cos(\text{lat})$$
* **Spatial Binning**: Aggregates native daily pixel burn flags into a regular $0.5^\circ \times 0.5^\circ$ grid over Iberia for the peak summer fire season (June, July, August — JJA).

---

### 2. Spatial Fire-Regime Clustering (`K-Means`)
To prevent data leakage from future time periods into historical spatial boundaries:
* **Training Window Filter**: Slices spatial burn history strictly to the training baseline ($2000\text{--}2019$).
* **Variance Stabilization**: Transforms skewed annual burn totals using $\log(1 + x)$ followed by standard z-score normalization (`StandardScaler`).
* **K-Means Partitioning**: Clusters grid cells into $k=3$ distinct fire regimes based on temporal burn patterns.
* **Spatial Morphological Cleaning**: Applies 8-neighbor component labeling (`scipy.ndimage.label`) to isolate the primary contiguous cluster body and reassign isolated spatial artifacts to adjacent majority clusters.

---

### 3. ERA5 Meteorological Feature Extraction
Extracts summer seasonal aggregations across target fire months (JJA):
* `t2m`: 2-meter surface air temperature (seasonal mean)
* `d2m`: 2-meter dewpoint temperature (seasonal mean)
* `tp`: Total accumulated precipitation (seasonal sum)
* `windspeed`: 10-meter surface wind vector magnitude, calculated as:
  $$\text{windspeed} = \sqrt{u_{10}^2 + v_{10}^2}$$

---

### 4. Tabular Dataset Assembly
Aligns spatial meteorology and MODIS grid cells into structured tabular DataFrames categorized into three distinct operational windows:
1. **Training Split (`2000–2019`)**: Used for model fitting and hyperparameter optimization.
2. **Test Split (`2020–2024`)**: Out-of-sample temporal model generalizability assessment.
3. **Evaluation Split (`2025`)**: Benchmark on an unseen recent year.

---

### 5. Annual $R^2$ Hyperparameter Tuning
* **Candidate Sampling**: Uses `ParameterSampler` to evaluate 40 randomized configurations spanning tree depth (`max_depth`), learning rate (`learning_rate`), tree count (`n_estimators`), subsample ratios, and L1/L2 regularization (`reg_alpha`, `reg_lambda`).
* **Chronological Cross-Validation Folds**: Evaluates model performance sequentially across time:
  * Fold 1: Train `2000–2013` $\rightarrow$ Validate `2014–2015`
  * Fold 2: Train `2000–2015` $\rightarrow$ Validate `2016–2017`
  * Fold 3: Train `2000–2018` $\rightarrow$ Validate `2018–2019`
* **Macro-Level Metric Optimization**: Pixel predictions within each validation fold are aggregated to compute total annual regional burned area ($\sum \text{burned area}$). Candidate configurations are ranked by **Annual Total $R^2$**, ensuring hyperparameter selection favors macro-level annual reconstruction fidelity over individual pixel noise.

---

### 6. Final Model Fitting & Metric Computation
* Fits an `XGBRegressor` on the complete $2000\text{--}2019$ dataset using the top-performing hyperparameter set.
* Enforces non-negative physical constraints by zero-clipping predictions (`np.clip(..., 0, None)`).
* Computes performance metrics across training, testing, and 2025 evaluation periods at both the pixel level and the aggregated annual total level.

---

### 7. Feature Importance Diagnostics
* **Tree Gain Importance**: Measures relative contribution of each meteorological driver toward reducing squared error during tree splits.
* **Test Permutation Importance**: Evaluates performance degradation on unseen test data ($2020\text{--}2024$) by randomly shuffling feature values across 5 iterations.

---

### 8. Annual Reconstruction Visualization
Generates a time-series line plot comparing total observed vs. predicted burned area from 2000 through 2025, complete with visually shaded bands distinguishing the **Test period ($2020\text{--}2024$)** and **2025 evaluation period**.

---

## Required Dependencies

* `python >= 3.9`
* `numpy`, `pandas`, `xarray`, `scipy`
* `scikit-learn`
* `xgboost`
* `matplotlib`, `cartopy`

---

## Execution & Output Files

Set `cluster_to_model` (e.g., `0`, `1`, `2`) near the top of the script and execute:

```bash
python xgb_burned_area_analysis.py
```

### Generated Outputs (Directory: `/xgb_cluster_{N}/`)
* `metrics.csv`: Detailed numerical evaluation table ($R^2$, RMSE, MAE).
* `xgboost_hyperparameter_search.csv`: Complete hyperparameter candidate search results.
* `best_xgboost_parameters.csv`: Winning hyperparameter set.
* `feature_importance.csv` / `.png`: XGBoost split importance.
* `permutation_importance_test.csv` / `.png`: Test set permutation importance.
* `training_table_2000_2019.csv`, `test_table_2020_2024.csv`, `evaluation_2025.csv`: Predictions per pixel/year.
* `annual_reconstruction_cluster_{N}.png`: Time-series reconstruction plot.