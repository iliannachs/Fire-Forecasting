"""XGBoost hyperparameter search (annual-total R2) and final model training."""
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import ParameterSampler
from xgboost import XGBRegressor


def sample_rows_by_year(frame, max_rows, seed):
    """Subsample to at most max_rows, with an equal quota of rows per year."""
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


class ModelTrainer:
    """
    Tunes and trains the XGBoost model. Results are stored in the experiment
    folder and reused on the next run unless OVERWRITE_MODEL = True:
        xgboost_hyperparameter_search.csv, best_xgboost_parameters.csv,
        best_params.json, xgb_model.json
    """

    def __init__(self, settings, experiment):
        self.s = settings
        self.exp = experiment
        # getattr keeps settings files that predate these options working.
        self.scheme = getattr(settings, "TUNING_SCHEME", "original")
        self.metric = getattr(settings, "TUNING_METRIC", "annual_r2")
        if self.scheme not in ("original", "pooled", "yearly_forward", "leave_one_year_out"):
            raise ValueError(f"Unknown TUNING_SCHEME: {self.scheme}")
        if self.metric not in ("annual_r2", "annual_rmse"):
            raise ValueError(f"Unknown TUNING_METRIC: {self.metric}")

    def make_xgb(self, params):
        return XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            tree_method="hist",
            random_state=self.s.RANDOM_STATE,
            n_jobs=1,
            verbosity=0,
            **params,
        )

    # ------------------------------------------------------------
    # hyperparameter search
    # ------------------------------------------------------------
    def _validation_folds(self, df_train):
        """
        (fit years, validation years) pairs for the chosen TUNING_SCHEME:
          original, pooled : VALIDATION_FOLDS from settings.py
          yearly_forward   : train on all years before Y, validate on Y, for
                             every Y from TUNING_FIRST_VALIDATION_YEAR onwards
          leave_one_year_out : validate on each training year Y, fit on all other
                             years except the TUNING_BUFFER_YEARS neighbours on
                             either side of Y (limits leakage via year-to-year memory)
        """
        s = self.s
        if self.scheme in ("original", "pooled"):
            return [(list(f), list(v)) for f, v in s.VALIDATION_FOLDS]

        years = sorted(df_train["year"].unique())
        if self.scheme == "leave_one_year_out":
            buf = s.TUNING_BUFFER_YEARS
            return [
                ([y for y in years if abs(y - year) > buf], [year])
                for year in years
            ]

        return [
            ([y for y in years if y < year], [year])
            for year in years
            if year >= s.TUNING_FIRST_VALIDATION_YEAR
        ]

    def tune(self, df_train, feature_cols):
        """
        Random search over PARAM_DISTRIBUTIONS. Candidates are ranked on the
        annual regional total of the validation years, according to
        TUNING_SCHEME and TUNING_METRIC (see settings.py). The default
        ("original", "annual_r2") is the mean over the folds of the annual-total R2.
        """
        s = self.s
        candidates = list(
            ParameterSampler(
                s.PARAM_DISTRIBUTIONS,
                n_iter=s.N_PARAMETER_CANDIDATES,
                random_state=s.RANDOM_STATE,
            )
        )
        folds = self._validation_folds(df_train)
        print(f"Tuning scheme: {self.scheme}, metric: {self.metric}, {len(folds)} folds")

        search_results = []
        for candidate_idx, params in enumerate(candidates, start=1):
            fold_r2, fold_rmse = [], []
            all_obs, all_pred, all_base = [], [], []
            print(f"\nCandidate {candidate_idx}/{len(candidates)}: {params}")

            for fold_idx, (fit_years, valid_years) in enumerate(folds, start=1):
                fit_df = df_train[df_train["year"].isin(fit_years)]
                valid_df = df_train[df_train["year"].isin(valid_years)].copy()

                if fit_df.empty or valid_df.empty:
                    raise ValueError(f"Empty data in validation fold {fold_idx}.")

                fit_sample = sample_rows_by_year(
                    fit_df, s.MAX_ROWS_PER_FOLD, s.RANDOM_STATE + fold_idx
                )

                model = self.make_xgb(params)
                model.fit(fit_sample[feature_cols], fit_sample["burned_area"], verbose=False)

                valid_df["pred"] = np.clip(model.predict(valid_df[feature_cols]), 0, None)

                # Performance is judged on the annual regional total.
                annual_obs = valid_df.groupby("year")["burned_area"].sum()
                annual_pred = valid_df.groupby("year")["pred"].sum()

                # R2 needs at least two points; single-year folds only contribute to pooled scores.
                r2 = r2_score(annual_obs, annual_pred) if len(annual_obs) > 1 else np.nan
                fold_r2.append(r2)
                fold_rmse.append(float(np.sqrt(np.mean((annual_obs - annual_pred) ** 2))))

                all_obs.append(annual_obs.values)
                all_pred.append(annual_pred.values)
                # Reference forecast for the skill score: mean annual total of the fit years.
                base = fit_df.groupby("year")["burned_area"].sum().mean()
                all_base.append(np.full(len(annual_obs), base))

                print(
                    f"   Fold {fold_idx}: fit {fit_years[0]}-{fit_years[-1]}, validation years "
                    f"{valid_years[0]}-{valid_years[-1]}, Annual R2={r2:.5f}"
                )

            obs, pred, base = map(np.concatenate, (all_obs, all_pred, all_base))
            pooled_r2 = float(r2_score(obs, pred))
            pooled_rmse = float(np.sqrt(np.mean((obs - pred) ** 2)))
            # Skill > 0: better than predicting the training-period mean annual total.
            pooled_skill = float(1 - np.sum((obs - pred) ** 2) / np.sum((obs - base) ** 2))

            if self.scheme == "original":
                mean_r2 = float(np.mean(fold_r2))
                std_r2 = float(np.std(fold_r2))
                mean_rmse = float(np.mean(fold_rmse))
                selection_score = mean_r2 if self.metric == "annual_r2" else mean_rmse
            else:
                mean_r2 = pooled_r2
                std_r2 = float(np.nanstd(fold_r2)) if not np.all(np.isnan(fold_r2)) else np.nan
                selection_score = pooled_r2 if self.metric == "annual_r2" else pooled_rmse

            search_results.append({
                "candidate": candidate_idx,
                "mean_validation_annual_r2": mean_r2,
                "std_validation_annual_r2": std_r2,
                "validation_annual_rmse": mean_rmse if self.scheme == "original" else pooled_rmse,
                "pooled_validation_annual_r2": pooled_r2,
                "pooled_validation_skill_vs_mean": pooled_skill,
                "selection_score": float(selection_score),
                "fold_annual_r2": fold_r2,
                "params": params,
            })
            print(
                f"   {self.metric} (selection) = {selection_score:.5f} | "
                f"annual R2 = {mean_r2:.5f}, pooled skill vs mean = {pooled_skill:.5f}"
            )

        # Best candidate first: highest R2, or lowest RMSE
        search_results.sort(
            key=lambda x: x["selection_score"], reverse=(self.metric == "annual_r2")
        )
        best_result = search_results[0]

        print("\n============================================")
        print(f"BEST XGBOOST HYPERPARAMETERS (BY {self.metric.upper()}, {self.scheme})")
        print("============================================")
        print(best_result["params"])
        print("Selection score:", best_result["selection_score"])
        print("Mean validation annual R2:", best_result["mean_validation_annual_r2"])
        print("Validation fold standard deviation:", best_result["std_validation_annual_r2"])

        best = {
            "params": best_result["params"],
            "best_mean_validation_annual_r2": best_result["mean_validation_annual_r2"],
            "validation_annual_r2_std": best_result["std_validation_annual_r2"],
            "tuning_scheme": self.scheme,
            "tuning_metric": self.metric,
            "best_selection_score": best_result["selection_score"],
            "best_validation_annual_rmse": best_result["validation_annual_rmse"],
            "best_pooled_validation_skill_vs_mean": best_result["pooled_validation_skill_vs_mean"],
        }
        self._save_search(search_results, best)
        return best

    def _save_search(self, search_results, best):
        pd.DataFrame([
            {
                "candidate": r["candidate"],
                "mean_validation_annual_r2": r["mean_validation_annual_r2"],
                "std_validation_annual_r2": r["std_validation_annual_r2"],
                "validation_annual_rmse": r["validation_annual_rmse"],
                "pooled_validation_annual_r2": r["pooled_validation_annual_r2"],
                "pooled_validation_skill_vs_mean": r["pooled_validation_skill_vs_mean"],
                "selection_score": r["selection_score"],
                **r["params"],
            }
            for r in search_results
        ]).to_csv(self.exp.path("xgboost_hyperparameter_search.csv"), index=False)

        pd.DataFrame([{
            **best["params"],
            **{k: v for k, v in best.items() if k != "params"},
        }]).to_csv(self.exp.path("best_xgboost_parameters.csv"), index=False)

        with open(self.exp.path("best_params.json"), "w") as f:
            json.dump(
                best, f, indent=2,
                default=lambda o: o.item() if hasattr(o, "item") else str(o),
            )

    def _load_best(self):
        with open(self.exp.path("best_params.json")) as f:
            return json.load(f)

    # ------------------------------------------------------------
    # final model
    # ------------------------------------------------------------
    def train(self, df_train, feature_cols):
        """
        Returns (model, best_result). Loads the stored model when present
        (and OVERWRITE_MODEL is False), otherwise tunes and fits on all
        training years and stores it.
        """
        model_path = self.exp.path("xgb_model.json")
        params_path = self.exp.path("best_params.json")

        if os.path.exists(model_path) and os.path.exists(params_path) and not self.s.OVERWRITE_MODEL:
            print(f"[model] loading stored model {model_path}")
            model = XGBRegressor()
            model.load_model(model_path)
            return model, self._load_best()

        if os.path.exists(params_path) and not self.s.OVERWRITE_MODEL:
            print("[model] reusing stored hyperparameters")
            best_result = self._load_best()
        else:
            best_result = self.tune(df_train, feature_cols)

        # Final fit on all training years (2000-2019) with the best parameters.
        model = self.make_xgb(best_result["params"])
        model.fit(df_train[feature_cols], df_train["burned_area"], verbose=False)
        model.save_model(model_path)
        return model, best_result
