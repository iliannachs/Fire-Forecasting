"""Evaluation metrics and feature importances."""
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


class Evaluator:
    def __init__(self, settings):
        self.s = settings

    def annual_totals(self, df_train, df_test, df_eval):
        """Total observed and predicted burned area per year over all periods."""
        df_all = pd.concat([df_train, df_test, df_eval], ignore_index=True)
        return df_all.groupby("year")[["burned_area", "predicted"]].sum()

    def metrics(self, df_train, df_test, df_eval, feature_cols, best_result):
        """Pixel-level and annual-total metrics (same definitions as the original script)."""
        s = self.s
        annual_totals = self.annual_totals(df_train, df_test, df_eval)
        train_ann = annual_totals.loc[annual_totals.index.isin(s.TRAIN_YEARS)]
        test_ann = annual_totals.loc[annual_totals.index.isin(s.TEST_YEARS)]

        metrics = {
            "cluster": s.CLUSTER_TO_MODEL,
            "features": ", ".join(feature_cols),

            "pixel_train_r2": r2_score(df_train["burned_area"], df_train["predicted"]),
            "pixel_test_r2": r2_score(df_test["burned_area"], df_test["predicted"]),
            "pixel_eval_2025_r2": r2_score(df_eval["burned_area"], df_eval["predicted"]),

            "annual_train_r2": r2_score(train_ann["burned_area"], train_ann["predicted"]),
            "annual_test_r2": r2_score(test_ann["burned_area"], test_ann["predicted"]),
            "annual_overall_r2": r2_score(annual_totals["burned_area"], annual_totals["predicted"]),

            "train_rmse": rmse(df_train["burned_area"], df_train["predicted"]),
            "test_rmse": rmse(df_test["burned_area"], df_test["predicted"]),
            "eval_2025_rmse": rmse(df_eval["burned_area"], df_eval["predicted"]),

            "observed_total_2025": float(df_eval["burned_area"].sum()),
            "predicted_total_2025": float(df_eval["predicted"].sum()),
            "best_mean_validation_annual_r2": best_result["best_mean_validation_annual_r2"],
            "tuning_scheme": best_result.get("tuning_scheme", "original"),
            "tuning_metric": best_result.get("tuning_metric", "annual_r2"),
            "best_validation_annual_rmse": best_result.get("best_validation_annual_rmse"),
            "best_pooled_validation_skill_vs_mean": best_result.get(
                "best_pooled_validation_skill_vs_mean"
            ),
        }
        return pd.DataFrame([metrics])

    def feature_importance(self, model, feature_cols):
        """XGBoost built-in importance."""
        return pd.DataFrame({
            "feature": feature_cols,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)

    def permutation_importance(self, model, df_test, feature_cols):
        """Decrease in test R2 when a feature is shuffled."""
        s = self.s
        perm = permutation_importance(
            model,
            df_test[feature_cols],
            df_test["burned_area"],
            n_repeats=s.PERMUTATION_N_REPEATS,
            random_state=s.RANDOM_STATE,
            n_jobs=1,
            scoring="r2",
        )
        return pd.DataFrame({
            "feature": feature_cols,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }).sort_values("importance_mean", ascending=False)
