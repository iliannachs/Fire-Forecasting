"""SHAP analysis of the trained model on the test years."""
import os

import matplotlib

matplotlib.use("Agg")   # no display needed on a cluster
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


class Explainer:
    """
    Default SHAP figures (beeswarm summary and mean |SHAP| bar plot) for the
    test years. SHAP values are stored in shap_values.csv and reused on the
    next run unless OVERWRITE_MODEL = True.
    """

    def __init__(self, settings, experiment):
        self.s = settings
        self.exp = experiment

    def run(self, model, df_test, feature_cols):
        csv_path = self.exp.path("shap_values.csv")
        X = df_test[feature_cols]

        if os.path.exists(csv_path) and not self.s.OVERWRITE_MODEL:
            print(f"[shap] loading stored values {csv_path}")
            stored = pd.read_csv(csv_path)
            # Only reuse if the stored rows belong to the current test table.
            if len(stored) == len(X) and all(f"shap_{f}" in stored for f in feature_cols):
                shap_values = stored[[f"shap_{f}" for f in feature_cols]].values
            else:
                shap_values = self._compute(model, X)
                self._save(csv_path, df_test, feature_cols, shap_values)
        else:
            shap_values = self._compute(model, X)
            self._save(csv_path, df_test, feature_cols, shap_values)

        self._plots(shap_values, X)

    def _compute(self, model, X):
        print("[shap] computing SHAP values on the test years")
        # TreeExplainer gives exact SHAP values for XGBoost.
        return shap.TreeExplainer(model).shap_values(X)

    def _save(self, path, df_test, feature_cols, shap_values):
        out = df_test[["year", "lat", "lon"]].reset_index(drop=True).copy()
        for i, f in enumerate(feature_cols):
            out[f"shap_{f}"] = shap_values[:, i]
        out.to_csv(path, index=False)

    def _plots(self, shap_values, X):
        s = self.s
        title = f"Cluster {s.CLUSTER_TO_MODEL}: {s.EXPERIMENT_NAME} (test years)"

        # Beeswarm summary plot (default SHAP figure)
        shap.summary_plot(shap_values, X, show=False)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(self.exp.path("shap_summary_beeswarm.png"), dpi=200, bbox_inches="tight")
        plt.close()

        # Bar plot of the mean absolute SHAP value per feature
        shap.summary_plot(shap_values, X, plot_type="bar", show=False)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(self.exp.path("shap_importance_bar.png"), dpi=200, bbox_inches="tight")
        plt.close()
