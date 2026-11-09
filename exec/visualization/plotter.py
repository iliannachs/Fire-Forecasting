"""Figures of the pipeline."""
import matplotlib

matplotlib.use("Agg")   # no display needed on a cluster
import matplotlib.pyplot as plt
import numpy as np


class Plotter:
    def __init__(self, settings, experiment):
        self.s = settings
        self.exp = experiment

    def feature_importance(self, fi):
        s = self.s
        plt.figure(figsize=(8, 5))
        plot_fi = fi.sort_values("importance")
        plt.barh(plot_fi["feature"], plot_fi["importance"])
        plt.xlabel("XGBoost feature importance")
        plt.title(f"Cluster {s.CLUSTER_TO_MODEL}: {s.EXPERIMENT_NAME}")
        plt.tight_layout()
        plt.savefig(self.exp.path("feature_importance.png"), dpi=200)
        plt.close()

    def permutation_importance(self, perm_df):
        s = self.s
        plt.figure(figsize=(8, 5))
        plot_perm = perm_df.sort_values("importance_mean")
        plt.barh(
            plot_perm["feature"],
            plot_perm["importance_mean"],
            xerr=plot_perm["importance_std"],
        )
        plt.xlabel("Decrease in test R2 after permutation")
        plt.title(f"Cluster {s.CLUSTER_TO_MODEL}: test permutation importance")
        plt.tight_layout()
        plt.savefig(self.exp.path("permutation_importance.png"), dpi=200)
        plt.close()

    def annual_reconstruction(self, annual_totals):
        """Observed vs predicted annual total burned area, test/evaluation periods shaded."""
        s = self.s
        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(
            annual_totals.index, annual_totals["burned_area"],
            marker="o", color="#1f77b4", linewidth=1.5, markersize=4, label="Observed total",
        )
        ax.plot(
            annual_totals.index, annual_totals["predicted"],
            marker="s", color="#ff7f0e", linewidth=1.5, markersize=4, label="Predicted total",
        )

        # Highlight test period and evaluation year
        t0, t1 = min(s.TEST_YEARS), max(s.TEST_YEARS)
        ax.axvspan(t0 - 0.5, t1 + 0.5, color="#dce8f5", alpha=0.6, label="Test period")
        ax.axvspan(
            s.EVALUATION_YEAR - 0.5, s.EVALUATION_YEAR + 0.5,
            color="#b8d3ec", alpha=0.8, label=f"{s.EVALUATION_YEAR} evaluation",
        )

        ax.set_title(
            f"Cluster {s.CLUSTER_TO_MODEL}: total burned area reconstruction", fontsize=13
        )
        ax.set_xlabel("Year", fontsize=11)
        ax.set_ylabel("Total burned area (ha)", fontsize=11)

        first, last = min(s.YEARS_TARGET), max(s.YEARS_TARGET)
        ax.set_xlim(first - 0.5, last + 0.5)
        ax.set_xticks(np.arange(first, last + 1, 5))
        ax.set_ylim(bottom=0)

        ax.grid(True, linestyle="-", linewidth=0.5, alpha=0.3)
        ax.legend(loc="upper left", frameon=True)

        plt.tight_layout()
        plt.savefig(
            self.exp.path(f"annual_reconstruction_cluster_{s.CLUSTER_TO_MODEL}.png"), dpi=300
        )
        plt.close()
