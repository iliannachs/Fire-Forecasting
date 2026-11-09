"""
Entry point of the pipeline. Run from anywhere with:

    python exec/run_pipeline.py                 # uses settings.py
    python exec/run_pipeline.py settings_sib4   # uses settings_sib4.py

All configuration is read from the chosen settings file in the project root
(default: settings.py). Paths and values are defined only in that file.

Stages: read data -> preprocess -> train (or load) model -> predict ->
evaluate -> save tables and figures.
"""
import sys
from pathlib import Path

# Make the project root (settings.py, exec/) importable independent of the working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import importlib  # noqa: E402

# Optional first argument: name of the settings module (without .py).
settings = importlib.import_module(sys.argv[1] if len(sys.argv) > 1 else "settings")

from exec.cache import DiskCache  # noqa: E402
from exec.data.reader import DataReader  # noqa: E402
from exec.evaluation.evaluator import Evaluator  # noqa: E402
from exec.evaluation.explainer import Explainer  # noqa: E402
from exec.experiment import Experiment  # noqa: E402
from exec.model.predictor import Predictor  # noqa: E402
from exec.model.trainer import ModelTrainer  # noqa: E402
from exec.preprocessing.preprocessor import Preprocessor  # noqa: E402
from exec.visualization.plotter import Plotter  # noqa: E402


def main():
    experiment = Experiment(settings)
    print(f"Experiment: {settings.EXPERIMENT_NAME}  ->  {experiment.dir}")

    cache = DiskCache(settings.CACHE_DIR, overwrite=settings.OVERWRITE_CACHE)

    # 1.-2. Read and preprocess (burned area grid, clusters, predictors, pixel tables)
    reader = DataReader(settings)
    preprocessor = Preprocessor(settings, reader, cache)
    df_train, df_test, df_eval, feature_cols = preprocessor.prepare()

    # 3. Hyperparameter tuning + final model (loaded from disk if already stored)
    model, best_result = ModelTrainer(settings, experiment).train(df_train, feature_cols)

    # 4. Predictions for training, test and evaluation years
    predictor = Predictor(model, feature_cols)
    df_train = predictor.predict(df_train)
    df_test = predictor.predict(df_test)
    df_eval = predictor.predict(df_eval)

    # 5. Evaluation
    evaluator = Evaluator(settings)
    metrics_df = evaluator.metrics(df_train, df_test, df_eval, feature_cols, best_result)
    metrics_df.to_csv(experiment.path("metrics.csv"), index=False)
    print("\nFINAL METRICS")
    print(metrics_df.T)

    fi = evaluator.feature_importance(model, feature_cols)
    fi.to_csv(experiment.path("feature_importance.csv"), index=False)

    perm_df = evaluator.permutation_importance(model, df_test, feature_cols)
    perm_df.to_csv(experiment.path("permutation_importance_test.csv"), index=False)

    # 6. Prediction tables
    df_train.to_csv(experiment.path("training_table_2000_2019.csv"), index=False)
    df_test.to_csv(experiment.path("test_table_2020_2024.csv"), index=False)
    df_eval.to_csv(experiment.path("evaluation_2025.csv"), index=False)

    # 7. Figures
    plotter = Plotter(settings, experiment)
    plotter.feature_importance(fi)
    plotter.permutation_importance(perm_df)
    plotter.annual_reconstruction(evaluator.annual_totals(df_train, df_test, df_eval))

    # 8. SHAP analysis on the test years (optional)
    if getattr(settings, "RUN_SHAP", False):
        Explainer(settings, experiment).run(model, df_test, feature_cols)

    print("\nDone.")
    print("Output directory:", experiment.dir)


if __name__ == "__main__":
    main()
