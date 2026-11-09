"""Prediction with a trained model."""
import numpy as np


class Predictor:
    """Applies a trained model to pixel tables."""

    def __init__(self, model, feature_cols):
        self.model = model
        self.feature_cols = feature_cols

    def predict(self, df):
        """Adds 'predicted' (clipped at 0, burned area cannot be negative) and 'difference'."""
        df = df.copy()
        df["predicted"] = np.clip(self.model.predict(df[self.feature_cols]), 0, None)
        df["difference"] = df["predicted"] - df["burned_area"]
        return df
