import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class PhysicsFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        torque = X["Torque"].to_numpy(dtype=float)
        rpm = X["Rotational speed"].to_numpy(dtype=float)
        X["Mechanical Power W"] = torque * rpm * (2 * np.pi / 60.0)
        X["Temperature Delta K"] = (
            X["Process temperature"].to_numpy(dtype=float)
            - X["Air temperature"].to_numpy(dtype=float)
        )
        X["Wear-Torque Interaction"] = (
            X["Tool wear"].to_numpy(dtype=float) * torque
        )
        return X


def add_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    return PhysicsFeatures().transform(df)
