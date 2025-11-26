# UTF-8 Python 3.13.5
# Author: Intan K. Wardhani
# Last modified: 25-11-2025

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error

class ModelTrainer:
    """
    Class to train and evaluate regression models for house price prediction.

    Features:
    - Preprocessing pipelines for numerical & categorical features
    - Handles imputation, optional scaling, and encoding
    - Supports K-Fold cross-validation on training data
    - Evaluates models on a held-out test set
    - Works with Ridge, RandomForest, and XGBoost regressors

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target (house prices)
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test target
    """

    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        # Define models
        self.models = {
            "Ridge": Ridge(alpha=1.0),
            "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
            "XGBoost": XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
        }

        # Column splitting
        self.numerical_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
        self.categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

        self._build_preprocessors()

    def _build_preprocessors(self):
        """Construct 2 separate preprocessors: scaled & non-scaled."""

        # ---- Shared categorical pipeline ----
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        # ---- Numerical pipeline for Ridge (scaled) ----
        num_scaled_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        # ---- Numerical pipeline for RF and XGBoost (NO scaling) ----
        num_noscale_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ])

        # Store two preprocessors
        self.preprocessor_scaled = ColumnTransformer([
            ("num", num_scaled_pipe, self.numerical_cols),
            ("cat", cat_pipe, self.categorical_cols)
        ])

        self.preprocessor_noscale = ColumnTransformer([
            ("num", num_noscale_pipe, self.numerical_cols),
            ("cat", cat_pipe, self.categorical_cols)
        ])


    def _get_preprocessor(self, model_name):
        """Return correct preprocessor for the model."""
        if model_name == "Ridge":
            return self.preprocessor_scaled
        else:  # RandomForest, XGBoost
            return self.preprocessor_noscale


    def cross_validate_models(self, cv=5):
        results = {}

        for name, model in self.models.items():
            preprocessor = self._get_preprocessor(name)

            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("regressor", model)
            ])

            scores = cross_val_score(
                pipeline,
                self.X_train,
                self.y_train,
                cv=cv,
                scoring="neg_root_mean_squared_error"
            )

            results[name] = {
                "cv_scores": -scores,
                "mean_cv_rmse": -scores.mean()
            }

        return results


    def train_and_evaluate(self):
        results = {}

        for name, model in self.models.items():
            preprocessor = self._get_preprocessor(name)

            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("regressor", model)
            ])

            pipeline.fit(self.X_train, self.y_train)
            y_pred = pipeline.predict(self.X_test)
            rmse = root_mean_squared_error(self.y_test, y_pred)

            results[name] = rmse

        return results


