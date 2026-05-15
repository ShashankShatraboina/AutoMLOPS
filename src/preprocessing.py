# =====================================================
# PREPROCESSING MODULE
# Converts raw dataset → numeric dataset for AutoML
# =====================================================

import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class DataPreprocessor:

    def __init__(self):

        self.numerical_cols = None
        self.categorical_cols = None
        self.preprocessor = None

    # --------------------------------------------------
    # Detect column types
    # --------------------------------------------------
    def detect_columns(self, X):

        self.numerical_cols = X.select_dtypes(
            include=["int64", "float64"]
        ).columns

        self.categorical_cols = X.select_dtypes(
            include=["object", "category"]
        ).columns

    # --------------------------------------------------
    # Build preprocessing pipeline
    # --------------------------------------------------
    def build_pipeline(self):

        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ])

        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, self.numerical_cols),
                ("cat", categorical_pipeline, self.categorical_cols)
            ]
        )

    # --------------------------------------------------
    # Fit + transform training data
    # --------------------------------------------------
    def fit_transform(self, X):

        self.detect_columns(X)

        self.build_pipeline()

        X_processed = self.preprocessor.fit_transform(X)

        feature_names = self.preprocessor.get_feature_names_out()

        return pd.DataFrame(X_processed, columns=feature_names)

    # --------------------------------------------------
    # Transform new data (for prediction)
    # --------------------------------------------------
    def transform(self, X):

        X_processed = self.preprocessor.transform(X)

        feature_names = self.preprocessor.get_feature_names_out()

        return pd.DataFrame(X_processed, columns=feature_names)