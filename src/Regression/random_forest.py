# ==========================================================
# RANDOM FOREST REGRESSION MODULE
# AutoML Compatible Version
# ==========================================================

import numpy as np

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_val_score,
    KFold
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer


class RandomForestModel:

    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state

        self.model = None
        self.best_params = None
        self.metrics = {}
        self.cv_scores = None

    def _build_preprocessor(self, X):

        numerical_features = X.select_dtypes(
            include=["int64", "float64"]
        ).columns

        categorical_features = X.select_dtypes(
            include=["object", "category"]
        ).columns

        numerical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ])

        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        return ColumnTransformer([
            ("num", numerical_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features)
        ])

    def train(self, X, y):

        preprocessor = self._build_preprocessor(X)

        rf = RandomForestRegressor(
            random_state=self.random_state,
            n_jobs=-1
        )

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", rf)
        ])

        kf = KFold(
            n_splits=5,
            shuffle=True,
            random_state=self.random_state
        )

        self.cv_scores = cross_val_score(
            pipeline,
            X,
            y,
            cv=kf,
            scoring="r2",
            n_jobs=-1
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state
        )

        param_grid = {
            "model__n_estimators": [200, 300],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5],
            "model__max_features": ["sqrt", "log2"]
        }

        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=kf,
            scoring="r2",
            n_jobs=-1,
            verbose=0
        )

        grid_search.fit(X_train, y_train)

        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_

        y_pred = self.model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        self.metrics = {
            "test_r2": r2,
            "cv_mean_r2": self.cv_scores.mean(),
            "cv_std_r2": self.cv_scores.std(),
            "rmse": rmse,
            "mae": mae
        }

        return self.model

    def get_metrics(self):
        return self.metrics

    def get_best_params(self):
        return self.best_params

    def get_cv_scores(self):
        return self.cv_scores