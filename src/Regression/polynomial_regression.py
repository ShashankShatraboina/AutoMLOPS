# =====================================================
# POLYNOMIAL REGRESSION MODULE
# Auto Degree Selection + 1-SE Rule
# AutoML Compatible Version
# =====================================================

import numpy as np
import pandas as pd

from functools import reduce
from operator import mul

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    train_test_split
)
from sklearn.metrics import mean_squared_error, r2_score


class PolynomialRegressionModel:

    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state

        self.best_degree = None
        self.model = None
        self.metrics = {}
        self.results = None

    def train(self, X, y):

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state
        )

        pipeline = Pipeline([
            ("poly", PolynomialFeatures()),
            ("model", LinearRegression())
        ])

        param_grid = {
            "poly__degree": list(range(1, 11))
        }

        search_space_size = reduce(
            mul,
            [len(v) for v in param_grid.values()]
        )

        if len(X_train) < 5000 and search_space_size <= 25:

            search = GridSearchCV(
                pipeline,
                param_grid,
                scoring="neg_mean_squared_error",
                cv=5,
                n_jobs=-1,
                return_train_score=True
            )

        else:

            search = RandomizedSearchCV(
                pipeline,
                param_grid,
                n_iter=min(20, search_space_size),
                scoring="neg_mean_squared_error",
                cv=5,
                random_state=self.random_state,
                n_jobs=-1,
                return_train_score=True
            )

        search.fit(X_train, y_train)

        results = pd.DataFrame(search.cv_results_)
        results["mean_mse"] = -results["mean_test_score"]
        results["std"] = results["std_test_score"]

        best_idx = results["mean_mse"].idxmin()
        best_error = results.loc[best_idx, "mean_mse"]
        best_std = results.loc[best_idx, "std"]

        threshold = best_error + best_std
        candidate_models = results[results["mean_mse"] <= threshold]

        self.best_degree = candidate_models["param_poly__degree"].min()

        self.model = Pipeline([
            ("poly", PolynomialFeatures(self.best_degree)),
            ("model", LinearRegression())
        ])

        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        self.metrics = {
            "mse": mse,
            "rmse": rmse,
            "r2": r2
        }

        self.results = results

        return self.model

    def get_equation(self, feature_names):

        poly = self.model.named_steps["poly"]
        linreg = self.model.named_steps["model"]

        names = poly.get_feature_names_out(feature_names)

        coefficients = linreg.coef_
        intercept = linreg.intercept_

        equation = f"y = {intercept:.4f}"

        for name, coef in zip(names[1:], coefficients[1:]):
            equation += f" + ({coef:.4f})*{name}"

        return equation

    def get_metrics(self):
        return self.metrics

    def get_cv_results(self):
        return self.results