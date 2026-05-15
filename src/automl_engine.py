# ==========================================================
# AUTOML ENGINE
# Supports Regression + Classification + Clustering
# ==========================================================

import numpy as np

from src.Regression.pattern_recognition import PatternRecognition
from src.Regression.polynomial_regression import PolynomialRegressionModel
from src.Regression.random_forest import RandomForestModel

from src.Classification.classification_models import ClassificationModels
from src.Clustering.clustering_models import ClusteringModels


class AutoMLEngine:

    def __init__(self):
        self.best_model = None
        self.best_model_name = None
        self.results = {}
        self.pattern = None
        self.pattern_scores = {}
        self.predictions = None

    # ------------------------------------------------------
    # REGRESSION
    # ------------------------------------------------------
    def run_regression(self, X_train, y_train):

        detector = PatternRecognition()
        self.pattern = detector.detect(X_train, y_train)
        self.pattern_scores = detector.get_scores()

        poly_obj = PolynomialRegressionModel()
        poly_model = poly_obj.train(X_train, y_train)
        poly_metrics = poly_obj.get_metrics()

        rf_obj = RandomForestModel()
        rf_model = rf_obj.train(X_train, y_train)
        rf_metrics = rf_obj.get_metrics()

        if poly_metrics["r2"] >= rf_metrics["test_r2"]:
            self.best_model = poly_model
            self.best_model_name = "Polynomial Regression"
            self.results = poly_metrics
        else:
            self.best_model = rf_model
            self.best_model_name = "Random Forest Regressor"
            self.results = rf_metrics

        self.predictions = self.best_model.predict(X_train)

        return {
            "problem_type": "Regression",  # ✅ ADDED
            "pattern": self.pattern,
            "pattern_scores": self.pattern_scores,
            "model_name": self.best_model_name,
            "metrics": self.results,
            "reason": f"{self.best_model_name} selected based on performance.",
            "confidence": 95,
            "model": self.best_model,
            "X": X_train,
            "y": y_train,
            "predictions": self.predictions
        }

    # ------------------------------------------------------
    # CLASSIFICATION
    # ------------------------------------------------------
    def run_classification(self, X, y):

        clf = ClassificationModels()
        best_model, best_name, metrics = clf.train(X, y)

        preds = best_model.predict(X)

        return {
            "problem_type": "Classification",  # ✅ ADDED
            "pattern": "Classification",
            "pattern_scores": {},
            "model_name": best_name,
            "metrics": metrics,
            "reason": f"{best_name} selected because it achieved highest accuracy.",
            "confidence": round(metrics["accuracy"] * 100, 2),
            "model": best_model,
            "X": X,
            "y": y,
            "predictions": preds
        }

    # ------------------------------------------------------
    # CLUSTERING
    # ------------------------------------------------------
    def run_clustering(self, X):

        clusterer = ClusteringModels()
        best_model, best_name, score, labels = clusterer.train(X)

        return {
            "problem_type": "Clustering",  # ✅ ADDED
            "pattern": "Clustering",
            "pattern_scores": {},
            "model_name": best_name,
            "metrics": {"silhouette_score": score},
            "reason": f"{best_name} selected based on silhouette score.",
            "confidence": max(0, score) * 100,
            "model": best_model,
            "X": X,
            "y": labels,
            "predictions": labels
        }