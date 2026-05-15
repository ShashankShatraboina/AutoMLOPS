# =====================================================
# PATTERN RECOGNITION MODULE
# Linear | Polynomial | Tree | Noisy
# AutoML Compatible Version
# =====================================================

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, KFold


class PatternRecognition:

    def __init__(self, margin=0.05):
        self.margin = margin
        self.best_degree = None
        self.scores = {}

    def detect(self, X, y):

        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        def cv_r2(model):
            return cross_val_score(model, X, y, cv=kf, scoring="r2").mean()

        # Linear
        linear_model = LinearRegression()
        r2_linear = cv_r2(linear_model)

        # Polynomial
        best_poly_score = -np.inf

        for degree in range(2, 6):
            poly_model = Pipeline([
                ("poly", PolynomialFeatures(degree)),
                ("model", LinearRegression())
            ])

            score = cv_r2(poly_model)

            if score > best_poly_score:
                best_poly_score = score
                self.best_degree = degree

        # Tree
        tree_model = DecisionTreeRegressor(max_depth=5, random_state=42)
        r2_tree = cv_r2(tree_model)

        self.scores = {
            "Linear": r2_linear,
            f"Polynomial_deg_{self.best_degree}": best_poly_score,
            "Tree": r2_tree
        }

        return self._decide_pattern(r2_linear, best_poly_score, r2_tree)

    def _decide_pattern(self, r2_linear, best_poly_score, r2_tree):

        if r2_linear < 0.2 and best_poly_score < 0.2 and r2_tree < 0.2:
            return "NOISY"

        elif r2_linear >= best_poly_score - self.margin and \
             r2_linear >= r2_tree - self.margin:
            return "LINEAR"

        elif best_poly_score > r2_linear + self.margin:
            return f"POLYNOMIAL_deg_{self.best_degree}"

        elif r2_tree > best_poly_score + self.margin:
            return "TREE"

        else:
            return "COMPLEX"

    def get_scores(self):
        return self.scores