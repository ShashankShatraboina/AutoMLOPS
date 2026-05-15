# ==========================================================
# VISUALIZATION MODULE (Regression)
# ==========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

try:
    matplotlib.use('QtAgg')
except ImportError:
    matplotlib.use('Agg')


# ----------------------------------------------------------
# 1️⃣ Actual vs Predicted
# ----------------------------------------------------------
def plot_actual_vs_predicted(X, y, y_pred):

    plt.figure(figsize=(6, 6))

    plt.scatter(y, y_pred, alpha=0.7)

    plt.plot(
        [y.min(), y.max()],
        [y.min(), y.max()],
        'r--'
    )

    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted")
    plt.grid(True)
    plt.show()


# ----------------------------------------------------------
# 2️⃣ Residual Plot
# ----------------------------------------------------------
def plot_residuals(y, y_pred):

    residuals = y - y_pred

    plt.figure(figsize=(7, 5))
    plt.scatter(y_pred, residuals, alpha=0.7)

    plt.axhline(0, linestyle="--")

    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.grid(True)
    plt.show()


# ----------------------------------------------------------
# 3️⃣ Feature Importance (Random Forest Only)
# ----------------------------------------------------------
def plot_feature_importance(model, feature_names):

    try:
        # if pipeline exists
        if hasattr(model, "named_steps"):
            importances = model.named_steps["model"].feature_importances_
        else:
            importances = model.feature_importances_

        plt.figure(figsize=(8, 5))
        plt.barh(feature_names, importances)
        plt.gca().invert_yaxis()

        plt.title("Feature Importance (Random Forest)")
        plt.xlabel("Importance")
        plt.show()

    except Exception:
        print("Feature importance not available for this model.")


# ----------------------------------------------------------
# 4️⃣ Polynomial Curve (Single Feature Only)
# ----------------------------------------------------------
def plot_polynomial_curve(model, X, y):

    # Only single feature supported
    if X.shape[1] != 1:
        return

    feature_name = X.columns[0]

    x_vals = np.linspace(
        X[feature_name].min(),
        X[feature_name].max(),
        300
    )

    X_smooth = pd.DataFrame(
        x_vals,
        columns=[feature_name]
    )

    y_smooth = model.predict(X_smooth)

    plt.figure(figsize=(8, 5))
    plt.scatter(X[feature_name], y, alpha=0.7)
    plt.plot(x_vals, y_smooth)

    plt.title("Polynomial Fit")
    plt.xlabel(feature_name)
    plt.ylabel("Target")
    plt.grid(True)
    plt.show()