# ==========================================================
# MAIN ENTRY POINT - REGRESSION AUTOML (FINAL STABLE VERSION)
# ==========================================================

import argparse
import pandas as pd

from src.automl_engine import AutoMLEngine
from src.visualization import (
    plot_actual_vs_predicted,
    plot_residuals,
    plot_feature_importance,
    plot_polynomial_curve
)


# --------------------------------------------------
# Load Dataset
# --------------------------------------------------
def load_data(path):

    print("\nLoading dataset...")
    df = pd.read_csv(path)

    print("Dataset shape:", df.shape)

    return df


# --------------------------------------------------
# Main Pipeline
# --------------------------------------------------
def main(data_path):

    print("\nAutoML Regression Pipeline Started")

    df = load_data(data_path)

    # Drop ID column if exists
    if "Id" in df.columns:
        df = df.drop(columns=["Id"])

    # Assume last column is target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    print("Feature shape:", X.shape)
    print("Target shape:", y.shape)

    # --------------------------------------------------
    # Run AutoML Engine
    # --------------------------------------------------
    engine = AutoMLEngine()

    result = engine.run_regression(X, y)
    best_model = result["model"]
    best_model_name = result["model_name"]
    y_pred = result["predictions"]

    print("\nFinal Best Model:", best_model_name)
    print("Final Metrics:", result["metrics"])

    # --------------------------------------------------
    # SAFE VISUALIZATION BLOCK
    # --------------------------------------------------
    print("\nGenerating Visualizations...")

    # Always safe
    plot_actual_vs_predicted(X, y, y_pred)
    plot_residuals(y, y_pred)

    # Only if Random Forest selected
    if "Random Forest" in best_model_name:
        plot_feature_importance(best_model, X.columns)

    # Only if Polynomial selected
    if "Polynomial" in best_model_name:
        plot_polynomial_curve(best_model, X, y)

    print("\nAutoML Regression Completed Successfully!")


# --------------------------------------------------
# CLI ENTRY
# --------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="AutoML Regression Pipeline"
    )

    parser.add_argument(
        "--data",
        required=True,
        help="Path to dataset CSV file"
    )

    args = parser.parse_args()

    main(args.data)