# =====================================================
# STREAMLIT FRONTEND - INTELLIGENT AUTOML PLATFORM
# =====================================================

import sys
import os
import time

# Add project root path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.automl_engine import AutoMLEngine
from src.preprocessing import DataPreprocessor


# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="AutoML System", layout="wide")
st.title("🚀 MLOPs-Enabled AutoML Pipeline - ML Factory")


# =====================================================
# SESSION STATE
# =====================================================
if "result" not in st.session_state:
    st.session_state.result = None

if "df" not in st.session_state:
    st.session_state.df = None

if "processed_df" not in st.session_state:
    st.session_state.processed_df = None


# =====================================================
# SIDEBAR NAVIGATION
# =====================================================
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "1️⃣ Dataset Upload",
        "2️⃣ Data Preview",
        "3️⃣ Pattern Detection",
        "4️⃣ Model Details",
        "5️⃣ Visualization",
        "6️⃣ Prediction"
    ]
)


# =====================================================
# 1️⃣ DATASET UPLOAD
# =====================================================
if page == "1️⃣ Dataset Upload":

    st.header("Upload Dataset")

    file = st.file_uploader("Upload CSV File", type=["csv"])

    if file:
        df = pd.read_csv(file)
        st.session_state.df = df

        st.success("Dataset Loaded Successfully ✅")
        st.dataframe(df.head(), width="stretch")

        problem_type = st.selectbox(
            "Select Problem Type",
            ["Regression", "Classification", "Clustering"]
        )

        if st.button("Run AutoML"):

            with st.spinner("Running AutoML Engine..."):

                engine = AutoMLEngine()

                if problem_type == "Regression":

                    X = df.iloc[:, :-1]
                    y = df.iloc[:, -1]

                    preprocessor = DataPreprocessor()
                    X = preprocessor.fit_transform(X)

                    st.session_state.processed_df = X

                    result = engine.run_regression(X, y)

                elif problem_type == "Classification":

                    X = df.iloc[:, :-1]
                    y = df.iloc[:, -1]

                    preprocessor = DataPreprocessor()
                    X = preprocessor.fit_transform(X)

                    st.session_state.processed_df = X

                    result = engine.run_classification(X, y)

                elif problem_type == "Clustering":

                    X = df

                    preprocessor = DataPreprocessor()
                    X = preprocessor.fit_transform(X)

                    st.session_state.processed_df = X

                    result = engine.run_clustering(X)

                st.session_state.result = result

            st.success("AutoML Completed ✅")

    st.info("Upload dataset and click Run AutoML.")


# =====================================================
# 2️⃣ DATA PREVIEW
# =====================================================
elif page == "2️⃣ Data Preview":

    st.header("🧾 Dataset Preview & Understanding")

    df = st.session_state.df
    processed_df = st.session_state.processed_df

    if df is None:
        st.warning("Upload dataset first.")
    else:

        st.caption(f"Dataset Size: {df.shape[0]} rows × {df.shape[1]} columns")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Rows", df.shape[0])
        c2.metric("Columns", df.shape[1])
        c3.metric("Missing Values", int(df.isnull().sum().sum()))

        st.divider()

        tab1, tab2, tab3, tab4 = st.tabs([
            "📄 Data Before Preprocessing",
            "⚙️ Data After Preprocessing",
            "📊 Dataset Information",
            "❗ Missing Values"
        ])

        with tab1:
            st.subheader("Raw Dataset")
            st.dataframe(df.head(10), width="stretch")

        with tab2:
            st.subheader("Processed Dataset")
            if processed_df is None:
                st.info("Run AutoML first.")
            else:
                st.dataframe(processed_df.head(10), width="stretch")

        with tab3:
            st.dataframe(df.describe(), width="stretch")

        with tab4:
            st.dataframe(df.isnull().sum())


# =====================================================
# 3️⃣ PATTERN DETECTION
# =====================================================
elif page == "3️⃣ Pattern Detection":

    st.header("🔍 Pattern Detection")

    result = st.session_state.result

    if result is None:
        st.warning("Run AutoML first.")
    else:
        st.success(f"Detected Pattern: {result['pattern']}")


# =====================================================
# 4️⃣ MODEL DETAILS
# =====================================================
elif page == "4️⃣ Model Details":

    st.markdown("## 🧠 Model Details")

    result = st.session_state.result

    if result is None:
        st.warning("Run AutoML first.")
    else:

        # =====================================================
        # FIX: Determine problem type correctly
        # =====================================================
        problem_type = result.get("problem_type", None)

        if problem_type is None:
            if result.get("pattern") == "Classification":
                problem_type = "Classification"
            elif result.get("pattern") == "Clustering":
                problem_type = "Clustering"
            else:
                problem_type = "Regression"

        # =====================================================
        # MODEL NAME CARD
        # =====================================================
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #1f2937, #111827);
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #374151;
            margin-bottom: 15px;
        ">
            <h3 style="color:#60a5fa; margin:0;">Selected Model</h3>
            <h2 style="color:white; margin:5px 0 0 0;">
                {result["model_name"]}
            </h2>
        </div>
        """, unsafe_allow_html=True)

        metrics = result.get("metrics", {})

        st.markdown("### 📊 Performance Metrics")

        # =====================================================
        # REGRESSION METRICS
        # =====================================================
        if problem_type == "Regression":

            r2 = metrics.get("r2", metrics.get("test_r2", 0))
            rmse = metrics.get("rmse", 0)

            c1, c2 = st.columns(2)
            c1.metric("📈 R² Score", f"{r2:.4f}")
            c2.metric("📉 RMSE", f"{rmse:.4f}")

        # =====================================================
        # CLASSIFICATION METRICS
        # =====================================================
        elif problem_type == "Classification":

            c1, c2, c3, c4 = st.columns(4)

            c1.metric("Accuracy", f"{metrics.get('accuracy',0):.4f}")
            c2.metric("Precision", f"{metrics.get('precision',0):.4f}")
            c3.metric("Recall", f"{metrics.get('recall',0):.4f}")
            c4.metric("ROC-AUC", f"{metrics.get('roc_auc',0):.4f}")

            st.metric("F1 Score", f"{metrics.get('f1_score',0):.4f}")

        # =====================================================
        # CLUSTERING METRICS
        # =====================================================
        else:

            score = metrics.get("silhouette_score", None)

            if score is not None:
                st.metric("🧩 Silhouette Score", f"{score:.4f}")
            else:
                st.warning("No clustering metrics available")

        # =====================================================
        # FULL METRICS VIEW
        # =====================================================
        with st.expander("🔍 View Full Metrics"):
            st.json(metrics)

        # =====================================================
        # EXTRA INFO
        # =====================================================
        if "reason" in result:
            st.markdown("### 🤖 AutoML Decision")
            st.info(result["reason"])

        if "confidence" in result:
            st.success(f"Confidence Level: {result['confidence']}%")

# =====================================================
# 5️⃣ VISUALIZATION
# =====================================================
elif page == "5️⃣ Visualization":

    st.header("📈 Model Visualizations")

    result = st.session_state.result

    if result is None:
        st.warning("Run AutoML first.")
    else:

        X = pd.DataFrame(result["X"])
        y = np.array(result["y"])
        preds = np.array(result["predictions"])

        # =====================================================
        # FIX: Determine problem type correctly
        # =====================================================
        problem_type = result.get("problem_type", None)

        if problem_type is None:
            if result.get("pattern") == "Classification":
                problem_type = "Classification"
            elif result.get("pattern") == "Clustering":
                problem_type = "Clustering"
            else:
                problem_type = "Regression"

        # =====================================================
        # Feature selection
        # =====================================================
        feature = st.selectbox("Select feature", X.columns)
        X_plot = X[feature].values

        # =====================================================
        # Plot options
        # =====================================================
        if problem_type == "Regression":
            plots = [
                "Actual vs Predicted",
                "Residual Plot",
                "Model Visualization"
            ]

        elif problem_type == "Classification":
            plots = [
                "Class Distribution",
                "Feature Visualization",
                "Prediction Plot"
            ]

        else:  # Clustering
            plots = [
                "Cluster Scatter",
                "Cluster Distribution",
                "Cluster View"
            ]

        # =====================================================
        # State
        # =====================================================
        if "plot_index" not in st.session_state:
            st.session_state.plot_index = 0

        if "auto_play" not in st.session_state:
            st.session_state.auto_play = False

        # =====================================================
        # Controls
        # =====================================================
        c1, c2, c3, c4 = st.columns([1,3,1,1])

        with c1:
            if st.button("⬅"):
                st.session_state.plot_index = max(
                    0,
                    st.session_state.plot_index - 1
                )

        with c3:
            if st.button("➡"):
                st.session_state.plot_index = min(
                    len(plots)-1,
                    st.session_state.plot_index + 1
                )

        with c4:
            st.session_state.auto_play = st.toggle(
                "Auto Play",
                value=st.session_state.auto_play
            )

        current_plot = plots[st.session_state.plot_index]

        st.subheader(current_plot)

        # =====================================================
        # Plotting (SMALL + CENTERED)
        # =====================================================
        fig, ax = plt.subplots(figsize=(4.5, 3))
        X_np = np.array(X)

        # ---------------- REGRESSION ----------------
        if problem_type == "Regression":

            if current_plot == "Actual vs Predicted":
                ax.scatter(y, preds, alpha=0.7)
                ax.plot([y.min(), y.max()], [y.min(), y.max()], "r--")

            elif current_plot == "Residual Plot":
                residuals = y - preds
                ax.scatter(preds, residuals)
                ax.axhline(0, color="red")

            elif current_plot == "Model Visualization":
                idx = np.argsort(X_plot)
                ax.scatter(X_plot, y)
                ax.plot(X_plot[idx], preds[idx], color="red")

        # ---------------- CLASSIFICATION ----------------
        elif problem_type == "Classification":

            if current_plot == "Class Distribution":
                vals, cnt = np.unique(preds, return_counts=True)
                ax.bar(vals, cnt)

            elif current_plot == "Feature Visualization":
                if X_np.shape[1] >= 2:
                    ax.scatter(X_np[:,0], X_np[:,1], c=preds, cmap="coolwarm")
                else:
                    ax.scatter(X_np[:,0], preds, c=preds)

            elif current_plot == "Prediction Plot":
                ax.scatter(y, preds, c=preds, cmap="coolwarm")

        # ---------------- CLUSTERING ----------------
        else:

            if current_plot == "Cluster Scatter":
                if X_np.shape[1] >= 2:
                    ax.scatter(X_np[:,0], X_np[:,1], c=preds, cmap="viridis")
                else:
                    ax.scatter(X_np[:,0], preds, c=preds)

            elif current_plot == "Cluster Distribution":
                vals, cnt = np.unique(preds, return_counts=True)
                ax.bar(vals, cnt)

            elif current_plot == "Cluster View":
                if X_np.shape[1] >= 2:
                    ax.scatter(X_np[:,0], X_np[:,1], c=preds, cmap="viridis")

        # =====================================================
        # Centered small plot
        # =====================================================
        left, center, right = st.columns([1,2,1])

        with center:
            st.pyplot(fig, width="content")

        # =====================================================
        # Auto Play
        # =====================================================
        if st.session_state.auto_play:
            time.sleep(1.5)
            st.session_state.plot_index = (
                st.session_state.plot_index + 1
            ) % len(plots)
            st.rerun()


# =====================================================
# 6️⃣ PREDICTION
# =====================================================
elif page == "6️⃣ Prediction":

    st.header("🔮 Prediction")

    result = st.session_state.result

    if result is None:
        st.warning("Run AutoML first.")
    else:

        model = result["model"]
        X = pd.DataFrame(result["X"])

        inputs = [st.number_input(col) for col in X.columns]

        if st.button("Predict"):
            pred = model.predict([inputs])
            st.success(pred)