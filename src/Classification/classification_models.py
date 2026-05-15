import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


class ClassificationModels:

    def __init__(self):

        self.models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "SVM": SVC(probability=True),
            "KNN": KNeighborsClassifier(),
            "Naive Bayes": GaussianNB()
        }

        self.results_df = None
        self.best_model = None
        self.best_model_name = None
        self.best_metrics = {}

    # ------------------------------------------------------
    # TRAIN MODELS
    # ------------------------------------------------------

    def train(self, X, y):

        # Encode target if categorical
        if y.dtype == "object":
            le = LabelEncoder()
            y = le.fit_transform(y)

        # Train test split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        # Feature scaling
        scaler = StandardScaler()

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        results = []

        best_accuracy = 0

        for name, model in self.models.items():

            model.fit(X_train, y_train)

            preds = model.predict(X_test)

            accuracy = accuracy_score(y_test, preds)

            precision = precision_score(
                y_test,
                preds,
                average="weighted",
                zero_division=0
            )

            recall = recall_score(
                y_test,
                preds,
                average="weighted",
                zero_division=0
            )

            f1 = f1_score(
                y_test,
                preds,
                average="weighted",
                zero_division=0
            )

            # ---------------------------
            # ROC AUC Calculation
            # ---------------------------
            try:

                probs = model.predict_proba(X_test)

                # Binary classification
                if len(np.unique(y_test)) == 2:
                    roc_auc = roc_auc_score(y_test, probs[:, 1])

                # Multi-class classification
                else:
                    roc_auc = roc_auc_score(
                        y_test,
                        probs,
                        multi_class="ovr"
                    )

            except:
                roc_auc = 0

            results.append({
                "Model": name,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1,
                "ROC-AUC": roc_auc
            })

            # Best model tracking
            if accuracy > best_accuracy:

                best_accuracy = accuracy

                self.best_model = model
                self.best_model_name = name

                self.best_metrics = {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "roc_auc": roc_auc
                }

        # Create leaderboard
        self.results_df = pd.DataFrame(results).sort_values(
            by="Accuracy",
            ascending=False
        )

        return self.best_model, self.best_model_name, self.best_metrics


    # ------------------------------------------------------
    # RETURN MODEL LEADERBOARD
    # ------------------------------------------------------

    def get_results(self):

        return self.results_df