import joblib
import json
import os
from datetime import datetime

def save_best_model(model):

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/best_model.pkl")

    print("✅ Model saved")


def log_experiment(results):

    os.makedirs("experiments", exist_ok=True)

    filename = f"experiments/run_{datetime.now().timestamp()}.json"

    with open(filename, "w") as f:
        json.dump(results, f, indent=4)

    print("✅ Experiment logged")
