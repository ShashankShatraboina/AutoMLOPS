# =====================================================
# H2O AutoML - Regression Example
# =====================================================

import h2o
from h2o.automl import H2OAutoML
import pandas as pd

# =====================================================
# 1️⃣ Start H2O Server
# =====================================================
h2o.init()

# =====================================================
# 2️⃣ Load Dataset (CSV)
# =====================================================
# Replace with your dataset path
df = pd.read_csv("linear.csv")

# Convert Pandas -> H2O Frame
hf = h2o.H2OFrame(df)

# =====================================================
# 3️⃣ Define Features and Target
# =====================================================
target = "target_column_name"   # Change this
features = [col for col in hf.columns if col != target]

# =====================================================
# 4️⃣ Split Data
# =====================================================
train, test = hf.split_frame(ratios=[0.8], seed=42)

# =====================================================
# 5️⃣ Run AutoML
# =====================================================
aml = H2OAutoML(
    max_models=10,        # Number of models to train
    seed=42,
    sort_metric="RMSE"    # For regression
)

aml.train(x=features, y=target, training_frame=train)

# =====================================================
# 6️⃣ Leaderboard
# =====================================================
print("📊 Leaderboard:")
print(aml.leaderboard)

# =====================================================
# 7️⃣ Best Model
# =====================================================
best_model = aml.leader
print("\n🏆 Best Model:")
print(best_model)

# =====================================================
# 8️⃣ Evaluate on Test Data
# =====================================================
performance = best_model.model_performance(test)

print("\n📈 Test Performance:")
print("RMSE:", performance.rmse())
print("MAE:", performance.mae())
print("R2:", performance.r2())

# =====================================================
# 9️⃣ Save Best Model
# =====================================================
model_path = h2o.save_model(best_model, path=".", force=True)
print("\n💾 Model saved at:", model_path)

# Shutdown if needed
# h2o.shutdown(prompt=False)