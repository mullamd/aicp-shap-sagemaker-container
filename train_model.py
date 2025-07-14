import xgboost as xgb
import pandas as pd
import numpy as np

FEATURE_NAMES = [
    "claim_amount",
    "estimated_damage",
    "vehicle_year",
    "days_since_policy_start",
    "location_risk_score"
]

np.random.seed(42)

# Generate synthetic data
X_train = pd.DataFrame({
    "claim_amount": np.random.uniform(1000, 10000, 2000),
    "estimated_damage": np.random.uniform(500, 9500, 2000),
    "vehicle_year": np.random.randint(2000, 2023, 2000),
    "days_since_policy_start": np.random.randint(1, 366, 2000),
    "location_risk_score": np.random.uniform(0, 5, 2000)
})

# Define fraud logic
y_train = (
    (X_train["claim_amount"] > 8000) &
    (X_train["estimated_damage"] < 2000) &
    (X_train["days_since_policy_start"] < 30) &
    (X_train["location_risk_score"] > 3)
).astype(int)

# Print fraud distribution
fraud_count = y_train.value_counts()
print("Fraud Label Distribution:")
print(fraud_count)

# Handle class imbalance
if 1 in fraud_count:
    scale_pos_weight = fraud_count[0] / fraud_count[1]
else:
    scale_pos_weight = 1  # fallback to avoid division by zero

print(f"Using scale_pos_weight: {scale_pos_weight:.2f}")

# Train XGBoost Classifier with weight adjustment
model = xgb.XGBClassifier(
    eval_metric='logloss',
    use_label_encoder=False,
    scale_pos_weight=scale_pos_weight
)
model.fit(X_train, y_train)

# Save model
model.save_model("xgboost-model.json")

print("✅ Model trained and saved with class balancing.")
