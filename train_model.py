import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load data
df = pd.read_csv("train_data_aicp.csv")
X = df.drop(columns=["fraud_label"])
y = df["fraud_label"]

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Train the model with class imbalance adjustment
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    eval_metric="logloss",
    scale_pos_weight=5  # To handle class imbalance
)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model in SageMaker-compatible format
model.get_booster().save_model("xgboost-model.json")
print("✅ Model saved as xgboost-model.json")
