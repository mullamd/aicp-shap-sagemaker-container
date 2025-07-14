import xgboost as xgb
import pandas as pd
import shap
import numpy as np

# ✅ Load trained model
model = xgb.XGBClassifier()
model.load_model("xgboost-model.json")

# 🔍 Sample input (you can modify this for different tests)
input_data = pd.DataFrame([{
    "claim_amount": 9500,
    "estimated_damage": 1000,
    "vehicle_year": 2004,
    "days_since_policy_start": 5,
    "location_risk_score": 4.8
}])

# ✅ SHAP explanation setup
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(input_data)[0]

# ✅ Get prediction score
score = model.predict_proba(input_data)[0][1]
prediction = "FRAUD" if score > 0.5 else "LEGIT"

# ✅ Sort and format all SHAP features
sorted_indices = np.argsort(np.abs(shap_values))[::-1]
all_features = [
    {
        "feature": input_data.columns[i],
        "impact": round(shap_values[i], 1)
    }
    for i in sorted_indices
]

# ✅ Build SHAP explanation string
shap_summary = [f"{f['feature']} ({f['impact']:+.1f})" for f in all_features]
tech_expl = ", ".join(shap_summary)

# ✅ Generate human-readable explanation
reason_map = {
    "claim_amount": "the claim has a high amount",
    "estimated_damage": "the estimated damage is low compared to the claim",
    "vehicle_year": "the vehicle is older",
    "days_since_policy_start": "the claim was filed soon after the policy started",
    "location_risk_score": "the claim occurred in a high-risk area"
}
reasons = [reason_map.get(f["feature"], f["feature"]) for f in all_features[:3]]  # top 3 for plain text

plain_text = (
    "The claim is suspicious because " + ", ".join(reasons) + "."
    if prediction == "FRAUD"
    else "The claim appears legitimate based on current features."
)

# ✅ Output
print(f"✅ Fraud Score: {score}")
print(f"✅ Prediction: {prediction}")
print(f"✅ All SHAP Features: {shap_summary}")
print(f"✅ Plain Explanation: {plain_text}")
print(f"✅ Full Explanation: Top features: {tech_expl}. {plain_text}")
