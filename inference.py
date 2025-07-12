import os
import xgboost as xgb
import shap
import numpy as np
import json

# Load model from model directory
def model_fn(model_dir):
    model = xgb.Booster()
    model.load_model(os.path.join(model_dir, "xgboost-model.json"))
    return model

# Parse input CSV string
def input_fn(request_body, request_content_type):
    if request_content_type == "text/csv":
        values = [float(x) for x in request_body.strip().split(",")]
        return np.array([values])
    raise ValueError(f"Unsupported content type: {request_content_type}")

# Make prediction and build explanation
def predict_fn(input_data, model):
    dmatrix = xgb.DMatrix(input_data)

    # Predict fraud score
    score = float(model.predict(dmatrix)[0])
    prediction = "FRAUD" if score > 0.5 else "LEGIT"

    # SHAP explanation
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(dmatrix)

    # Define feature names (must match training order)
    feature_names = [
        "claim_amount",
        "estimated_damage",
        "vehicle_year",
        "days_since_policy_start",
        "location_risk_score"
    ]

    top_indices = np.argsort(np.abs(shap_values[0]))[::-1][:3]
    top_features = [
        {
            "feature": feature_names[i],
            "impact": round(shap_values[0][i], 4)
        }
        for i in top_indices
    ]

    # Build technical explanation (SHAP format)
    tech_expl = ", ".join([f"{f['feature']} ({f['impact']:+})" for f in top_features])

    # Plain English mapping
    reason_map = {
        "claim_amount": "the claim has a high amount",
        "estimated_damage": "the estimated damage is low compared to the claim",
        "vehicle_year": "the vehicle is older",
        "days_since_policy_start": "the claim was filed soon after the policy started",
        "location_risk_score": "the claim occurred in a high-risk area"
    }

    reason_phrases = [reason_map.get(f['feature'], f['feature']) for f in top_features]
    plain_text = "The claim is suspicious because " + ", ".join(reason_phrases) + "."

    # Combine into one explanation field
    full_explanation = f"Top features: {tech_expl}. {plain_text}"

    return {
        "fraud_score": round(score, 4),
        "fraud_prediction": prediction,
        "fraud_explanation": full_explanation,
        "shap_feature_1": top_features[0]["feature"],
        "shap_impact_1": top_features[0]["impact"],
        "shap_feature_2": top_features[1]["feature"],
        "shap_impact_2": top_features[1]["impact"],
        "shap_feature_3": top_features[2]["feature"],
        "shap_impact_3": top_features[2]["impact"]
    }

# Format response for SageMaker
def output_fn(prediction, response_content_type):
    return json.dumps(prediction)
