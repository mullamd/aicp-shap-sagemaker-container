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

# Parse input CSV or JSON
def input_fn(request_body, request_content_type):
    expected_features = [
        "claim_amount",
        "estimated_damage",
        "vehicle_year",
        "days_since_policy_start",
        "location_risk_score"
    ]

    if request_content_type == "application/json":
        data = json.loads(request_body)
        values = [float(data[f]) for f in expected_features]
        return np.array([values])

    elif request_content_type == "text/csv":
        values = [float(x) for x in request_body.strip().split(",")]
        return np.array([values])

    raise ValueError(f"Unsupported content type: {request_content_type}")

# Perform prediction and explanation
def predict_fn(input_data, model):
    if input_data.shape[1] != 5:
        raise ValueError(f"Expected 5 features, got {input_data.shape[1]}")

    dmatrix = xgb.DMatrix(input_data)
    score = float(model.predict(dmatrix)[0])
    prediction = "FRAUD" if score > 0.5 else "LEGIT"

    feature_names = [
        "claim_amount",
        "estimated_damage",
        "vehicle_year",
        "days_since_policy_start",
        "location_risk_score"
    ]

    # SHAP explanation (wrapped in try block)
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(dmatrix)
        shap_values = shap_values[0]  # for single prediction

        top_indices = np.argsort(np.abs(shap_values))[::-1][:3]
        top_features = [
            {
                "feature": feature_names[i],
                "impact": float(round(shap_values[i], 4))
            }
            for i in top_indices
        ]

        tech_expl = ", ".join([f"{f['feature']} ({f['impact']:+})" for f in top_features])

        reason_map = {
            "claim_amount": "the claim has a high amount",
            "estimated_damage": "the estimated damage is low compared to the claim",
            "vehicle_year": "the vehicle is older",
            "days_since_policy_start": "the claim was filed soon after the policy started",
            "location_risk_score": "the claim occurred in a high-risk area"
        }

        reason_phrases = [reason_map.get(f["feature"], f["feature"]) for f in top_features]
        plain_text = "The claim is suspicious because " + ", ".join(reason_phrases) + "."

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

    except Exception as e:
        return {
            "fraud_score": round(score, 4),
            "fraud_prediction": prediction,
            "fraud_explanation": "SHAP explanation not available due to internal error.",
            "error": str(e)
        }

# Format output
def output_fn(prediction, response_content_type):
    return json.dumps(prediction)
