import os
import xgboost as xgb
import shap
import numpy as np
import json
from flask import Flask, request, jsonify

# Load model from /opt/ml/model directory
def model_fn(model_dir):
    model = xgb.Booster()
    model_path = os.path.join(model_dir, "xgboost-model.json")
    model.load_model(model_path)
    return model

# Parse input
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

# Perform prediction + SHAP explanation
def predict_fn(input_data, model):
    if input_data.shape[1] != 5:
        raise ValueError(f"Expected 5 features, got {input_data.shape[1]}")

    feature_names = [
        "claim_amount",
        "estimated_damage",
        "vehicle_year",
        "days_since_policy_start",
        "location_risk_score"
    ]

    # ✅ Fix: Pass feature names to DMatrix
    dmatrix = xgb.DMatrix(input_data, feature_names=feature_names)
    score = float(model.predict(dmatrix)[0])
    prediction = "FRAUD" if score > 0.5 else "LEGIT"

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(dmatrix)
        shap_values = shap_values[0]  # Single row

        # Sort by absolute SHAP impact
        sorted_indices = np.argsort(np.abs(shap_values))[::-1]
        all_features = [
            {
                "feature": feature_names[i],
                "impact": round(shap_values[i], 1)
            }
            for i in sorted_indices
        ]

        # Build technical explanation string
        tech_expl = ", ".join([f"{f['feature']} ({f['impact']:+.1f})" for f in all_features])

        # Plain English explanation
        reason_map = {
            "claim_amount": {
                "pos": "the claim has a high amount",
                "neg": "the claim amount is reasonable"
            },
            "estimated_damage": {
                "pos": "the estimated damage is low compared to the claim",
                "neg": "the estimated damage matches the claim"
            },
            "vehicle_year": {
                "pos": "the vehicle is older",
                "neg": "the vehicle is newer"
            },
            "days_since_policy_start": {
                "pos": "the claim was filed soon after the policy started",
                "neg": "the policy has been active for a long time"
            },
            "location_risk_score": {
                "pos": "the claim occurred in a high-risk area",
                "neg": "the location is low-risk"
            }
        }

        top3 = all_features[:3]
        plain_reasons = [
            reason_map[f["feature"]]["pos" if f["impact"] > 0 else "neg"]
            for f in top3
        ]

        plain_text = (
            "The claim is suspicious because " + ", ".join(plain_reasons) + "."
            if prediction == "FRAUD"
            else "The claim appears legitimate based on current features."
        )

        return {
            "fraud_score": round(score, 4),
            "fraud_prediction": prediction,
            "fraud_explanation": plain_text,
            "shap_features": [f"{f['feature']} ({f['impact']:+.1f})" for f in all_features]
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

# ========== Flask App for Local & SageMaker ========== #
app = Flask(__name__)
model = model_fn("/opt/ml/model")

@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "ok"}), 200

@app.route("/invocations", methods=["POST"])
def invoke():
    try:
        content_type = request.content_type
        data = request.data.decode("utf-8")
        input_data = input_fn(data, content_type)
        prediction = predict_fn(input_data, model)
        return output_fn(prediction, content_type), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ✅ Local test entry point
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
