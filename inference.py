from flask import Flask, request, jsonify
import xgboost as xgb
import shap
import numpy as np
import os
import traceback

# ─────────────── Flask App ─────────────── #
app = Flask(__name__)

# ─────────────── Load Model ─────────────── #
model_path = os.path.join(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"), "xgboost-model.json")
model = xgb.Booster()
model.load_model(model_path)
explainer = shap.Explainer(model)

# ─────────────── Feature Order ─────────────── #
features = [
    "claim_to_damage_ratio",
    "vehicle_age",
    "previous_claims_count",
    "days_since_policy_start",
    "location_risk_score",
    "incident_time_hour"
]

# ─────────────── Dynamic Explanation Logic ─────────────── #
def get_dynamic_explanation(feature, shap_value):
    if feature == "location_risk_score":
        return "The accident occurred in a location that is considered high-risk for fraud." if shap_value > 0 else \
               "The accident occurred in a location that is not considered high-risk for fraud."
    elif feature == "previous_claims_count":
        return "This claimant has filed previous claims, which signals higher risk." if shap_value > 0 else \
               "This claimant has never filed a previous claim, which signals lower risk."
    elif feature == "claim_to_damage_ratio":
        return "The claimed amount is significantly higher than the estimated damage, which is suspicious." if shap_value > 0 else \
               "The claimed amount closely matches the estimated damage, indicating a reasonable request."
    elif feature == "vehicle_age":
        return "The vehicle's age appears unusual and may indicate potential risk." if shap_value > 0 else \
               "The vehicle's age appears typical and does not raise suspicion."
    elif feature == "days_since_policy_start":
        return "The claim was made soon after the policy started, which can be risky." if shap_value > 0 else \
               "The policy had been active for a sufficient period before the claim was made."
    elif feature == "incident_time_hour":
        return "The time of the incident falls outside normal hours, which may be suspicious." if shap_value > 0 else \
               "The time of the incident falls within normal hours and does not appear unusual."
    return f"Key factor: {feature}"

# ─────────────── Health Check ─────────────── #
@app.route("/ping", methods=["GET"])
def ping():
    return jsonify(status="ok"), 200

# ─────────────── Prediction Endpoint ─────────────── #
@app.route("/invocations", methods=["POST"])
def invoke():
    try:
        input_data = request.get_json()

        # Extract feature values in order
        input_values = [
            float(input_data.get("claim_to_damage_ratio", 0)),
            int(input_data.get("vehicle_age", 0)),
            int(input_data.get("previous_claims_count", 0)),
            int(input_data.get("days_since_policy_start", 0)),
            float(input_data.get("location_risk_score", 0.6)),
            int(input_data.get("incident_time_hour", 8))
        ]

        dmatrix = xgb.DMatrix(np.array([input_values]), feature_names=features)
        fraud_prob = model.predict(dmatrix)[0]
        fraud_class = int(fraud_prob > 0.5)
        shap_values = explainer(dmatrix)

        shap_score_map = dict(zip(
            features,
            [round(float(val) * 10, 2) for val in shap_values.values[0]]
        ))

        # Generate explanation from top 3 contributing features
        top_indices = np.argsort(np.abs(shap_values.values[0]))[::-1][:3]
        explanations = [
            get_dynamic_explanation(features[i], shap_values.values[0][i])
            for i in top_indices
        ]
        fraud_explanation = " | ".join(explanations)

        result = {
            "fraud_prediction": "Fraud" if fraud_class else "Not Fraud",
            "fraud_score": round(float(fraud_prob) * 10, 1),
            "fraud_explanation": fraud_explanation,
            "shap_values": shap_score_map
        }

        return jsonify(result), 200

    except Exception as e:
        return jsonify(error=str(e), traceback=traceback.format_exc()), 500

# ─────────────── Start Server ─────────────── #
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
