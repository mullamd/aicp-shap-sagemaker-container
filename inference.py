from flask import Flask, request, jsonify
import boto3
import xgboost as xgb
import shap
import numpy as np
import os
import traceback
import json
from datetime import datetime
from pytz import timezone

# ───────── Flask App ───────── #
app = Flask(__name__)
s3 = boto3.client("s3")

bucket = "aicp-claims-data"
input_prefix = "processed/DQ-validated-claims-data/"
output_prefix = "processed/fraud-predicted-claims-data/"

# ───────── Load Model ───────── #
model_path = os.path.join(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"), "xgboost-model.json")
model = xgb.Booster()
model.load_model(model_path)
explainer = shap.Explainer(model)

# ───────── Features ───────── #
features = [
    "claim_to_damage_ratio",
    "vehicle_age",
    "previous_claims_count",
    "days_since_policy_start",
    "location_risk_score",
    "incident_time_hour"
]

# ───────── SHAP Explanation Logic ───────── #
def get_dynamic_explanation(feature, shap_value):
    explanations = {
        "location_risk_score": (
            "The accident occurred in a location that is considered high-risk for fraud."
            if shap_value > 0 else
            "The accident occurred in a location that is not considered high-risk for fraud."
        ),
        "previous_claims_count": (
            "This claimant has filed previous claims, which signals higher risk."
            if shap_value > 0 else
            "This claimant has never filed a previous claim, which signals lower risk."
        ),
        "claim_to_damage_ratio": (
            "The claimed amount is significantly higher than the estimated damage, which is suspicious."
            if shap_value > 0 else
            "The claimed amount closely matches the estimated damage, indicating a reasonable request."
        ),
        "vehicle_age": (
            "The vehicle's age appears unusual and may indicate potential risk."
            if shap_value > 0 else
            "The vehicle's age appears typical and does not raise suspicion."
        ),
        "days_since_policy_start": (
            "The claim was made soon after the policy started, which can be risky."
            if shap_value > 0 else
            "The policy had been active for a sufficient period before the claim was made."
        ),
        "incident_time_hour": (
            "The time of the incident falls outside normal hours, which may be suspicious."
            if shap_value > 0 else
            "The time of the incident falls within normal hours and does not appear unusual."
        )
    }
    return explanations.get(feature, f"Key factor: {feature}")

# ───────── Decision Logic ───────── #
def determine_decision(prediction, score, ratio, risk, prev_claims):
    if prediction == "Fraud" or score >= 8.0:
        return "Suspicious"
    elif (
        prediction == "Not Fraud"
        and score <= 2.0
        and ratio <= 1.5
        and risk < 0
        and prev_claims == 0
    ):
        return "Auto-Approve"
    else:
        return "Manual Review"

# ───────── Helpers ───────── #
def get_latest_claim_file(claim_id):
    prefix = f"{input_prefix}clean-claim-{claim_id}"
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    objects = response.get("Contents", [])
    if not objects:
        return None
    return sorted(objects, key=lambda x: x["LastModified"], reverse=True)[0]["Key"]

def get_timestamp_str():
    return datetime.now(timezone("US/Eastern")).strftime("%B-%d-%Y_%I-%M-%p")

# ───────── Main Prediction Logic ───────── #
def process_claim(claim_id):
    key = get_latest_claim_file(claim_id)
    if not key:
        raise FileNotFoundError(f"No file found for claim ID: {claim_id}")

    raw = s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8").strip()
    claim_data = json.loads(raw.splitlines()[0]) if "\n" in raw else json.loads(raw)

    current_year = datetime.now().year

    # ───── Location Risk Logic ───── #
    accident_location = claim_data["accident_location"].lower()
    high_risk_map = {
        "new york": 0.95,
        "chicago": 0.93,
        "los angeles": 0.91,
        "houston": 0.89,
        "detroit": 0.9,
        "miami": 0.85,
        "baltimore": 0.75,
        "new orleans": 0.72,
        "oakland": 0.7,
        "philadelphia": 0.68
    }
    location_risk_score = 0.6  # default low-risk
    for city, score in high_risk_map.items():
        if city in accident_location:
            location_risk_score = score
            break

    # ───── Dynamic Incident Time ───── #
    incident_time_str = claim_data.get("incident_time", "08:00 AM")
    try:
        incident_time_hour = datetime.strptime(incident_time_str, "%I:%M %p").hour
    except:
        incident_time_hour = 8  # fallback

    # ───── Feature Engineering ───── #
    damage = claim_data["estimated_damage_cost"]
    claim = claim_data["claim_amount_requested"]
    claim_to_damage_ratio = round(claim / damage, 2) if damage else 0.0

    previous_claims = claim_data.get("previous_claims_count", 0)
    vehicle_year = int(claim_data.get("vehicle_year", current_year))
    policy_days = (datetime.strptime(claim_data["date_of_loss"], "%Y-%m-%d") -
                   datetime.strptime(claim_data["policy_start_date"], "%Y-%m-%d")).days

    features_values = [
        claim_to_damage_ratio,
        current_year - vehicle_year,
        previous_claims,
        policy_days,
        location_risk_score,
        incident_time_hour
    ]

    dmatrix = xgb.DMatrix(np.array([features_values]), feature_names=features)
    fraud_prob = model.predict(dmatrix)[0]
    fraud_class = int(fraud_prob > 0.5)
    shap_values = explainer(dmatrix)

    shap_score_map = dict(zip(features, [round(float(val) * 10, 2) for val in shap_values.values[0]]))
    top_indices = np.argsort(np.abs(shap_values.values[0]))[::-1][:3]
    fraud_explanation = " | ".join([
        get_dynamic_explanation(features[i], shap_values.values[0][i]) for i in top_indices
    ])

    fraud_prediction = "Fraud" if fraud_class else "Not Fraud"
    fraud_score = round(float(fraud_prob) * 10, 1)

    claim_status = determine_decision(
        prediction=fraud_prediction,
        score=fraud_score,
        ratio=claim_to_damage_ratio,
        risk=location_risk_score,
        prev_claims=previous_claims
    )

    result = {
        "claim_id": claim_id,
        "fraud_prediction": fraud_prediction,
        "fraud_score": fraud_score,
        "fraud_explanation": fraud_explanation,
        "shap_values": shap_score_map,
        "claim_status": claim_status
    }

    output_filename = f"fraud-claim-{claim_id}__{get_timestamp_str()}.json"
    output_key = f"{output_prefix}{output_filename}"
    s3.put_object(Bucket=bucket, Key=output_key, Body=json.dumps(result), ContentType="application/json")

    print(f"✅ Saved result to s3://{bucket}/{output_key}")
    return result

# ───────── Flask Routes ───────── #
@app.route("/ping", methods=["GET"])
def ping():
    return jsonify(status="ok"), 200

@app.route("/invocations", methods=["POST"])
def invoke():
    try:
        payload = request.get_json()
        claim_id = payload.get("claim_id")
        if not claim_id:
            return jsonify(error="Missing claim_id"), 400

        result = process_claim(claim_id)
        return jsonify(result), 200

    except Exception as e:
        return jsonify(error=str(e), traceback=traceback.format_exc()), 500

# ───────── ECS Automation ───────── #
if __name__ == "__main__":
    run_trigger = os.environ.get("RUN_ECS_TRIGGER", "false").lower() == "true"
    claim_id = os.environ.get("CLAIM_ID")

    if run_trigger:
        if not claim_id:
            print("❌ CLAIM_ID not found in environment variables.")
            exit(1)

        print(f"🚀 Triggering fraud prediction for claim: {claim_id}")
        try:
            result = process_claim(claim_id)
            print("✅ Prediction result:", result)
        except Exception as e:
            print("❌ Prediction failed:", str(e))
        exit(0)
    else:
        app.run(host="0.0.0.0", port=8080)
