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
import time
import requests
from threading import Thread

# ────────── Flask App ────────── #
app = Flask(__name__)
s3 = boto3.client("s3")

bucket = "aicp-claims-data"
input_prefix = "processed/DQ-validated-claims-data/"
output_prefix = "processed/fraud-predicted-claims-data/"

# ────────── Load Model ────────── #
model_path = os.path.join(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"), "xgboost-model.json")
model = xgb.Booster()
model.load_model(model_path)
explainer = shap.Explainer(model)

# ────────── Feature Definitions ────────── #
features = [
    "claim_to_damage_ratio",
    "vehicle_age",
    "previous_claims_count",
    "days_since_policy_start",
    "location_risk_score",
    "incident_time_hour"
]

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

# ────────── Updated Claim File Lookup ────────── #
def get_latest_claim_file(claim_id):
    prefix = f"{input_prefix}clean-claim-{claim_id}__"
    try:
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        if "Contents" in response:
            sorted_objects = sorted(response["Contents"], key=lambda x: x["LastModified"], reverse=True)
            return sorted_objects[0]["Key"]
    except Exception as e:
        print(f"❌ S3 list_objects_v2 failed: {e}")
    return None

def get_timestamp_str():
    return datetime.now(timezone("US/Eastern")).strftime("%B-%d-%Y_%I-%M-%p")

# ────────── Routes ────────── #
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

        key = get_latest_claim_file(claim_id)
        if not key:
            return jsonify(error=f"Claim file not found for {claim_id}"), 404

        raw = s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8").strip()
        claim_data = json.loads(raw.splitlines()[0]) if "\n" in raw else json.loads(raw)

        current_year = datetime.now().year
        features_values = [
            round(claim_data["claim_amount_requested"] / claim_data["estimated_damage_cost"], 2),
            current_year - int(claim_data["vehicle_year"]),
            claim_data.get("previous_claims_count", 0),
            (datetime.strptime(claim_data["date_of_loss"], "%Y-%m-%d") -
             datetime.strptime(claim_data["policy_start_date"], "%Y-%m-%d")).days,
            0.95 if "chicago" in claim_data["accident_location"].lower() else 0.6,
            3  # default 3 AM
        ]

        dmatrix = xgb.DMatrix(np.array([features_values]), feature_names=features)
        fraud_prob = model.predict(dmatrix)[0]
        fraud_class = int(fraud_prob > 0.5)
        shap_values = explainer(dmatrix)

        shap_score_map = dict(zip(features, [round(float(val) * 10, 2) for val in shap_values.values[0]]))
        top_indices = np.argsort(np.abs(shap_values.values[0]))[::-1][:3]
        explanations = [get_dynamic_explanation(features[i], shap_values.values[0][i]) for i in top_indices]
        fraud_explanation = " | ".join(explanations)

        result = {
            "claim_id": claim_id,
            "fraud_prediction": "Fraud" if fraud_class else "Not Fraud",
            "fraud_score": round(float(fraud_prob) * 10, 1),
            "fraud_explanation": fraud_explanation,
            "shap_values": shap_score_map
        }

        output_filename = f"fraud-claim-{claim_id}__{get_timestamp_str()}.json"
        output_key = f"{output_prefix}{output_filename}"

        s3.put_object(
            Bucket=bucket,
            Key=output_key,
            Body=json.dumps(result),
            ContentType="application/json"
        )

        print(f"✅ Saved result to s3://{bucket}/{output_key}")
        return jsonify(result), 200

    except Exception as e:
        return jsonify(error=str(e), traceback=traceback.format_exc()), 500

# ────────── ECS Trigger (Optional) ────────── #
def wait_for_server(url="http://localhost:8080/ping", timeout=10):
    for _ in range(timeout):
        try:
            r = requests.get(url)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False

if __name__ == "__main__" and os.getenv("RUN_ECS_TRIGGER", "false").lower() == "true":
    # Start Flask server in background
    Thread(target=lambda: app.run(host="0.0.0.0", port=8080)).start()

    # Wait until server is ready
    if not wait_for_server():
        print("❌ Server did not start within timeout.")
        os._exit(1)

    claim_id = os.getenv("CLAIM_ID")
    if not claim_id:
        print("❌ CLAIM_ID environment variable not set.")
        os._exit(1)

    try:
        print(f"🚀 Triggering fraud prediction for {claim_id}")
        response = requests.post("http://localhost:8080/invocations", json={"claim_id": claim_id})
        print("✅ Prediction result:", response.json())
    except Exception as e:
        print("❌ Internal invocation failed:", str(e))

    # Final exit for ECS
    os._exit(0)
