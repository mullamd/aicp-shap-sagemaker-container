import boto3
import json
import shap
import xgboost as xgb
import numpy as np
from datetime import datetime
from dateutil import parser
from pytz import timezone

# --- CONFIG ---
s3_bucket = "aicp-claims-data"
input_prefix = "processed/DQ-validated-claims-data/"
output_prefix = "processed/fraud-predicted-claims-data/"

# --- Load XGBoost Booster model ---
model = xgb.Booster()
model.load_model("xgboost-model.json")

# --- SHAP Explainer ---
explainer = shap.Explainer(model)

# --- Model Feature Names (must match training exactly) ---
features = [
    "claim_to_damage_ratio",
    "vehicle_age",
    "previous_claims_count",
    "days_since_policy_start",
    "location_risk_score",
    "incident_time_hour"
]

# --- S3 Client ---
s3 = boto3.client("s3")

# --- Helper: convert time string to hour ---
def time_to_hour(t_str):
    try:
        return datetime.strptime(t_str, "%I:%M %p").hour
    except:
        return 8

# --- Dynamic explanations based on SHAP sign ---
def get_dynamic_explanation(feature, shap_value):
    if feature == "location_risk_score":
        if shap_value > 0:
            return "The accident occurred in a location that is considered high-risk for fraud."
        else:
            return "The accident occurred in a location that is not considered high-risk for fraud."
    elif feature == "previous_claims_count":
        if shap_value > 0:
            return "This claimant has filed previous claims, which signals higher risk."
        else:
            return "This claimant has never filed a previous claim, which signals lower risk."
    elif feature == "claim_to_damage_ratio":
        if shap_value > 0:
            return "The claimed amount is significantly higher than the estimated damage, which is suspicious."
        else:
            return "The claimed amount closely matches the estimated damage, indicating a reasonable request."
    elif feature == "vehicle_age":
        if shap_value > 0:
            return "The vehicle's age appears unusual and may indicate potential risk."
        else:
            return "The vehicle's age appears typical and does not raise suspicion."
    elif feature == "days_since_policy_start":
        if shap_value > 0:
            return "The claim was made soon after the policy started, which can be risky."
        else:
            return "The policy had been active for a sufficient period before the claim was made."
    elif feature == "incident_time_hour":
        if shap_value > 0:
            return "The time of the incident falls outside normal hours, which may be suspicious."
        else:
            return "The time of the incident falls within normal hours and does not appear unusual."
    else:
        return f"Key factor: {feature}"

# --- Generate timestamped string ---
def get_timestamp_str():
    local_now = datetime.now(timezone("US/Eastern"))
    return local_now.strftime("%B-%d-%Y_%I-%M-%p")

# --- Get input claim files from S3 ---
response = s3.list_objects_v2(Bucket=s3_bucket, Prefix=input_prefix)
claim_files = [
    obj["Key"] for obj in response.get("Contents", [])
    if obj["Key"].endswith(".json") and "clean-claim-" in obj["Key"]
]

results = []

for s3_key in claim_files:
    try:
        response = s3.get_object(Bucket=s3_bucket, Key=s3_key)
        body = response["Body"].read().decode("utf-8").strip()
        try:
            claim_data = json.loads(body)
        except json.JSONDecodeError:
            claim_data = json.loads(body.splitlines()[0])

        # Defensive field checks
        if "vehicle_year" not in claim_data or "claim_id" not in claim_data:
            print(f"⚠️ Skipping file {s3_key} due to missing keys.")
            continue

        # Feature engineering
        current_year = datetime.now().year
        claim_to_damage_ratio = claim_data["claim_amount_requested"] / claim_data["estimated_damage_cost"]
        vehicle_age = current_year - int(claim_data["vehicle_year"])
        previous_claims_count = claim_data.get("previous_claims_count", 0)
        date_of_loss = parser.parse(claim_data["date_of_loss"])
        policy_start = parser.parse(claim_data["policy_start_date"])
        days_since_policy_start = (date_of_loss - policy_start).days

        location = claim_data["accident_location"].lower()
        location_risk_map = {
            "new york": 0.9,
            "los angeles": 0.85,
            "chicago": 0.95,
            "houston": 0.75
        }
        location_risk_score = 0.6  # default risk
        for city, score in location_risk_map.items():
            if city in location:
                location_risk_score = score
                break

        incident_time_str = datetime.now().strftime("%I:%M %p")
        incident_time_hour = time_to_hour(incident_time_str)

        input_values = [
            round(claim_to_damage_ratio, 2),
            vehicle_age,
            previous_claims_count,
            days_since_policy_start,
            location_risk_score,
            incident_time_hour
        ]

        # Create DMatrix with named features
        X_input = xgb.DMatrix(
            data=np.array([input_values]),
            feature_names=features
        )

        # Model prediction
        fraud_prob = model.predict(X_input)[0]
        fraud_class = int(fraud_prob > 0.5)
        shap_values = explainer(X_input)

        # Top 3 SHAP features and dynamic explanations
        top_indices = np.argsort(np.abs(shap_values.values[0]))[::-1][:3]
        explanations = [
            get_dynamic_explanation(features[i], shap_values.values[0][i])
            for i in top_indices
        ]
        fraud_explanation = " | ".join(explanations)

        shap_score_map = dict(zip(
            features,
            [round(float(val) * 10, 2) for val in shap_values.values[0]]
        ))

        # Build output record
        result = {
            "claim_id": claim_data["claim_id"],
            "fraud_score": round(float(fraud_prob) * 10, 1),
            "fraud_prediction": "Fraud" if fraud_class == 1 else "Not Fraud",
            "fraud_explanation": fraud_explanation,
            "shap_values": shap_score_map
        }

        # Save result to S3
        output_filename = f"fraud-claim-{result['claim_id']}__{get_timestamp_str()}.json"
        output_key = f"{output_prefix}{output_filename}"
        s3.put_object(
            Bucket=s3_bucket,
            Key=output_key,
            Body=json.dumps(result),
            ContentType="application/json"
        )

        results.append(result)
        print(f"✅ Processed {result['claim_id']}")

    except Exception as e:
        print(f"❌ Error processing file {s3_key}: {str(e)}")
        continue

# Optionally save summary locally
with open("batch_fraud_predictions.json", "w") as f:
    json.dump(results, f, indent=2)
