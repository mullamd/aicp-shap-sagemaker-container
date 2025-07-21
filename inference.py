import json
import pandas as pd
import boto3
import os
import joblib
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Expected feature names from the cleaned S3 claim data
expected_features = [
    "claim_amount_requested",
    "estimated_damage_cost",
    "claim_to_damage_ratio",
    "days_since_policy_start",
    "vehicle_age",
    "location_risk_score",
    "previous_claims_count"
]

# Load model artifacts
model_path = "/opt/ml/model/fraud_model.joblib"
explainer_path = "/opt/ml/model/shap_explainer.joblib"
model = joblib.load(model_path)
explainer = joblib.load(explainer_path)

def input_fn(input_data, content_type):
    data = json.loads(input_data)
    logger.info(f"📥 Parsing content type: {content_type}")
    values = [float(data[f]) for f in expected_features]
    df = pd.DataFrame([values], columns=expected_features)
    return df, data

def predict_fn(input_data, model):
    return model.predict_proba(input_data)

def explain_fn(input_data):
    shap_values = explainer.shap_values(input_data)
    return shap_values

def format_explanation(shap_values, input_data):
    impact = shap_values[1][0]
    features = input_data.columns
    explanation = []
    for feature, val in zip(features, impact):
        sign = "+" if val >= 0 else "-"
        explanation.append(f"{feature} ({sign}{abs(round(val, 2))})")
    return explanation

def save_prediction_to_s3(result_dict, claim_id):
    bucket = os.environ.get("S3_BUCKET", "aicp-claims-data")
    prefix = "processed/fraud-predicted-claims-data"
    filename = f"fraud_prediction_ecs-test-{claim_id}.json"
    key = f"{prefix}/{filename}"

    s3 = boto3.client("s3")
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(result_dict),
        ContentType="application/json"
    )
    logger.info(f"✅ Saved prediction to s3://{bucket}/{key}")

if __name__ == "__main__":
    claim_id = os.environ.get("CLAIM_ID")
    if not claim_id:
        raise ValueError("CLAIM_ID environment variable not found.")

    s3 = boto3.client("s3")
    bucket = "aicp-claims-data"
    prefix = "processed/DQ-validated-claims-data/"
    logger.info("🚀 Starting batch fraud predictor...")

    # Search for the latest matching cleaned claim file
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    matched_key = None
    for obj in sorted(response.get("Contents", []), key=lambda x: x["LastModified"], reverse=True):
        key = obj["Key"]
        if claim_id in key and key.endswith(".json"):
            matched_key = key
            break

    if not matched_key:
        logger.error(f"❌ Failed to load enriched claim JSON from S3: No matching claim file found in S3 for {claim_id}")
        raise FileNotFoundError(f"No matching claim file found in S3 for {claim_id}")

    logger.info(f"📄 Processing latest file: {matched_key}")

    response = s3.get_object(Bucket=bucket, Key=matched_key)
    content = response['Body'].read().decode('utf-8')

    try:
        input_data, original_input = input_fn(content, "application/json")
        proba = predict_fn(input_data, model)[0][1]
        prediction = "FRAUD" if proba >= 0.5 else "LEGIT"
        explanation = format_explanation(explain_fn(input_data), input_data)

        result = {
            "claim_id": claim_id,
            "fraud_score": round(proba, 4),
            "fraud_prediction": prediction,
            "fraud_explanation": (
                "The claim appears suspicious based on SHAP impact scores." if prediction == "FRAUD"
                else "The claim appears legitimate based on current features."
            ),
            "shap_features": explanation
        }

        save_prediction_to_s3(result, claim_id)

    except Exception as e:
        logger.exception(f"🔥 Failed to run inference: {e}")
        raise
