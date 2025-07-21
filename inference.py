import os
import json
import uuid
import boto3
import xgboost as xgb
import shap
import numpy as np
import logging

# ✅ Configure logging for CloudWatch visibility
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ Load model from /opt/ml/model
def model_fn(model_dir):
    model = xgb.Booster()
    model_path = os.path.join(model_dir, "xgboost-model.json")
    logger.info(f"🔁 Loading XGBoost model from: {model_path}")
    model.load_model(model_path)
    return model

# ✅ Load input JSON from S3
def input_fn(s3_bucket, s3_prefix, claim_id):
    s3 = boto3.client("s3")

    logger.info(f"📥 Looking for claim file in {s3_prefix} with ID: {claim_id}")
    response = s3.list_objects_v2(Bucket=s3_bucket, Prefix=s3_prefix)

    matched_key = None
    for obj in response.get("Contents", []):
        key = obj["Key"]
        if claim_id in key and key.endswith(".json"):
            matched_key = key
            break

    if not matched_key:
        raise FileNotFoundError(f"No matching claim file found in S3 for {claim_id}")

    obj = s3.get_object(Bucket=s3_bucket, Key=matched_key)
    body = obj["Body"].read().decode("utf-8")
    data = json.loads(body)
    logger.info(f"✅ Loaded claim data from {matched_key}")

    # Extract and validate required features
    required_features = [
        "claim_amount_requested",
        "estimated_damage_cost",
        "vehicle_year",
        "days_since_policy_start",
        "location_risk_score"
    ]

    feature_values = [float(data[f]) for f in required_features]
    input_array = np.array([feature_values])
    return input_array, data

# ✅ Predict and explain
def predict_fn(input_data, model):
    feature_names = [
        "claim_amount_requested",
        "estimated_damage_cost",
        "vehicle_year",
        "days_since_policy_start",
        "location_risk_score"
    ]

    dmatrix = xgb.DMatrix(input_data, feature_names=feature_names)
    score = float(model.predict(dmatrix)[0])
    prediction = "FRAUD" if score > 0.5 else "LEGIT"

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(dmatrix)[0]

    sorted_indices = np.argsort(np.abs(shap_values))[::-1]
    all_features = [
        {
            "feature": feature_names[i],
            "impact": round(shap_values[i], 1)
        }
        for i in sorted_indices
    ]

    reason_map = {
        "claim_amount_requested": {
            "pos": "the claim has a high amount",
            "neg": "the claim amount is reasonable"
        },
        "estimated_damage_cost": {
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

# ✅ Output function
def output_fn(result, enriched_data, claim_id):
    enriched_data.update(result)
    enriched_data["claim_id"] = claim_id

    output_key = f"processed/fraud-predicted-claims-data/fraud_prediction_{claim_id}.json"
    s3 = boto3.client("s3")
    s3.put_object(
        Bucket="aicp-claims-data",
        Key=output_key,
        Body=json.dumps(enriched_data)
    )
    logger.info(f"✅ Saved final fraud prediction to: {output_key}")

# ✅ Main ECS entry point
if __name__ == "__main__":
    logger.info("🚀 Starting batch fraud predictor...")

    # Load environment and model
    claim_id = os.environ.get("CLAIM_ID", f"unknown-{uuid.uuid4()}")
    model = model_fn("/opt/ml/model")

    try:
        input_data, enriched_data = input_fn(
            s3_bucket="aicp-claims-data",
            s3_prefix="processed/DQ-validated-claims-data/",
            claim_id=claim_id
        )

        result = predict_fn(input_data, model)
        output_fn(result, enriched_data, claim_id)

    except Exception as e:
        logger.error(f"🔥 Failed to run inference: {str(e)}")
        raise
