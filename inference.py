import os
import json
import uuid
import boto3
import xgboost as xgb
import shap
import numpy as np
import logging

# ✅ Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model
def model_fn(model_dir):
    logger.info("🔁 Loading model from: %s", model_dir)
    model = xgb.Booster()
    model.load_model(os.path.join(model_dir, "xgboost-model.json"))
    logger.info("✅ Model loaded successfully.")
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

    logger.info("📥 Parsing content type: %s", request_content_type)

    if request_content_type == "application/json":
        data = json.loads(request_body)
        values = [float(data[f]) for f in expected_features]
        return np.array([values]), data

    elif request_content_type == "text/csv":
        values = [float(x) for x in request_body.strip().split(",")]
        return np.array([values]), {}

    raise ValueError(f"Unsupported content type: {request_content_type}")

# Predict and explain
def predict_fn(input_data, model):
    if input_data.shape[1] != 5:
        raise ValueError(f"Expected 5 features, got {input_data.shape[1]}")

    logger.info("🔮 Running fraud prediction + SHAP")

    feature_names = [
        "claim_amount",
        "estimated_damage",
        "vehicle_year",
        "days_since_policy_start",
        "location_risk_score"
    ]

    dmatrix = xgb.DMatrix(input_data, feature_names=feature_names)
    score = float(model.predict(dmatrix)[0])
    prediction = "FRAUD" if score > 0.5 else "LEGIT"

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(dmatrix)
        shap_values = shap_values[0]

        sorted_indices = np.argsort(np.abs(shap_values))[::-1]
        all_features = [
            {
                "feature": feature_names[i],
                "impact": round(shap_values[i], 1)
            }
            for i in sorted_indices
        ]

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

        explanation = (
            "The claim is suspicious because " + ", ".join(plain_reasons) + "."
            if prediction == "FRAUD"
            else "The claim appears legitimate based on current features."
        )

        return {
            "fraud_score": round(score, 4),
            "fraud_prediction": prediction,
            "fraud_explanation": explanation,
            "shap_features": [f"{f['feature']} ({f['impact']:+.1f})" for f in all_features]
        }

    except Exception as e:
        logger.error("❌ SHAP explanation failed: %s", str(e))
        return {
            "fraud_score": round(score, 4),
            "fraud_prediction": prediction,
            "fraud_explanation": "SHAP explanation not available due to internal error.",
            "error": str(e)
        }

# ========= Entry point for Batch ECS or Local run ========= #
if __name__ == "__main__":
    logger.info("🚀 Starting batch fraud predictor...")

    s3 = boto3.client("s3")
    bucket = "aicp-claims-data"
    prefix = "processed/DQ-validated-claims-data/"

    try:
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        files = sorted(
            [obj['Key'] for obj in response.get("Contents", []) if obj['Key'].endswith(".json")],
            key=lambda x: s3.head_object(Bucket=bucket, Key=x)["LastModified"],
            reverse=True
        )

        if not files:
            logger.warning("⚠️ No DQ-validated claims found.")
            exit(0)

        latest_key = files[0]
        logger.info("📄 Processing latest file: %s", latest_key)

        obj = s3.get_object(Bucket=bucket, Key=latest_key)
        content = obj["Body"].read().decode("utf-8")
        input_data, original_input = input_fn(content, "application/json")

        model = model_fn("/opt/ml/model")
        prediction = predict_fn(input_data, model)

        claim_id = original_input.get("claim_id", f"batch-{uuid.uuid4()}")
        prediction["claim_id"] = claim_id

        output_key = f"processed/fraud-predicted-claims-data/fraud_prediction_{claim_id}.json"
        s3.put_object(Bucket=bucket, Key=output_key, Body=json.dumps(prediction))
        logger.info("✅ Prediction saved: %s", output_key)

    except Exception as e:
        logger.error("🔥 Failed to run inference: %s", str(e))
        raise
