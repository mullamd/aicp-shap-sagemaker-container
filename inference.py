import os
import json
import uuid
import boto3
import xgboost as xgb
import shap
import numpy as np
import logging

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load XGBoost model
def load_model():
    model = xgb.Booster()
    model.load_model("/opt/ml/model/xgboost-model.json")
    logger.info("✅ Model loaded.")
    return model

# Perform prediction
def predict(model, input_features, claim_id="test-claim-id"):
    feature_names = [
        "claim_amount",
        "estimated_damage",
        "vehicle_year",
        "days_since_policy_start",
        "location_risk_score"
    ]
    dmatrix = xgb.DMatrix([input_features], feature_names=feature_names)
    score = float(model.predict(dmatrix)[0])
    prediction = "FRAUD" if score > 0.5 else "LEGIT"

    # SHAP explanation
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
        "claim_id": claim_id,
        "fraud_score": round(score, 4),
        "fraud_prediction": prediction,
        "fraud_explanation": explanation,
        "shap_features": [f"{f['feature']} ({f['impact']:+.1f})" for f in all_features]
    }

# Main execution
if __name__ == "__main__":
    logger.info("🚀 Starting ECS batch prediction...")

    # Sample input (you can replace this with reading from S3 or ENV)
    sample_input = [7000, 3000, 2010, 30, 8.5]
    claim_id = f"ecs-test-{uuid.uuid4()}"

    try:
        model = load_model()
        result = predict(model, sample_input, claim_id)

        # Upload to S3
        s3 = boto3.client("s3")
        output_key = f"processed/fraud-predicted-claims-data/fraud_prediction_{claim_id}.json"
        s3.put_object(
            Bucket="aicp-claims-data",
            Key=output_key,
            Body=json.dumps(result)
        )
        logger.info(f"✅ Saved prediction to S3: {output_key}")

    except Exception as e:
        logger.error(f"❌ Error during prediction: {e}")
        raise
