# deploy_sagemaker.py

from sagemaker import Model
import sagemaker
import os
import time

# --- Config ---
aws_account_id = os.environ.get("AWS_ACCOUNT_ID", "461512246753")
aws_region = os.environ.get("AWS_REGION", "us-east-1")
ecr_repo = os.environ.get("ECR_REPOSITORY", "aicp-shap-sagemaker-container")
role = os.environ.get("SAGEMAKER_ROLE", "arn:aws:iam::461512246753:role/service-role/AmazonSageMakerServiceCatalogProductsUseRole")

# Unique endpoint name
timestamp = int(time.time())
endpoint_name = f"aicp-fraud-endpoint-{timestamp}"

# Image URI
image_uri = f"{aws_account_id}.dkr.ecr.{aws_region}.amazonaws.com/{ecr_repo}:latest"
print(f"📦 Using image: {image_uri}")

# SageMaker session
sagemaker_session = sagemaker.Session()

# Create model
model = Model(
    image_uri=image_uri,
    role=role,
    sagemaker_session=sagemaker_session
)

# Deploy
try:
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type="ml.m5.large",
        endpoint_name=endpoint_name
    )
    print(f"✅ SageMaker endpoint deployed: {endpoint_name}")
except Exception as e:
    print(f"❌ Deployment failed: {str(e)}")

print(f"🧪 Invoke this endpoint via: {endpoint_name}")
