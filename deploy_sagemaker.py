# deploy_sagemaker.py

from sagemaker import Model
import sagemaker
import os

# --- Config from environment variables ---
image_uri = f"{os.environ['AWS_ACCOUNT_ID']}.dkr.ecr.{os.environ['AWS_REGION']}.amazonaws.com/{os.environ['ECR_REPOSITORY']}:latest"
role = os.environ['SAGEMAKER_ROLE']
endpoint_name = "aicp-fraud-endpoint-1752473908"  # ✅ Your existing endpoint

# --- SageMaker Session ---
sagemaker_session = sagemaker.Session()

# --- Create Model object ---
model = Model(
    image_uri=image_uri,
    role=role,
    sagemaker_session=sagemaker_session
)

# --- Redeploy (update existing endpoint) ---
predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",
    endpoint_name=endpoint_name,
    update_endpoint=True  # ✅ This forces redeployment on the same endpoint
)

print(f"✅ SageMaker endpoint updated: {endpoint_name}")
