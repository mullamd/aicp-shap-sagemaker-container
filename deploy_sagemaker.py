# 🔁 Triggering GitHub Actions redeploy
from sagemaker import Model
import sagemaker
import os

# ✅ Use environment variables from GitHub Secrets
image_uri = f"{os.environ['AWS_ACCOUNT_ID']}.dkr.ecr.{os.environ['AWS_REGION']}.amazonaws.com/{os.environ['ECR_REPOSITORY']}:latest"
role = os.environ['SAGEMAKER_ROLE']
endpoint_name = "aicp-fraud-endpoint-1752473908"

# ✅ Create SageMaker model and deploy
model = Model(
    image_uri=image_uri,
    role=role,
    sagemaker_session=sagemaker.Session()
)

predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",
    endpoint_name=endpoint_name,
    update_endpoint=True
)

print(f"✅ SageMaker endpoint redeployed: {endpoint_name}")
