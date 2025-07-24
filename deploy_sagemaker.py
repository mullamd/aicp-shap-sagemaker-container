# 🔁 Deploy new SageMaker endpoint (no reuse of old config)
from sagemaker import Model
import sagemaker
import os
import time

# Read values from environment (GitHub secrets)
image_uri = f"{os.environ['AWS_ACCOUNT_ID']}.dkr.ecr.{os.environ['AWS_REGION']}.amazonaws.com/{os.environ['ECR_REPOSITORY']}:latest"
role = os.environ['SAGEMAKER_ROLE']

# Unique model and endpoint names using timestamp
timestamp = int(time.time())
model_name = f"aicp-fraud-model-{timestamp}"
endpoint_name = f"aicp-fraud-endpoint-{timestamp}"

# Create SageMaker model
model = Model(
    image_uri=image_uri,
    role=role,
    name=model_name,
    sagemaker_session=sagemaker.Session()
)

# Deploy new endpoint
predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",
    endpoint_name=endpoint_name
)

print(f"✅ New SageMaker endpoint deployed: {endpoint_name}")
