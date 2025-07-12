# Use lightweight Python base image
FROM python:3.8-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Create SageMaker-required folders
WORKDIR /opt/ml
RUN mkdir -p /opt/ml/model /opt/ml/code

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ✅ Copy inference script to /opt/ml/code
COPY inference.py /opt/ml/code/

# ✅ SageMaker needs to know what script to run
ENV SAGEMAKER_PROGRAM=inference.py
ENV PYTHONPATH=/opt/ml/code

# ✅ Required: tell SageMaker this is the entrypoint
ENV SM_NUM_GPUS=0
ENV SM_MODEL_DIR=/opt/ml/model
ENV SM_OUTPUT_DATA_DIR=/opt/ml/output
ENV SM_INPUT_DATA_DIR=/opt/ml/input
ENV SM_CHANNEL_TRAINING=/opt/ml/input/data/training

# ✅ Start inference service
ENTRYPOINT ["python", "/opt/ml/code/inference.py"]
