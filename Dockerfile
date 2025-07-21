# Use a lightweight base Python image
FROM python:3.8-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Create SageMaker-required folders
WORKDIR /opt/ml
RUN mkdir -p /opt/ml/model /opt/ml/code

# Copy dependency file and install
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ✅ Copy your inference script and model file into the container
COPY inference.py /opt/ml/code/
COPY xgboost-model.json /opt/ml/model/

# ✅ Set SageMaker environment variables
ENV SAGEMAKER_PROGRAM=inference.py
ENV PYTHONPATH=/opt/ml/code

# ✅ Additional SageMaker expected variables
ENV SM_NUM_GPUS=0
ENV SM_MODEL_DIR=/opt/ml/model
ENV SM_OUTPUT_DATA_DIR=/opt/ml/output
ENV SM_INPUT_DATA_DIR=/opt/ml/input
ENV SM_CHANNEL_TRAINING=/opt/ml/input/data/training

# ✅ Start inference script directly (works locally & in SageMaker)
ENTRYPOINT ["python", "/opt/ml/code/inference.py"]
lets