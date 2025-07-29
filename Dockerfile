# ───── Base Image ───── #
FROM python:3.8-slim

# ───── Env Settings ───── #
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ───── Create Required Dirs ───── #
WORKDIR /opt/ml
RUN mkdir -p /opt/ml/model /opt/ml/code

# ───── Install Python Dependencies ───── #
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ───── Copy App Code and Model ───── #
COPY inference.py /opt/ml/code/
COPY xgboost-model.json /opt/ml/model/

# ───── Set SageMaker Environment Variables ───── #
ENV SAGEMAKER_PROGRAM=inference.py
ENV PYTHONPATH=/opt/ml/code

# ───── Optional SageMaker Directories ───── #
ENV SM_NUM_GPUS=0
ENV SM_MODEL_DIR=/opt/ml/model
ENV SM_OUTPUT_DATA_DIR=/opt/ml/output
ENV SM_INPUT_DATA_DIR=/opt/ml/input
ENV SM_CHANNEL_TRAINING=/opt/ml/input/data/training

# ───── Run App ───── #
ENTRYPOINT ["python", "/opt/ml/code/inference.py"]
