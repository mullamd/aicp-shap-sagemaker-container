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

# Copy inference script to the code directory
COPY inference.py /opt/ml/code/

# Set required SageMaker environment variable
ENV SAGEMAKER_PROGRAM=inference.py
ENV PYTHONPATH=/opt/ml/code
