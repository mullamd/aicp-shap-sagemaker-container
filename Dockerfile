# Use Amazon SageMaker Python base image (Python 3.8)
FROM python:3.8-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create working directory
WORKDIR /opt/program

# Install required libraries
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy inference code
COPY inference.py .

# Define the entry point for inference
ENV SAGEMAKER_PROGRAM=inference.py

# (Optional) Expose port if debugging locally
EXPOSE 8080
