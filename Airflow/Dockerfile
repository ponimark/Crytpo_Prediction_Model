# Start with the official Airflow image
FROM apache/airflow:2.8.1-python3.10

# Switch to root to allow system updates if needed
USER root

# (Optional) Install system dependencies
# RUN apt-get update && apt-get install -y git

# Switch back to airflow user for pip installs
USER airflow

# Copy the requirements file from your local 'airflow' folder into the container
COPY requirements.txt /requirements.txt

# Install all Python libraries (TensorFlow, SQLAlchemy, yfinance, etc.)
RUN pip install --no-cache-dir -r /requirements.txt

# Install PyTorch with CUDA 12.1 support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
