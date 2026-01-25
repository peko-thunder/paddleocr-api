FROM python:3.12-slim

WORKDIR /workspace

# Install system dependencies for PaddleOCR and OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo \
    ccache \
    libxcb1 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt /workspace/.devcontainer/
RUN pip install \
    --upgrade pip \
    --no-cache-dir \
    -r /workspace/.devcontainer/requirements.txt

# Pre-download PaddleOCR models during build
# This ensures models are available immediately at runtime
COPY preload_models.py /workspace/.devcontainer/
RUN python3 /workspace/.devcontainer/preload_models.py

# Expose FastAPI port
EXPOSE 8000

# Run any command to initialize the container
CMD ["bash"]
