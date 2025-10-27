# Use slim Python base
FROM python:3.10-slim

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system deps for OpenCV and git
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create workdir
WORKDIR /app

# Copy dependency list and install first (for better caching)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Fetch U-2-Net repo for model definition
RUN git clone https://github.com/xuebinqin/U-2-Net.git /opt/U-2-Net
ENV PYTHONPATH="/opt/U-2-Net:${PYTHONPATH}"

# Copy application code
COPY app /app/app
COPY weights /app/weights

# Environment for weights
ENV WEIGHTS_PATH=/app/weights/u2net_clothing.pth
ENV WEIGHTS_URL=

# Expose port and run server
EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--log-level", "info"]
