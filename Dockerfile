FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (cached layer)
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source
COPY backend/ /app/backend/

# NOTE: Frontend is deployed to Vercel separately — not bundled here

# Copy data files
# NOTE: Large files (schedules.json, Train_details_22122017.csv) must exist
# in your project folder when building — they are bundled into the image.
COPY data/stations.json              /app/data/stations.json
COPY data/train_delay_data_rich.csv       /app/data/train_delay_data_rich.csv

# Copy rich training data if exists (generated locally)
# If not present, the app generates it on first startup
COPY data/Train_details_22122017.csv /app/data/Train_details_22122017.csv
COPY data/schedules.json             /app/data/schedules.json

# Create models cache directory
RUN mkdir -p /app/data/models

# Set environment
ENV RAILWAYS_DATA_DIR=/app/data
ENV PYTHONPATH=/app

EXPOSE 8000

# Render sets PORT automatically; fallback to 8000 for local Docker
CMD ["sh", "-c", "python -m uvicorn backend.app:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --timeout-keep-alive 120"]
