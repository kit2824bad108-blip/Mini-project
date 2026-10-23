# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Bind to all interfaces so the container is reachable from outside
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 \
    PORT=8000

WORKDIR /app

# Build tools are only needed if a dependency has to compile from source
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

# Install runtime dependencies first so code edits don't bust the cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY . .

# Trained artifacts are git-ignored, so create the directory and train a small
# model at build time. Users can also mount their own models/ over this.
RUN mkdir -p models \
    && python train_model.py --n-samples 3000 --models xgboost --output models/best_xgboost.pkl

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status == 200 else 1)"

CMD ["python", "app.py"]
