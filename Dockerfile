# ══════════════════════════════════════════════════════════════════════════════
# Intrinsic AI — Production Dockerfile
# ══════════════════════════════════════════════════════════════════════════════
FROM python:3.11-slim

# System deps for numpy/scipy/sklearn compilation
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Create directories for runtime data
RUN mkdir -p instance models logs

# Train model if not already present
RUN python train_model.py || echo "Model training skipped (models may already exist)"

EXPOSE 5000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/')" || exit 1

# Production: Gunicorn with 2 workers, 120s timeout (for yfinance + Mistral calls)
CMD ["gunicorn", \
     "--bind", "0.0.0.0:5000", \
     "--workers", "2", \
     "--threads", "4", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "app:app"]