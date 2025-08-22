# Dockerfile for Real-Time Alpaca Trading Bot
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY realtime_alpaca_trader.py .
COPY alpaca_config.json .
COPY alpaca_integration.py .

# Copy any ML models or data files if needed
COPY *.py .
COPY *.json .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TZ=America/New_York

# Expose port for monitoring (optional)
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health')" || exit 1

# Run the trading bot
CMD ["python", "realtime_alpaca_trader.py"]
