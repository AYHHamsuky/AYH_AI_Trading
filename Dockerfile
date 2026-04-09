FROM python:3.11-slim

# System deps for TA-Lib
RUN apt-get update && apt-get install -y \
    build-essential wget curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
# Remove MetaTrader5 line on Linux (MT5 is Windows-only; use REST API bridge)
RUN sed -i '/MetaTrader5/d' requirements.txt && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Create persistent dirs
RUN mkdir -p saved_models logs data_cache

# Environment (override via docker run -e or .env file)
ENV MT5_LOGIN=""
ENV MT5_PASSWORD=""
ENV MT5_SERVER=""
ENV TELEGRAM_BOT_TOKEN=""
ENV TELEGRAM_CHAT_ID=""

# Default: run backtest to validate
CMD ["python", "main.py", "--mode", "backtest", "--symbols", "EURUSD", "XAUUSD", "--tf", "H1"]
