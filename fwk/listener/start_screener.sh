#!/bin/bash

# Candle Data Screener Startup Script

set -e

cd "$(dirname "$0")"

export CONFIG_PATH="${CONFIG_PATH:-config/downloader.yaml}"
export BASE_FOLDER="${BASE_FOLDER:-./data/candles}"

echo "=========================================="
echo "  Candle Data Screener"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  CONFIG_PATH: $CONFIG_PATH"
echo "  BASE_FOLDER: $BASE_FOLDER"
echo ""

if [ ! -f "$CONFIG_PATH" ]; then
    echo "ERROR: Config file not found: $CONFIG_PATH"
    echo ""
    echo "Please ensure the config file exists or set CONFIG_PATH environment variable."
    exit 1
fi

if [ ! -d "$BASE_FOLDER" ]; then
    echo "WARNING: Data folder not found: $BASE_FOLDER"
    echo "  The screener will start but may return empty results."
    echo ""
fi

echo "Starting screener on port 8004..."
echo "Dashboard: http://localhost:8004/screener/dashboard_screener"
echo "API Docs: http://localhost:8004/docs"
echo "Health: http://localhost:8004/screener/health"
echo ""

/home/brian/open_singularity/fwk/singularity_listener/.venv/bin/python -m uvicorn app.screener_app:app --host 0.0.0.0 --port 8002 --reload
