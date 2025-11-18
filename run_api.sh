#!/bin/bash

# Robinson Crusoe API Startup Script
# Starts the FastAPI server with production settings

set -e  # Exit on error

echo "========================================="
echo "Robinson Crusoe Adaptation Detection API"
echo "========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} Python 3 found"

# Check if virtual environment exists
if [ ! -d "venv" ] && [ ! -d "env" ]; then
    echo -e "${YELLOW}No virtual environment found. Creating one...${NC}"
    python3 -m venv venv
    echo -e "${GREEN}✓${NC} Virtual environment created"
fi

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d "env" ]; then
    source env/bin/activate
fi

echo -e "${GREEN}✓${NC} Virtual environment activated"

# Install/upgrade dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
echo -e "${GREEN}✓${NC} Dependencies installed"

# Check if model exists
if [ ! -f "models/final_model.keras" ] && [ ! -f "models/best_model.keras" ]; then
    echo -e "${YELLOW}Warning: Model file not found${NC}"
    echo "Expected location: models/final_model.keras or models/best_model.keras"
    echo "Please ensure you have trained the model first."
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo -e "${GREEN}✓${NC} Model file found"
fi

# Create cache directory if it doesn't exist
mkdir -p data/cache
echo -e "${GREEN}✓${NC} Cache directory ready"

# Load environment variables if .env exists
if [ -f ".env" ]; then
    echo -e "${GREEN}✓${NC} Loading environment variables from .env"
    export $(cat .env | grep -v '^#' | xargs)
else
    echo -e "${YELLOW}No .env file found, using defaults${NC}"
fi

# Set defaults
export API_HOST=${API_HOST:-0.0.0.0}
export API_PORT=${API_PORT:-8000}
export LOG_LEVEL=${LOG_LEVEL:-info}

echo ""
echo "========================================="
echo "Starting API Server..."
echo "========================================="
echo "Host: $API_HOST"
echo "Port: $API_PORT"
echo "Log Level: $LOG_LEVEL"
echo ""
echo "API will be available at:"
echo -e "${GREEN}http://localhost:$API_PORT${NC}"
echo ""
echo "Documentation available at:"
echo -e "${GREEN}http://localhost:$API_PORT/docs${NC}"
echo ""
echo "Press Ctrl+C to stop the server"
echo "========================================="
echo ""

# Start the server
uvicorn api.main:app \
    --host "$API_HOST" \
    --port "$API_PORT" \
    --log-level "$LOG_LEVEL" \
    --reload
