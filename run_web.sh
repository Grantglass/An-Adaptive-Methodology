#!/bin/bash

# Robinson Crusoe Web Interface Startup Script
# Starts the Streamlit dashboard

set -e  # Exit on error

echo "========================================="
echo "Robinson Crusoe Web Interface"
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

# Check if API is running
API_URL=${API_URL:-http://localhost:8000}
if curl -s "$API_URL/health" > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} API is reachable at $API_URL"
else
    echo -e "${YELLOW}Warning: API is not reachable at $API_URL${NC}"
    echo "Please ensure the API is running:"
    echo "  ./run_api.sh"
    echo "  or"
    echo "  uvicorn api.main:app --reload"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if secrets file exists
if [ ! -f ".streamlit/secrets.toml" ]; then
    echo -e "${YELLOW}Creating .streamlit/secrets.toml from example...${NC}"
    mkdir -p .streamlit
    cp .streamlit/secrets.toml.example .streamlit/secrets.toml
    echo -e "${GREEN}✓${NC} Secrets file created"
    echo "You can edit .streamlit/secrets.toml to configure API URL"
fi

echo ""
echo "========================================="
echo "Starting Streamlit Web Interface..."
echo "========================================="
echo ""
echo "The interface will open in your browser at:"
echo -e "${GREEN}http://localhost:8501${NC}"
echo ""
echo "Press Ctrl+C to stop the server"
echo "========================================="
echo ""

# Start Streamlit
streamlit run app.py \
    --server.address localhost \
    --server.port 8501 \
    --browser.gatherUsageStats false
