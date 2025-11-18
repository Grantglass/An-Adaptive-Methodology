#!/bin/bash

# Setup script for Robinson Crusoe Adaptation Detection project
# Installs dependencies and prepares the environment

set -e  # Exit on error

echo "========================================="
echo "Robinson Crusoe Adaptation Detection"
echo "Setup Script"
echo "========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check Python version
echo -e "${BLUE}Checking Python version...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed${NC}"
    echo "Please install Python 3.9 or higher"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo -e "${GREEN}✓${NC} Python $PYTHON_VERSION found"

# Check if Python version is 3.9+
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 9 ]); then
    echo -e "${RED}Error: Python 3.9 or higher is required${NC}"
    echo "Current version: $PYTHON_VERSION"
    exit 1
fi

# Create virtual environment
echo ""
echo -e "${BLUE}Creating virtual environment...${NC}"
if [ -d "venv" ]; then
    echo -e "${YELLOW}Virtual environment already exists${NC}"
    read -p "Recreate it? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf venv
        python3 -m venv venv
        echo -e "${GREEN}✓${NC} Virtual environment recreated"
    fi
else
    python3 -m venv venv
    echo -e "${GREEN}✓${NC} Virtual environment created"
fi

# Activate virtual environment
echo ""
echo -e "${BLUE}Activating virtual environment...${NC}"
source venv/bin/activate
echo -e "${GREEN}✓${NC} Virtual environment activated"

# Upgrade pip
echo ""
echo -e "${BLUE}Upgrading pip...${NC}"
pip install --upgrade pip --quiet
echo -e "${GREEN}✓${NC} Pip upgraded"

# Install dependencies
echo ""
echo -e "${BLUE}Installing dependencies...${NC}"
echo "This may take several minutes (installing TensorFlow, PyTorch, etc.)"
pip install -r requirements.txt --quiet

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} All dependencies installed successfully"
else
    echo -e "${RED}Error installing dependencies${NC}"
    exit 1
fi

# Create necessary directories
echo ""
echo -e "${BLUE}Creating project directories...${NC}"
mkdir -p data/cache
mkdir -p models
mkdir -p logs
mkdir -p .streamlit
echo -e "${GREEN}✓${NC} Directories created"

# Copy environment file if it doesn't exist
echo ""
echo -e "${BLUE}Setting up configuration files...${NC}"
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo -e "${GREEN}✓${NC} .env file created from example"
    echo -e "${YELLOW}Please edit .env to configure your settings${NC}"
else
    echo -e "${YELLOW}.env file already exists${NC}"
fi

# Copy Streamlit secrets if they don't exist
if [ ! -f ".streamlit/secrets.toml" ]; then
    cp .streamlit/secrets.toml.example .streamlit/secrets.toml
    echo -e "${GREEN}✓${NC} Streamlit secrets file created from example"
else
    echo -e "${YELLOW}Streamlit secrets file already exists${NC}"
fi

# Make scripts executable
echo ""
echo -e "${BLUE}Making scripts executable...${NC}"
chmod +x run_api.sh
chmod +x run_web.sh
chmod +x setup.sh
echo -e "${GREEN}✓${NC} Scripts are now executable"

# Check for model files
echo ""
echo -e "${BLUE}Checking for model files...${NC}"
if [ -f "models/final_model.keras" ] || [ -f "models/best_model.keras" ]; then
    echo -e "${GREEN}✓${NC} Model file found"
else
    echo -e "${YELLOW}No model file found${NC}"
    echo "You'll need to train a model before running the API."
    echo "Run the training notebook: Notebooks/train.ipynb"
fi

# Summary
echo ""
echo "========================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo "========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Train the model (if not done already):"
echo "   jupyter notebook Notebooks/train.ipynb"
echo ""
echo "2. Start the API server:"
echo "   ./run_api.sh"
echo ""
echo "3. Start the web interface (in another terminal):"
echo "   ./run_web.sh"
echo ""
echo "4. Or use Docker:"
echo "   docker-compose up -d"
echo ""
echo "Documentation:"
echo "  - Main README: README.md"
echo "  - API docs: api/README.md"
echo "  - Web interface: WEB_INTERFACE.md"
echo ""
echo "========================================="
echo ""
