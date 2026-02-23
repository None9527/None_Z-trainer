#!/bin/bash
# ============================================================================
# None Trainer - Linux/Mac Setup Script
# ============================================================================

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "================================================"
echo "   None Trainer - Setup"
echo "================================================"
echo ""

# [1/7] Check Python
echo "[1/7] Check Python..."
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo -e "${RED}[ERROR] Python not found!${NC}"
    exit 1
fi
$PYTHON --version

# [2/7] Create / Activate venv
echo ""
echo "[2/7] Setup virtual environment..."
VENV_DIR="$SCRIPT_DIR/venv"
if [ -d "$VENV_DIR" ]; then
    echo "  venv already exists, activating..."
else
    echo "  Creating venv..."
    $PYTHON -m venv "$VENV_DIR" --system-site-packages
    echo -e "${GREEN}  venv created${NC}"
fi
source "$VENV_DIR/bin/activate"
PYTHON="$VENV_DIR/bin/python"
echo "  Python: $(which python) ($($PYTHON --version 2>&1))"

# [3/7] Check CUDA
echo ""
echo "[3/7] Check CUDA..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
else
    echo -e "${YELLOW}[WARNING] No NVIDIA GPU detected${NC}"
fi

# [4/7] Check PyTorch
echo ""
echo "[4/7] Check PyTorch..."
if ! $PYTHON -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')" 2>/dev/null; then
    echo -e "${RED}[ERROR] PyTorch not installed!${NC}"
    echo "Install manually (activate venv first):"
    echo "  source venv/bin/activate"
    echo "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128"
    exit 1
fi

# [5/7] Install dependencies
echo ""
echo "[5/7] Install Python dependencies..."
$PYTHON -m pip install -r requirements.txt -q
echo "  Installing diffusers (git latest)..."
$PYTHON -m pip install git+https://github.com/huggingface/diffusers.git -q

# [6/7] Create .env
if [ ! -f ".env" ]; then
    # Strip Windows \r line endings during copy
    sed 's/\r$//' env.example > .env
    echo "  Created .env config file"
fi

# [7/7] Build frontend
echo ""
echo "[7/7] Build frontend..."
if command -v npm &> /dev/null; then
    cd "$SCRIPT_DIR/frontend"
    [ ! -d "node_modules" ] && npm install --silent
    npm run build --silent
    cd "$SCRIPT_DIR"
    echo -e "${GREEN}  Frontend build done${NC}"
else
    echo -e "${YELLOW}  [WARNING] npm not found, skip frontend build${NC}"
fi

echo ""
echo "================================================"
echo -e "${GREEN}  Setup complete!${NC}"
echo "================================================"
echo ""
echo "Next steps:"
echo "  1. Edit .env to set model path"
echo "  2. source venv/bin/activate"
echo "  3. Run ./start.sh to launch"
echo ""
