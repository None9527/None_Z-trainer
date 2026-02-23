#!/bin/bash
# ============================================================================
# None Trainer - Linux/Mac Startup Script
# ============================================================================
#
# Usage:
#   ./start.sh              # Default startup
#   ./start.sh --port 8080  # Custom port
#   ./start.sh --dev        # Development mode (hot reload)
#
# ============================================================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Defaults
DEFAULT_PORT=28000
DEFAULT_HOST="0.0.0.0"
DEV_MODE=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --port|-p) TRAINER_PORT="$2"; shift 2 ;;
        --host|-H) TRAINER_HOST="$2"; shift 2 ;;
        --dev|-d) DEV_MODE=1; shift ;;
        --help|-h)
            echo "None Trainer Startup Script"
            echo ""
            echo "Usage: ./start.sh [options]"
            echo "  --port, -p PORT    (default: 28000)"
            echo "  --host, -H HOST    (default: 0.0.0.0)"
            echo "  --dev, -d          Development mode (hot reload)"
            exit 0 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

# Load .env (strip Windows \r if present)
if [ -f "$SCRIPT_DIR/.env" ]; then
    # Remove carriage returns to prevent paths like "datasets\r"
    set -a; source <(sed 's/\r$//' "$SCRIPT_DIR/.env"); set +a
fi

# Apply defaults
export TRAINER_PORT=${TRAINER_PORT:-$DEFAULT_PORT}
export TRAINER_HOST=${TRAINER_HOST:-$DEFAULT_HOST}
export MODEL_PATH=${MODEL_PATH:-"$SCRIPT_DIR/Z-Image"}
export DATASET_PATH=${DATASET_PATH:-"$SCRIPT_DIR/datasets"}
export OUTPUT_PATH=${OUTPUT_PATH:-"$SCRIPT_DIR/model-output"}
export OLLAMA_HOST=${OLLAMA_HOST:-"http://127.0.0.1:11434"}

mkdir -p "$DATASET_PATH" "$OUTPUT_PATH/logs"

# Activate venv if exists
VENV_DIR="$SCRIPT_DIR/venv"
[ -d "$VENV_DIR" ] && source "$VENV_DIR/bin/activate"

# Python path - v2 uses trainer_core for zimage_trainer
export PYTHONPATH="$SCRIPT_DIR/backend/trainer_core:$PYTHONPATH"
export PYTHONUNBUFFERED=1

# Banner
clear
echo -e "${CYAN}"
echo "  _   _                    _____          _                 "
echo " | \ | |                  |_   _|        (_)                "
echo " |  \| | ___  _ __   ___    | |_ __ __ _ _ _ __   ___ _ __ "
echo " | . \` |/ _ \| '_ \ / _ \   | | '__/ _\` | | '_ \ / _ \ '__|"
echo " | |\  | (_) | | | |  __/   | | | | (_| | | | | |  __/ |   "
echo " |_| \_|\___/|_| |_|\___|   \_/_|  \__,_|_|_| |_|\___|_|   "
echo "                                                            "
echo -e "${NC}"

# Config display
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  Service Configuration${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "  Port:       ${YELLOW}$TRAINER_PORT${NC}"
echo -e "  Host:       ${YELLOW}$TRAINER_HOST${NC}"
echo -e "  Models:     ${YELLOW}$MODEL_PATH${NC}"
echo -e "  Datasets:   ${YELLOW}$DATASET_PATH${NC}"
echo -e "  Output:     ${YELLOW}$OUTPUT_PATH${NC}"
echo -e "  Ollama:     ${YELLOW}$OLLAMA_HOST${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Check services
echo -e "${BLUE}[Check Services]${NC}"
if curl -s "$OLLAMA_HOST/api/tags" > /dev/null 2>&1; then
    echo -e "  Ollama: ${GREEN}✓ Running${NC}"
else
    echo -e "  Ollama: ${YELLOW}✗ Not running${NC}"
fi

if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
    echo -e "  GPU: ${GREEN}✓ $GPU_NAME ($GPU_MEM)${NC}"
else
    echo -e "  GPU: ${RED}✗ Not detected${NC}"
fi
echo ""

# Check frontend build
DIST_DIR="$SCRIPT_DIR/frontend/dist"
SRC_DIR="$SCRIPT_DIR/frontend/src"

check_frontend_build() {
    [ ! -d "$DIST_DIR" ] && return 0
    NEWEST_SRC=$(find "$SRC_DIR" -type f \( -name "*.vue" -o -name "*.ts" \) -newer "$DIST_DIR/index.html" 2>/dev/null | head -1)
    [ -n "$NEWEST_SRC" ] && return 0
    return 1
}

if check_frontend_build; then
    echo -e "${YELLOW}[Frontend Build] Rebuilding...${NC}"
    cd "$SCRIPT_DIR/frontend"
    if command -v npm &> /dev/null; then
        npm run build --silent
        echo -e "${GREEN}[Frontend Build] ✓ Done${NC}"
    else
        echo -e "${RED}[Frontend Build] ✗ npm not found${NC}"
    fi
    cd "$SCRIPT_DIR"
fi
echo ""

# Get local IP
get_local_ip() {
    if command -v hostname &> /dev/null; then
        hostname -I 2>/dev/null | awk '{print $1}'
    else
        echo "127.0.0.1"
    fi
}
LOCAL_IP=$(get_local_ip)

# Start TensorBoard (scan entire output dir to find per-run logs)
LOG_DIR="$OUTPUT_PATH"
mkdir -p "$LOG_DIR"
echo -e "${GREEN}Starting TensorBoard...${NC}"
if lsof -Pi :6006 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "  Port 6006 in use, skipping"
else
    tensorboard --logdir "$LOG_DIR" --port 6006 --host 0.0.0.0 > /dev/null 2>&1 &
    TB_PID=$!
    echo -e "  Address: ${CYAN}http://localhost:6006${NC}"
fi

# Cleanup
cleanup() {
    echo -e "\n${YELLOW}Stopping services...${NC}"
    [ -n "$TB_PID" ] && kill "$TB_PID" 2>/dev/null
}
trap cleanup EXIT

# Start v2 server
echo -e "${GREEN}Starting Web UI...${NC}"
echo -e "  Local:  ${CYAN}http://localhost:$TRAINER_PORT${NC}"
if [ "$TRAINER_HOST" = "0.0.0.0" ] && [ -n "$LOCAL_IP" ] && [ "$LOCAL_IP" != "127.0.0.1" ]; then
    echo -e "  LAN:    ${CYAN}http://$LOCAL_IP:$TRAINER_PORT${NC}"
fi
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
echo ""

# Run from v2 root using module path
cd "$SCRIPT_DIR"
if [ "$DEV_MODE" -eq 1 ]; then
    echo -e "${YELLOW}[Dev Mode] Hot reload enabled${NC}"
    python -m uvicorn backend.interface.main:app \
        --host "$TRAINER_HOST" \
        --port "$TRAINER_PORT" \
        --reload \
        --reload-dir "$SCRIPT_DIR/backend" \
        --log-level info
else
    python -m uvicorn backend.interface.main:app \
        --host "$TRAINER_HOST" \
        --port "$TRAINER_PORT" \
        --log-level warning
fi
