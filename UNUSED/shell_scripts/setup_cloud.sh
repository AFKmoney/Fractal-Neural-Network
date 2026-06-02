#!/bin/bash
# ════════════════════════════════════════════════════════════════════════════
#  NFN Cloud GPU Setup — one command to go from bare server to training
# ════════════════════════════════════════════════════════════════════════════
#
#  Run this on any Ubuntu/Debian cloud GPU instance (Lambda Labs, RunPod,
#  Vast.ai, AWS, GCP, Azure…):
#
#    bash setup_cloud.sh [OPTIONS]
#
#  Or via curl (easier):
#
#    curl -fsSL https://raw.githubusercontent.com/AFKmoney/FNN/main/setup_cloud.sh | bash
#
#  Options:
#    --preset <name>       Training preset (default: auto-detect from VRAM)
#    --dataset <name>      Dataset to download (default: from preset)
#    --config <size>       nano | small | medium | large (default: from preset)
#    --no-train            Just set up, don't start training
#    --tmux                Run training inside a tmux session (survives SSH disconnect)
#    --help                Show this help
#
#  Examples:
#    bash setup_cloud.sh                              # Auto-detect + start training
#    bash setup_cloud.sh --preset small_wikipedia     # Specific preset
#    bash setup_cloud.sh --no-train                   # Just install, train manually later
#    bash setup_cloud.sh --tmux                       # Train in tmux session
# ════════════════════════════════════════════════════════════════════════════

set -e   # exit on first error

# ── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'  # no colour

ok()   { echo -e "${GREEN}✓${NC} $1"; }
info() { echo -e "${CYAN}→${NC} $1"; }
warn() { echo -e "${YELLOW}⚠${NC} $1"; }
err()  { echo -e "${RED}✗${NC} $1"; exit 1; }
hdr()  { echo -e "\n${BOLD}${BLUE}══ $1 ══${NC}"; }

# ── Parse args ────────────────────────────────────────────────────────────────
PRESET=""
DATASET=""
CONFIG=""
DO_TRAIN=true
USE_TMUX=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --preset)    PRESET="$2";   shift 2 ;;
        --dataset)   DATASET="$2";  shift 2 ;;
        --config)    CONFIG="$2";   shift 2 ;;
        --no-train)  DO_TRAIN=false; shift ;;
        --tmux)      USE_TMUX=true;  shift ;;
        --help|-h)
            head -40 "$0" | grep "^#" | sed 's/^# \{0,3\}//'
            exit 0
            ;;
        *) warn "Unknown option: $1"; shift ;;
    esac
done

# ── Banner ────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║   NFN — Neural Fractal Network  v5.0   Cloud GPU Setup          ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# ── Detect OS ─────────────────────────────────────────────────────────────────
hdr "System check"

OS=$(uname -s)
if [[ "$OS" != "Linux" ]]; then
    warn "This script is designed for Linux. macOS users: run setup manually."
fi

# Check for NVIDIA GPU
if command -v nvidia-smi &>/dev/null; then
    GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "Unknown")
    VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "0")
    VRAM_GB=$(echo "scale=1; $VRAM / 1024" | bc 2>/dev/null || echo "?")
    ok "GPU: $GPU ($VRAM_GB GB VRAM)"
    HAS_GPU=true
else
    warn "No NVIDIA GPU detected — will use CPU (training will be slow)"
    VRAM_GB=0
    HAS_GPU=false
fi

# Python version check
PY=$(python3 --version 2>/dev/null || python --version 2>/dev/null || echo "not found")
if [[ "$PY" == "not found" ]]; then
    err "Python not found. Install Python 3.10+ first."
fi
ok "Python: $PY"

# ── Dependencies ──────────────────────────────────────────────────────────────
hdr "Installing dependencies"

# Upgrade pip
info "Upgrading pip …"
python3 -m pip install --upgrade pip --quiet

# Install PyTorch with CUDA if GPU available
if $HAS_GPU; then
    # Detect CUDA version
    CUDA_VER=$(nvidia-smi | grep "CUDA Version" | awk '{print $NF}' 2>/dev/null || echo "12.1")
    CUDA_MAJOR=$(echo $CUDA_VER | cut -d. -f1)

    info "Installing PyTorch with CUDA $CUDA_VER …"
    if [[ "$CUDA_MAJOR" -ge 12 ]]; then
        python3 -m pip install torch torchvision torchaudio \
            --index-url https://download.pytorch.org/whl/cu121 --quiet
    elif [[ "$CUDA_MAJOR" -eq 11 ]]; then
        python3 -m pip install torch torchvision torchaudio \
            --index-url https://download.pytorch.org/whl/cu118 --quiet
    else
        python3 -m pip install torch torchvision torchaudio --quiet
    fi
else
    info "Installing PyTorch (CPU) …"
    python3 -m pip install torch torchvision torchaudio --quiet
fi
ok "PyTorch installed"

# Install NFN requirements
if [[ -f requirements.txt ]]; then
    info "Installing NFN requirements …"
    python3 -m pip install -r requirements.txt --quiet
    ok "NFN requirements installed"
fi

# Install in editable mode
if [[ -f setup.py ]]; then
    info "Installing NFN package …"
    python3 -m pip install -e . --quiet 2>/dev/null || true
    ok "NFN installed"
fi

# Optional but useful extras
info "Installing extras (tqdm, datasets) …"
python3 -m pip install tqdm datasets --quiet 2>/dev/null || true

# ── Clone repo (if running via curl, not yet in the repo dir) ─────────────────
if [[ ! -f "cloud_train.py" ]]; then
    hdr "Cloning NFN repository"
    if command -v git &>/dev/null; then
        git clone https://github.com/AFKmoney/FNN.git NFN
        cd NFN
        ok "Repository cloned to ./NFN"
        # Re-install inside the repo
        python3 -m pip install -r requirements.txt --quiet 2>/dev/null || true
        python3 -m pip install -e . --quiet 2>/dev/null || true
    else
        err "git not found. Install git with: apt-get install -y git"
    fi
fi

# ── Auto-select preset based on VRAM ─────────────────────────────────────────
if [[ -z "$PRESET" && -z "$DATASET" ]]; then
    hdr "Auto-selecting preset"

    VRAM_INT=${VRAM_GB%.*}
    if (( VRAM_INT >= 40 )); then
        PRESET="large_pile"
    elif (( VRAM_INT >= 20 )); then
        PRESET="medium_openwebtext"
    elif (( VRAM_INT >= 10 )); then
        PRESET="medium_wikipedia"
    elif (( VRAM_INT >= 4 )); then
        PRESET="small_wikipedia"
    else
        PRESET="nano_shakespeare"
    fi

    ok "Selected preset: $PRESET (based on ${VRAM_GB}GB VRAM)"
fi

# ── tmux setup ───────────────────────────────────────────────────────────────
if $USE_TMUX; then
    if ! command -v tmux &>/dev/null; then
        info "Installing tmux …"
        apt-get install -y tmux --quiet 2>/dev/null || yum install -y tmux --quiet 2>/dev/null || true
    fi
fi

# ── Summary ───────────────────────────────────────────────────────────────────
hdr "Ready to train"
echo ""
if [[ -n "$PRESET" ]]; then
    echo -e "  ${BOLD}Preset${NC}   : $PRESET"
fi
if [[ -n "$DATASET" ]]; then
    echo -e "  ${BOLD}Dataset${NC}  : $DATASET"
fi
if [[ -n "$CONFIG" ]]; then
    echo -e "  ${BOLD}Config${NC}   : $CONFIG"
fi
echo ""

if ! $DO_TRAIN; then
    ok "Setup complete. Start training manually with:"
    echo ""
    if [[ -n "$PRESET" ]]; then
        echo "    python cloud_train.py --preset $PRESET"
    elif [[ -n "$DATASET" ]]; then
        CMD="python cloud_train.py --dataset $DATASET"
        [[ -n "$CONFIG" ]] && CMD="$CMD --config $CONFIG"
        echo "    $CMD"
    else
        echo "    python cloud_train.py --list-presets"
        echo "    python cloud_train.py --preset small_wikipedia"
    fi
    echo ""
    exit 0
fi

# ── Build training command ────────────────────────────────────────────────────
TRAIN_CMD="python3 cloud_train.py"
if [[ -n "$PRESET" ]]; then
    TRAIN_CMD="$TRAIN_CMD --preset $PRESET"
elif [[ -n "$DATASET" ]]; then
    TRAIN_CMD="$TRAIN_CMD --dataset $DATASET"
    [[ -n "$CONFIG" ]] && TRAIN_CMD="$TRAIN_CMD --config $CONFIG"
fi
TRAIN_CMD="$TRAIN_CMD --tmux 2>&1 | tee train_$(date +%Y%m%d_%H%M%S).log"

# ── Start training ────────────────────────────────────────────────────────────
hdr "Starting training"

if $USE_TMUX; then
    SESSION="nfn_train"
    tmux new-session -d -s "$SESSION" "$TRAIN_CMD" 2>/dev/null || \
    tmux new-session -d -s "${SESSION}_$(date +%s)" "$TRAIN_CMD"

    ok "Training started in tmux session '$SESSION'"
    echo ""
    echo -e "  ${BOLD}Monitor training:${NC}"
    echo "    tmux attach -t $SESSION"
    echo ""
    echo -e "  ${BOLD}Detach (keep training):${NC}"
    echo "    Ctrl+B then D"
    echo ""
    echo -e "  ${BOLD}Resume after SSH disconnect:${NC}"
    echo "    tmux attach -t $SESSION"
    echo ""
    echo -e "  ${BOLD}Checkpoints saved to:${NC} checkpoints/"
else
    echo -e "  ${BOLD}Tip:${NC} Add --tmux to keep training after SSH disconnect"
    echo ""
    eval "$TRAIN_CMD"
fi
