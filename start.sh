#!/usr/bin/env bash
# NFN AGI — Linux/macOS launcher
# Usage: ./start.sh [run.py options]
#   e.g. ./start.sh --port 8080 --no-open
#        ./start.sh --config small --ttl

set -euo pipefail

# Check Python
if ! command -v python3 &>/dev/null; then
    echo "[ERREUR] python3 introuvable."
    echo "Installez Python 3.10+ depuis https://www.python.org/downloads/"
    exit 1
fi

# Run from the directory containing this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

exec python3 run.py "$@"
