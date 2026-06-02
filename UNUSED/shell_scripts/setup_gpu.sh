#!/bin/bash
set -e
echo "=== Setup FNN 3B on GPU ==="

cd /root/FNN

# Install deps (RunPod template already has torch, just need extras)
pip install nltk fpdf2 -q --break-system-packages

# Download words
python3 -c "
import nltk; nltk.download('words', quiet=True)
from nltk.corpus import words
print(f'Dictionary: {len(words.words()):,} words')
"

# Check GPU
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

echo "=== Setup complete ==="
