#!/bin/bash
# Fix arXiv — run from /tmp to avoid FNN datasets/ conflict
cd /tmp
python3 << 'EOF'
import sys, pickle
# Avoid FNN's datasets/ folder shadowing
sys.path = [p for p in sys.path if 'FNN' not in p]
sys.path.insert(0, '/usr/local/lib/python3.12/dist-packages')

from datasets import load_dataset
print("Loading arXiv via HuggingFace...", flush=True)

ds = load_dataset("scientific_papers", "arxiv", split="train", streaming=True)
all_text = ""
count = 0
for item in ds:
    text = item.get('article', str(item))
    all_text += text[:5000]
    count += 1
    if count % 100 == 0:
        print(f"  {count} papers, {len(all_text):,} chars", flush=True)
    if count >= 300:
        break

GEM_SHIFT = 256
GEM_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz .,!?;:\"-()[]{}<>/@#%^&*_~+=|\t\n0123456789"

def gem_encode(t):
    ids = [1]
    for ch in t:
        idx = GEM_CHARS.find(ch)
        ids.append(idx + GEM_SHIFT if idx >= 0 else GEM_SHIFT)
    ids.append(2)
    return ids

tokens = gem_encode(all_text)
with open('/workspace/arxiv_gematria.pkl', 'wb') as f:
    pickle.dump(tokens, f)

import os
sz = os.path.getsize('/workspace/arxiv_gematria.pkl')
print(f"ArXiv: {count} papers, {len(tokens):,} tokens ({sz/1e6:.1f}MB)", flush=True)
EOF
