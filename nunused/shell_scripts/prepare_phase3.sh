#!/bin/bash
# Phase 3: arXiv via HuggingFace datasets
cd /root/FNN

pip install datasets -q --break-system-packages 2>/dev/null

python3 << 'EOF'
import pickle
import os

GEM_SHIFT = 256
GEM_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz .,!?;:\"-()[]{}<>/@#%^&*_~+=|\t\n0123456789"

def gem_encode(text):
    ids = [1]
    for ch in text:
        idx = GEM_CHARS.find(ch)
        ids.append(idx + GEM_SHIFT if idx >= 0 else GEM_SHIFT)
    ids.append(2)
    return ids

print("Trying HuggingFace scientific_papers...", flush=True)
try:
    from datasets import load_dataset
    ds = load_dataset("scientific_papers", "arxiv", split="train", streaming=True)
    all_text = ""
    count = 0
    for item in ds:
        text = item.get('article', item.get('text', str(item)))
        all_text += text[:5000]  # first 5000 chars per paper
        count += 1
        if count % 100 == 0:
            print(f"  {count} papers, {len(all_text):,} chars", flush=True)
        if count >= 500:
            break
    print(f"Downloaded {count} papers, {len(all_text):,} chars", flush=True)
    tokens = gem_encode(all_text)
    with open('/workspace/arxiv_gematria.pkl', 'wb') as f:
        pickle.dump(tokens, f)
    print(f"ArXiv: {len(tokens):,} tokens saved", flush=True)
except Exception as e:
    print(f"HF failed: {e}", flush=True)
    
    # Fallback: use local arXiv-like text (the docs in the FNN repo)
    print("Fallback: using FNN docs as 'scientific' corpus...", flush=True)
    import glob
    all_text = ""
    for p in glob.glob('/root/FNN/docs/*.md') + ['/root/FNN/README.md']:
        try:
            with open(p, 'r') as f:
                all_text += f.read() + "\n"
        except:
            pass
    tokens = gem_encode(all_text)
    with open('/workspace/arxiv_gematria.pkl', 'wb') as f:
        pickle.dump(tokens, f)
    print(f"Fallback: {len(tokens):,} tokens saved", flush=True)
EOF
