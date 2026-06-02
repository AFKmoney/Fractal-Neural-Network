#!/bin/bash
# Phase 4: Code corpus (Python repos)
cd /tmp

python3 << 'EOF'
import os, pickle, subprocess, glob

GEM_SHIFT = 256
GEM_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz .,!?;:\"-()[]{}<>/@#%^&*_~+=|\t\n0123456789"

def gem_encode(text):
    ids = [1]
    for ch in text:
        idx = GEM_CHARS.find(ch)
        ids.append(idx + GEM_SHIFT if idx >= 0 else GEM_SHIFT)
    ids.append(2)
    return ids

repos = [
    ("https://github.com/numpy/numpy.git", "numpy"),
    ("https://github.com/psf/requests.git", "requests"),
    ("https://github.com/pallets/flask.git", "flask"),
    ("https://github.com/python/cpython.git", "cpython", 2),  # depth 2
]

all_text = ""
total_chars = 0

for repo_info in repos:
    url = repo_info[0]
    name = repo_info[1]
    depth = repo_info[2] if len(repo_info) > 2 else 1
    
    print(f"Cloning {name}...", flush=True)
    try:
        subprocess.run(["git", "clone", "--depth", str(depth), url, f"/tmp/{name}"], 
                      capture_output=True, timeout=120)
        for f in glob.glob(f"/tmp/{name}/**/*.py", recursive=True):
            try:
                with open(f, 'r', encoding='utf-8', errors='ignore') as fh:
                    code = fh.read()
                    all_text += code[:3000] + "\n"
                    total_chars += min(len(code), 3000)
            except:
                pass
        print(f"  {name}: {total_chars:,} chars so far", flush=True)
    except Exception as e:
        print(f"  {name} failed: {e}", flush=True)

# Also add FNN codebase
print("Adding FNN codebase...", flush=True)
for f in glob.glob("/root/FNN/nfn/*.py") + glob.glob("/root/FNN/*.py"):
    try:
        with open(f, 'r', encoding='utf-8') as fh:
            all_text += fh.read()[:2000] + "\n"
            total_chars += min(len(fh.read()), 2000)
    except:
        pass

tokens = gem_encode(all_text)
with open('/workspace/code_gematria.pkl', 'wb') as f:
    pickle.dump(tokens, f)
print(f"Code corpus: {len(tokens):,} tokens ({total_chars:,} chars)", flush=True)
EOF
