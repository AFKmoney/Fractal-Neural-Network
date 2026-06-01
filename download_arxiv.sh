#!/bin/bash
# Download arXiv papers directly on server
cd /root/FNN

pip install arxiv -q --break-system-packages 2>/dev/null

python3 << 'EOF'
import arxiv, pickle, time, random

GEM_SHIFT = 256
GEM_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz .,!?;:\"-()[]{}<>/@#%^&*_~+=|\t\n0123456789"

def gem_encode(text):
    ids = [1]
    for ch in text:
        idx = GEM_CHARS.find(ch)
        ids.append(idx + GEM_SHIFT if idx >= 0 else GEM_SHIFT)
    ids.append(2)
    return ids

categories = ["cs.AI", "cs.LG", "cs.CL", "math.*", "stat.ML", "physics.*"]
all_text = ""
total_papers = 0

print("Downloading arXiv papers...", flush=True)

for cat in categories:
    try:
        search = arxiv.Search(
            query=f"cat:{cat}",
            max_results=200,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        for paper in search.results():
            title = paper.title.replace('\n', ' ')
            summary = paper.summary.replace('\n', ' ')
            text = f"{title}. {summary} "
            all_text += text
            total_papers += 1
            if total_papers % 50 == 0:
                print(f"  Downloaded {total_papers} papers...", flush=True)
            time.sleep(0.5)  # Rate limit
    except Exception as e:
        print(f"  Category {cat}: {e}", flush=True)
        continue

print(f"\nTotal: {total_papers} papers, {len(all_text):,} chars", flush=True)

# Encode gematria
tokens = gem_encode(all_text)
print(f"Gematria tokens: {len(tokens):,}", flush=True)

# Save
with open('/workspace/arxiv_gematria.pkl', 'wb') as f:
    pickle.dump(tokens, f)

import os
print(f"Saved: /workspace/arxiv_gematria.pkl ({os.path.getsize('/workspace/arxiv_gematria.pkl')/1e6:.1f}MB)", flush=True)
EOF
