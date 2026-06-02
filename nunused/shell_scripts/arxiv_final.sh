#!/bin/bash
cd /tmp
pip install arxiv -q --break-system-packages 2>/dev/null

python3 << 'EOF'
import arxiv, pickle, os, time

GEM_SHIFT=256
CH="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz .,!?;:\"-()[]{}<>/@#%^&*_~+=|\t\n0123456789"
def ge(t):
    ids=[1]
    for ch in t:
        idx=CH.find(ch)
        ids.append(idx+GEM_SHIFT if idx>=0 else GEM_SHIFT)
    ids.append(2)
    return ids

all_text=""
count=0
cats=["cs.AI","cs.LG","cs.CL","math.NT","stat.ML","physics.comp-ph"]

for cat in cats:
    try:
        s=arxiv.Search(query="cat:"+cat, max_results=50, sort_by=arxiv.SortCriterion.SubmittedDate)
        for p in s.results():
            all_text += p.title + ". " + p.summary[:1500] + " "
            count += 1
            if count%25==0: print(f"  {count} papers...", flush=True)
            time.sleep(0.3)
    except Exception as e:
        print(f"  {cat}: {e}", flush=True)

tokens=ge(all_text)
with open("/workspace/arxiv_gematria.pkl","wb") as f: pickle.dump(tokens,f)
sz=os.path.getsize("/workspace/arxiv_gematria.pkl")
print(f"ArXiv: {count} papers, {len(tokens):,} tokens ({sz/1e6:.1f}MB)", flush=True)
EOF
