#!/bin/bash
cd /root/FNN

# Generate Shakespeare+Gutenberg on server
python3 << 'EOF'
import nltk, pickle
nltk.download('shakespeare', quiet=True)
nltk.download('gutenberg', quiet=True)
from nltk.corpus import shakespeare, gutenberg

GEM_SHIFT=256
CH="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz .,!?;:\"-()[]{}<>/@#%^&*_~+=|\t\n0123456789"
def ge(t):
    ids=[1]
    for ch in t:
        idx=CH.find(ch)
        ids.append(idx+GEM_SHIFT if idx>=0 else GEM_SHIFT)
    ids.append(2)
    return ids

all_text = ""
for f in shakespeare.fileids():
    all_text += shakespeare.raw(f)
for f in gutenberg.fileids():
    all_text += gutenberg.raw(f)

tokens = ge(all_text)
with open('/workspace/texts_gematria.pkl','wb') as f:
    pickle.dump(tokens, f)
import os
print(f"Corpus: {len(tokens):,} tokens ({os.path.getsize('/workspace/texts_gematria.pkl')/1e6:.1f}MB)")
EOF

echo "Corpus ready. Launching Phase 2..."
# Update phase2 script to load texts
python3 -c "
c = open('phase2_quick.py').read()
c = c.replace('import nltk; nltk.download', '# Phase 2 with texts\nimport nltk; nltk.download')
# Add text loading before model.train()
insert = '''# Load text corpus
try:
    with open('/workspace/texts_gematria.pkl','rb') as f: data = pickle.load(f)
    text_t = torch.tensor(data, dtype=torch.long, device=device)
    ST = len(text_t)
    print(f'Text corpus: {ST:,} tokens', flush=True)
except:
    text_t = None
    ST = 0
    print('No text corpus', flush=True)
'''
c = c.replace('task_names = [', insert + '\ntask_names = [')
# Update task 5 to use text corpus
c = c.replace('''else:
        tokens = [1]
        for _ in range(random.randint(5,12)):
            w = random.choice(gw); tokens.extend(w[1:])
        tokens.append(2)
        sl = min(cfg.max_seq_len, len(tokens))
        x = torch.tensor([tokens[:sl]], dtype=torch.long, device=device)
        y = torch.full((1,sl), -1, dtype=torch.long, device=device)
        y[0,:sl-1] = torch.tensor(tokens[1:sl], dtype=torch.long)''',
'''else:
        if text_t is not None and ST > cfg.max_seq_len:
            sl = min(cfg.max_seq_len, ST-2)
            start = random.randint(0, ST - sl - 1)
            chunk = text_t[start:start+sl+1]
            x = chunk[:sl].unsqueeze(0)
            y = torch.full((1,sl), -1, dtype=torch.long, device=device)
            y[0,:sl-1] = chunk[1:sl]
        else:
            tokens = [1]
            for _ in range(random.randint(5,12)):
                w = random.choice(gw); tokens.extend(w[1:])
            tokens.append(2)
            sl = min(cfg.max_seq_len, len(tokens))
            x = torch.tensor([tokens[:sl]], dtype=torch.long, device=device)
            y = torch.full((1,sl), -1, dtype=torch.long, device=device)
            y[0,:sl-1] = torch.tensor(tokens[1:sl], dtype=torch.long)''')
open('phase2_quick.py','w').write(c)
print('Phase 2 updated with text corpus')
"

nohup python3 -u phase2_quick.py > /root/phase2.log 2>&1 &
echo "Phase 2 PID: $!"
