#!/bin/bash
python3 << 'EOF'
import pickle, os
with open('/workspace/shakespeare_gematria.pkl','rb') as f: s=pickle.load(f)
with open('/workspace/gutenberg_gematria.pkl','rb') as f: g=pickle.load(f)
combined = s + g
with open('/workspace/texts_gematria.pkl','wb') as f: pickle.dump(combined, f)
sz = os.path.getsize('/workspace/texts_gematria.pkl')
print(f'Combined: {len(combined):,} tokens ({sz/1e6:.1f}MB)')
EOF
