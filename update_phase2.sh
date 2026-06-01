#!/bin/bash
python3 << 'EOF'
c = open('/root/FNN/train_phase2.py').read()
c = c.replace('shakespeare_gematria.pkl','texts_gematria.pkl')
c = c.replace('shakes_toks','text_toks')
c = c.replace('shakes_t','text_t')
c = c.replace('ST = len(shakes_t)','ST = len(text_t)')
c = c.replace('Shakespeare:','Corpus:')
c = c.replace('SH=','TX=')
open('/root/FNN/train_phase2.py','w').write(c)
print('Phase 2 updated — using 13.5M token corpus')
EOF
tail -3 /root/train.log
