#!/bin/bash
pkill -f train_phase2 2>/dev/null || true
sleep 2

# Fix checkpoint path
python3 << 'EOF'
c = open('/root/FNN/train_phase2.py').read()
c = c.replace('nfn_20000.pt', 'NFNmini.pt')
open('/root/FNN/train_phase2.py','w').write(c)
print('Fixed checkpoint -> NFNmini.pt')
EOF

cd /root/FNN
nohup python3 -u train_phase2.py > /root/train_phase2.log 2>&1 &
echo "Phase 2 PID: $!"
