#!/bin/bash
cd /root/FNN
pkill -f train_final 2>/dev/null || true
sleep 2
python3 << 'EOF'
with open('/root/FNN/train_final.py') as f: c = f.read()
c = c.replace('use_episodic_memory=True','use_episodic_memory=False')
with open('/root/FNN/train_final.py','w') as f: f.write(c)
print('Fixed.')
EOF
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup python3 -u train_final.py > /root/train.log 2>&1 &
echo "PID: $!"
