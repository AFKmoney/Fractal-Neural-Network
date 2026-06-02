#!/bin/bash
cd /root/FNN
pkill -f train_final 2>/dev/null || true
sleep 2

python3 << 'EOF'
with open('/root/FNN/train_final.py') as f: c = f.read()
# Disable all aux modules that produce NaN
off = ['use_causal_graph','use_goal_predictor','use_free_energy',
       'use_self_model','use_nonlinear_causal']
for s in off:
    c = c.replace(f'{s}=True', f'{s}=False')
with open('/root/FNN/train_final.py','w') as f: f.write(c)
print('NaN modules disabled')
EOF

cd /root/FNN
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup python3 -u train_final.py > /root/train.log 2>&1 &
echo "PID: $!"
sleep 3
