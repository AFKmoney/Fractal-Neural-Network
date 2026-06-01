#!/bin/bash
cd /root/FNN
pkill -f train_final 2>/dev/null || true
sleep 2

# Direct fix with Python
python3 << 'EOF'
with open('/root/FNN/train_final.py') as f: c = f.read()
c = c.replace('d_model=1024, n_blocks=12, d_ff=4096', 'd_model=768, n_blocks=8, d_ff=3072')
c = c.replace('n_heads=16', 'n_heads=12')
c = c.replace('moe_d_ff_per_expert=2048', 'moe_d_ff_per_expert=1536')
off = ['use_multi_token_pred','use_recursive_reasoning','use_self_consistency','use_mixture_of_depths','use_plan_executor','use_predictive_coding']
for s in off:
    c = c.replace(f'{s}=True', f'{s}=False')
with open('/root/FNN/train_final.py','w') as f: f.write(c)
print('Config fixed.')
EOF

echo "Launching..."
cd /root/FNN
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup python3 -u train_final.py > /root/train.log 2>&1 &
echo "PID: $!"
