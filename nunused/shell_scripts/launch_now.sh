#!/bin/bash
cd /root/FNN
python3 << 'EOF'
import nltk
nltk.download('words', quiet=True)
print('NLTK words ready')
EOF

nohup python3 -u train_phase1.py > /root/train.log 2>&1 &
echo "PID: $!"
