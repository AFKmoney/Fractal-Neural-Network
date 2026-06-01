#!/bin/bash
cd /root/FNN
pkill -f train_final 2>/dev/null || true
sleep 2

python3 << 'EOF'
c = open('/root/FNN/train_final.py').read()

# Remove autocast
c = c.replace("with torch.amp.autocast('cuda'):", "# NO AUTOCAST - fp32")
c = c.replace("    logits, aux = model(x, targets=y, write_memory=(step % 4 == 0))", "    logits, aux = model(x, targets=y, write_memory=(step % 4 == 0))")
# The indentation needs fixing - remove the extra indent from the autocast block
import re
# Fix: de-indent the block that was inside autocast
lines = c.split('\n')
new_lines = []
skip_indent = False
for line in lines:
    if '# NO AUTOCAST - fp32' in line:
        skip_indent = True
        new_lines.append(line)
        continue
    if skip_indent and line.startswith('        '):
        line = line[4:]  # remove one level of indent
    elif skip_indent and not line.startswith('        '):
        skip_indent = False
    new_lines.append(line)
c = '\n'.join(new_lines)

# Also remove autocast from generate
c = c.replace("with torch.no_grad(), torch.amp.autocast('cuda'):", "with torch.no_grad():")

open('/root/FNN/train_final.py','w').write(c)
print('fp32 mode active')
EOF

cd /root/FNN
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup python3 -u train_final.py > /root/train.log 2>&1 &
echo "PID: $!"
