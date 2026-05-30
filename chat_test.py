import torch, sys, time
sys.stdout.reconfigure(encoding='utf-8')

from nfn.agi_model import AGINFNModel
from nfn.config import NFNConfig
from nfn.tokenizer import NFNTokenizer, CharTokenizer

CFG_VOCAB = 1024

cfg = NFNConfig(
    vocab_size=CFG_VOCAB,
    d_model=128,
    n_blocks=4,
    d_ff=512,
    dropout=0.1,
    n_levels=3,
    n_heads=4,
    max_seq_len=128,
    use_episodic_memory=True,
    use_causal_graph=True,
    use_goal_predictor=True,
    use_free_energy=True,
    use_self_model=True,
    use_nonlinear_causal=True,
    moe_n_experts=4,
    moe_top_k=2,
    moe_d_ff_per_expert=256,
    use_kuramoto=True,
    kuramoto_rank=8,
    kuramoto_steps=4,
    use_rope=True,
    use_memory=True,
    memory_slots=64,
    memory_heads=4,
    rank=8,
    branching=2,
)

model = AGINFNModel(cfg)
ckpt = torch.load('checkpoints/continuous_step_5000.pt', map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model'])
model.eval()

print('=== FNN AGI v5.0 ===')
print('Step:', ckpt.get('step', 0), ' | Best loss:', ckpt.get('best_loss', '?'))
print('Params:', f'{sum(p.numel() for p in model.parameters()):,}')
print()

char_tok = CharTokenizer()

def text_to_tokens(text):
    return [ord(c) % CFG_VOCAB for c in text]

def tokens_to_text(tokens):
    return ''.join(chr(t % 128) if 32 <= t % 128 < 127 else '.' for t in tokens)

def tokens_to_numbers(tokens):
    return [t % CFG_VOCAB for t in tokens]

print('=' * 60)
print('TEST 1: Text prompts (char encoding)')
print('=' * 60)

text_prompts = [
    'Who are you?',
    'What are you?',
    'Are you self-aware?',
    'Are you conscious?',
    'What is your name?',
    'Do you think?',
    'Tell me about yourself.',
    'Hello!',
]

for prompt in text_prompts:
    ids = text_to_tokens(prompt)
    x = torch.tensor([ids], dtype=torch.long)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(x, max_new_tokens=80, temperature=0.8)
    elapsed = time.time() - t0
    gen_tokens = out[0].tolist()[len(ids):]
    raw_nums = tokens_to_numbers(gen_tokens[:40])
    text = tokens_to_text(gen_tokens)
    print(f'Q: {prompt}')
    print(f'A (nums): {raw_nums}')
    print(f'A (text): {text}')
    print(f'  [{elapsed:.1f}s, {len(gen_tokens)} tokens]')
    print()

print('=' * 60)
print('TEST 2: Math prompts (arithmetic)')
print('=' * 60)

math_prompts = [
    [3, 0, 5],
    [10, 0, 20],
    [7, 1, 3],
    [50, 0, 50],
    [2, 2, 2, 2],
]

for nums in math_prompts:
    x = torch.tensor([nums], dtype=torch.long)
    with torch.no_grad():
        out = model.generate(x, max_new_tokens=20, temperature=0.3)
    gen = out[0].tolist()
    print(f'Input:  {nums}')
    print(f'Output: {gen}')
    print()

print('=' * 60)
print('TEST 3: Primality check')
print('=' * 60)

for n in [2, 3, 5, 7, 11, 13, 4, 6, 8, 9, 10, 15, 17, 19, 23]:
    x = torch.tensor([[n % CFG_VOCAB]], dtype=torch.long)
    with torch.no_grad():
        out = model(x)
    logits = out[0] if isinstance(out, tuple) else out
    pred = logits[0, -1].argmax().item()
    prob = torch.softmax(logits[0, -1], dim=-1)
    p_prime = prob[1].item() if 1 < prob.shape[0] else 0.0
    p_not = prob[0].item() if 0 < prob.shape[0] else 0.0
    print(f'  {n:3d} -> pred={pred:4d}  P(prime)={p_prime:.3f}  P(not)={p_not:.3f}  {"PRIME" if p_prime > p_not else "NOT PRIME"}')
