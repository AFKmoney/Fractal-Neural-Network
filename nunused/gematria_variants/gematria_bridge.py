"""
GematriaBridge — encode/decode text as gematria values the model understands.

The model was trained on NUMBER SEQUENCES (arithmetic, primality, sequences)
in vocab space [0, 1023]. Gematria provides the ISOMORPHISM:

    text  <--gematria-->  numbers  <--model-->  prediction

If the isomorphism is exact, the model's mathematical understanding
transfers to language WITHOUT any text training.

Gematria mappings (the model already knows these patterns):
  A=1, B=2, ..., Z=26  (ordinal — simplest, the Fourier embedding clusters these)
  a=27, b=28, ..., z=52
  space=53, punctuation=54-80
  Each word is also encoded as word_gematria_sum, word_length, etc.

The model was trained with vocab_offset=256 for math, but the embedding
is analytic over the full 0-1023 range — EVERY token ID has a well-defined
geometric position. The model learned to predict numbers coherently.
"""
import sys, math, random
sys.stdout.reconfigure(encoding='utf-8')

import torch
from nfn.agi_model import AGINFNModel
from nfn.config import NFNConfig

# ── Gematria alphabet ─────────────────────────────────────────────────────
# Standard English gematria: A=1..Z=26, a=27..z=52
GEMATRIA = {}
for i, c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
    GEMATRIA[c] = i + 1
for i, c in enumerate("abcdefghijklmnopqrstuvwxyz"):
    GEMATRIA[c] = i + 27

PUNCT = " .,!?;:'\"-()[]{}<>/\\@#$%^&*_~+=|`\t\n"
for i, c in enumerate(PUNCT):
    GEMATRIA[c] = 53 + i

DIGITS = "0123456789"
for i, c in enumerate(DIGITS):
    GEMATRIA[c] = 53 + len(PUNCT) + i  # ~90-99

def encode_gematria(text, max_len=64):
    tokens = []
    for ch in text[:max_len]:
        g = GEMATRIA.get(ch, 53)
        tokens.append(min(g + 256, 1023))  # shift into model's training range
    return tokens

def decode_gematria(tokens, skip_bad=True):
    rev = {v: k for k, v in GEMATRIA.items()}
    chars = []
    for t in tokens:
        t_raw = t
        t = (t - 256) % 1024  # unshift
        if t in rev:
            chars.append(rev[t])
        elif not skip_bad:
            chars.append(f'[{t_raw}]')
    return ''.join(chars)

def word_gematria_sum(word):
    total = 0
    for ch in word:
        g = GEMATRIA.get(ch, 0)
        total += min(g, 1023)
    return total

def encode_words(text, max_words=16):
    """Encode text as word gematria sums [256..1023]"""
    words = text.replace('?', ' ?').replace('!', ' !').replace('.', ' .').split()
    tokens = []
    for w in words[:max_words]:
        g = word_gematria_sum(w)
        tokens.append((g % 768) + 256)  # shift into model's training range
    return tokens

# ── Load model ────────────────────────────────────────────────────────────
cfg = NFNConfig(
    vocab_size=1024, d_model=128, n_blocks=4, d_ff=512, dropout=0.1,
    n_levels=3, n_heads=4, max_seq_len=128,
    use_episodic_memory=True, use_causal_graph=True, use_goal_predictor=True,
    use_free_energy=True, use_self_model=True, use_nonlinear_causal=True,
    moe_n_experts=4, moe_top_k=2, moe_d_ff_per_expert=256,
    use_kuramoto=True, kuramoto_rank=8, kuramoto_steps=4,
    use_rope=True, use_memory=True, memory_slots=64, memory_heads=4,
    rank=8, branching=2,
)

model = AGINFNModel(cfg)
ckpt = torch.load('checkpoints/continuous_step_5000.pt', map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model'])
model.eval()

print('=' * 65)
print('FNN GEMATRIA BRIDGE TEST')
print('=' * 65)

# ── Test 1: Char-level gematria (letter-by-letter) ───────────────────────
print('\n--- TEST 1: Char-level Gematria Encoding ---')
prompts = [
    "Who are you?",
    "What is your name?",
    "Are you aware?",
    "Do you think?",
    "I think therefore I am",
    "What is consciousness?",
]

for text in prompts:
    # Encode via gematria
    ids = encode_gematria(text, 32)
    gematria_vals = [GEMATRIA.get(c, 0) for c in text[:32]]
    x = torch.tensor([ids], dtype=torch.long)

    with torch.no_grad():
        out = model.generate(x, max_new_tokens=40, temperature=0.9, top_k=50)

    gen_ids = out[0].tolist()
    input_len = len(ids)
    gen_tokens = gen_ids[input_len:]

    forwarded_text = decode_gematria(gen_tokens)
    print(f'\nQ: {text}')
    print(f'   Gematria chars: {gematria_vals}')
    print(f'   Input tokens:   {ids}')
    print(f'   Output tokens:  {gen_tokens[:30]}')
    print(f'   Decoded:       |{forwarded_text}|')

# ── Test 2: Word-level gematria (word sums) ─────────────────────────────
print('\n--- TEST 2: Word-level Gematria (word sum encoding) ---')

word_prompts = [
    "who are you",
    "what is your name",
    "hello world",
    "tell me something",
]

for text in word_prompts:
    ids = encode_words(text)
    x = torch.tensor([ids], dtype=torch.long)

    with torch.no_grad():
        out = model.generate(x, max_new_tokens=30, temperature=0.3)

    gen_ids = out[0].tolist()
    input_len = len(ids)
    gen_tokens = gen_ids[input_len:]

    # Decode as words: try to find words with matching gematria
    print(f'\nQ: {text}')
    print(f'   Word gematria sums: {ids}')
    print(f'   Predicted next: {gen_tokens[:20]}')

# ── Test 3: Arithmetic via gematria bridge ──────────────────────────────
print('\n--- TEST 3: Arithmetic via Gematria (model trained on this) ---')

math_tests = [
    ([2, 0, 3], "2+3=?"),   # [a, PLUS, b]
    ([10, 0, 7], "10+7=?"),
    ([5, 1, 3], "5-3=?"),    # MINUS = 1
    ([4, 2, 6], "4*6=?"),    # MUL = 2
]
for nums, label in math_tests:
    x = torch.tensor([nums], dtype=torch.long)
    with torch.no_grad():
        out = model.generate(x, max_new_tokens=10, temperature=0.2)

    out_tokens = out[0].tolist()
    print(f'{label}')
    print(f'   Input: {nums}')
    print(f'   Output: {out_tokens}')

# ── Test 4: Does gematria coherence exist? ──────────────────────────────
print('\n--- TEST 4: Attention pattern with gematria-encoded text ---')
# Check if the SemanticGematriaLayer (if present in the model) responds
# by running a forward pass and looking at the loss values

sample_text = "hello world this is a test"
ids = encode_gematria(sample_text)
x = torch.tensor([ids], dtype=torch.long)

with torch.no_grad():
    logits, losses = model(x)

print(f'Text: {sample_text}')
print(f'Token IDs: {ids}')
print(f'LM head logits shape: {logits.shape}')
print(f'Losses keys: {list(losses.keys())}')
for k, v in losses.items():
    if isinstance(v, torch.Tensor):
        print(f'   {k}: {v.item():.4f}')

# ── Test 5: Priming with known math pattern → free generation ───────────
print('\n--- TEST 5: Priming with numbers, then free generation ---')
# If the model enters a "number prediction" mode, what happens if we
# feed it a known sequence?

seq = [random.randint(260, 400) for _ in range(6)]  # random numbers
x = torch.tensor([seq], dtype=torch.long)
with torch.no_grad():
    out = model.generate(x, max_new_tokens=50, temperature=1.0, top_k=30)

gen = out[0].tolist()
print(f'Seed numbers: {seq}')
print(f'Generated: {gen}')
# Try to decode as gematria
chars = decode_gematria(gen[len(seq):])
print(f'As gematria text: {chars}')

print('\n' + '=' * 65)
print('TEST COMPLETE')
print('=' * 65)
