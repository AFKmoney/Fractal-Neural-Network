# NFN — Complete API Reference

**Author:** Philippe-Antoine Robert
**Version:** 3.2
**Date:** 2026-05-03 07:22:48 UTC

---

## Table of Contents

1. [Python API — Configuration](#1-python-api--configuration)
2. [Python API — Models](#2-python-api--models)
3. [Python API — Tokenizer](#3-python-api--tokenizer)
4. [Python API — Training](#4-python-api--training)
5. [Python API — Inference](#5-python-api--inference)
6. [Python API — Memory](#6-python-api--memory)
7. [HTTP API — FastAPI Server](#7-http-api--fastapi-server)
8. [WebSocket — Streaming](#8-websocket--streaming)
9. [CLI — Command Line Interface](#9-cli--command-line-interface)
10. [Complete Examples](#10-complete-examples)

---

## 1. Python API — Configuration

### `NFNConfig`

```python
from nfn.config import NFNConfig

cfg = NFNConfig(
    vocab_size=512, pad_token_id=0, bos_token_id=1, eos_token_id=2,
    d_model=256, d_ff=1024, n_blocks=4, n_heads=4, dropout=0.1,
    n_levels=4, branching=2, motifs=["binary_tree", "cantor"],
    rank=8, omega_base=10_000.0, lambda_scale=2.0, damping=True, gamma_init=0.1,
    use_rope=True, rope_base=10_000.0, rope_scale_factor=1.0,
    max_seq_len=4096, context_len=32768,
    use_flash_attn=True,
    use_kuramoto=True, kuramoto_rank=8, kuramoto_steps=4, kuramoto_n_max=512,
    use_memory=True, memory_slots=64, memory_heads=4, memory_per_level=True,
    n_time_steps=4, alpha=0.9,
    lambda_phase=0.01, lambda_freq=0.001, lambda_spectral=0.0001,
    temperature=0.8, top_k=50, top_p=0.95,
    use_nfmc=False, nfmc_n_rff=256, nfmc_n_scales=8, nfmc_rank=64,
    nfmc_n_phases=8, nfmc_lock_iter=8, nfmc_eta=0.15, nfmc_lambda_phase=0.005,
    nfmc_hopfield_n=256, nfmc_zipf_alpha=1.0,
    moe_n_experts=8, moe_top_k=2, moe_d_ff_per_expert=256,
)

d = cfg.to_dict()
cfg2 = NFNConfig.from_dict(d)
cfg.n_motifs    # len(motifs)
cfg.top_len     # L / b^K
```

---

## 2. Python API — Models

### `NFNLanguageModel` (v2.0)

```python
from nfn.network import NFNLanguageModel
model = NFNLanguageModel(cfg)
logits, aux = model(input_ids, targets=None)
# logits: [B, L, V]  aux: {phases, loss_aux: {total, task, phase, freq, spectral}}
output_ids = model.generate(input_ids, max_new_tokens=200, temperature=0.8, top_k=50, top_p=0.95)
model.reset_memory(batch_size=1)
states = model.save_memory()
model.load_memory(states)
```

### `ZeroShotNFMC` (v3.1)

```python
from nfn.nfmc import ZeroShotNFMC
model = ZeroShotNFMC(cfg, use_zero_shot_embed=True)
info = model.param_summary()
# {total_params, trainable_params, analytic_buffers, knowledge_total, trainable_%_of_knowledge}
optimizer = AdamW(model.learnable_params(), lr=1e-3)
```

### `EfficientNFNLanguageModel` (v3.2)

```python
from nfn.efficient_block import EfficientNFNLanguageModel
model = EfficientNFNLanguageModel(cfg, use_analytic_embed=True)
model.condense_from_text(corpus, tokenizer=tok)
info = model.param_summary()
# {total_params, embed: 0, attn_per_block, moe_per_block, soliton/block, lm_head, analytic_buffers}
```

---

## 3. Python API — Tokenizer

```python
from nfn.tokenizer import load_tokenizer, BPETokenizer
tok = load_tokenizer(path=None, prefer="auto")  # tiktoken > bpe > char
ids = tok.encode("Hello world", add_bos=True, add_eos=False, max_length=512)
text = tok.decode([1, 45, 92, 2], skip_special=True)
tok.vocab_size  # int

tok = BPETokenizer(vocab_size=32_000)
tok.train(corpus_text)
tok.save("tokenizer.json")
tok2 = BPETokenizer.load("tokenizer.json")
```

---

## 4. Python API — Training

```python
from training.trainer import NFNTrainer
trainer = NFNTrainer(
    model=model, tokenizer=tok, cfg=cfg,
    lr=3e-4, weight_decay=0.1, max_grad_norm=1.0,
    dtype=torch.float32, output_dir="checkpoints",
    grad_accumulation_steps=4, use_grad_checkpointing=False, compile_model=False,
)
history = trainer.train(
    text=open("corpus.txt").read(), n_epochs=3, seq_len=512, batch_size=8,
    n_warmup_steps=200, eval_text=open("val.txt").read(), save_every=500, log_every=10,
)
path = trainer.save(tag="best")
trainer2 = NFNTrainer.load(path, device=torch.device("cuda"))

# Multi-GPU
from training.distributed import DistributedNFNTrainer, launch
trainer = DistributedNFNTrainer(model=model, tokenizer=tok, cfg=cfg, strategy="ddp")
launch("train.py", n_gpus=4, extra_args=["--config", "configs/medium.json"])
```

---

## 5. Python API — Inference

```python
from inference.engine import NFNInferenceEngine
engine = NFNInferenceEngine(model, tokenizer, device=torch.device("cuda"))
text = engine.generate(
    prompt="Once upon a time", max_new_tokens=500, temperature=0.8,
    top_k=50, top_p=0.95, strategy="top_p",  # greedy|top_p|beam|mirostat
    beam_width=4, mirostat_tau=5.0, mirostat_eta=0.1,
)
for token in engine.stream("Hello", max_new_tokens=200):
    print(token, end="", flush=True)
async def handler():
    async for token in engine.astream("Hello"):
        await ws.send_text(token)
ppl = engine.perplexity(text, stride=64)
response = engine.chat(messages=[{"role":"user","content":"What is NFN?"}], system="You are NFN.")
```

---

## 6. Python API — Memory

```python
model.reset_memory(batch_size=1)
states = model.save_memory()
torch.save(states, "memory.pt")
states = torch.load("memory.pt")
model.load_memory(states)
```

---

## 7. HTTP API — FastAPI Server

Launch: `python run.py --config configs/small.json --port 8000`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/status` | GET | Model status, param count, config |
| `/api/chat` | POST | Multi-turn chat, returns `{response, tokens_generated, elapsed_ms}` |
| `/api/generate` | POST | Generation from prompt, returns `{text, tokens}` |
| `/api/code` | POST | Code completion, returns `{completion}` |
| `/api/train/start` | POST | Starts training in background |
| `/api/train/stop` | POST | Stops training |
| `/api/train/status` | GET | Real-time metrics |
| `/api/load_model` | POST | Load a checkpoint |
| `/api/save_model` | POST | Save the model |
| `/api/condense` | POST | One-shot condensation from corpus |
| `/api/memory/reset` | POST | Resets memory to zero |
| `/api/memory/state` | GET | Current state of memory banks |
| `/api/agent/run` | POST | Run an agent with an objective |

---

## 8. WebSocket — Streaming

```javascript
const ws = new WebSocket("ws://localhost:8000/ws/stream");
ws.send(JSON.stringify({
  mode: "chat", messages: [{role:"user",content:"Hello"}],
  max_tokens: 300, temperature: 0.8, strategy: "top_p"
}));
ws.onmessage = (e) => {
  const msg = JSON.parse(e.data);
  if (msg.type === "token") process.stdout.write(msg.text);
  if (msg.type === "end") console.log("\nFull:", msg.full);
};
```

---

## 9. CLI — Command Line Interface

```bash
# Training
python train.py --config configs/small.json --data corpus.txt \
  --epochs 3 --batch 8 --seq_len 512 --lr 3e-4 --warmup 200 \
  --dtype bfloat16 --accumulate 4 --compile --distributed ddp

# Web server
python run.py --config configs/small.json --checkpoint checkpoints/nfn_final.pt --port 8000

# Multi-GPU
torchrun --nproc_per_node=4 train.py --config configs/medium.json --distributed ddp
torchrun --nproc_per_node=8 train.py --config configs/large.json --distributed fsdp
```

---

## 10. Complete Examples

### Example 1 — ZeroShot Pipeline

```python
from nfn.config import NFNConfig
from nfn.nfmc import ZeroShotNFMC
from nfn.tokenizer import load_tokenizer
from torch.optim import AdamW
import torch

cfg = NFNConfig(d_model=256, n_blocks=4, n_heads=4, nfmc_n_rff=256, nfmc_rank=64, max_seq_len=512)
tok = load_tokenizer()
model = ZeroShotNFMC(cfg, use_zero_shot_embed=True)
corpus = open("corpus.txt").read()
model.condense_from_text(corpus, tok, max_tokens=100_000)
model.condense_vocabulary(corpus, tok, max_tokens=200_000)

opt = AdamW(model.learnable_params(), lr=1e-3)
ids = torch.tensor(tok.encode(corpus[:50_000]))
for step in range(500):
    i = (step * 32) % (len(ids) - 33)
    x, y = ids[i:i+32].unsqueeze(0), ids[i+1:i+33].unsqueeze(0)
    opt.zero_grad()
    _, aux = model(x, targets=y)
    aux["loss_aux"]["total"].backward()
    torch.nn.utils.clip_grad_norm_(model.learnable_params(), 1.0)
    opt.step()
    if step % 50 == 0:
        print(f"Step {step}: loss={aux['loss_aux']['task'].item():.4f}")

model.eval()
with torch.no_grad():
    out = model.generate(torch.tensor([tok.encode("Hello,")]), max_new_tokens=200)
print(tok.decode(out[0].tolist()))
```

### Example 2 — EfficientNFN Production

```python
from nfn.config import NFNConfig
from nfn.efficient_block import EfficientNFNLanguageModel
from training.trainer import NFNTrainer
from nfn.tokenizer import load_tokenizer

cfg = NFNConfig(
    d_model=512, n_blocks=6, n_heads=8,
    max_seq_len=4096, context_len=32768,
    moe_n_experts=8, moe_top_k=2, moe_d_ff_per_expert=512,
    nfmc_n_rff=256, nfmc_rank=64, use_flash_attn=True, use_rope=True,
)
tok = load_tokenizer()
model = EfficientNFNLanguageModel(cfg, use_analytic_embed=True)
trainer = NFNTrainer(model, tok, cfg, lr=2e-4, dtype=torch.bfloat16,
                     grad_accumulation_steps=8, compile_model=True)
trainer.train(open("corpus.txt").read(), n_epochs=2, seq_len=2048, batch_size=4)
```

### Example 3 — Persistent Memory

```python
engine = NFNInferenceEngine(model, tok)
rep1 = engine.chat([{"role":"user","content":"My name is Philippe."}])
torch.save(model.save_memory(), "session.pt")
model.load_memory(torch.load("session.pt"))
rep2 = engine.chat([{"role":"user","content":"What is my name?"}])
# NFN remembers: "Philippe"
```

### Example 4 — WebSocket Streaming

```python
import asyncio, json, websockets
async def stream_chat():
    async with websockets.connect("ws://localhost:8000/ws/stream") as ws:
        await ws.send(json.dumps({"mode":"chat",
            "messages":[{"role":"user","content":"Explain fractals"}],
            "max_tokens":400,"temperature":0.7}))
        async for raw in ws:
            msg = json.loads(raw)
            if msg["type"] == "token": print(msg["text"], end="", flush=True)
            elif msg["type"] == "end": break
asyncio.run(stream_chat())
```

---

*Philippe-Antoine Robert — 2026-05-03 07:22:48 UTC*
