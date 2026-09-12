#!/usr/bin/env python3
"""FNN live train — perpetual. Never reset tokens_seen if RESUME.progress > 0."""
from __future__ import annotations
import json, re, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch
from nfn.config import FNNConfig
from nfn.model import FNNModel
from nfn.tokenizer import BPETokenizer

ART = Path('.')
CKPT = ART / 'fnn_d256.pt'
RESUME = ART / 'RESUME_d256.json'
TOKFILE = ART / 'fnn_talk_bpe.json'
LOG = ART / 'fnn_d256.log'
CORPUS = Path('shakespeare.txt')
RUN_NAME = 'fnn-d256-live'

def log(msg):
    print(msg, flush=True)
    with LOG.open('a') as f:
        f.write(msg + '\n')

def clean(text):
    text = text.replace('</w>', ' ')
    return re.sub(r'[ \t]+', ' ', text).replace(' \n', '\n').strip()

def make_cfg(tok):
    return FNNConfig(
        vocab_size=tok.vocab_size, pad_token_id=tok.pad_token_id,
        bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
        d_model=256, d_ff=512, n_blocks=2, n_heads=4, n_levels=2,
        dropout=0.0, max_seq_len=256, moe_n_experts=4, moe_top_k=2,
        moe_d_ff_per_expert=128, use_linear_attn=True, use_gematria=False,
        use_kuramoto=True, use_goal_forcing=False, use_causal_graph=False,
        use_self_model=False, use_working_memory=False, use_episodic_memory=False,
        use_analytic_embed=True, use_condensate=False, use_auto_genesis=False,
        use_self_modification=False, use_hyperbolic_gematria=False,
        use_ads_cft=False, use_mera=False, use_godel_loop=False, use_rg_flow=False,
        use_rope=False, use_ssm=False, use_mixture_of_depths=False,
        use_predictive_coding=False, use_mtp=False,
    )

def load_resume():
    if not RESUME.exists():
        return {'run': RUN_NAME, 'tokens_seen': 0, 'step': 0, 'loss': None, 'd_model': 256, 'n_blocks': 2}
    data = json.loads(RESUME.read_text())
    if int(data.get('tokens_seen', 0)) < 0 or int(data.get('step', 0)) < 0:
        raise SystemExit('RESUME corrupt')
    return data

def save_resume(state, model):
    if state['tokens_seen'] <= 0 and CKPT.exists():
        raise SystemExit('refuse tokens_seen=0 over existing brain')
    tmp = RESUME.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(RESUME)
    torch.save({'state': model.state_dict(), 'tokens_seen': state['tokens_seen'],
                'step': state['step'], 'loss': state['loss'], 'run': RUN_NAME}, CKPT)

def encode_corpus(tok, text):
    ids = []
    for i in range(0, len(text), 6000):
        ids.extend(tok.encode(text[i:i+6000], add_bos=False, add_eos=False))
    return torch.tensor(ids, dtype=torch.long)

def sample(model, tok, prompt, temperature=0.4, top_k=12, n=80):
    p = torch.tensor(tok.encode(prompt, add_bos=True), dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        out = model.generate(p, max_new_tokens=n, temperature=temperature, top_k=top_k)
    return clean(tok.decode(out[0].tolist()))

def main():
    tok = BPETokenizer.load(str(TOKFILE))
    raw = CORPUS.read_text(encoding='utf-8', errors='ignore')
    ids = encode_corpus(tok, raw)
    cfg = make_cfg(tok)
    model = FNNModel(cfg)
    total = sum(p.numel() for p in model.parameters())
    moe = sum(p.numel() for n,p in model.named_parameters() if '.moe.' in n)
    active = int(total - moe * 0.5)
    man = load_resume()
    if CKPT.exists():
        blob = torch.load(CKPT, map_location='cpu', weights_only=False)
        model.load_state_dict(blob['state'])
        man['tokens_seen'] = max(int(man.get('tokens_seen', 0)), int(blob.get('tokens_seen', 0)))
        man['step'] = max(int(man.get('step', 0)), int(blob.get('step', 0)))
        log(f"OPEN-HEART resume tokens_seen={man['tokens_seen']} step={man['step']}")
    else:
        if man['tokens_seen'] > 0 or man['step'] > 0:
            raise SystemExit('RESUME progress but missing .pt')
        log('NEW brain fnn_d256.pt')
    log(json.dumps({'total': total, 'active': active, 'vocab': tok.vocab_size}))
    batch, seq = 4, 128
    opt = torch.optim.AdamW(model.parameters(), lr=6e-4)
    n = len(ids); t0 = time.time(); last = man.get('loss')
    step = int(man['step']); seen = int(man['tokens_seen'])
    session_steps = 800
    model.train()
    for i in range(session_steps):
        starts = torch.randint(0, max(1, n-seq-1), (batch,))
        x = torch.stack([ids[int(s):int(s)+seq] for s in starts])
        y = torch.stack([ids[int(s)+1:int(s)+seq+1] for s in starts])
        opt.zero_grad(set_to_none=True)
        _, losses = model(x, targets=y)
        losses['lm'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last = float(losses['lm'].detach())
        step += 1; seen += batch * seq
        if i % 50 == 0 or i == session_steps-1:
            man = {'run': RUN_NAME, 'tokens_seen': seen, 'step': step, 'loss': last,
                   'd_model': 256, 'n_blocks': 2, 'active_params': active, 'total_params': total}
            save_resume(man, model)
            log(f'step={step:06d} loss={last:.4f} tok={seen} t={time.time()-t0:.1f}s')
    model.eval()
    for prompt in ['ROMEO:\n', 'JULIET:\n', 'To be or not to be']:
        log('GEN ' + prompt.strip() + ' -> ' + sample(model, tok, prompt)[:300])
    log('SLICE_DONE next launch continues')

if __name__ == '__main__':
    main()
