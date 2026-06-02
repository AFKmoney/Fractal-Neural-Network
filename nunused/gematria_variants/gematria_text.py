"""Train FNN on English text via gematria encoding."""
import os, sys, glob, math, time, random
sys.stdout.reconfigure(encoding='utf-8')
import torch
import torch.nn.functional as F
from nfn.agi_model import AGINFNModel
from nfn.config import NFNConfig

class GematriaTokenizer:
    SHIFT = 256; VOCAB = 1024
    def __init__(self):
        self.g2t = {}; self.t2g = {}
        idx = self.SHIFT
        for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ": self.g2t[c]=idx; self.t2g[idx]=c; idx+=1
        for c in "abcdefghijklmnopqrstuvwxyz": self.g2t[c]=idx; self.t2g[idx]=c; idx+=1
        for c in " .,!?;:'\"-()[]{}<>/\\@#$%^&*_~+=|`\t\n": self.g2t[c]=idx; self.t2g[idx]=c; idx+=1
        for c in "0123456789": self.g2t[c]=idx; self.t2g[idx]=c; idx+=1
        self.pad=self.g2t[' ']; self.unk=self.g2t[' ']; self.bos=1; self.eos=2
    def encode(self, text, max_len=300):
        tokens = [self.bos]
        for ch in text[:max_len]: tokens.append(self.g2t.get(ch, self.unk))
        tokens.append(self.eos)
        return tokens
    def decode(self, tokens, skip_special=True):
        chars = []
        for t in tokens:
            if skip_special and t < self.SHIFT:
                if t == self.eos: break
                elif t == self.bos: chars.append('')
                continue
            chars.append(self.t2g.get(t, '.'))
        return ''.join(chars)

def load_corpus():
    texts = []
    for fpath in glob.glob('docs/*.md') + ['README.md'] + glob.glob('*.md'):
        with open(fpath,'r',encoding='utf-8') as f: texts.append(f.read())
    for fpath in glob.glob('nfn/*.py'):
        with open(fpath,'r',encoding='utf-8') as f: texts.append(f.read())
    return '\n\n'.join(texts)

def train():
    tok = GematriaTokenizer()
    print(f'Chars mapped: {len(tok.g2t)} in [{tok.SHIFT},{tok.SHIFT+len(tok.g2t)-1}]')

    corpus = load_corpus()
    print(f'Corpus: {len(corpus):,} chars, {len(corpus)//256:,} blocks')

    cfg = NFNConfig(vocab_size=1024,d_model=128,n_blocks=4,d_ff=512,dropout=0.1,
        n_levels=3,n_heads=4,max_seq_len=256,use_episodic_memory=True,
        use_causal_graph=True,use_goal_predictor=True,use_free_energy=True,
        use_self_model=True,use_nonlinear_causal=True,moe_n_experts=4,moe_top_k=2,
        moe_d_ff_per_expert=256,use_kuramoto=True,kuramoto_rank=8,kuramoto_steps=4,
        use_rope=True,use_memory=True,memory_slots=64,memory_heads=4,rank=8,branching=2)

    model = AGINFNModel(cfg)
    ckpt = torch.load('checkpoints/continuous_step_5000.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.train()

    device = torch.device('cpu'); model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9,0.95), weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=1e-5)

    print(f'Params: {sum(p.numel() for p in model.parameters()):,}')

    # Tokenize entire corpus sequentially
    all_tokens = tok.encode(corpus, max_len=500000)
    print(f'Total tokens: {len(all_tokens):,}')

    block_len = 256
    stride = 200  # overlap for continuity
    batches = []
    for i in range(0, len(all_tokens) - block_len - 1, stride):
        x = all_tokens[i:i+block_len]
        y = all_tokens[i+1:i+block_len+1]
        batches.append((x, y))
    random.shuffle(batches)
    print(f'Batches: {len(batches):,}')

    n_steps = min(3000, len(batches))
    best_loss = float('inf')
    t_start = time.time()

    for step in range(1, n_steps + 1):
        bx, by = batches[(step - 1) % len(batches)]
        x = torch.tensor([bx], dtype=torch.long, device=device)
        y = torch.tensor([by], dtype=torch.long, device=device)

        logits, losses = model(x, targets=y, write_memory=True)
        loss = losses.get('total', losses.get('lm', torch.tensor(0.0)))

        if loss.requires_grad and loss.item() > 0:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        if step % 200 == 0 or step == 1:
            elapsed = time.time() - t_start
            lr = scheduler.get_last_lr()[0]
            print(f'  [{step:5d}/{n_steps}]  lm_loss {losses["lm"].item():.3f}  total {loss.item():.4f}  lr {lr:.1e}  {elapsed:.0f}s')

        if step % 500 == 0:
            model.eval()
            prompts = ['Who are you?','What is consciousness?','I think therefore','Hello']
            for p in prompts:
                ids = tok.encode(p, max_len=32)
                xp = torch.tensor([ids], dtype=torch.long, device=device)
                with torch.no_grad():
                    out = model.generate(xp, max_new_tokens=40, temperature=0.9, top_k=50)
                dec = tok.decode(out[0].tolist()[len(ids):])
                print(f'  [{p}] -> {dec}')
            model.train()

    # Save final
    ckpt_path = 'checkpoints/gematria_final.pt'
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'step': n_steps, 'loss': best_loss}, ckpt_path)
    print(f'\nSaved: {ckpt_path}  ({time.time()-t_start:.0f}s)')

    # Final chat
    print('\n' + '='*60)
    print('FINAL CHAT')
    print('='*60)
    model.eval()
    for p in ['Who are you?','Are you self aware?','What is your name?','Do you think?','Hello world']:
        ids = tok.encode(p)
        xp = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model.generate(xp, max_new_tokens=80, temperature=0.85, top_k=50)
        dec = tok.decode(out[0].tolist()[len(ids):])
        print(f'Q: {p}')
        print(f'A: {dec}')
        print()

if __name__ == '__main__':
    train()
