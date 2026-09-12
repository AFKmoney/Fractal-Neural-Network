#!/usr/bin/env python3
"""Mini FNN train+test. Core flags only."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from nfn.config import FNNConfig
from nfn.model import FNNModel
from nfn.tokenizer import NFNTokenizer

CORPUS = (
    "Le Fractal Neural Network utilise une attention lineaire et une phase Kuramoto.\n"
    "Les experts sont routes par von Mises, pas par un gate Mixtral.\n"
    "Ce n'est pas un transformer softmax. Le bloc est FractalLinearAttention.\n"
    "PhaseSoliton amplifie les motifs coherents. Le MoE est creux top-2.\n"
) * 40

def main():
    tok = NFNTokenizer()
    cfg = FNNConfig(
        vocab_size=tok.vocab_size,
        pad_token_id=tok.pad_token_id,
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        d_model=64, d_ff=128, n_blocks=2, n_heads=2, n_levels=2,
        dropout=0.0, max_seq_len=128,
        moe_n_experts=4, moe_top_k=2, moe_d_ff_per_expert=64,
        use_linear_attn=True, use_gematria=False, use_kuramoto=True,
        use_goal_forcing=False, use_causal_graph=False, use_self_model=False,
        use_working_memory=False, use_episodic_memory=False,
        use_analytic_embed=True, use_condensate=False,
        use_auto_genesis=False, use_self_modification=False,
        use_hyperbolic_gematria=False, use_ads_cft=False, use_mera=False,
        use_godel_loop=False, use_rg_flow=False, use_rope=False,
        use_ssm=False, use_mixture_of_depths=False,
        use_predictive_coding=False, use_mtp=False,
    )
    model = FNNModel(cfg)
    print("params", sum(p.numel() for p in model.parameters()))
    ids = torch.tensor(tok.encode(CORPUS, add_bos=True, add_eos=False), dtype=torch.long)
    seq_len = 64
    x = ids[:seq_len].unsqueeze(0)
    y = ids[1:seq_len+1].unsqueeze(0)
    logits, losses = model(x, targets=y)
    print("loss_before", float(losses["lm"]))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    last = float(losses["lm"])
    for step in range(40):
        start = int(torch.randint(0, max(1, len(ids) - seq_len - 1), (1,)).item())
        x = ids[start:start+seq_len].unsqueeze(0)
        y = ids[start+1:start+seq_len+1].unsqueeze(0)
        opt.zero_grad(set_to_none=True)
        logits, losses = model(x, targets=y)
        losses["lm"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last = float(losses["lm"].detach())
        if step % 10 == 0:
            print(step, last)
    print("loss_after", last)

if __name__ == "__main__":
    main()
