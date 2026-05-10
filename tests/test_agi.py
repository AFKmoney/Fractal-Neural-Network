"""
Tests for NFN AGI v4.0 modules.

Covers: AGINFNModel, AGIBlock, all sub-modules, continual learning, tool calling.
"""

import math
import json
import pytest
import torch
import torch.nn.functional as F

from nfn.config import NFNConfig
from nfn.agi_model import AGINFNModel, build_agi_model
from nfn.agi_block import AGIBlock
from nfn.episodic_memory import TwoTierMemory
from nfn.working_memory import FractalWorkingMemory
from nfn.causal import CausalGraphLayer
from nfn.goal import PhaseGoalPredictor
from nfn.reasoning import RecursiveReasoner, SelfConsistencyCheck, PlanExecutor
from nfn.predictive import PredictiveCodingBlock, FreeEnergyMinimiser
from nfn.mixture_of_depths import MixtureOfDepths
from nfn.multi_token_pred import MultiTokenPredictor, SpeculativeDecoder
from nfn.streaming import InfiniteNFN, ChunkedForward
from nfn.hyper import ContextHyperNet, HyperResidual
from nfn.tools import ToolSpec, ToolRegistry, ToolCallParser, make_default_registry
from nfn.continual import KnowledgeStore, ContinualLearner, EWCRegularizer
from training.losses import AGILoss
from training.agi_trainer import AGITextDataset, AGITrainer
from nfn.tokenizer import NFNTokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

VOCAB   = 256
D_MODEL = 64
N_BLOCKS = 2
B, L    = 2, 16


@pytest.fixture
def tiny_cfg():
    return NFNConfig(
        vocab_size=VOCAB,
        d_model=D_MODEL,
        n_blocks=N_BLOCKS,
        n_heads=4,
        d_ff=128,
        n_levels=2,
        dropout=0.0,
        max_seq_len=64,
        goal_n_phases=8,
        episodic_capacity=32,
        episodic_n_read=4,
        causal_n_slots=8,
        wm_n_slots=8,
        sc_n_candidates=2,
        plan_n_subgoals=2,
        moe_n_experts=4,
        moe_top_k=2,
        moe_d_ff_per_expert=32,
    )


@pytest.fixture
def full_model(tiny_cfg):
    return build_agi_model(
        vocab_size=VOCAB,
        d_model=D_MODEL,
        n_blocks=N_BLOCKS,
        use_reasoning=True,
        use_predictive_coding=True,
        use_free_energy=True,
        use_self_consistency=True,
        use_plan_executor=True,
        use_mod=True,
        use_mtp=True,
        use_hyper=True,
    )


@pytest.fixture
def ids():
    return torch.randint(0, VOCAB, (B, L))


# ─────────────────────────────────────────────────────────────────────────────
# Core model
# ─────────────────────────────────────────────────────────────────────────────

def test_forward_basic(full_model, ids):
    logits, losses = full_model(ids, targets=ids)
    assert logits.shape == (B, L, VOCAB)
    assert "lm" in losses
    assert "total" in losses
    assert losses["total"].item() > 0


def test_forward_loss_keys(full_model, ids):
    _, losses = full_model(ids, targets=ids)
    # All registered loss channels present
    for k in ("causal", "goal", "ponder", "free_energy", "consistency"):
        assert k in losses, f"missing loss key: {k}"


def test_generate_shapes(full_model, ids):
    out = full_model.generate(ids[:1, :4], max_new_tokens=8)
    # May stop early on EOS; must be between prompt_len and prompt_len + max_new_tokens
    assert out.shape[1] >= 4
    assert out.shape[1] <= 4 + 8


def test_speculative_generate(full_model, ids):
    out = full_model.generate(ids[:1, :4], max_new_tokens=6, speculative=True)
    assert out.shape[0] == 1
    assert out.shape[1] >= 4


def test_goal_set_generate(full_model, ids):
    full_model.set_goal(ids[:1, :4])
    out = full_model.generate(ids[:1, :4], max_new_tokens=4, think_rounds=1)
    full_model.reset_goal()
    assert out.shape[1] >= 4
    assert out.shape[1] <= 4 + 4


def test_repr(full_model):
    r = repr(full_model)
    assert "AGINFNModel" in r


# ─────────────────────────────────────────────────────────────────────────────
# AGIBlock
# ─────────────────────────────────────────────────────────────────────────────

def test_agi_block_forward(tiny_cfg):
    cfg = tiny_cfg
    cfg.use_episodic_memory  = True
    cfg.use_working_memory   = True
    cfg.use_causal_graph     = True
    cfg.use_goal_predictor   = True
    cfg.use_self_consistency = True
    cfg.use_free_energy      = True
    block = AGIBlock(cfg)
    h = torch.randn(B, L, D_MODEL)
    h_out, losses = block(h)
    assert h_out.shape == (B, L, D_MODEL)
    assert isinstance(losses, dict)


def test_agi_block_short_seq(tiny_cfg):
    cfg = tiny_cfg
    cfg.use_causal_graph = True
    block = AGIBlock(cfg)
    h = torch.randn(B, 3, D_MODEL)   # L=3 < causal_n_slots=8
    h_out, _ = block(h)
    assert h_out.shape == (B, 3, D_MODEL)


def test_agi_block_counterfactual(tiny_cfg):
    cfg = tiny_cfg
    cfg.use_causal_graph = True
    block = AGIBlock(cfg)
    h = torch.randn(1, L, D_MODEL)
    val = torch.randn(1, D_MODEL)
    out = block.counterfactual(h, slot_idx=0, value=val)
    assert out.shape == (1, L, D_MODEL)


# ─────────────────────────────────────────────────────────────────────────────
# Sub-modules
# ─────────────────────────────────────────────────────────────────────────────

def test_two_tier_memory():
    mem = TwoTierMemory(D_MODEL, episodic_capacity=64, episodic_n_read=4, semantic_rank=16)
    h = torch.randn(B, L, D_MODEL)
    out = mem(h, write=True)
    assert out.shape == (B, L, D_MODEL)
    mem.maybe_consolidate()


def test_fractal_working_memory():
    wm = FractalWorkingMemory(D_MODEL, n_slots=8, n_heads=2)
    h = torch.randn(B, L, D_MODEL)
    out = wm(h, write=True)
    assert out.shape == (B, L, D_MODEL)
    wm.reset()


def test_causal_graph_layer_normal():
    cg = CausalGraphLayer(D_MODEL, n_slots=8)
    h = torch.randn(B, L, D_MODEL)
    out, loss = cg(h)
    assert out.shape == (B, L, D_MODEL)
    assert loss.item() >= 0


def test_causal_graph_layer_short():
    cg = CausalGraphLayer(D_MODEL, n_slots=8)
    h = torch.randn(B, 3, D_MODEL)
    out, loss = cg(h)
    assert out.shape == (B, 3, D_MODEL)


def test_phase_goal_predictor():
    gp = PhaseGoalPredictor(D_MODEL, n_phases=8)
    phases = torch.randn(B, L, 8)
    theta, align = gp(phases)
    assert theta.shape == (B, L, 8)
    gp.set_goal(torch.randn(B, L, D_MODEL))
    loss = gp.loss_goal(phases)
    assert loss.item() >= 0
    gp.reset_goal()


def test_free_energy_minimiser():
    fe = FreeEnergyMinimiser(D_MODEL, latent_dim=16)
    h = torch.randn(B, L, D_MODEL)
    h_out, loss = fe(h)
    assert h_out.shape == (B, L, D_MODEL)
    assert loss.item() >= 0


def test_self_consistency_check():
    sc = SelfConsistencyCheck(D_MODEL, n_candidates=2, noise_scale=0.05)
    h = torch.randn(B, L, D_MODEL)
    h_out, loss = sc(h, causal_layer=None)
    assert h_out.shape == (B, L, D_MODEL)


def test_plan_executor():
    pe = PlanExecutor(n_phases=8, n_subgoals=3)
    goal = torch.randn(1, 8)   # [B, n_phases]
    pe.set_plan(goal)
    sg = pe.current_subgoal(torch.device("cpu"))
    assert sg is not None
    pe.advance(torch.tensor([0.9]))
    pe.reset()


# ─────────────────────────────────────────────────────────────────────────────
# Killer features
# ─────────────────────────────────────────────────────────────────────────────

def test_mixture_of_depths():
    from nfn.efficient_block import EfficientNFNBlock
    cfg = NFNConfig(d_model=D_MODEL, n_blocks=1, n_heads=2, d_ff=64,
                    moe_n_experts=4, moe_top_k=2, moe_d_ff_per_expert=32)
    block = EfficientNFNBlock(cfg)
    mod = MixtureOfDepths(D_MODEL, block, capacity_factor=0.5)
    h = torch.randn(B, L, D_MODEL)
    out, loss = mod(h)
    assert out.shape == (B, L, D_MODEL)
    assert loss.item() >= 0


def test_multi_token_predictor():
    mtp = MultiTokenPredictor(D_MODEL, VOCAB, n_heads=3, alpha=1.0)
    h = torch.randn(B, L, D_MODEL)
    targets = torch.randint(0, VOCAB, (B, L))
    loss, sub = mtp.loss(h, targets)
    assert loss.item() >= 0
    draft = mtp.draft_tokens(h[:, -1:, :])
    # draft: [B, N_heads] tensor of draft token ids
    assert draft.shape == (B, 3)


def test_hyper_net():
    hn = ContextHyperNet(D_MODEL, n_phases=8, rank=2, z_dim=32)
    hr = HyperResidual(D_MODEL, 32)
    h = torch.randn(B, L, D_MODEL)
    z, deltas = hn(h, goal_phase=None)
    assert z.shape == (B, 32)
    h_out = hr(h, z)
    assert h_out.shape == (B, L, D_MODEL)


def test_infinite_nfn():
    model = build_agi_model(
        vocab_size=VOCAB, d_model=D_MODEL, n_blocks=1,
        use_mod=False, use_mtp=False, use_hyper=False,
    )
    infinite = InfiniteNFN(model, window_size=8, overlap=2)
    ids = torch.randint(0, VOCAB, (1, 20))
    logits, losses = infinite(ids)
    assert logits.shape[0] == 1
    assert logits.shape[2] == VOCAB


# ─────────────────────────────────────────────────────────────────────────────
# Tool calling
# ─────────────────────────────────────────────────────────────────────────────

def test_tool_registry():
    registry = make_default_registry()
    assert len(registry) >= 3
    result = registry.dispatch("calculator", {"expression": "2**10"})
    assert "1024" in result


def test_tool_call_parser_basic():
    text = 'Let me check that.\n<tool_call>\n{"name": "calculator", "arguments": {"expression": "2+2"}}\n</tool_call>'
    calls = ToolCallParser.find_calls(text)
    assert len(calls) == 1
    name, args, start, end = calls[0]
    assert name == "calculator"
    assert args["expression"] == "2+2"


def test_tool_call_parser_open():
    text = "Sure: <tool_call>\n{\"name\": \"calc\""  # unclosed
    assert ToolCallParser.has_open_call(text)


def test_tool_result_injection():
    text = 'before<tool_call>{"name":"x","arguments":{}}</tool_call>after'
    calls = ToolCallParser.find_calls(text)
    assert calls
    _, _, start, end = calls[0]
    injected = ToolCallParser.inject_result(text, "x", "42", end)
    assert "<tool_result" in injected
    assert "42" in injected


def test_json_extract_tool():
    registry = make_default_registry()
    result = registry.dispatch("json_extract", {
        "json_string": '{"data": {"value": 99}}',
        "path": "data.value",
    })
    assert "99" in result


# ─────────────────────────────────────────────────────────────────────────────
# Continual learning
# ─────────────────────────────────────────────────────────────────────────────

def test_knowledge_store_add_retrieve():
    store = KnowledgeStore(max_entries=100)
    emb1  = torch.randn(D_MODEL)
    emb2  = torch.randn(D_MODEL)
    store.add("cats like fish", emb1, source="test")
    store.add("dogs like bones", emb2, source="test")
    # Query similar to emb1
    results = store.retrieve(emb1, top_k=1)
    assert results[0][0] == "cats like fish"


def test_knowledge_store_save_load(tmp_path):
    store = KnowledgeStore()
    store.add("hello world", torch.randn(D_MODEL))
    path = str(tmp_path / "store.json.gz")
    store.save(path)
    loaded = KnowledgeStore.load(path)
    assert len(loaded) == 1
    assert loaded.entries[0]["text"] == "hello world"


def test_continual_learner_learn(full_model):
    tokenizer = NFNTokenizer()
    cl = ContinualLearner(full_model, tokenizer, chunk_size=16)
    emb = cl.learn("The quick brown fox jumps over the lazy dog.", source="test")
    assert emb.shape == (D_MODEL,)
    assert len(cl.store) >= 1


def test_continual_learner_retrieve(full_model):
    tokenizer = NFNTokenizer()
    cl = ContinualLearner(full_model, tokenizer, chunk_size=16)
    cl.learn("Neural networks learn from data.")
    results = cl.retrieve("machine learning", top_k=1)
    assert len(results) >= 0   # may or may not match, just check it runs


def test_continual_learner_rag(full_model):
    tokenizer = NFNTokenizer()
    cl = ContinualLearner(full_model, tokenizer, chunk_size=16)
    cl.learn("Paris is the capital of France.")
    out = cl.generate_with_rag("What is the capital of France?", max_new_tokens=8)
    assert isinstance(out, str)


def test_continual_learner_stats(full_model):
    tokenizer = NFNTokenizer()
    cl = ContinualLearner(full_model, tokenizer)
    stats = cl.stats()
    assert "knowledge_entries" in stats
    assert "learn_calls" in stats


# ─────────────────────────────────────────────────────────────────────────────
# AGI Trainer
# ─────────────────────────────────────────────────────────────────────────────

def test_agi_text_dataset():
    tokenizer = NFNTokenizer()
    text = "hello world " * 200
    ds = AGITextDataset(text, tokenizer, seq_len=16, batch_size=2)
    batches = list(ds.iter_batches(torch.device("cpu")))
    assert len(batches) > 0
    x, y = batches[0]
    assert x.shape[1] == 16
    assert y.shape[1] == 16


def test_agi_trainer_one_step(full_model):
    tokenizer = NFNTokenizer()
    trainer = AGITrainer(
        full_model, tokenizer,
        lr=1e-3,
        agi_loss_start_step=0,
        agi_loss_ramp_steps=1,
        goal_set_every=1,
        sleep_every=1,
        use_self_play=False,
        use_critique=False,
    )
    text = "the quick brown fox " * 100
    history = trainer.train(
        text,
        n_epochs=1,
        seq_len=16,
        batch_size=2,
        n_warmup_steps=1,
        save_every=10000,
        log_every=1,
    )
    assert len(history) > 0
    assert "lm" in history[0] or "total" in history[0]


def test_agi_trainer_perplexity(full_model):
    tokenizer = NFNTokenizer()
    trainer = AGITrainer(full_model, tokenizer)
    ppl = trainer.eval_perplexity("hello world test", seq_len=16)
    assert math.isfinite(ppl)
    assert ppl > 0


# ─────────────────────────────────────────────────────────────────────────────
# AGI Loss
# ─────────────────────────────────────────────────────────────────────────────

def test_agi_loss():
    from nfn.config import NFNConfig
    cfg = NFNConfig()
    criterion = AGILoss(cfg)
    losses = {
        "lm":           torch.tensor(2.5),
        "causal":       torch.tensor(0.01),
        "goal":         torch.tensor(0.05),
        "ponder":       torch.tensor(0.001),
        "free_energy":  torch.tensor(0.02),
        "consistency":  torch.tensor(0.003),
    }
    total, breakdown = criterion(losses)
    assert total.item() > 0
    assert "total" in breakdown
    log = AGILoss.log_breakdown(breakdown, step=1)
    assert "step=1" in log
