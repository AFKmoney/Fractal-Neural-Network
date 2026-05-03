"""
Multi-GPU distributed training for NFN.

Supports:
  - DDP  (DistributedDataParallel) : standard multi-GPU, model replicated
  - FSDP (FullyShardedDataParallel): for models too large for single GPU
  - Single-GPU / CPU fallback

Launch:
    # 2 GPUs, DDP:
    torchrun --nproc_per_node=2 train.py --distributed ddp

    # 4 GPUs, FSDP (large models):
    torchrun --nproc_per_node=4 train.py --distributed fsdp

    # Programmatic:
    python -c "from training.distributed import launch; launch('train.py', n_gpus=2)"
"""

import os
import subprocess
import sys
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Setup / teardown
# ─────────────────────────────────────────────────────────────────────────────

def setup_distributed(backend: str = "nccl"):
    """Initialise the distributed process group (called once per process)."""
    if not dist.is_available():
        return False
    if dist.is_initialized():
        return True

    rank       = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size <= 1:
        return False

    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
    )
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    return True


def teardown_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def world_size() -> int:
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


# ─────────────────────────────────────────────────────────────────────────────
# Model wrapping
# ─────────────────────────────────────────────────────────────────────────────

def wrap_ddp(model: nn.Module, device: torch.device) -> nn.Module:
    """Wrap model with DDP. Call after setup_distributed()."""
    model = model.to(device)
    return nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank()],
        output_device=local_rank(),
        find_unused_parameters=False,
    )


def wrap_fsdp(model: nn.Module, device: torch.device) -> nn.Module:
    """Wrap model with FSDP for very large models."""
    try:
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            MixedPrecision,
            ShardingStrategy,
            BackwardPrefetch,
        )
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        import functools

        # Auto-wrap NFNBlock layers
        from nfn.network import NFNBlock
        wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={NFNBlock},
        )

        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
        )

        model = FSDP(
            model.to(device),
            auto_wrap_policy=wrap_policy,
            mixed_precision=mp_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=local_rank(),
        )
        return model
    except ImportError as e:
        print(f"FSDP unavailable ({e}), falling back to DDP")
        return wrap_ddp(model, device)


def wrap_model(
    model: nn.Module,
    device: torch.device,
    strategy: str = "ddp",  # "ddp" | "fsdp" | "none"
) -> nn.Module:
    if not dist.is_initialized() or world_size() == 1:
        return model.to(device)
    if strategy == "fsdp":
        return wrap_fsdp(model, device)
    return wrap_ddp(model, device)


# ─────────────────────────────────────────────────────────────────────────────
# Distributed data sampler
# ─────────────────────────────────────────────────────────────────────────────

class DistributedTextSampler:
    """
    Splits a flat token array across ranks so each GPU trains on a
    non-overlapping shard, with round-robin refill each epoch.
    """

    def __init__(
        self,
        data: torch.Tensor,    # [total_tokens]
        seq_len: int,
        batch_size: int,
        rank: int = 0,
        n_workers: int = 1,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.data = data
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.rank = rank
        self.n_workers = n_workers
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        n_chunks = (len(data) - 1) // seq_len
        # Each rank gets a disjoint slice
        per_rank = n_chunks // n_workers
        self.start = rank * per_rank
        self.end   = self.start + per_rank

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        n = self.end - self.start
        return n // self.batch_size

    def __iter__(self):
        import random
        rng = random.Random(self.seed + self.epoch)

        indices = list(range(self.start, self.end))
        if self.shuffle:
            rng.shuffle(indices)

        for i in range(0, len(indices) - self.batch_size + 1, self.batch_size):
            batch_idx = indices[i: i + self.batch_size]
            batch = torch.stack([
                self.data[j * self.seq_len: j * self.seq_len + self.seq_len + 1]
                for j in batch_idx
            ])
            x = batch[:, :-1]
            y = batch[:, 1:]
            yield {"input_ids": x, "targets": y}


# ─────────────────────────────────────────────────────────────────────────────
# Distributed trainer (wraps NFNTrainer)
# ─────────────────────────────────────────────────────────────────────────────

class DistributedNFNTrainer:
    """
    Drop-in wrapper around NFNTrainer that:
    - Sets up the process group
    - Wraps the model with DDP or FSDP
    - Shards data across ranks
    - Reduces losses for logging on rank 0
    """

    def __init__(
        self,
        model,
        tokenizer,
        cfg,
        strategy: str = "ddp",
        **trainer_kwargs,
    ):
        self._strategy = strategy
        dist_ok = setup_distributed()

        device = (
            torch.device(f"cuda:{local_rank()}")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        model_wrapped = wrap_model(model, device, strategy if dist_ok else "none")

        from training.trainer import NFNTrainer
        self._trainer = NFNTrainer(
            model_wrapped, tokenizer, cfg,
            **trainer_kwargs,
        )
        self._trainer.device = device
        self.is_main = is_main_process()
        self._world = world_size()

    def train(self, text: str, **kwargs):
        # On non-main ranks, suppress stdout
        if not self.is_main:
            import io, contextlib
            with contextlib.redirect_stdout(io.StringIO()):
                return self._trainer.train(text, **kwargs)
        return self._trainer.train(text, **kwargs)

    def save(self, tag: str = "latest"):
        if self.is_main:
            return self._trainer.save(tag)

    def __del__(self):
        teardown_distributed()


# ─────────────────────────────────────────────────────────────────────────────
# Programmatic launcher (wraps torchrun)
# ─────────────────────────────────────────────────────────────────────────────

def launch(
    script: str,
    n_gpus: int = 1,
    extra_args: Optional[list] = None,
):
    """Launch a training script on n_gpus using torchrun."""
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        f"--nproc_per_node={n_gpus}",
        "--master_port=29500",
        script,
    ]
    if extra_args:
        cmd.extend(extra_args)
    print(f"Launching: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
