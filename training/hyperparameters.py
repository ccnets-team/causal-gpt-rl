"""
Causal-GPT-RL Training Hyperparameters

Hyperparameter schema for a Causal-GPT-RL training job. Instantiate
``Hyperparameters()`` to get the default training recipe, set the fields
you want to override, and submit the result as the job payload. Any
field left at its default is treated as "use the recipe default".

Quick-Reference for Key Fields:
-------------------------------------------------------------------

data_source          : (str)    "byo" trains on your own Minari datasets
                                supplied via the job's training channel;
                                "minari_remote" downloads from the public
                                Minari registry.

dataset_ids          : (list)   Minari dataset identifiers. Required when
                                data_source="byo".

td_lambda            : (float)  TD(lambda) bootstrap coefficient used in
                                return estimation. Typically near 1.0 for
                                longer-horizon tasks. Accepted alias:
                                "lambd".

context_length       : (int)    Model context window: the trajectory length
                                the transformer operates over. Applies both
                                at training time and to the exported model at
                                inference.

tau                  : (float)  "Soft update" factor for the target network.
                                Lower means slower, more stable updates.

max_grad_norm        : (float)  Gradient clipping threshold. Lower means more
                                stable training.

scheduler_type       : (str)    LR scheduler after warmup. One of
                                "linear", "exponential", "cosine".

network_name         : (str)    Backbone architecture (e.g. "Llama").

d_model / num_heads  : (int)    Hidden dim and attention head count.
                                d_model must be divisible by num_heads.
-------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class Hyperparameters:
    # Data
    data_source: str = "byo"
    dataset_ids: Optional[list[str]] = None
    env_id: Optional[str] = None
    difficulties: list[str] = field(default_factory=lambda: ["simple", "medium"])
    no_download: bool = False

    # Run
    seed: int = 42
    device: Optional[str] = None
    max_iters: int = 200_000

    # Training
    batch_size: int = 128
    gamma: float = 0.99
    td_lambda: float = 0.95
    extra_td_ratio: float = 0.25
    init_log_std: float = -1.0

    # Optimization
    learning_rate: float = 1e-4
    tau: float = 0.005
    scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    lr_decay_rate: float = 0.01
    max_grad_norm: float = 1.0

    # Network
    network_name: str = "Llama"
    num_layers: int = 4
    d_model: int = 256
    num_heads: int = 8
    dropout: float = 0.05
    rope_theta: float = 1e3
    context_length: int = 32
    intermediate_size: Optional[int] = None
    max_position_embeddings: Optional[int] = None

    def set_config(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: No attribute '{key}' in Hyperparameters")

    def to_dict(self) -> dict:
        return asdict(self)
