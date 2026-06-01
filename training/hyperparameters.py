"""
Causal-GPT-RL Training Hyperparameters

Hyperparameter schema for a Causal-GPT-RL training job. Instantiate
``Hyperparameters()`` to get the default training recipe, set the fields
you want to override, and submit the result as the job payload. Any
field left at its default is treated as "use the recipe default".

Quick-Reference for Domain-Specific Fields:
-------------------------------------------------------------------

dataset_ids          : (list)   Minari dataset identifiers to train on.
                                Required. Your datasets are supplied via the
                                job's training channel (bring-your-own).

env_id               : (str)    Optional Gymnasium environment id for
                                environment-based evaluation. Omit for
                                offline/data-only evaluation.

context_length       : (int)    RL trajectory context window passed to the
                                Hugging Face Transformers Llama backbone.
                                Applies both at training time and to the
                                exported model at inference.
-------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class Hyperparameters:

    # 1) Data - managed jobs are bring-your-own: data arrives via the training
    # channel, so dataset_ids is required.
    dataset_ids: Optional[list[str]] = None                     # required BYO Minari dataset ids
    env_id: Optional[str]  = None                               # optional Gymnasium env id for evaluation

    # 2) Training
    seed: int = 42                                              # reproducibility seed
    max_steps: int = 100_000                                    # optimizer updates; 0 is allowed for smoke tests
    batch_size: int = 128                                       # minibatch size
    gamma: float = 0.99                                         # RL discount factor
    td_lambda: float = 0.95                                     # TD(lambda) coefficient; JSON key "lambda" is also accepted

    # 3) Optimization
    learning_rate: float = 1e-4                                 # peak LR (after warmup)
    min_lr: float = 1e-6                                        # LR decay floor (absolute); 0 < min_lr <= learning_rate
    lr_scheduler_type: str = "cosine"                           # LR scheduler choice: "linear" | "cosine"
    warmup_ratio: float = 0.05                                  # fraction of training steps used for LR warmup
    max_grad_norm: float = 1.0                                  # gradient clipping threshold

    # 4) Network
    d_model: int = 256                                          # model width; maps to HF LlamaConfig.hidden_size
    num_heads: int = 8                                          # attention heads; d_model must be divisible by this
    num_layers: int = 4                                         # transformer layer count
    context_length: int = 32                                    # trajectory timesteps visible to the policy

    # -----------------------
    # Methods
    # -----------------------
    def set_config(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                print(f"Warning: No attribute '{k}' in Hyperparameters")

    def to_dict(self) -> dict:
        return asdict(self)
