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
                                job's training channel (bring-your-own). The
                                target environment is read from each dataset's
                                Minari metadata; managed jobs always evaluate
                                offline (data-only), so no env id is supplied.

context_length       : (int)    RL trajectory context window passed to the
                                Hugging Face Transformers Llama backbone.
                                Applies both at training time and to the
                                exported model at inference. Managed entry
                                normalizes to 16/24/32/48/64, then may lower it
                                further within that set for dataset capacity.

gamma / td_lambda    : (float)  Managed bounded preferences. Effective ranges
                                are gamma=[0.98, 0.995] and
                                td_lambda=[0.90, 0.975].

dropout              : (float)  Strict [0.0, 1.0]; out-of-range values fail.

learning_rate        : (float)  Managed clamp: [1e-6, 1e-3].
min_lr               : (float)  Managed clamp: [1e-8, 1e-3], then capped at
                                effective learning_rate. Set equal to
                                learning_rate for constant LR after warmup.
lr_scheduler_type    : (str)    linear/cosine; unsupported values use cosine.

Network capacity     : (limit)  The customer-defined Llama backbone must not
                                exceed 20M parameters. Oversized requests fail
                                before data loading; Network values are never
                                adjusted.

archive_steps        : (list)   Extra training steps to preserve permanently,
                                on top of the automatic evenly-spaced archive
                                points. Each preserved step ships a resume
                                checkpoint and a loadable inference bundle that
                                are never rotated away. Invalid/unreachable
                                values are ignored, duplicates collapse, and
                                the 10 numerically largest valid unique steps
                                are retained after sorting.
-------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class Hyperparameters:

    # 1) Data - managed jobs are bring-your-own: data arrives via the training
    # channel, so dataset_ids is required. The target env is read from the
    # dataset's own Minari metadata (env_spec), so no env_id field is needed.
    dataset_ids: Optional[list[str]] = None                     # required BYO Minari dataset ids

    # 2) Training
    seed: int = 42                                              # reproducibility seed
    max_steps: int = 100_000                                    # optimizer updates; 0 is allowed for smoke tests
    batch_size: int = 128                                       # managed values: 32/64/128/256/512 + capacity fallback
    context_length: int = 32                                    # managed values: 16/24/32/48/64 + capacity fallback
    gamma: float = 0.99                                         # managed clamp: [0.98, 0.995]
    td_lambda: float = 0.95                                     # managed clamp: [0.90, 0.975]; "lambda" alias accepted

    # 3) Optimization
    learning_rate: float = 1e-4                                 # managed clamp: [1e-6, 1e-3]
    min_lr: float = 1e-6                                        # clamp [1e-8, 1e-3], then cap at learning_rate
    lr_scheduler_type: str = "cosine"                           # linear/cosine; unsupported -> cosine

    # 4) Network - customer-defined and strict: values are never rounded or
    # capacity-adjusted; invalid structure/ranges and backbones over 20M fail.
    num_layers: int = 4                                         # transformer layer count
    d_model: int = 256                                          # model width; HF LlamaConfig.hidden_size alias accepted
    num_heads: int = 8                                          # any positive integer; d_model divisibility required
    dropout: float = 0.05                                       # strict HF/PyTorch range: [0.0, 1.0]

    # 5) Checkpointing - evenly spaced archive points are automatic; these are
    # extra steps you want preserved permanently. Bounded because archive points
    # are never reclaimed from the training volume.
    archive_steps: Optional[list[int]] = None                   # 10 numerically largest valid unique steps are retained

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
        """Job payload, with list fields flattened to comma-separated strings.

        SageMaker hyperparameters are ``Map<String,String>``: a managed run reads
        ``/opt/ml/input/config/hyperparameters.json``, so whatever is submitted
        arrives as a string. A Python list submitted here would arrive as its
        ``repr`` -- ``"['a', 'b']"`` -- which the training job splits on commas
        into broken values. Comma-separated is the only form that survives the
        wire, so it is what this emits.

        Unset stays ``None``; the training job reads that as "use the default".
        """
        payload = asdict(self)
        for field in ("dataset_ids", "archive_steps"):
            payload[field] = _as_csv(payload[field])
        return payload


def _as_csv(value) -> Optional[str]:
    """Comma-join a list field, passing an already-joined string through."""
    if value is None:
        return None
    if isinstance(value, str):
        return value or None
    return ",".join(str(item) for item in value) or None
