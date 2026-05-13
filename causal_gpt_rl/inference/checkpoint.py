"""Inference-only checkpoint utilities.

Loads model weights and state-normalizer statistics from a training
checkpoint. Optimizer / scheduler / sum-reward / dataset-return-range
state are training-only and are intentionally ignored.

Author:
    PARK, Jun-Ho, junho@ccnets.org

Copyright (c) 2026 CCNets, Inc. All rights reserved.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from .state_normalizer import StateNormalizer


def load_inference_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> dict:
    """Load a training checkpoint and return only inference-relevant pieces.

    Returns a dict with at least `model_state`, plus `state_normalizer_state`
    if the checkpoint includes one.
    """
    checkpoint = torch.load(str(path), map_location=map_location)
    if "model_state" not in checkpoint:
        raise KeyError(f"Checkpoint at {path} has no 'model_state' entry.")

    out: dict = {"model_state": checkpoint["model_state"]}
    if "state_normalizer_state" in checkpoint:
        out["state_normalizer_state"] = checkpoint["state_normalizer_state"]
    return out


def load_state_normalizer(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> Optional[StateNormalizer]:
    """Convenience helper: load only the StateNormalizer from a checkpoint.

    Returns None when the checkpoint has no normalizer state.
    """
    ckpt = load_inference_checkpoint(path, map_location=map_location)
    state = ckpt.get("state_normalizer_state")
    if state is None:
        return None
    return StateNormalizer.from_state_dict(state)
