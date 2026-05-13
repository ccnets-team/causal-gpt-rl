"""Stable public inference surface for the autoregressive policy.

This package exposes the pieces needed to run a trained model: bundle
loading/export, the high-level PolicyRunner, and the eval episode loop.

Training-time logic (online normalizer updates, optimizer/scheduler state,
dataset/return-range and sum-reward normalization, replay buffers, etc.) is
intentionally absent and lives elsewhere.

Author:
    PARK, Jun-Ho, junho@ccnets.org

Copyright (c) 2026 CCNets, Inc. All rights reserved.
"""
from .bundle import (
    BUNDLE_FORMAT_VERSION,
    convert_legacy_bundle_to_safetensors,
    export_bundle,
    load_runner,
    load_runner_from_hub,
)
from .evaluation import run_episodes
from .runner import PolicyRunner

__all__ = [
    "BUNDLE_FORMAT_VERSION",
    "PolicyRunner",
    "convert_legacy_bundle_to_safetensors",
    "export_bundle",
    "load_runner",
    "load_runner_from_hub",
    "run_episodes",
]
