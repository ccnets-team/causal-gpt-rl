"""Where the hand-delivered oracle bundles live.

A few tests cross-check this runtime against real exported bundles plus a
`meta.json` oracle. Those bundles are delivered by hand and kept out of git, so
their location is a property of the machine, not of the repository: point
`CAUSAL_GPT_RL_FIXTURE_BUNDLES` at the directory holding them.

Unset, the tests that need them skip — the same behaviour a fresh clone or CI
has always had.

Author:
    PARK, Jun-Ho, junho@ccnets.org

Copyright (c) 2026 CCNets, Inc. All rights reserved.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

ENV_VAR = "CAUSAL_GPT_RL_FIXTURE_BUNDLES"

_UNSET = (
    f"set {ENV_VAR} to the directory holding the delivered oracle bundles "
    "to run this check"
)


def fixture_bundles_dir() -> Optional[Path]:
    """The configured fixture directory, or `None` when unconfigured."""
    raw = os.environ.get(ENV_VAR, "").strip()
    return Path(raw).expanduser() if raw else None


def require_fixture_bundles(pytest_module) -> Path:
    """The fixture directory, skipping the calling test when unavailable."""
    base = fixture_bundles_dir()
    if base is None:
        pytest_module.skip(_UNSET)
    if not base.is_dir():
        pytest_module.skip(f"{ENV_VAR} points at a missing directory: {base}")
    return base


def require_output_dir() -> Path:
    """The fixture directory for a generator script, or a clear failure."""
    base = fixture_bundles_dir()
    if base is None:
        raise SystemExit(_UNSET.replace("run this check", "choose an output directory"))
    return base


__all__ = [
    "ENV_VAR",
    "fixture_bundles_dir",
    "require_fixture_bundles",
    "require_output_dir",
]
