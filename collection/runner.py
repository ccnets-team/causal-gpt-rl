"""Record the episodes a `PolicyRunner` drives, in the input contract's shape.

`CollectionRunner` wraps a loaded bundle's runner and writes what it drives as
`ep_*.npz` plus a `spec.json`, so the policy that came out of training can
produce the dataset for the next round of it. The wrapped runner is not
modified: this is a recorder around the base form.

It integrates with exactly one policy source — ours — because that is the one
whose interface this repository owns and versions. Every other source keeps the
contract (`docs/01-the-input-contract.md`) instead of an integration.

Author:
    PARK, Jun-Ho, junho@ccnets.org

Copyright (c) 2026 CCNets, Inc. All rights reserved.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import numpy as np

from causal_gpt_rl import __version__ as _RUNTIME_VERSION
from causal_gpt_rl.inference.spaces import serialize_space

from ._contract import ContractError, normalize_spec, validate_episode

# `spec.json` keys the packager reads. Everything else in the file is
# provenance, which `_contract.normalize_spec` passes through untouched.
_CONTRACT_KEYS = (
    "action_kind",
    "obs_channels",
    "act_dim",
    "branches",
    "continuous_size",
    "act_low",
    "act_high",
)
_SPEC_FILENAME = "spec.json"
_EPISODE_STEM = re.compile(r"ep_(\d+)")


def _space_leaves(space):
    """Leaf spaces in `gym.spaces.flatten` order — the order values arrive in."""
    if isinstance(space, gym.spaces.Tuple):
        for sub in space.spaces:
            yield from _space_leaves(sub)
    elif isinstance(space, gym.spaces.Dict):
        for sub in space.spaces.values():
            yield from _space_leaves(sub)
    else:
        yield space


class _ObservationEncoder:
    """Declared observation space -> the contract's flat `[obs_width]` row.

    Records in **declared** (`gym.spaces.flatten`) order, not the model's
    continuous-first canonical order: the file describes the spaces `spec.json`
    declares, and re-deriving the model's layout from those is training's job.
    Each leaf becomes one `obs_channels` entry, so a multi-sensor observation
    stays split per sensor through packaging.
    """

    def __init__(self, obs_space, state_size: int):
        self.space = obs_space
        self.obs_channels = (
            [int(state_size)]
            if obs_space is None
            else [int(gym.spaces.flatdim(leaf)) for leaf in _space_leaves(obs_space)]
        )
        self.width = sum(self.obs_channels)
        if self.width != int(state_size):
            raise ContractError(
                f"observation space flattens to {self.width} values but the "
                f"bundle expects {int(state_size)}"
            )

    def __call__(self, observation) -> np.ndarray:
        if self.space is None:
            flat = np.asarray(observation, dtype=np.float32).reshape(-1)
        else:
            flat = np.asarray(
                gym.spaces.flatten(self.space, observation), dtype=np.float32
            )
        if flat.shape != (self.width,):
            raise ContractError(
                f"observation flattened to {flat.shape}, expected ({self.width},)"
            )
        return flat


def _continuous_fields(space: gym.spaces.Box, *, what: str):
    """1-D `Box` -> (`act_dim` / bounds spec fields, encode)."""
    if len(space.shape) != 1:
        raise ContractError(
            f"{what} must be a 1-D Box for the contract, got shape {space.shape}"
        )
    low = np.asarray(space.low, dtype=np.float32).reshape(-1)
    high = np.asarray(space.high, dtype=np.float32).reshape(-1)
    if not (np.all(np.isfinite(low)) and np.all(np.isfinite(high))):
        # The contract stores declared bounds and checks every recorded value
        # against them, so an unbounded action space has nothing to declare.
        raise ContractError(
            f"{what} has non-finite bounds ({space}); the contract needs finite "
            "act_low / act_high"
        )
    return (
        {
            "act_dim": int(space.shape[0]),
            "act_low": [float(v) for v in low],
            "act_high": [float(v) for v in high],
        },
        lambda action: np.asarray(action, dtype=np.float32).reshape(-1),
    )


def _categorical_fields(space):
    """Categorical leaf -> (`branches`, encode), or `(None, None)`.

    The contract stores one 0-based **index** per branch. A `Discrete` /
    `MultiDiscrete` `start` offset is env-facing only — the model is trained on
    0-based classes — so it is subtracted here and kept in the provenance's
    serialized space. `MultiBinary(n)` is `n` independent two-way branches,
    which is exactly `[2] * n` in that convention.
    """
    if isinstance(space, gym.spaces.Discrete):
        start = int(space.start)
        return [int(space.n)], lambda a: np.array([int(a) - start], dtype=np.int64)
    if isinstance(space, gym.spaces.MultiDiscrete):
        start = np.asarray(getattr(space, "start", 0), dtype=np.int64).reshape(-1)
        branches = [int(n) for n in np.asarray(space.nvec).reshape(-1)]
        return branches, lambda a: np.asarray(a, dtype=np.int64).reshape(-1) - start
    if isinstance(space, gym.spaces.MultiBinary):
        n = int(np.prod(space.shape))
        return [2] * n, lambda a: np.asarray(a, dtype=np.int64).reshape(-1)
    return None, None


def _action_encoding(space):
    """Declared action space -> (`spec.json` fields, encode) for the contract."""
    if isinstance(space, gym.spaces.Box):
        fields, encode = _continuous_fields(space, what="the action space")
        return {"action_kind": "continuous", **fields}, encode

    branches, encode = _categorical_fields(space)
    if branches is not None:
        return {"action_kind": "discrete", "branches": branches}, encode

    if isinstance(space, gym.spaces.Tuple) and len(space.spaces) == 2:
        continuous_space, categorical_space = space.spaces
        branches, encode_categorical = _categorical_fields(categorical_space)
        if isinstance(continuous_space, gym.spaces.Box) and branches is not None:
            fields, encode_continuous = _continuous_fields(
                continuous_space, what="the action space's continuous part"
            )

            def encode_hybrid(action):
                continuous, categorical = action
                # One array carries both parts, so the indices ride along as
                # integral floats; the packager casts them back.
                return np.concatenate(
                    [
                        encode_continuous(continuous),
                        encode_categorical(categorical).astype(np.float32),
                    ]
                )

            return (
                {
                    "action_kind": "hybrid",
                    "continuous_size": fields["act_dim"],
                    "branches": branches,
                    "act_low": fields["act_low"],
                    "act_high": fields["act_high"],
                },
                encode_hybrid,
            )

    raise ContractError(
        f"Action space {space} has no form in the input contract, which covers "
        "Box, Discrete, MultiDiscrete, MultiBinary, and Tuple(Box, categorical). "
        "Record it with your own loop against docs/01-the-input-contract.md."
    )


def _declared_action_space(runner):
    """The runner's action space, or the one its action heads imply.

    Bundles predating declared spaces carry only the per-head schedule, and the
    runner's decode for those emits exactly what a declared space with a zero
    `start` would — so a single-family schedule names the space unambiguously.
    A mixed schedule does not: with no declared container the runner returns a
    bare list of heads, which has no contract form either.
    """
    if runner.action_space is not None:
        return runner.action_space

    schedule = runner.action_schedule
    kinds = {head_type for head_type, _, _, _ in schedule}
    sizes = [size for _, size, _, _ in schedule]
    if kinds == {"continuous"}:
        bounds = [low for _, _, low, _ in schedule] + [
            high for _, _, _, high in schedule
        ]
        if any(bound is None for bound in bounds):
            raise ContractError(
                "The bundle declares no action_space and its continuous heads "
                "carry no bounds; the contract needs finite act_low / act_high."
            )
        return gym.spaces.Box(
            np.concatenate([low for _, _, low, _ in schedule]),
            np.concatenate([high for _, _, _, high in schedule]),
            dtype=np.float32,
        )
    if kinds == {"discrete"} and len(sizes) == 1:
        return gym.spaces.Discrete(sizes[0])
    if kinds == {"multi_discrete"}:
        return gym.spaces.MultiDiscrete(sizes)
    if kinds == {"multi_binary"}:
        return gym.spaces.MultiBinary(sum(sizes))
    raise ContractError(
        "The bundle declares no action_space and its action heads are mixed, so "
        "the emitted action has no declared structure to record against. Export "
        "the bundle with its Gymnasium spaces."
    )


class CollectionRunner:
    """Record what a `PolicyRunner` drives, one environment at a time.

    Wraps the runner and keeps its calls — `reset` / `act` / `observe`. What it
    adds is the episode file: the pairing, the terminal observation, and the two
    ways an episode can close, so the loop stays the one `docs/spaces.md`
    documents.

        runner = CollectionRunner(load_runner(bundle), "raw/")

        for episode in range(100):
            observation, _ = env.reset(seed=episode)
            runner.reset(observation, record=True)
            done = False
            while not done:
                action = runner.act()
                observation, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                runner.observe(observation, reward, terminated, truncated)

    The action recorded for a step is the one `act()` returned for the state
    last given, paired at this boundary rather than read back out of the model's
    context — that context token is `(state, previous action)`, one step off
    from what the contract wants, and holds model-space values besides.

    The terminal observation is recorded but not fed back: the contract needs
    `T + 1` observations, and the runner has no use for the last one.

    Any entry point that leaves the recording state ambiguous is refused rather
    than inherited — `act(state)`, and an `observe()` before the first `reset()`
    — because both would drive the policy and drop the step without an error.

    Boundaries: one environment (`num_envs == 1`); an auto-resetting env must
    hand `observe` the true final observation, since only the caller can still
    reach it; and the action is recorded exactly as emitted, so perturbing it
    before `env.step` would record a policy that never ran.
    """

    def __init__(
        self,
        runner,
        out_dir: str | Path,
        *,
        bundle: Optional[str] = None,
        resume: bool = False,
    ):
        if int(getattr(runner, "num_envs", 1)) != 1:
            raise ContractError(
                f"CollectionRunner records one environment, but this runner has "
                f"num_envs={runner.num_envs}. A batched rollout does not reduce "
                "identically to a single-env one, so it is not a faster way to "
                "record the same data."
            )
        self.runner = runner
        self.out_dir = Path(out_dir)

        self._encode_observation = _ObservationEncoder(
            runner.obs_space, runner.state_size
        )
        action_space = _declared_action_space(runner)
        action_fields, self._encode_action = _action_encoding(action_space)
        self._action_kind = action_fields["action_kind"]
        self.spec: dict[str, Any] = {
            "action_kind": self._action_kind,
            "obs_channels": list(self._encode_observation.obs_channels),
            **{k: v for k, v in action_fields.items() if k != "action_kind"},
        }
        # The declaration is checked before the run rather than after it: the
        # same check the packager makes, made once, here.
        self._normalized_spec = normalize_spec(
            self.spec,
            observation_width=self._encode_observation.width,
            action_width=self._action_width(),
        )

        self.out_dir.mkdir(parents=True, exist_ok=True)
        existing = self._existing_episode_indices()
        if existing and not resume:
            raise FileExistsError(
                f"{self.out_dir} already holds {len(existing)} episodes; pass "
                "resume=True to add to them, or choose another directory"
            )
        self._next_index = max(existing) + 1 if existing else 0
        self.episodes_written = 0
        self._write_spec(
            provenance=self._provenance(bundle=bundle, action_space=action_space)
        )

        self._recording = False
        self._episode_open = False
        self._pending_action: Optional[np.ndarray] = None
        self._observations: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self._rewards: list[float] = []

    # -- the loop --------------------------------------------------------

    def reset(self, observation, *, record: bool = True) -> None:
        """Start an episode: flush the previous one, then seed the runner.

        `record` sits on this call because it is already the episode boundary —
        the same call clears the runner's context, so the two spans coincide.
        """
        self._close_episode(aborted=True)
        self.runner.reset(observation)
        self._pending_action = None
        self._episode_open = True
        self._recording = bool(record)
        if self._recording:
            self._observations = [self._encode_observation(observation)]
            self._actions = []
            self._rewards = []

    def act(self, state=None):
        """The next action, unchanged — and the one this step records."""
        if state is not None:
            # `PolicyRunner.act(state)` is observe(state) then act(): it feeds
            # the runner while bypassing the reward and flag channel, so the
            # step would be driven but never recorded.
            raise RuntimeError(
                "act(state) would feed the runner without recording the step. "
                "Pass the observation to observe(observation, reward, "
                "terminated, truncated) and call act() with no argument."
            )
        if not self._episode_open:
            raise RuntimeError("Call reset(observation) before act().")
        if self._pending_action is not None:
            raise RuntimeError(
                "act() called twice in a row; each recorded action needs the "
                "observe(...) of what it did."
            )
        action = self.runner.act()
        if self._recording:
            self._pending_action = self._encode_action(action)
        return action

    def observe(
        self,
        observation,
        reward: float = 0.0,
        terminated: bool = False,
        truncated: bool = False,
    ) -> None:
        """Record the step the last `act()` produced, and close on a flag."""
        if not self._episode_open:
            raise RuntimeError("Call reset(observation) before observe().")
        done = bool(terminated) or bool(truncated)
        if self._recording:
            if self._pending_action is None:
                raise RuntimeError(
                    "observe() without a preceding act(); there is no action for "
                    "this observation to pair with."
                )
            self._observations.append(self._encode_observation(observation))
            self._actions.append(self._pending_action)
            self._rewards.append(float(reward))
            self._pending_action = None
        if not done:
            # The runner has no use for a terminal state; the contract does.
            self.runner.observe(observation)
        else:
            self._close_episode(terminated=bool(terminated), truncated=bool(truncated))

    def close(self) -> None:
        """Flush an episode still open, the way a mid-episode reset would."""
        self._close_episode(aborted=True)

    def __enter__(self) -> "CollectionRunner":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # -- episode files ---------------------------------------------------

    def _close_episode(
        self,
        *,
        terminated: bool = False,
        truncated: bool = False,
        aborted: bool = False,
    ) -> None:
        if aborted and self._recording:
            # A dangling act() has no transition to belong to, and an episode
            # cut short really is a truncation — the honest flag, not a stand-in.
            self._pending_action = None
            terminated, truncated = False, True
        if self._recording and self._actions:
            self._write_episode(terminated=terminated, truncated=truncated)
        self._recording = False
        self._episode_open = False
        self._observations = []
        self._actions = []
        self._rewards = []
        self._pending_action = None

    def _write_episode(self, *, terminated: bool, truncated: bool) -> None:
        transitions = len(self._actions)
        dtype = np.int64 if self._action_kind == "discrete" else np.float32
        terminations = np.zeros(transitions, dtype=bool)
        truncations = np.zeros(transitions, dtype=bool)
        terminations[-1] = terminated
        truncations[-1] = truncated
        arrays = {
            "observations": np.stack(self._observations).astype(np.float32),
            "actions": np.stack(self._actions).astype(dtype),
            "rewards": np.asarray(self._rewards, dtype=np.float32),
            "terminations": terminations,
            "truncations": truncations,
        }
        path = self.out_dir / f"ep_{self._next_index:06d}.npz"
        # Checked before it is written, so a recording run cannot leave behind a
        # directory the packager refuses.
        validate_episode(arrays, self._normalized_spec, source=path.name)
        np.savez(path, **arrays)
        self._next_index += 1
        self.episodes_written += 1

    def _existing_episode_indices(self) -> list[int]:
        indices = []
        for path in self.out_dir.glob("ep_*.npz"):
            match = _EPISODE_STEM.fullmatch(path.stem)
            if match is not None:
                indices.append(int(match.group(1)))
        return indices

    def _action_width(self) -> int:
        if self._action_kind == "continuous":
            return int(self.spec["act_dim"])
        if self._action_kind == "discrete":
            return len(self.spec["branches"])
        return int(self.spec["continuous_size"]) + len(self.spec["branches"])

    # -- provenance ------------------------------------------------------

    def _provenance(self, *, bundle: Optional[str], action_space) -> dict[str, Any]:
        """What produced this directory — which policy, run at what retention.

        `spec.json` ignores keys it does not know, and without these there is no
        way to tell afterwards what wrote the episodes.
        """
        obs_space = self.runner.obs_space
        return {
            "recorder": "collection.CollectionRunner",
            "causal_gpt_rl": _RUNTIME_VERSION,
            "bundle": bundle,
            "started_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "context_length": int(self.runner.context_length),
            "kv_cache_max_len": int(self.runner.kv_cache_max_len),
            "bos_cache_mode": getattr(self.runner, "bos_cache_mode", None),
            "use_windowed": bool(getattr(self.runner, "use_windowed", False)),
            "observation_space": (
                None if obs_space is None else serialize_space(obs_space)
            ),
            "action_space": serialize_space(action_space),
            "action_space_declared": self.runner.action_space is not None,
        }

    def _write_spec(self, *, provenance: dict[str, Any]) -> None:
        path = self.out_dir / _SPEC_FILENAME
        history: list[dict[str, Any]] = []
        if path.is_file():
            existing = json.loads(path.read_text(encoding="utf-8"))
            declared = {k: v for k, v in existing.items() if k in _CONTRACT_KEYS}
            if declared != self.spec:
                raise ContractError(
                    f"{path} declares {declared}, which is not what this runner "
                    f"records ({self.spec}). One raw directory holds one set of "
                    "spaces; record into another directory."
                )
            history = list(existing.get("provenance", []))
        # A list, because a resumed directory holds more than one recording run
        # and each one's retention is worth keeping.
        history.append(provenance)
        path.write_text(
            json.dumps({**self.spec, "provenance": history}, indent=2),
            encoding="utf-8",
        )

    # -- everything else is the wrapped runner's -------------------------

    def __getattr__(self, name: str):
        try:
            runner = self.__dict__["runner"]
        except KeyError:  # during __init__, before the runner is attached
            raise AttributeError(name) from None
        return getattr(runner, name)

    def __repr__(self) -> str:
        return (
            f"CollectionRunner(out_dir={str(self.out_dir)!r}, "
            f"action_kind={self._action_kind!r}, "
            f"obs_channels={self.spec['obs_channels']}, "
            f"episodes_written={self.episodes_written}, runner={self.runner!r})"
        )
