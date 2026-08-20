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

from ._internal.contract import ContractError, normalize_spec, validate_episode

# `spec.json` keys the packager reads. Everything else in the file is
# provenance, which `_internal.contract.normalize_spec` passes through untouched.
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
    """Record what a `PolicyRunner` drives, one episode file per env row.

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

    A batched runner records the same files from a vectorized env — one episode
    per row, `observe` taking the arrays `step` returns::

        runner = CollectionRunner(
            load_runner(bundle, num_envs=8), "raw/", episodes_per_row=1
        )
        observations, _ = venv.reset(seed=list(range(8)))
        runner.reset(observations)
        while not runner.all_rows_retired:
            actions = runner.act()
            observations, rewards, terms, truncs, _ = venv.step(actions)
            runner.observe(observations, rewards, terms, truncs)
        runner.close()

    Rows end at different steps, so each closes and writes on its own step;
    `ep_%06d.npz` are numbered in the order they finish, not by row.

    `episodes_per_row` is each row's share, and it is what makes a count exact.
    A row that has written its share is *retired*: it keeps being driven, because
    the batch is one forward and a single row cannot leave it, but nothing it
    does after that reaches a file — including the flush `close()` performs.
    Without a share, a run stopped on a total instead overshoots twice over:
    rows whose episodes are short churn through several while long ones are
    still on their first, and whatever is in flight when the total lands is
    flushed as a truncation.

    The share is also what keeps seeds meaningful. A vector env is seeded once,
    at `reset`, so only each row's *first* episode carries a seed the caller
    chose; every later one is the env restarting itself, and its trajectory
    depends on how many steps the policy took before it — which is exactly what
    differs between two runs being compared. `episodes_per_row=1` is therefore
    the form to record with when the seeds have to line up across runs, and it
    makes `num_envs` the episode count.

    Gymnasium's vector autoreset is `NEXT_STEP`: the step a row ends on carries
    its true final observation and flags, and the *following* step carries the
    new episode's first observation with a zero reward, no flags, and that row's
    action ignored. The recorder owns that one-step wait — it closes the row on
    the first, and on the second seeds a new episode without recording a
    transition, calling the runner's `reset_rows` so the ended episode leaves
    the row's context before it does. A batched run therefore needs no
    `final_observation` handling from the caller.

    The action recorded for a step is the one `act()` returned for the state
    last given, paired at this boundary rather than read back out of the model's
    context — that context token is `(state, previous action)`, one step off
    from what the contract wants, and holds model-space values besides.

    The terminal observation is recorded but not fed back: the contract needs
    `T + 1` observations, and the runner has no use for the last one. In a batch
    a row that ended does carry it for one step, until the `reset_rows` above
    wipes the row — the runner is fed the whole batch or none of it.

    Any entry point that leaves the recording state ambiguous is refused rather
    than inherited — `act(state)`, and an `observe()` before the first `reset()`
    — because both would drive the policy and drop the step without an error.

    Boundaries: `num_envs == 1` keeps the single-env contract exactly, autoreset
    included — a closed episode ends the run and the next `reset()` starts the
    next one, so a one-row vector env should use the loop above rather than the
    batched one. A single-env auto-resetting env must still hand `observe` the
    true final observation, since only the caller can reach it. And the action
    is recorded exactly as emitted, so perturbing it before `env.step` would
    record a policy that never ran.
    """

    def __init__(
        self,
        runner,
        out_dir: str | Path,
        *,
        bundle: Optional[str] = None,
        resume: bool = False,
        episodes_per_row: Optional[int] = None,
    ):
        self.num_envs = int(getattr(runner, "num_envs", 1))
        if self.num_envs <= 0:
            raise ContractError(f"num_envs must be > 0, got {self.num_envs}")
        # The batched flow restarts rows in place; without that call there is no
        # way to end one row's episode without ending every row's.
        if self.num_envs > 1 and not callable(getattr(runner, "reset_rows", None)):
            raise ContractError(
                f"Recording {self.num_envs} envs needs a runner with "
                "reset_rows(done_mask) to restart rows in place; this one has "
                "none. Record it one environment at a time."
            )
        self.runner = runner
        # Per-row episode boundaries only mean something with rows to stagger.
        # At one env the shipped single-env contract is kept unchanged.
        self._vectorized = self.num_envs > 1
        self.out_dir = Path(out_dir)

        if episodes_per_row is not None and int(episodes_per_row) < 1:
            raise ContractError(
                f"episodes_per_row must be >= 1, got {episodes_per_row}"
            )
        self.episodes_per_row = (
            None if episodes_per_row is None else int(episodes_per_row)
        )
        # What each row has written, and which rows are done writing. A retired
        # row keeps being driven — the batch is one forward and one row cannot
        # leave it — but nothing it does after its share reaches a file.
        self._row_episodes = np.zeros(self.num_envs, dtype=np.int64)
        self._retired = np.zeros(self.num_envs, dtype=bool)

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
        self._started = False
        self._open = np.zeros(self.num_envs, dtype=bool)
        # Rows whose episode ended last step; the next observation is the new
        # episode's first, not a transition of the old one.
        self._awaiting = np.zeros(self.num_envs, dtype=bool)
        self._pending_action: list[Optional[np.ndarray]] = [None] * self.num_envs
        self._observations: list[list[np.ndarray]] = [[] for _ in range(self.num_envs)]
        self._actions: list[list[np.ndarray]] = [[] for _ in range(self.num_envs)]
        self._rewards: list[list[float]] = [[] for _ in range(self.num_envs)]

    @property
    def row_episodes(self) -> np.ndarray:
        """How many episodes each row has written, as a copy."""
        return self._row_episodes.copy()

    @property
    def all_rows_retired(self) -> bool:
        """True once every row has written its `episodes_per_row` share.

        Always False without a share to reach, so a loop driven by this needs
        `episodes_per_row` set — which is the point: it is the only stop
        condition that lands on an exact count instead of overshooting one.
        """
        if self.episodes_per_row is None:
            return False
        return bool(self._retired.all())

    def _records(self, row: int) -> bool:
        """Whether this row's steps still reach a file."""
        return self._recording and not self._retired[row]

    # -- batch rows ------------------------------------------------------

    def _observation_rows(self, observation) -> list:
        """The batch's observation -> one per-env observation per row.

        The convention `PolicyRunner._format_state` reads, because the object
        handed here is the one passed straight through to the runner: a declared
        observation space means `num_envs` structured observations, a bundle
        without one means an array whose leading axis is the batch.
        """
        if self.num_envs == 1:
            return [observation]
        if self.runner.obs_space is None:
            arr = np.asarray(observation, dtype=np.float32)
            if arr.ndim < 1 or arr.shape[0] != self.num_envs:
                raise ContractError(
                    f"Expected observations for {self.num_envs} envs, got an "
                    f"array of shape {arr.shape}"
                )
            return [arr[i] for i in range(self.num_envs)]
        rows = list(observation)
        if len(rows) != self.num_envs:
            raise ContractError(
                f"Expected {self.num_envs} per-env observations, got {len(rows)}"
            )
        return rows

    def _action_rows(self, action) -> list:
        """`act()`'s return -> one env-ready action per row.

        Every action space the contract covers decodes to a batched form whose
        leading axis is the row: an array for the homogeneous families, a list
        of `num_envs` containers when the bundle declared a Tuple. The per-head
        list `PolicyRunner` returns for an undeclared mixed schedule is not one
        of them, and cannot arrive here — `_declared_action_space` refuses that
        bundle first. The length check is what keeps that reasoning falsifiable.
        """
        if self.num_envs == 1:
            return [action]
        if len(action) != self.num_envs:
            raise ContractError(
                f"act() returned {len(action)} actions for {self.num_envs} envs"
            )
        return [action[i] for i in range(self.num_envs)]

    def _step_values(self, values, what: str) -> np.ndarray:
        """A step's reward / flag, scalar or per row, as a `(num_envs,)` array."""
        arr = np.asarray(values).reshape(-1)
        if arr.shape[0] != self.num_envs:
            raise ContractError(
                f"Expected {what} for {self.num_envs} envs, got {arr.shape[0]}"
            )
        return arr

    # -- the loop --------------------------------------------------------

    def reset(self, observation, *, record: bool = True) -> None:
        """Start an episode on every row: flush what is open, then seed.

        `record` sits on this call because it is already the episode boundary —
        the same call clears the runner's context, so the two spans coincide.
        """
        self._close_open_rows(aborted=True)
        self.runner.reset(observation)
        self._started = True
        self._recording = bool(record)
        self._awaiting[:] = False
        self._pending_action = [None] * self.num_envs
        rows = self._observation_rows(observation) if self._recording else None
        for row in range(self.num_envs):
            self._open[row] = True
            if self._records(row):
                self._seed_row(row, rows[row])

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
        if not self._started:
            raise RuntimeError("Call reset(observation) before act().")
        if any(pending is not None for pending in self._pending_action):
            raise RuntimeError(
                "act() called twice in a row; each recorded action needs the "
                "observe(...) of what it did."
            )
        action = self.runner.act()
        if self._recording:
            rows = self._action_rows(action)
            for row in range(self.num_envs):
                # A row mid-autoreset has its action ignored by the env, and the
                # step it belongs to seeds the new episode rather than being a
                # transition of either one.
                if not self._awaiting[row] and self._records(row):
                    self._pending_action[row] = self._encode_action(rows[row])
        return action

    def observe(
        self,
        observation,
        reward: float = 0.0,
        terminated: bool = False,
        truncated: bool = False,
    ) -> None:
        """Record the step the last `act()` produced, and close rows on a flag.

        Batched, each argument carries one value per row; single-env, each is
        the scalar it has always been.
        """
        if not self._started:
            raise RuntimeError("Call reset(observation) before observe().")
        rewards = self._step_values(reward, "rewards")
        terminations = self._step_values(terminated, "terminations").astype(bool)
        truncations = self._step_values(truncated, "truncations").astype(bool)
        done = terminations | truncations

        # Rows that ended last step: this observation is their new episode's
        # first. Restart them in the runner before it is fed, so the ended
        # episode leaves their context instead of preceding the new one.
        restarting = self._awaiting.copy()
        if restarting.any():
            self.runner.reset_rows(restarting)
            self._awaiting[:] = False

        rows = self._observation_rows(observation) if self._recording else None
        for row in range(self.num_envs):
            if restarting[row]:
                self._open[row] = True
                if self._records(row):
                    self._seed_row(row, rows[row])
                continue
            if not self._records(row):
                continue
            if self._pending_action[row] is None:
                raise RuntimeError(
                    "observe() without a preceding act(); there is no action for "
                    "this observation to pair with."
                )
            self._observations[row].append(self._encode_observation(rows[row]))
            self._actions[row].append(self._pending_action[row])
            self._rewards[row].append(float(rewards[row]))
            self._pending_action[row] = None

        # A seeded row reports no flags on the step that seeds it, so a `done`
        # there would be the previous episode's; `restarting` masks it out.
        closing = done & ~restarting
        for row in np.flatnonzero(closing):
            self._close_row(
                int(row),
                terminated=bool(terminations[row]),
                truncated=bool(truncations[row]),
            )

        if self._vectorized:
            self._awaiting |= closing
        elif closing.all():
            # One env has no row to stagger against: the episode ending ends the
            # run, and the next `reset()` starts the next one.
            self._recording = False
            self._started = False

        if not done.all():
            # The runner has no use for a terminal state; the contract does. It
            # is fed the whole batch or none of it, so a row that ended keeps its
            # terminal observation until the `reset_rows` above wipes the row.
            self.runner.observe(observation)

    def close(self) -> None:
        """Flush the episodes still open, the way a mid-episode reset would."""
        self._close_open_rows(aborted=True)

    def __enter__(self) -> "CollectionRunner":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # -- episode files ---------------------------------------------------

    def _seed_row(self, row: int, observation) -> None:
        """Begin a row's episode at `observation`, discarding what it held."""
        self._observations[row] = [self._encode_observation(observation)]
        self._actions[row] = []
        self._rewards[row] = []
        self._pending_action[row] = None

    def _close_open_rows(self, *, aborted: bool = False) -> None:
        for row in range(self.num_envs):
            if self._open[row]:
                self._close_row(row, aborted=aborted)
        self._recording = False
        self._started = False
        self._awaiting[:] = False

    def _close_row(
        self,
        row: int,
        *,
        terminated: bool = False,
        truncated: bool = False,
        aborted: bool = False,
    ) -> None:
        if aborted and self._records(row):
            # A dangling act() has no transition to belong to, and an episode
            # cut short really is a truncation — the honest flag, not a stand-in.
            self._pending_action[row] = None
            terminated, truncated = False, True
        if self._records(row) and self._actions[row]:
            self._write_episode(row, terminated=terminated, truncated=truncated)
        self._open[row] = False
        self._observations[row] = []
        self._actions[row] = []
        self._rewards[row] = []
        self._pending_action[row] = None

    def _write_episode(self, row: int, *, terminated: bool, truncated: bool) -> None:
        transitions = len(self._actions[row])
        dtype = np.int64 if self._action_kind == "discrete" else np.float32
        terminations = np.zeros(transitions, dtype=bool)
        truncations = np.zeros(transitions, dtype=bool)
        terminations[-1] = terminated
        truncations[-1] = truncated
        arrays = {
            "observations": np.stack(self._observations[row]).astype(np.float32),
            "actions": np.stack(self._actions[row]).astype(dtype),
            "rewards": np.asarray(self._rewards[row], dtype=np.float32),
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
        self._row_episodes[row] += 1
        if (
            self.episodes_per_row is not None
            and self._row_episodes[row] >= self.episodes_per_row
        ):
            self._retired[row] = True

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
            "num_envs": self.num_envs,
            "episodes_per_row": self.episodes_per_row,
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
            f"obs_channels={self.spec['obs_channels']}, num_envs={self.num_envs}, "
            f"episodes_written={self.episodes_written}, runner={self.runner!r})"
        )
