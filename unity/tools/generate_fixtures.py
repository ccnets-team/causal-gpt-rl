"""Generate deterministic ONNX Runtime fixtures for the Unity runtime.

The window and decode semantics are IMPORTED from the shipping reference
(``examples/unity/evaluate_onnx.py``) rather than reimplemented here. A local
copy could diverge from that reference and leave the Unity C# runtime validated
against stale truth. Importing removes that duplication; fixture schema, the C#
consumer, and changes in the reference itself still require explicit tests.

Emits:
  forward-input.json / forward-expected.json  one-forward parity
  rollout-expected.json                       N-step autoregressive parity
  rowreset-expected.json                      per-row episode restart (batch > 1)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

REFERENCE_RELPATH = Path("examples") / "unity"
REFERENCE_MODULE = "evaluate_onnx"


def _reference_dir() -> Path:
    """Locate the shipping reference by searching upward for its directory.

    Deliberately does NOT assume this file's depth in the tree: the same helper
    must work from the staging area and from its promoted location without edits.
    """
    for candidate in Path(__file__).resolve().parents:
        directory = candidate / REFERENCE_RELPATH
        if (directory / f"{REFERENCE_MODULE}.py").is_file():
            return directory
    raise RuntimeError(
        f"Could not locate {REFERENCE_RELPATH / (REFERENCE_MODULE + '.py')} in any "
        f"parent of {Path(__file__).resolve()}."
    )


sys.path.insert(0, str(_reference_dir()))
import evaluate_onnx as reference  # noqa: E402

Window = reference.Window
decode = reference._decode

ROLLOUT_STEPS = 10

# The bundle's declared bounds are passed to the reference decode rather than
# left to its ML-Agents default of [-1, 1]. A bundle whose environment uses
# another range -- MuJoCo's Humanoid actuators are [-0.4, 0.4] -- would
# otherwise be decoded against bounds it never declared, so the fixture and the
# C# decode (which reads the declared bounds) would disagree.
#
# A null bound is *unbounded* in a v2 bundle, not a shorthand for the unit
# interval. Reading it as [-1, 1] here would let this generator emit a fixture
# for a bundle `BundleValidator` then refuses to load, which is the one thing
# the fixtures exist to rule out: they must never disagree with the decode that
# will consume them.
FIXTURE_VERSION = "2"


def _bound(value, unbounded: float) -> float:
    """Read one declared bound. Null means unbounded, matching `BundleConfig`."""
    if value is None:
        return unbounded
    return float(value)


# Every file this generator can emit. Anything here that a given run does not
# produce is deleted from the output directory so a stale fixture from an
# earlier model can never be staged alongside a newer one.
OUTPUT_FILES = (
    "policy.onnx",
    "config.json",
    "forward-input.json",
    "forward-expected.json",
    "rollout-expected.json",
    "rowreset-expected.json",
)
CPU_TOLERANCE = 1.0e-4
GPU_TOLERANCE = 1.0e-3


def _shape(value) -> list[int]:
    shape = []
    for dimension in value:
        if not isinstance(dimension, int):
            raise ValueError(f"Fixture generation requires static shapes, got {value!r}")
        shape.append(dimension)
    return shape


def _tensor(name: str, value: np.ndarray) -> dict:
    return {
        "name": name,
        "shape": list(value.shape),
        "data": value.astype(np.float32, copy=False).reshape(-1).tolist(),
    }


def _action_layout(config: dict, action_size: int) -> dict:
    """Derive (continuous_size, branches, bounds) from the bundle's action specs.

    Mirrors how ``evaluate_onnx.main`` derives the same values from the live
    ML-Agents behavior spec, but sources them from the bundle so the Unity
    runtime never needs a live environment to know its own contract.
    """
    continuous_size = 0
    branches: list[int] = []
    low: list[float] = []
    high: list[float] = []
    for spec in config.get("action_specs", []):
        kind = spec.get("type")
        size = int(spec["size"])
        if kind == "continuous":
            if branches:
                # ``_decode`` emits every continuous column before any branch, so a
                # schedule that interleaves them would be silently reordered.
                raise ValueError(
                    "The shipping reference decodes continuous-first; this bundle "
                    "declares a continuous head after a discrete one and cannot be "
                    "expressed as a fixture."
                )
            continuous_size += size
            spec_low = spec.get("low") or [None] * size
            spec_high = spec.get("high") or [None] * size
            for index, value in enumerate(spec_low):
                floor = _bound(value, -math.inf)
                ceiling = _bound(spec_high[index], math.inf)
                if not math.isfinite(floor) or not math.isfinite(ceiling):
                    raise ValueError(
                        f"Continuous bound {index} is unbounded ([{floor}, {ceiling}]); "
                        "the decode has nothing to clip against, and the Unity runtime "
                        "refuses such a bundle rather than serving it."
                    )
                if not floor < ceiling:
                    raise ValueError(
                        f"Continuous bound {index} declares low {floor} and high "
                        f"{ceiling}; the decode would clip every value to a point."
                    )
                low.append(floor)
                high.append(ceiling)
        elif kind in ("discrete", "multi_discrete"):
            branches.append(size)
        else:
            raise ValueError(
                f"Unity fixtures do not cover action spec type {kind!r}. "
                "Supported: continuous, discrete, multi_discrete."
            )
    expected = continuous_size + sum(branches)
    if expected != action_size:
        raise ValueError(
            f"Bundle action specs total {expected} but the ONNX action width is "
            f"{action_size}."
        )
    return {
        "continuous_size": continuous_size,
        "branches": branches,
        "low": np.asarray(low, np.float32),
        "high": np.asarray(high, np.float32),
    }


def _check_action_container(config: dict) -> None:
    """Refuse a declared action space the reference would decode into the wrong action.

    A ``Dict``/``Tuple`` container is NOT such a case. ``evaluate_onnx`` never reads the
    bundle's container: it takes the branch layout from the live environment's action
    spec and emits a flat environment action, which is what ``env.step`` consumes. The
    published multi-agent bundles declare ``{"agents": {"agent_0": ...}}`` and run fine
    through that path, so refusing them here would refuse working models.

    What the container does decide is the class ``start`` offset. ``SpaceSpec`` drops it
    and ``PolicyRunner`` reads it back from the declared space instead (see
    ``causal_gpt_rl/inference/runner.py``, ``_discrete_start`` / ``_multi_discrete_start``).
    ``evaluate_onnx._decode`` emits bare 0-based argmax indices and never adds an offset,
    so a non-zero start anywhere in the tree would be decoded into the wrong environment
    action with no error.
    """
    _KNOWN_KINDS = {"Box", "Discrete", "MultiDiscrete", "MultiBinary", "Tuple", "Dict"}

    def visit(space, path: str) -> None:
        if not isinstance(space, dict):
            raise ValueError(f"Unrecognised action_container at {path}: {space!r}")

        kind = space.get("type")
        if kind not in _KNOWN_KINDS:
            raise ValueError(f"Unsupported action_container type {kind!r} at {path}.")

        start = space.get("start", 0)
        starts = start if isinstance(start, (list, tuple)) else [start]
        if any(int(value) != 0 for value in starts):
            raise ValueError(
                f"Bundle declares a {kind} action container with start={start} at "
                f"{path}; the shipping reference emits 0-based indices and never adds "
                "the offset, so the fixture would record the wrong environment action."
            )

        leaves = space.get("spaces")
        if kind == "Tuple":
            if not isinstance(leaves, list):
                raise ValueError(f"Tuple container at {path} is missing its 'spaces' list.")
            for index, leaf in enumerate(leaves):
                visit(leaf, f"{path}[{index}]")
        elif kind == "Dict":
            # Serialized as ordered [key, subspace] pairs so key order round-trips.
            if not isinstance(leaves, list):
                raise ValueError(f"Dict container at {path} is missing its 'spaces' list.")
            for pair in leaves:
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    raise ValueError(
                        f"Dict container at {path} must serialize leaves as "
                        f"[key, subspace] pairs; got {pair!r}."
                    )
                key, leaf = pair
                visit(leaf, f"{path}.{key}")

    container = config.get("action_container")
    if container is None:
        return
    visit(container, "action_container")


def _env_action_width(layout: dict) -> int:
    return layout["continuous_size"] + len(layout["branches"])


def _window_tensors(window: Window) -> dict:
    return {name: value.copy() for name, value in window.inputs().items()}


# Fields the C# runtime is responsible for maintaining across a decision. A
# fixture that does not react to them constrains nothing about that bookkeeping,
# so `_assert_bookkeeping_visible` corrupts each one wholesale and requires the
# output to move by more than the tolerance the fixture ships.
BOOKKEEPING_PROBES = (("actions", 0.0), ("is_bos", 1.0), ("mask", 1.0))


def _state_normalizer(onnx_path: Path):
    """Read the bundle's embedded state normalizer out of the ONNX graph.

    A v2 bundle folds ``(states - mean) / std`` into the graph, so the statistics
    the model was trained under are available without a second generator input.
    Best-effort by design -- a bundle exported without normalization has no such
    pair. What makes a miss safe rather than silent is the sensitivity gate: a
    fixture drawn from the wrong distribution fails there instead of shipping.
    """
    graph = onnx.load(str(onnx_path)).graph
    subtract = next(
        (n for n in graph.node if n.op_type == "Sub" and n.input and n.input[0] == "states"),
        None,
    )
    if subtract is None:
        return None
    divide = next(
        (n for n in graph.node
         if n.op_type == "Div" and n.input and n.input[0] == subtract.output[0]),
        None,
    )
    if divide is None:
        return None
    constants = {i.name: i for i in graph.initializer}
    if subtract.input[1] not in constants or divide.input[1] not in constants:
        return None
    mean = np.asarray(numpy_helper.to_array(constants[subtract.input[1]]), np.float32)
    std = np.asarray(numpy_helper.to_array(constants[divide.input[1]]), np.float32)
    return mean.reshape(-1), std.reshape(-1)


def _draw_states(rng, shape, stats, fallback_sigma: float) -> np.ndarray:
    """Draw fixture observations the model was actually trained to see.

    Drawing from a fixed spread instead puts every channel the training data held
    constant far outside its recorded range. Those channels normalize by a std of
    ``sqrt(0) + 1e-8``, so the embedded normalizer emits ~1e8, the residual stream
    swamps every attention and MLP output below float32 precision, and the graph
    collapses into a function of the newest state alone.
    """
    if stats is None:
        return rng.normal(0.0, fallback_sigma, shape).astype(np.float32)
    mean, std = stats
    return (mean + std * rng.normal(0.0, 1.0, shape)).astype(np.float32)


def _tensors_to_arrays(entries) -> dict:
    return {
        entry["name"]: np.asarray(entry["data"], np.float32).reshape(entry["shape"])
        for entry in entries
    }


def _assert_bookkeeping_visible(session, inputs: dict, label: str) -> list[float]:
    """Refuse a fixture whose output ignores the autoregressive bookkeeping.

    A parity fixture only constrains the runtime where the graph reacts to what
    the runtime maintains. Where it does not, `AutoregressiveParity` and
    `RowReset` pass no matter how wrong the feedback action, the BOS flag, or the
    padding mask are -- they degrade into a state-in/action-out check. Measured,
    not assumed: every published bundle looks identical from the outside.
    """
    baseline = session.run(["action"], inputs)[0].astype(np.float32, copy=False)
    deltas = []
    for name, value in BOOKKEEPING_PROBES:
        corrupted = dict(inputs)
        corrupted[name] = np.full_like(inputs[name], value)
        output = session.run(["action"], corrupted)[0].astype(np.float32, copy=False)
        deltas.append(float(np.abs(output - baseline).max()))
    if max(deltas) < CPU_TOLERANCE:
        raise ValueError(
            f"The {label} fixture does not observe the autoregressive bookkeeping: "
            f"zeroing actions moves the output by {deltas[0]:.2e}, forcing is_bos=1 "
            f"by {deltas[1]:.2e}, and unmasking the window by {deltas[2]:.2e} -- all "
            f"below the {CPU_TOLERANCE:.0e} tolerance this fixture ships with. "
            "Emitting it would record parity the C# runtime cannot fail."
        )
    return deltas


def _rollout(
    session: ort.InferenceSession,
    shapes: dict,
    layout: dict,
    config: dict,
    rng: np.random.Generator,
    reset_schedule: dict[int, list[int]] | None = None,
    *,
    stats=None,
) -> dict:
    """Replay a deterministic rollout through the reference window.

    ``reset_schedule`` maps a step index to the rows whose episode ends on that
    step. Those rows follow ``PolicyRunner.reset_rows`` semantics: the row's
    buffered trajectory is wiped, its carried feedback action is zeroed, and the
    row is marked BOS for the NEXT observe rather than immediately.
    """
    batch, context, state_size = shapes["states"]
    action_size = shapes["actions"][2]
    bos_cache_mode = config.get("serving", {}).get("bos_cache_mode", "discard")

    window = Window(batch, context, state_size, action_size, bos_cache_mode=bos_cache_mode)
    initial_state = _draw_states(rng, (batch, state_size), stats, 0.35)
    window.update(initial_state, np.zeros((batch, action_size), np.float32), is_bos=1.0)

    steps = []
    for index in range(ROLLOUT_STEPS):
        inputs = _window_tensors(window)
        raw = session.run(["action"], inputs)[0].astype(np.float32, copy=False)
        env_action, feedback = decode(
            raw,
            layout["continuous_size"],
            layout["branches"],
            low=layout["low"] if layout["continuous_size"] else None,
            high=layout["high"] if layout["continuous_size"] else None,
        )

        window.after_act()

        reset_rows = list((reset_schedule or {}).get(index, []))
        if reset_rows:
            # PolicyRunner.reset_rows: wipe the row's context, disown its history,
            # zero its carried action, and seed BOS for the next observe.
            rows = np.asarray(reset_rows, dtype=np.int64)
            window.states[rows] = 0.0
            window.actions[rows] = 0.0
            window.mask[rows] = 0.0
            window.is_bos[rows] = 1.0
            feedback = feedback.copy()
            feedback[rows] = 0.0

        is_bos = np.zeros((batch,), np.float32)
        if reset_rows:
            is_bos[np.asarray(reset_rows, dtype=np.int64)] = 1.0

        next_state = _draw_states(rng, (batch, state_size), stats, 0.35)
        window.update(next_state, feedback, is_bos=is_bos)
        updated = _window_tensors(window)

        steps.append(
            {
                "index": index,
                "reset_rows": reset_rows,
                "inputs": [_tensor(name, inputs[name]) for name in ("states", "actions", "is_bos", "mask")],
                "raw_output": _tensor("action", raw),
                "feedback_action": _tensor("feedback_action", feedback),
                "environment_action": _tensor("environment_action", env_action),
                "next_state": _tensor("next_state", next_state),
                "updated_inputs": [_tensor(name, updated[name]) for name in ("states", "actions", "is_bos", "mask")],
            }
        )

    return {
        "fixture_version": FIXTURE_VERSION,
        "reference_backend": "onnxruntime:CPUExecutionProvider",
        "reference_source": "examples/unity/evaluate_onnx.py",
        "bos_cache_mode": bos_cache_mode,
        "continuous_size": layout["continuous_size"],
        "branches": layout["branches"],
        "env_action_width": _env_action_width(layout),
        "action_low": layout["low"].tolist(),
        "action_high": layout["high"].tolist(),
        "initial_state": _tensor("initial_state", initial_state),
        "cpu_max_abs_tolerance": CPU_TOLERANCE,
        "gpu_max_abs_tolerance": GPU_TOLERANCE,
        "steps": steps,
    }


def _forward_documents(session, shapes, onnx_path: Path, *, stats=None) -> tuple[dict, dict]:
    """One forward over a partially filled window (exercises the padding mask)."""
    rng = np.random.default_rng(20260821)
    states = _draw_states(rng, shapes["states"], stats, 0.25)
    actions = rng.uniform(-0.75, 0.75, shapes["actions"]).astype(np.float32)
    is_bos = np.zeros(shapes["is_bos"], dtype=np.float32)
    mask = np.zeros(shapes["mask"], dtype=np.float32)

    context = shapes["mask"][1]
    active = min(8, context)
    first_active = context - active
    mask[:, first_active:] = 1.0
    is_bos[:, first_active, 0] = 1.0

    feed = {"states": states, "actions": actions, "is_bos": is_bos, "mask": mask}
    output = session.run(["action"], feed)[0].astype(np.float32, copy=False)
    digest = hashlib.sha256(onnx_path.read_bytes()).hexdigest().upper()

    return (
        {
            "fixture_version": FIXTURE_VERSION,
            "onnx_sha256": digest,
            "inputs": [_tensor(name, feed[name]) for name in ("states", "actions", "is_bos", "mask")],
        },
        {
            "fixture_version": FIXTURE_VERSION,
            "reference_backend": "onnxruntime:CPUExecutionProvider",
            "onnx_sha256": digest,
            "output": _tensor("action", output),
            "cpu_max_abs_tolerance": CPU_TOLERANCE,
            "gpu_max_abs_tolerance": GPU_TOLERANCE,
        },
    )


def _copy_input(source: Path, target: Path) -> None:
    """Copy an input into the fixture set, tolerating an in-place refresh.

    Re-running against the fixtures directory itself is the natural way to
    refresh a fixture set after the generator changes, and there the source and
    target are the same file.
    """
    if source.resolve() == target.resolve():
        return
    shutil.copy2(source, target)


def _write(path: Path, document: dict) -> None:
    path.write_text(
        json.dumps(document, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )


def generate(onnx_path: Path, config_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    config = json.loads(config_path.read_text(encoding="utf-8"))

    shapes = {value.name: _shape(value.shape) for value in session.get_inputs()}
    required = {"states", "actions", "is_bos", "mask"}
    if set(shapes) != required:
        raise ValueError(f"Expected ONNX inputs {sorted(required)}, got {sorted(shapes)}")
    outputs = session.get_outputs()
    if not outputs or outputs[0].name != "action":
        raise ValueError("Expected an ONNX output named 'action'.")

    batch, context, state_size = shapes["states"]
    action_size = shapes["actions"][2]
    if int(config.get("context_length", context)) != context:
        raise ValueError(
            f"Bundle context_length {config.get('context_length')} != ONNX context {context}"
        )
    _check_action_container(config)
    layout = _action_layout(config, action_size)

    stats = _state_normalizer(onnx_path)
    input_document, expected_document = _forward_documents(
        session, shapes, onnx_path, stats=stats
    )
    rollout = _rollout(
        session, shapes, layout, config, np.random.default_rng(20260821001), stats=stats
    )
    forward_deltas = _assert_bookkeeping_visible(
        session, _tensors_to_arrays(input_document["inputs"]), "forward"
    )
    rollout_deltas = _assert_bookkeeping_visible(
        session, _tensors_to_arrays(rollout["steps"][-1]["inputs"]), "rollout"
    )

    _copy_input(onnx_path, output_dir / "policy.onnx")
    _copy_input(config_path, output_dir / "config.json")
    _write(output_dir / "forward-input.json", input_document)
    _write(output_dir / "forward-expected.json", expected_document)
    _write(output_dir / "rollout-expected.json", rollout)

    produced = {"policy.onnx", "config.json", "forward-input.json",
                "forward-expected.json", "rollout-expected.json"}

    rowreset = None
    if batch > 1:
        # Deterministic staggered restarts: never all rows at once, and one step
        # where two rows restart together while the rest keep their history.
        schedule = {3: [0], 5: [1, batch - 1]}
        rowreset = _rollout(
            session, shapes, layout, config, np.random.default_rng(20260821002), schedule,
            stats=stats,
        )
        rowreset["reset_schedule"] = {str(k): v for k, v in schedule.items()}
        _write(output_dir / "rowreset-expected.json", rowreset)
        produced.add("rowreset-expected.json")

    # A previous run against a different model may have left fixtures this run
    # does not produce (e.g. rowreset from a batched policy followed by a batch-1
    # one). Staging copies whatever it finds, so a stale file would travel into
    # the test project and be validated against the wrong graph.
    for stale in sorted(set(OUTPUT_FILES) - produced):
        path = output_dir / stale
        if path.exists():
            path.unlink()
            print(f"removed stale:  {stale}")

    print(f"fixture:        {output_dir}")
    print(f"onnx_sha256:    {input_document['onnx_sha256']}")
    print(f"shapes:         batch={batch} context={context} state={state_size} action={action_size}")
    print(
        f"action layout:  continuous={layout['continuous_size']} "
        f"branches={layout['branches']} env_width={_env_action_width(layout)}"
    )
    print(f"bos_cache_mode: {rollout['bos_cache_mode']}")
    print(f"rollout steps:  {len(rollout['steps'])}")
    print(f"rowreset:       {'yes' if rowreset else 'skipped (batch == 1)'}")
    print(
        f"state draw:     "
        + ("bundle normalizer stats" if stats is not None else "N(0, sigma) -- no embedded normalizer")
    )
    print(
        "bookkeeping:    forward "
        + " / ".join(f"{d:.2e}" for d in forward_deltas)
        + "  rollout "
        + " / ".join(f"{d:.2e}" for d in rollout_deltas)
        + f"  (actions / is_bos / mask, tolerance {CPU_TOLERANCE:.0e})"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    generate(args.onnx.resolve(), args.config.resolve(), args.out.resolve())


if __name__ == "__main__":
    main()
