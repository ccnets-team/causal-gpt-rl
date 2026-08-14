"""`docs/api.md` is checked against the code it documents.

A reference that drifts is worse than none — a reader trusts it and writes code
against a signature that no longer exists. These tests make the document
self-verifying: every signature block in it is parsed and compared to
`inspect.signature`, the documented import list is compared to the package's
`__all__`, and the documented return keys, attributes, and exceptions are
exercised against a real runner.

Adding a parameter to a public function therefore fails here until the document
is updated, which is the point.
"""
import ast
import inspect
import re
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

import causal_gpt_rl.inference as api
from causal_gpt_rl.inference.runner import PolicyRunner
from causal_gpt_rl.inference.spaces import extract_data_specs_from_space
from causal_gpt_rl.model.autoregressive_model import AutoregressiveModel
from causal_gpt_rl.model.schema import ModelConfig, SpaceSpec

DOC = Path(__file__).resolve().parents[1] / "docs" / "api.md"

_PY_BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)


def _python_blocks() -> list[str]:
    return [m.group(1) for m in _PY_BLOCK.finditer(DOC.read_text(encoding="utf-8"))]


def _as_signature(block: str):
    """Return the parsed FunctionDef for a bare signature block, else None.

    A signature block is prose-free — `name(args) -> ret` — so prefixing `def`
    turns it into a parseable definition. Example blocks (assignments, loops,
    the import list) fail to parse and are skipped.
    """
    try:
        tree = ast.parse("def " + block.strip() + ":\n    pass")
    except SyntaxError:
        return None
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        return None
    return tree.body[0]


def _documented_params(node: ast.FunctionDef) -> list[tuple]:
    """(name, kind, default) triples in declaration order.

    `default` is the literal value, or `inspect.Parameter.empty` when the
    document shows none.
    """
    empty = inspect.Parameter.empty
    out = []
    args = node.args
    pos_defaults = [None] * (len(args.args) - len(args.defaults)) + list(args.defaults)
    for arg, default in zip(args.args, pos_defaults):
        value = empty if default is None else ast.literal_eval(default)
        out.append((arg.arg, "positional", value))
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        value = empty if default is None else ast.literal_eval(default)
        out.append((arg.arg, "keyword", value))
    return out


def _actual_params(obj) -> list[tuple]:
    empty = inspect.Parameter.empty
    out = []
    for name, param in inspect.signature(obj).parameters.items():
        if name == "self":
            continue
        kind = (
            "keyword"
            if param.kind is inspect.Parameter.KEYWORD_ONLY
            else "positional"
        )
        out.append((name, kind, param.default))
    return out


def _resolve(name: str):
    """Map a documented symbol to the module function or runner method it names."""
    if hasattr(api, name):
        return getattr(api, name)
    if hasattr(PolicyRunner, name):
        return getattr(PolicyRunner, name)
    return None


def _model() -> AutoregressiveModel:
    torch.manual_seed(0)
    return AutoregressiveModel(
        ModelConfig(d_model=32, num_heads=4),
        state_specs=[
            SpaceSpec(
                type="continuous",
                size=2,
                dtype=torch.float32,
                low=[-1.0, -1.0],
                high=[1.0, 1.0],
            )
        ],
        action_specs=extract_data_specs_from_space(
            gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        ),
        device=torch.device("cpu"),
    )


def _runner(num_envs: int = 1) -> PolicyRunner:
    return PolicyRunner(
        model=_model(),
        action_schedule=[("continuous", 2, None, None)],
        state_size=2,
        context_length=6,
        num_envs=num_envs,
    )


class _StubEnv:
    """Minimal Gymnasium-shaped env: fixed-length episodes, unit reward."""

    def __init__(self, episode_len: int = 3):
        self.episode_len = episode_len
        self._t = 0

    def reset(self, **kwargs):
        self._t = 0
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action):
        self._t += 1
        done = self._t >= self.episode_len
        return np.zeros(2, dtype=np.float32), 1.0, done, False, {}


def test_documented_signature_blocks_are_recognized():
    """Guard the parser itself: the document must still contain signatures.

    Without this, a rename that stops every block from parsing would leave the
    signature test vacuously green.
    """
    named = [
        node.name for node in map(_as_signature, _python_blocks()) if node is not None
    ]
    assert len(named) >= 10, f"only {len(named)} signature blocks parsed: {named}"


@pytest.mark.parametrize(
    "block", [b for b in _python_blocks() if _as_signature(b) is not None]
)
def test_documented_signature_matches_code(block):
    node = _as_signature(block)
    obj = _resolve(node.name)
    assert obj is not None, f"docs/api.md documents unknown symbol {node.name!r}"
    assert _documented_params(node) == _actual_params(obj), (
        f"docs/api.md signature for {node.name!r} does not match the code"
    )


def test_documented_imports_match_public_surface():
    """The import block at the top of the document is the package's `__all__`."""
    block = next(b for b in _python_blocks() if b.startswith("from causal_gpt_rl"))
    tree = ast.parse(block)
    documented = {alias.name for alias in tree.body[0].names}
    assert documented == set(api.__all__)


def test_documented_attributes_exist():
    """Every row of the PolicyRunner attribute table resolves on a real runner."""
    documented = _doc_table_column("| Attribute | Description |")
    runner = _runner()
    missing = [name for name in documented if not hasattr(runner, name)]
    assert not missing, f"documented attributes absent from PolicyRunner: {missing}"
    assert "num_envs" in documented, "attribute table lost its rows"


def _doc_table_column(header: str) -> list[str]:
    """First-column entries of the markdown table introduced by `header`."""
    lines = DOC.read_text(encoding="utf-8").splitlines()
    start = lines.index(header) + 2  # skip the header and its separator row
    out = []
    for line in lines[start:]:
        if not line.startswith("|"):
            break
        cell = line.split("|")[1].strip()
        out.extend(part.strip().strip("`") for part in cell.split("/"))
    return out


def test_run_episodes_returns_documented_keys():
    documented = {
        "num_episodes",
        "returns",
        "lengths",
        "return_mean",
        "return_std",
        "length_mean",
        "length_std",
    }
    stats = api.run_episodes(_StubEnv(), _runner(), num_episodes=2, seed=0)
    assert set(stats) == documented


def test_documented_exceptions():
    """Each exception the document names is raised where it says it is."""
    with pytest.raises(FileNotFoundError):
        api.load_runner(Path(__file__).parent / "no-such-bundle")

    runner = _runner()
    with pytest.raises(RuntimeError):
        runner.act()  # documented: act() without a state before reset()

    batched = _runner(num_envs=2)
    with pytest.raises(RuntimeError):
        batched.reset_rows(np.array([True, False]))
    with pytest.raises(RuntimeError):
        batched.add_rows(np.zeros((1, 2), dtype=np.float32))

    batched.reset(np.zeros((2, 2), dtype=np.float32))
    batched.act()
    with pytest.raises(ValueError):
        batched.reset_rows(np.array([True, False, True]))  # length != num_envs

    with pytest.raises(ValueError):
        api.run_episodes(_StubEnv(), _runner(), num_episodes=0)
    with pytest.raises(ValueError):
        api.run_episodes(_StubEnv(), _runner(num_envs=2), num_episodes=1)


def test_documented_bundle_format_version():
    block = next(b for b in _python_blocks() if b.startswith("BUNDLE_FORMAT_VERSION"))
    documented = ast.literal_eval(block.split("=")[1].strip())
    assert documented == api.BUNDLE_FORMAT_VERSION
