"""Every shipped notebook keeps the conventions the others already follow.

Two of them are cheap to break and expensive to notice: a committed output cell
(which bakes the author's paths and a stale result into the diff), and a Colab
badge copied from a sibling notebook, which then opens the wrong file.
"""

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted(ROOT.glob("examples/**/*.ipynb"))

_BADGE_HREF = re.compile(
    r"https://colab\.research\.google\.com/github/[^/]+/[^/]+/blob/[^/]+/(\S+?\.ipynb)"
)


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_repository_ships_notebooks():
    assert NOTEBOOKS, "no notebooks found under examples/"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_a_notebook_is_committed_without_its_outputs(path):
    notebook = _load(path)
    assert notebook["nbformat"] == 4
    for index, cell in enumerate(notebook["cells"]):
        assert not cell.get("outputs"), f"{path.name} cell {index} carries output"
        assert cell.get("execution_count") is None, (
            f"{path.name} cell {index} carries an execution count"
        )


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_a_colab_badge_opens_its_own_notebook(path):
    source = json.dumps(_load(path)["cells"])
    targets = set(_BADGE_HREF.findall(source))
    if not targets:
        pytest.skip("no Colab badge")
    expected = path.relative_to(ROOT).as_posix()
    assert targets == {expected}, f"{path.name} badge points at {targets}"
