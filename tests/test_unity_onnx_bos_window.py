"""BOS retention parity for the stateless Unity ONNX window."""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


def _window_class():
    previous_ort = sys.modules.get("onnxruntime")
    if previous_ort is None:
        sys.modules["onnxruntime"] = types.ModuleType("onnxruntime")
    path = Path(__file__).resolve().parents[1] / "examples" / "unity" / "evaluate_onnx.py"
    spec = importlib.util.spec_from_file_location("unity_evaluate_onnx_test", path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        if previous_ort is None:
            sys.modules.pop("onnxruntime", None)
        else:
            sys.modules["onnxruntime"] = previous_ort
    return module.Window


def test_discard_masks_bos_after_first_action():
    window = _window_class()(1, 4, 2, 1, bos_cache_mode="discard")
    state0 = np.asarray([[1.0, 2.0]], np.float32)
    window.update(state0, np.zeros((1, 1), np.float32), is_bos=1.0)

    first = window.inputs()
    assert first["mask"].sum() == 1
    assert first["is_bos"][first["mask"].astype(bool)].item() == 1.0

    window.after_act()
    assert window.inputs()["mask"].sum() == 0

    state1 = np.asarray([[3.0, 4.0]], np.float32)
    action0 = np.asarray([[0.5]], np.float32)
    window.update(state1, action0, is_bos=0.0)
    second = window.inputs()
    assert second["mask"].sum() == 1
    assert second["is_bos"][second["mask"].astype(bool)].item() == 0.0
    # The project buffer pairs the previous state with its emitted action;
    # state1 remains in the trailing staged slot until the next update.
    np.testing.assert_array_equal(second["states"][0, -1], state0[0])
    np.testing.assert_array_equal(second["actions"][0, -1], action0[0])


def test_retain_keeps_bos_in_later_windows():
    window = _window_class()(1, 4, 2, 1, bos_cache_mode="retain")
    window.update(
        np.asarray([[1.0, 2.0]], np.float32),
        np.zeros((1, 1), np.float32),
        is_bos=1.0,
    )
    window.after_act()
    window.update(
        np.asarray([[3.0, 4.0]], np.float32),
        np.asarray([[0.5]], np.float32),
        is_bos=0.0,
    )

    context = window.inputs()
    assert context["mask"].sum() == 2
    assert context["is_bos"][context["mask"].astype(bool)].reshape(-1).tolist() == [1.0, 0.0]
