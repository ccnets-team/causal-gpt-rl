"""Expose ML-Agents discrete branch logits for reproducible temperature tiers.

The stock ONNX action sampler is left untouched.  This adds one extra output,
``tier_logits``, with branches concatenated in branch-index order.  A distinct
name avoids colliding with the first branch's internal ``logits`` tensor.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper, shape_inference
import onnxruntime as ort


def expose(source: Path, destination: Path) -> tuple[int, ...]:
    model = shape_inference.infer_shapes(onnx.load(str(source)))
    graph = model.graph
    if any(output.name == "tier_logits" for output in graph.output):
        raise ValueError("tier_logits output already exists")
    value_info = {item.name: item for item in [*graph.input, *graph.value_info, *graph.output]}
    branches = []
    for node in graph.node:
        if node.op_type != "Gemm":
            continue
        for operand in node.input:
            if "_discrete_distribution.branches." in operand and operand.endswith(".weight"):
                index = int(operand.split(".branches.", 1)[1].split(".", 1)[0])
                name = node.output[0]
                width = value_info[name].type.tensor_type.shape.dim[-1].dim_value
                branches.append((index, name, width))
    branches.sort()
    if not branches or [row[0] for row in branches] != list(range(len(branches))):
        raise ValueError("Could not identify an ordered discrete branch set")
    names = [row[1] for row in branches]
    graph.node.append(
        helper.make_node("Concat" if len(names) > 1 else "Identity",
                         inputs=names, outputs=["tier_logits"],
                         **({"axis": 1} if len(names) > 1 else {}))
    )
    graph.output.append(helper.make_tensor_value_info(
        "tier_logits", TensorProto.FLOAT, ["batch", sum(row[2] for row in branches)]
    ))
    onnx.checker.check_model(model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(destination))
    session = ort.InferenceSession(str(destination), providers=["CPUExecutionProvider"])
    actual = next(item.shape[-1] for item in session.get_outputs()
                  if item.name == "tier_logits")
    if actual != sum(row[2] for row in branches):
        raise ValueError("Patched logits output has the wrong width")
    return tuple(row[2] for row in branches)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(f"branch widths: {expose(args.source, args.destination)}", flush=True)


if __name__ == "__main__":
    main()
