#!/usr/bin/env python3
"""Generate tiny ONNX models for tests and examples (requires `onnx`)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from onnx import TensorProto, helper, numpy_helper


def _write_linear(path: Path, in_dim: int, out_dim: int) -> None:
    """Single FC layer: ``Y = X @ W + B`` with ``X`` [1, in], ``W`` [in, out]."""
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, in_dim])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, out_dim])

    w = np.random.randn(in_dim, out_dim).astype(np.float32)
    b = np.random.randn(out_dim).astype(np.float32)

    w_init = numpy_helper.from_array(w, name="W")
    b_init = numpy_helper.from_array(b, name="B")

    matmul = helper.make_node("MatMul", ["X", "W"], ["pre"])
    add = helper.make_node("Add", ["pre", "B"], ["Y"])

    graph = helper.make_graph(
        [matmul, add],
        f"linear_{in_dim}_{out_dim}",
        [x],
        [y],
        initializer=[w_init, b_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    path.parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(model.SerializeToString())


def main() -> None:
    root = Path(__file__).resolve().parent / "sample_models"
    _write_linear(root / "dummy_high.onnx", 64, 64)
    _write_linear(root / "dummy_medium.onnx", 16, 16)
    _write_linear(root / "dummy_low.onnx", 4, 4)
    print(f"Wrote models under {root}")


if __name__ == "__main__":
    main()
