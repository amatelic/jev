#!/usr/bin/env python
"""Quantize the exported kev ONNX graph to q4 + q4f16 (docs/TRAINING.md §5).

  q4    : int4 MatMul (block_size 32, symmetric, accuracy_level 4) — Node CPU default
  q4f16 : int4 FIRST, then fp16-convert the rest — WebGPU (order matters)

The pointer head (head_q/head_k projections, 1024->256) stays fp32: int4 on
the readout collapses calibration (docs/TRAINING.md §5/§9).

Usage (from vendor/kev):
  uv run python ../../scripts/quantize-kev-onnx.py ../../models/events-v1
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
from onnxconverter_common import float16
from onnxruntime.quantization import matmul_nbits_quantizer


def quantize_q4(src: Path, dst: Path, block_size=32):
    model = onnx.load(str(src))

    # MatMul nodes whose weight belongs to the pointer head keep fp32.
    head_nodes = [
        n.name for n in model.graph.node
        if n.op_type in ("MatMul", "Gemm") and "head" in (n.name or "")
    ]
    print(f"keeping fp32 (pointer head): {head_nodes or 'none matched — CHECK NODE NAMES'}")

    q = matmul_nbits_quantizer.MatMulNBitsQuantizer(
        model, bits=4, block_size=block_size, is_symmetric=True,
        accuracy_level=4, nodes_to_exclude=head_nodes or None,
    )
    q.process()
    onnx.save(q.model.model, str(dst))
    size = dst.stat().st_size
    print(f"wrote {dst} ({size / 1e6:.0f} MB" + (f" + {dst.with_suffix(dst.suffix + '_data').stat().st_size / 1e6:.0f} MB external" if dst.with_suffix(dst.suffix + '_data').exists() else "") + ")")


def quantize_q4f16(q4_src: Path, dst: Path):
    # int4 already applied to the MatMuls; convert the remaining fp32 ops to fp16.
    model = onnx.load(str(q4_src))
    fp16_model = float16.convert_float_to_float16(
        model, keep_io_types=True,
        node_block_list=None,
    )
    onnx.save(fp16_model, str(dst))
    size = dst.stat().st_size
    print(f"wrote {dst} ({size / 1e6:.0f} MB)")


def verify(path: Path):
    import onnxruntime as ort

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(path), sess_options, providers=["CPUExecutionProvider"])
    inputs = {i.name: i.type for i in session.get_inputs()}
    outputs = [(o.name, o.shape) for o in session.get_outputs()]
    print(f"verify {path.name}: inputs={inputs} outputs={outputs}")

    # smoke run
    ids = np.ones((1, 64), dtype=np.int64)
    att = np.ones((1, 64), dtype=np.int64)
    out = session.run(None, {"input_ids": ids, "attention_mask": att})
    assert out[0].shape == (1, 64), out[0].shape
    assert np.isfinite(out[0]).all(), "non-finite logits"
    print(f"  smoke run ✓ logits finite, shape {out[0].shape}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", help="directory containing model.onnx")
    ap.add_argument("--block-size", type=int, default=32)
    args = ap.parse_args()

    d = Path(args.dir)
    src = d / "model.onnx"
    if not src.exists():
        sys.exit(f"missing {src} — run export-kev-onnx.py first")

    q4 = d / "model_q4.onnx"
    quantize_q4(src, q4, block_size=args.block_size)
    verify(q4)

    q4f16 = d / "model_q4f16.onnx"
    quantize_q4f16(q4, q4f16)
    verify(q4f16)

    print("done — next: craft config.json + tokenizer files (scripts/package-kev-model.mjs)")


if __name__ == "__main__":
    main()
