#!/usr/bin/env python
"""Export a kev training run to the ONNX graph contract the open-jev JS
runtime loads (docs/TRAINING.md §2/§3/§5):

  inputs : input_ids [B,S] int64, attention_mask [B,S] int64  (nothing else)
  graph  : derives seg / block-causal mask / positions in-graph from the
           delimiter token ids, runs the (merged) Qwen3 backbone, applies
           the pointer head (</opt> hidden ⋅ <decide> hidden) * scale
  output : logits [B,S] — one scalar per token; the runtime reads the value
           at each option's </opt> position and softmaxes within the question

Usage (from the jev repo root, uses vendor/kev's uv env):
  cd vendor/kev && uv run python ../../scripts/export-kev-onnx.py runs/events-v1 ../../models/events-v1

Steps: merge LoRA → wrap → torch.onnx.export (opset 17) → parity check vs
the torch model on real records → leave fp32 model.onnx for quantization.
"""

import argparse
import sys
from pathlib import Path

VENDOR_KEV = Path(__file__).resolve().parent.parent / "vendor" / "kev"
sys.path.insert(0, str(VENDOR_KEV))

import torch
import torch.nn as nn

from kev.checkpoint import Checkpoint
from kev.model import SPECIAL, DecisionModel, encode  # noqa: E402


class KevExport(nn.Module):
    """The exported decision graph: (input_ids, attention_mask) -> logits [B,S]."""

    def __init__(self, lm, head, delim_ids, temperature=1.0):
        super().__init__()
        self.lm = lm
        self.head_q = head.q
        self.head_k = head.k
        self.register_buffer("scale", torch.tensor(head.scale))
        self.state_id, self.q_id, self.opt_id, self.end_id, self.decide_id = delim_ids
        self.register_buffer("temperature", torch.tensor(float(temperature)))

    def forward(self, input_ids, attention_mask):
        B, S = input_ids.shape
        # seg: 0 for state (before the first <|fim_middle|>), k for branch k
        seg = torch.cumsum((input_ids == self.q_id).to(torch.int64), dim=1)

        # block-causal additive mask: attend(i,j) iff j<=i and (seg[j]==0 or seg[j]==seg[i]),
        # padded keys masked (kev.model.branch_mask_batch semantics)
        causal = torch.tril(torch.ones(S, S, dtype=torch.bool, device=input_ids.device))
        same = (seg.unsqueeze(1) == seg.unsqueeze(2)) | (seg.unsqueeze(1) == 0)
        valid_key = attention_mask.unsqueeze(1).to(torch.bool)
        allow = causal.unsqueeze(0) & same & valid_key
        allow = allow | torch.eye(S, dtype=torch.bool, device=input_ids.device).unsqueeze(0)
        mask = torch.zeros(B, S, S, dtype=torch.float32, device=input_ids.device)
        mask = mask.masked_fill(~allow, torch.finfo(torch.float32).min).unsqueeze(1)

        pos = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, S)
        h = self.lm(input_ids=input_ids, position_ids=pos, attention_mask=mask).last_hidden_state

        # decide position for each token: the first <|fim_suffix|> at index >= i
        # (every </opt> precedes its branch's <decide>, which ends the branch)
        idx = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, S)
        is_decide = input_ids == self.decide_id
        masked = torch.where(is_decide, idx, torch.full_like(idx, S - 1))
        suffix = torch.flip(torch.cummin(torch.flip(masked, dims=[1]), dim=1)[0], dims=[1])

        # pointer readout: k(h_i) · q(h_decide(i)) * scale, read at </opt> positions
        qd = self.head_q(h)
        ko = self.head_k(h)
        q_at_decide = torch.gather(qd, 1, suffix.unsqueeze(-1).expand(B, S, qd.shape[-1]))
        logits = (ko * q_at_decide).sum(-1) * self.scale
        logits = logits / self.temperature
        logits = torch.where(input_ids == self.end_id, logits, torch.zeros_like(logits))
        return logits


def load_merged(run: str, temperature: float):
    """Checkpoint -> merged fp32 backbone + loaded head, on CPU, eval mode."""
    ck = Checkpoint(run)
    tok, model = ck.load("cpu")
    model.head.temperature = 1.0  # graph bakes it instead (JS kev family hardcodes T=1)
    # merge LoRA into the backbone for export
    lm = model.lm
    if hasattr(lm, "merge_and_unload"):
        lm = lm.merge_and_unload()
    lm = lm.float().eval()
    model.head.float()
    delim = [tok.convert_tokens_to_ids(t) for t in SPECIAL]
    for d in delim:
        assert isinstance(d, int) and d >= 0, f"delimiter {d} not in tokenizer"
    wrap = KevExport(lm, model.head, delim, temperature=temperature)
    wrap.eval()
    return tok, wrap, model, ck


def export_onnx(wrap, out_dir: Path, seq_len=256):
    import torch

    out_dir.mkdir(parents=True, exist_ok=True)
    dummy_ids = torch.ones(1, seq_len, dtype=torch.int64)
    dummy_mask = torch.ones(1, seq_len, dtype=torch.int64)
    path = out_dir / "model.onnx"
    with torch.no_grad():
        torch.onnx.export(
            wrap, (dummy_ids, dummy_mask), str(path),
            input_names=["input_ids", "attention_mask"], output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "logits": {0: "batch", 1: "sequence"},
            },
            opset_version=17, do_constant_folding=True,
        )
    print(f"exported {path} ({path.stat().st_size / 1e6:.0f} MB)")


def parity_check(tok, wrap, reference: DecisionModel, records, n=8, tol=2e-3):
    """Same records through the torch scoring path and the exported wrapper."""
    import torch

    print(f"parity check on {min(n, len(records))} records...")
    worst = 0.0
    for rec in records[:n]:
        enc = encode(tok, rec)
        ids = torch.tensor([enc["ids"]], dtype=torch.int64)
        att = torch.ones_like(ids)
        with torch.no_grad():
            got = wrap(ids, att)[0].tolist()
            # reference: DecisionModel forward + head readout at T=1
            ref_model = reference
            ref_model.head.temperature = 1.0
            z = ref_model.probs(enc)  # not logits; recompute logits directly:
        h = ref_model.hidden(enc)
        read = ref_model._readout(h, enc)
        flat_ref = {}
        for qi, (z_q) in enumerate(read):
            for oi, val in enumerate(z_q.tolist()):
                flat_ref[enc["opt_idx"][qi][oi]] = val
        for pos, ref_val in flat_ref.items():
            diff = abs(got[pos] - ref_val)
            worst = max(worst, diff)
    print(f"  worst |Δlogit| at option-end positions: {worst:.5f} (fp32 noise expected < {tol})")
    if worst > tol:
        raise SystemExit(f"parity FAILED: {worst} > {tol}")
    print("  parity ✓")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="training run dir (vendor/kev runs/…)")
    ap.add_argument("out", help="output dir for the ONNX model")
    ap.add_argument("--temperature", type=float, default=1.0, help="bake this calibration temperature into the graph")
    ap.add_argument("--seq", type=int, default=256, help="export dummy sequence length")
    ap.add_argument("--parity-data", default="", help="kev-format JSONL for the parity check (default: the run's training data)")
    args = ap.parse_args()

    tok, wrap, reference, ck = load_merged(args.run, args.temperature)

    data_path = args.parity_data
    if not data_path:
        from kev.data import load_records
        # parity on the run's own data if recorded, else the kev-train file next to this repo
        cand = Path(__file__).resolve().parent.parent / "data" / "kev-train.jsonl"
        data_path = str(cand)
    from kev.data import load_records
    records = load_records(data_path)[:8]
    parity_check(tok, wrap, reference, records)

    export_onnx(wrap, Path(args.out), seq_len=args.seq)
    print("done — next: quantize to q4/q4f16 (scripts/quantize-kev-onnx.py)")


if __name__ == "__main__":
    main()
