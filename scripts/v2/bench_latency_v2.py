#!/usr/bin/env python3
"""Single-frame latency of texture stages (batch 1, the closed-loop regime) on CPU and MPS, the v1 protocol of
scripts/v1/bench_latency.py (warm-up, then timed translate_aligned calls with an MPS synchronize), extended with
p95 and run for several stages in ONE process so the ratio to v1 is measured under the same conditions.

  python -I scripts/v2/bench_latency_v2.py --stage v1:sonospade:runs/v2/v1_repro/sonospade_s0.pt \
      --stage v2:sonospade_v2:runs/v2/....pt --out results/v2/latency.csv
"""
from __future__ import annotations

import argparse
import csv
import datetime
import os
import time

import numpy as np


def bench(stage, imgs, labs, device_str, n=120, warmup=15, repeats=3):
    import torch
    sync = (lambda: torch.mps.synchronize()) if device_str == "mps" else (lambda: None)
    for i in range(warmup):
        stage(imgs[i % len(imgs)], labs[i % len(labs)])
    sync()
    best = None
    for _ in range(repeats):                         # best of `repeats` runs, robust to OS noise
        ts = []
        for i in range(n):
            t0 = time.perf_counter()
            stage(imgs[i % len(imgs)], labs[i % len(labs)])
            sync()
            ts.append((time.perf_counter() - t0) * 1e3)
        ts = np.asarray(ts)
        if best is None or ts.mean() < best.mean():
            best = ts
    return dict(mean_ms=float(best.mean()), p95_ms=float(np.percentile(best, 95)),
                median_ms=float(np.median(best)), fps=float(1000.0 / best.mean()))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", action="append", required=True, help="name:kind:ckpt")
    ap.add_argument("--pairs", default="data/dataset/pairs_med.npz")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--threads", type=int, default=0, help="CPU threads (0 = torch default)")
    ap.add_argument("--out", default="results/v2/latency.csv")
    args = ap.parse_args(argv)

    import torch
    from sonospadepp.v2.evaldata import Stage, load_pairs
    if args.threads:
        torch.set_num_threads(args.threads)
    imgs, labs, _ = load_pairs(args.pairs, args.n + 20)
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    rows = []
    for dev in devices:
        for spec in args.stage:
            name, kind, ckpt = spec.split(":", 2)
            st = Stage(kind, ckpt or None, device=torch.device(dev))
            r = bench(st, imgs, labs, dev, n=args.n)
            r.update(stage=name, kind=kind, device=dev, threads=torch.get_num_threads(), torch=torch.__version__,
                     date=datetime.date.today().isoformat())
            rows.append(r)
            print(f"{dev:4s} {name:16s} {r['mean_ms']:7.2f} ms (p95 {r['p95_ms']:6.2f})  {r['fps']:7.1f} fps")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    new = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["stage", "kind", "device", "mean_ms", "p95_ms", "median_ms", "fps",
                                           "threads", "torch", "date"])
        if new:
            w.writeheader()
        w.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
