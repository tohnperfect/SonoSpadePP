#!/usr/bin/env python3
"""End-to-end frame time of the simulator: CT label slice at the probe pose,
physics render, resize and sector warp, and the SonoSPADE translator, at 128x128, batch 1, through USB Gym's own frame
path (usbg.gym_env.ViewPipeline.render -> usbg.view_match.curvi_render). Same protocol as bench_latency_v2.py: 15
warm-up frames, then the best of three runs of 120 frames, with an MPS synchronize per frame. The translator's share is
timed inside the same calls (its own synchronize), so the three numbers come from one set of frames.

With --ort-threads N, a third stage on the CPU runs the same generator weights through ONNX Runtime (exported with
torch.onnx, opset 17) with N intra-op threads, and its frames are compared with the PyTorch stage's (max absolute
difference on [0, 1]; the target fixed before the run allows 1/255).

Probe poses: a probe moving over one CT under ViewMatch-v0's random actions (12 steps per episode, liver and kidney
targets), collected once and replayed identically on every device and stage.

  USBG_SEAT=anatomical PYTHONPATH=$USBG_GYM_ROOT/src $USBG_GYM_ROOT/.venv/bin/python -I \\
      scripts/v2/bench_endtoend.py --ckpt $USBG_GYM_ROOT/data/dataset/v2fm/sonospade_v1r_curvi.pt \\
      --ort-threads 4 --out results/v2/latency_endtoend.csv
"""
from __future__ import annotations

import argparse
import csv
import datetime
import os
import platform
import subprocess
import time

import numpy as np


def collect_poses(case, n, seed=0):
    from usbg.gym_env import ViewMatchEnv
    env = ViewMatchEnv(patients=[case], texture="physics", mask="ct", device="cpu")
    rng = np.random.default_rng(seed)
    poses = []
    k = 0
    while len(poses) < n:
        env.reset(seed=seed + k, options={"target": ("liver", "kidney")[k % 2]})
        k += 1
        poses.append((env._lv, np.array(env._probe.world_from_probe), env._Wfv, env._cfg_ep))
        for _ in range(env.n_steps):
            _, _, _, trunc, _ = env.step(rng.uniform(-1, 1, 6).astype(np.float32))
            poses.append((env._lv, np.array(env._probe.world_from_probe), env._Wfv, env._cfg_ep))
            if trunc:
                break
    env.close()
    return poses[:n]


class OrtSonoSPADE:
    """USB Gym's SonoSPADETranslator with the generator run by ONNX Runtime on the CPU: same weights, same
    preprocessing (one-hot labels, [-1, 1] input) and postprocessing ([0, 1], clipped)."""

    wants_label = True

    def __init__(self, torch_translator, onnx_path, threads):
        import onnxruntime as ort
        import torch
        G = torch_translator.G.eval().to("cpu")
        self.n_classes = torch_translator.n_classes
        a = torch.zeros(1, 1, 128, 128)
        seg = torch.zeros(1, self.n_classes, 128, 128)
        torch.onnx.export(G, (a, seg), onnx_path, input_names=["img", "seg"], output_names=["out"],
                          opset_version=17, dynamo=False)
        so = ort.SessionOptions()
        so.intra_op_num_threads, so.inter_op_num_threads = int(threads), 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])

    def translate_aligned(self, img01, aligned_label):
        from usbg.texture_gan import onehot_labels
        img = np.asarray(img01, np.float32)
        seg = onehot_labels(np.asarray(aligned_label), self.n_classes, "cpu").numpy().astype(np.float32)
        out = self.sess.run(None, {"img": (img * 2 - 1)[None, None], "seg": seg})[0][0, 0]
        return np.clip((out + 1) / 2, 0, 1).astype(np.float32)

    def __call__(self, img01, label_slice):                   # render_lotus's call; curvi_render uses translate_aligned
        from usbg.segmenter import align_label_to_image
        img = np.asarray(img01, np.float32)
        return self.translate_aligned(img, align_label_to_image(label_slice, img.shape))


def timed_translator(tr, sync, acc):
    f = tr.translate_aligned

    def wrapped(img, lab):
        t0 = time.perf_counter()
        out = f(img, lab)
        sync()
        acc.append((time.perf_counter() - t0) * 1e3)
        return out
    tr.translate_aligned = wrapped


def bench(pipe, poses, sync, acc, n=120, warmup=15, repeats=3):
    for i in range(warmup):
        lv, T, W, cfg = poses[i % len(poses)]
        pipe.render(lv, T, W, cfg)
    sync()
    best = None
    for _ in range(repeats):
        acc.clear()
        ts = []
        for i in range(n):
            lv, T, W, cfg = poses[i % len(poses)]
            t0 = time.perf_counter()
            pipe.render(lv, T, W, cfg)
            sync()
            ts.append((time.perf_counter() - t0) * 1e3)
        ts = np.asarray(ts)
        tr = np.asarray(acc) if acc else np.zeros(0)
        if best is None or ts.mean() < best[0].mean():
            best = (ts, tr)
    ts, tr = best
    return dict(mean_ms=float(ts.mean()), p95_ms=float(np.percentile(ts, 95)), median_ms=float(np.median(ts)),
                fps=float(1000.0 / ts.mean()), translator_ms=float(tr.mean()) if len(tr) else 0.0)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="a SonoSPADE checkpoint in USB Gym's layout")
    ap.add_argument("--case", default="s0011")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--ort-threads", type=int, default=0, help="also time ONNX Runtime on the CPU with N threads")
    ap.add_argument("--out", default="results/v2/latency_endtoend.csv")
    args = ap.parse_args(argv)

    import torch
    from usbg.gym_env import ViewPipeline
    poses = collect_poses(args.case, args.n + 15)
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    try:
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    except OSError:
        chip = platform.processor()
    rows = []
    for dev in devices:
        sync = (lambda: torch.mps.synchronize()) if dev == "mps" else (lambda: None)
        for stage in ("physics", "sonospade"):
            acc = []
            pipe = ViewPipeline(stage, "ct", texture_ckpt=args.ckpt if stage == "sonospade" else None, device=dev,
                                image_size=128)
            if stage == "sonospade":
                timed_translator(pipe.translator, sync, acc)
            r = bench(pipe, poses, sync, acc, n=args.n)
            r.update(stage=stage, device=dev, size=128, case=args.case, threads=torch.get_num_threads(),
                     torch=torch.__version__, machine=chip, os=platform.platform(terse=True),
                     ckpt=os.path.basename(args.ckpt) if stage == "sonospade" else "",
                     date=datetime.date.today().isoformat())
            r["runtime"] = "pytorch"
            rows.append(r)
            print(f"{dev:4s} {stage:9s} total {r['mean_ms']:7.2f} ms (p95 {r['p95_ms']:6.2f}) {r['fps']:7.1f} fps"
                  f"  translator {r['translator_ms']:6.2f} ms")
    if args.ort_threads:                                     # the same weights through ONNX Runtime, CPU only
        import tempfile
        import onnxruntime as ort
        torch_pipe = ViewPipeline("sonospade", "ct", texture_ckpt=args.ckpt, device="cpu", image_size=128)
        with tempfile.TemporaryDirectory() as tmp:
            otr = OrtSonoSPADE(torch_pipe.translator, os.path.join(tmp, "g.onnx"), args.ort_threads)
            pipe = ViewPipeline(otr, "ct", device="cpu", image_size=128)
            diff = max(float(np.abs(pipe.render(lv, T, W, cfg)[0] - torch_pipe.render(lv, T, W, cfg)[0]).max())
                       for lv, T, W, cfg in poses[:args.n])
            acc = []
            timed_translator(otr, lambda: None, acc)
            r = bench(pipe, poses, lambda: None, acc, n=args.n)
        r.update(stage="sonospade_ort", device="cpu", size=128, case=args.case, threads=args.ort_threads,
                 torch=torch.__version__, machine=chip, os=platform.platform(terse=True),
                 ckpt=os.path.basename(args.ckpt), date=datetime.date.today().isoformat(),
                 runtime="onnxruntime " + ort.__version__, max_diff_vs_pytorch=diff)
        rows.append(r)
        print(f"cpu  sonospade_ort total {r['mean_ms']:7.2f} ms (p95 {r['p95_ms']:6.2f}) {r['fps']:7.1f} fps"
              f"  translator {r['translator_ms']:6.2f} ms  max diff vs PyTorch {diff:.2e}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["stage", "device", "size", "mean_ms", "p95_ms", "median_ms", "fps",
                                           "translator_ms", "case", "threads", "torch", "machine", "os", "ckpt",
                                           "date", "runtime", "max_diff_vs_pytorch"])
        w.writeheader()
        w.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
