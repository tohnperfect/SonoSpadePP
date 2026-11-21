#!/usr/bin/env python3
"""T1G, tier 1 in the USB Gym closed loop: export a tier-1 model (train_m7 --mode lab, started
from v1r) to USB Gym's SonoSPADE checkpoint layout, and gate G1: USB Gym's own SonoSPADETranslator, loading the
export, reproduces the tier-1 model on 50 frames of the gym's curvilinear pairs (max abs difference <= 1/255).

In lab mode the CT channels of the tier-1 generator are zero, so dropping their weights (the extra input channels
of the stem conv and of every SPADE shared conv) leaves the output unchanged.

  python -I scripts/v2/usbgym/export_t1g.py --start V1R.pt --tier1 runs/t1g/m7/t1g_s0_f0.pt --out EXPORT.pt \\
      --pairs $USBG_GYM_ROOT/data/dataset/v2/pairs_med_curvi.npz --gate results/v2/gate_t1g.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

GYM = os.environ.get("USBG_GYM_ROOT", "")          # a USB Gym checkout (https://github.com/tohnperfect/USBGym)
if not GYM:
    raise SystemExit("set USBG_GYM_ROOT to a USB Gym checkout")
CHECK = r"""
import sys, numpy as np
sys.path.insert(0, sys.argv[1] + "/src")
from usbg import texture_gan as TG
import usbg, os
assert os.path.realpath(usbg.__file__).startswith(os.path.realpath(sys.argv[1])), usbg.__file__
d = np.load(sys.argv[3])
tr = TG.SonoSPADETranslator(sys.argv[2], device="cpu")
out = np.stack([tr.translate_aligned(i, l) for i, l in zip(d["img"], d["lab"])])
print(float(np.abs(out - d["ref"]).max()))
"""


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--tier1", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--gate", required=True)
    ap.add_argument("--n", type=int, default=50)
    args = ap.parse_args(argv)

    import torch
    from sonospadepp import paired_translator as PT

    st = torch.load(args.start, map_location="cpu", weights_only=False)
    t1 = torch.load(args.tier1, map_location="cpu", weights_only=True)
    g = {}
    for k, v in st["G_spade"].items():
        w = t1[k]
        if w.shape != v.shape:                              # extra CT input channels: zero inputs in lab mode
            assert w.shape[0] == v.shape[0] and w.shape[2:] == v.shape[2:] and w.shape[1] == v.shape[1] + PT.N_CT, k
            w = w[:, :v.shape[1]]
        g[k] = w.clone()
    assert set(t1) == set(st["G_spade"]), "tier-1 and start generators differ in keys"
    ex = dict(st)
    ex["G_spade"] = g
    ex["G_raw"] = {k: v.clone() for k, v in g.items()}
    ex["t1g"] = {"start": os.path.abspath(args.start), "tier1": os.path.abspath(args.tier1)}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(ex, args.out)

    # reference: the tier-1 model as trained (CT channels zero), as eval_v2's paired_lab stage runs it
    G = PT.build(args.start, "cpu")
    G.load_state_dict(t1)
    G.eval()
    d = np.load(args.pairs, allow_pickle=False)
    img = d["images"][: args.n].astype(np.float32) / 255.0
    lab = d["labels"][: args.n].astype(np.int64)
    z = np.zeros_like(img)
    with torch.no_grad():
        c, s = PT.inputs(img, lab, z, z, False, "cpu")
        ref = PT.forward01(G, c, s).numpy()
    with tempfile.TemporaryDirectory() as tmp:
        f = os.path.join(tmp, "ref.npz")
        np.savez(f, img=img, lab=lab, ref=ref)
        r = subprocess.run([f"{GYM}/.venv/bin/python", "-I", "-c", CHECK, GYM, args.out, f],
                           capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(r.stderr)
    diff = float(r.stdout.strip().splitlines()[-1])
    gate = {"G1_export_max_abs_diff": diff, "G1_limit": 1 / 255, "G1_pass": diff <= 1 / 255, "n_frames": int(len(img)),
            "export": os.path.abspath(args.out)}
    prev = json.load(open(args.gate)) if os.path.exists(args.gate) else {}
    prev.update(gate)
    json.dump(prev, open(args.gate, "w"), indent=1)
    print(json.dumps(gate))
    return 0 if gate["G1_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
