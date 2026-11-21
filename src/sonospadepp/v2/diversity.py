"""WP2 metrics for style-coded models: per-tissue diversity (LPIPS between style samples for one label) and
style-swap locality (change one tissue's code; the change must stay inside that tissue).

LPIPS (Zhang et al. 2018, AlexNet backbone, `lpips` 0.1.4 with its bundled linear weights) is computed as a
spatial map and averaged inside each tissue's mask, over all pairs of K samples per content frame.
"""
from __future__ import annotations

import itertools

import numpy as np

TISSUES = {1: "liver", 2: "gallbladder", 3: "vessel", 4: "kidney", 7: "spleen", 12: "soft_tissue"}


def _lpips_model(device):
    import lpips
    return lpips.LPIPS(net="alex", spatial=True, verbose=False).to(device).eval()


def eval_diversity(R, tr, phys, labs, args, n_frames=20, k=4, seed=0):
    import torch
    from sonospadepp.v2 import metrics as M
    dev = tr.device
    net = _lpips_model(dev)
    rng = np.random.default_rng(seed)
    per = {t: [] for t in TISSUES}
    for img, lab in zip(phys[:n_frames], labs[:n_frames]):
        groups = sorted({int(g) for g in np.unique(tr.group_map[lab])})
        outs = [tr.translate_aligned(img, lab, style={g: tr.sample_style(g, rng) for g in groups}, noise_seed=0)
                for _ in range(k)]                    # same noise: diversity measures the style codes only
        with torch.no_grad():
            for a, b in itertools.combinations(outs, 2):
                ta = torch.from_numpy(a * 2 - 1)[None, None].repeat(1, 3, 1, 1).to(dev)
                tb = torch.from_numpy(b * 2 - 1)[None, None].repeat(1, 3, 1, 1).to(dev)
                d = net(ta, tb)[0, 0].cpu().numpy()
                for t in TISSUES:
                    m = lab == t
                    if m.sum() >= 50:
                        per[t].append(float(d[m].mean()))
    vals = []
    for t, name in TISSUES.items():
        if per[t]:
            R.add("diversity_lpips", float(np.mean(per[t])), tissue=name, n=len(per[t]), protocol=f"K={k} samples")
            vals.append(float(np.mean(per[t])))
    R.add("diversity_lpips", float(np.mean(vals)) if vals else float("nan"), tissue="all", n=len(vals),
          protocol=f"K={k} samples, mean over tissues")

    # style-swap locality: bank-mean styles vs one tissue's code sampled; outside an 8 px band
    loc = {t: [] for t in TISSUES}
    for img, lab in zip(phys[:args.n_locality], labs[:args.n_locality]):
        for t in TISSUES:
            g = int(tr.group_map[t])
            m = lab == t
            if m.sum() < 200:
                continue
            base = tr.translate_aligned(img, lab, noise_seed=0)          # same injected noise in both
            alt = tr.translate_aligned(img, lab, style={g: tr.sample_style(g, rng)}, noise_seed=0)
            r = M.locality_band(base, alt, m, band=args.band)
            if np.isfinite(r["locality"]):
                loc[t].append(r["locality"])
    vals = []
    for t, name in TISSUES.items():
        if loc[t]:
            R.add("style_swap_locality", float(np.mean(loc[t])), tissue=name, n=len(loc[t]),
                  protocol=f"band{args.band}")
            vals.append(float(np.mean(loc[t])))
    R.add("style_swap_locality", float(np.mean(vals)) if vals else float("nan"), tissue="all", n=len(vals),
          protocol=f"band{args.band}, mean over tissues")
