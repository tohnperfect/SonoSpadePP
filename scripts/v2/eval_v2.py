#!/usr/bin/env python3
"""Single evaluation entry point for SonoSPADE v2: one texture stage (checkpoint) -> long-format CSV rows,
one row per (model, seed, metric, tissue[, extractor, dr_db]), written to results/v2/rows/<model>_s<seed>.csv.
scripts/v2/make_v2_tables.py concatenates the rows into results/v2/metrics.csv.

Metrics. v1's metrics call usbg.texture_eval unchanged, on v1's protocol
(first 200 content pairs of pairs_med.npz, the 60 annotated real frames, locality on the first 40 pairs):
  edge_iou            structure vs the physics input (v1)
  organ_w1            intensity W1 inside each annotated organ, mean over organs (v1)
  locality_band0      label-swap locality, v1 definition exactly
  locality_band8      the same with an 8 px boundary band excluded from the outside region
  tstr_dice           train on synthetic, test on the 60 real frames (v1)
New:
  tissue_kid / tissue_fd       per-tissue KID (x1e3) / FD on crops inside eroded masks, per extractor
  frame_kid / frame_fd         whole-frame KID (x1e3) / FD, raw (v1 style) and fan-masked
  nakagami_m_w1 / log_omega_w1 W1 between real and fake window-level Nakagami m / log Omega, per tissue
  speckle_size_err             |FWHM fake - FWHM real|, axial and lateral, depth-binned, per tissue
  diversity_lpips              per-tissue LPIPS between style samples (0 for deterministic stages)
Real masks: the annotated frames' masks. Fake masks: the input label. Both restricted to the real pool's
fan (sonospadepp.v2.evaldata.fan_mask) for the crop and speckle metrics.

  python -I scripts/v2/eval_v2.py --model v1 --seed 0 --kind sonospade --ckpt runs/v2/v1_repro/sonospade_s0.pt
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import os

import numpy as np

TISSUE_NAMES = {0: "background", 1: "liver", 2: "gallbladder", 3: "vessel", 4: "kidney", 5: "gi",
                6: "pancreas", 7: "spleen", 8: "muscle", 9: "fat", 10: "bone", 11: "lung",
                12: "soft_tissue"}
FIELDS = ["model", "seed", "metric", "tissue", "value", "ci_lo", "ci_hi", "n", "extractor", "dr_db",
          "protocol", "ckpt_sha256", "date"]


def sha256(path):
    if not path or not os.path.exists(path):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class Rows:
    def __init__(self, model, seed, ckpt_sha):
        self.model, self.seed, self.sha = model, seed, ckpt_sha
        self.rows = []
        self.date = datetime.date.today().isoformat()

    def add(self, metric, value, tissue="all", ci=(float("nan"), float("nan")), n="", extractor="",
            dr_db="", protocol=""):
        self.rows.append(dict(model=self.model, seed=self.seed, metric=metric, tissue=tissue,
                              value=float(value), ci_lo=float(ci[0]), ci_hi=float(ci[1]), n=n,
                              extractor=extractor, dr_db=dr_db, protocol=protocol, ckpt_sha256=self.sha,
                              date=self.date))

    def write(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(self.rows)


def tname(c):
    return TISSUE_NAMES.get(int(c), str(c))


# ------------------------------------------------------------------------------------------------
def eval_v1_metrics(R, stage, phys, labs, trans, real, rlabs, args, M):
    from usbg import texture_eval as TE
    from usbg.volume import BACKGROUND

    # edge IoU (v1)
    e = np.array([TE.edge_iou(p, t) for p, t in zip(phys, trans)])
    R.add("edge_iou", e.mean(), ci=M.bootstrap_ci(lambda i: e[i].mean(), len(e), args.n_boot),
          n=len(e), protocol="v1")

    # organ intensity W1 (v1): per class, mean over non-background classes
    w1 = TE.intensity_w1_per_class(trans, labs, real, rlabs)
    organs = {c: v for c, v in w1.items() if int(c) != BACKGROUND}
    for c, v in organs.items():
        R.add("organ_w1", v, tissue=tname(c), protocol="v1")

    def w1_mean(fi, ri):
        d = TE.intensity_w1_per_class([trans[k] for k in fi], [labs[k] for k in fi],
                                      [real[k] for k in ri], [rlabs[k] for k in ri])
        vals = [v for c, v in d.items() if int(c) != BACKGROUND]
        return float(np.mean(vals)) if vals else float("nan")

    rng = np.random.default_rng(1)
    boots = [w1_mean(rng.integers(0, len(trans), len(trans)), rng.integers(0, len(real), len(real)))
             for _ in range(args.n_boot_heavy)]
    R.add("organ_w1", np.mean(list(organs.values())), ci=tuple(np.nanquantile(boots, [0.025, 0.975])),
          n=len(organs), protocol="v1")

    # label-swap locality: v1 definition (band 0) and with an 8 px boundary band
    if stage.label_aware:
        per = {0: [], args.band: []}
        noise_ctx = stage.fixed_noise(0)
        noise_ctx.__enter__()
        for img, lab in zip(phys[:args.n_locality], labs[:args.n_locality]):
            base = stage(img, lab)
            fr = {0: {}, args.band: {}}
            for c in range(13):
                m = lab == c
                if c == 1 or int(m.sum()) < 200 or not (~m).any():
                    continue
                sw = lab.copy()
                sw[m] = 1
                alt = stage(img, sw)
                for b in fr:
                    fr[b][c] = M.locality_band(base, alt, m, band=b)["locality"]
            for b in fr:
                per[b].append(fr[b])
        noise_ctx.__exit__(None, None, None)
        for b, frames in per.items():
            name = f"locality_band{b}"
            summ = TE.summarize_per_class(frames)
            for c, v in summ.items():
                R.add(name, v, tissue=tname(c), protocol="v1" if b == 0 else f"band{b}")

            def loc_mean(idx, frames=frames):
                s = TE.summarize_per_class([frames[k] for k in idx])
                return float(np.mean(list(s.values()))) if s else float("nan")

            R.add(name, np.mean(list(summ.values())), n=len(frames),
                  ci=M.bootstrap_ci(loc_mean, len(frames), args.n_boot),
                  protocol="v1" if b == 0 else f"band{b}")
    else:
        R.add("locality_band0", 0.0, protocol="v1 (label-free: 0 by construction)")
        R.add(f"locality_band{args.band}", 0.0, protocol=f"band{args.band} (label-free: 0 by construction)")

    # TSTR Dice (v1)
    if not args.skip_tstr:
        for ts in args.tstr_seeds:
            md, pc = TE.tstr_dice(trans, labs, real, rlabs, iters=args.tstr_iters, seed=ts)
            R.add("tstr_dice", md, n=len(trans), protocol=f"v1 tstr_seed={ts}")
            for c, v in pc.items():
                R.add("tstr_dice", v, tissue=tname(c), protocol=f"v1 tstr_seed={ts}")


# ------------------------------------------------------------------------------------------------
def eval_realism(R, trans, labs, real, rlabs, fan, args, M, F, encoders):
    from sonospadepp.v2.evaldata import REAL_ORGANS
    tf = [t * fan for t in trans]
    rf = [r * fan for r in real]
    for name, enc, crop in [(n, e, c) for n, e in encoders.items() for c in [args.crop] + args.crop_sens]:
        # per-tissue crops (same crop rng for every model, so the real crops are identical)
        tag = "" if crop == args.crop else "crop-size sensitivity, "
        per_t = {}
        for t in REAL_ORGANS:
            rng = np.random.default_rng(100 + t)
            rc, ri = [], []
            for k, (img, lab) in enumerate(zip(rf, rlabs)):
                cr = M.tissue_crops(img, (lab == t) & fan, size=crop, max_crops=args.crops_per_frame,
                                    erode=args.crop_erode, min_cover=args.crop_cover, rng=rng)
                rc.append(cr); ri += [k] * len(cr)
            rng = np.random.default_rng(200 + t)
            fc, fi = [], []
            for k, (img, lab) in enumerate(zip(tf, labs)):
                cr = M.tissue_crops(img, (lab == t) & fan, size=crop, max_crops=args.crops_per_frame,
                                    erode=args.crop_erode, min_cover=args.crop_cover, rng=rng)
                fc.append(cr); fi += [k] * len(cr)
            rc, fc = np.concatenate(rc), np.concatenate(fc)
            ri, fi = np.asarray(ri), np.asarray(fi)
            if len(rc) < args.min_crops or len(fc) < args.min_crops:
                R.add("tissue_kid", float("nan"), tissue=tname(t), n=f"{len(rc)}/{len(fc)}", extractor=name,
                      protocol=f"{tag}crop{crop} skipped: too few crops")
                continue
            fr_, fg_ = F.extract_global(enc, rc), F.extract_global(enc, fc)
            kv, ksd = M.kid(fr_, fg_)

            def kid_boot(idx_r, idx_f):
                a = fr_[np.concatenate([np.nonzero(ri == k)[0] for k in idx_r])]
                b = fg_[np.concatenate([np.nonzero(fi == k)[0] for k in idx_f])]
                return M.kid(a, b, n_subsets=10)[0] if len(a) > 2 and len(b) > 2 else float("nan")

            rng = np.random.default_rng(3)
            boots = [kid_boot(rng.integers(0, len(rf), len(rf)), rng.integers(0, len(tf), len(tf)))
                     for _ in range(args.n_boot_kid)]
            ci = tuple(1e3 * np.nanquantile(boots, [0.025, 0.975]))
            R.add("tissue_kid", 1e3 * kv, tissue=tname(t), ci=ci, n=f"{len(rc)}/{len(fc)}", extractor=name,
                  protocol=f"{tag}crop{crop} x1e3")
            R.add("tissue_fd", M.fd(fr_, fg_), tissue=tname(t), n=f"{len(rc)}/{len(fc)}", extractor=name,
                  protocol=f"{tag}crop{crop}")
            per_t[t] = 1e3 * kv
        if per_t:
            R.add("tissue_kid", np.mean(list(per_t.values())), tissue="mean", n=len(per_t), extractor=name,
                  protocol=f"{tag}crop{crop} x1e3, mean over tissues with >= {args.min_crops} crops")
        if crop != args.crop:
            continue
        # whole frame, raw (v1 style) and fan-masked
        for tag, A, B in [("raw", real, trans), ("fan", rf, tf)]:
            fa, fb = F.extract_global(enc, np.asarray(A)), F.extract_global(enc, np.asarray(B))
            kv, ksd = M.kid(fa, fb)
            R.add("frame_kid", 1e3 * kv, n=f"{len(A)}/{len(B)}", extractor=name, protocol=f"{tag} x1e3")
            R.add("frame_fd", M.fd(fa, fb), n=f"{len(A)}/{len(B)}", extractor=name, protocol=tag)


# Fixed in WP0 from real-frame coverage alone: 7 px windows cover liver and kidney.
NK = dict(win=7, stride=3, erode=1)
SZ = dict(patch=8, stride=4, erode=1)
SPECKLE_METRICS = ("nakagami", "log_omega", "speckle_")


def eval_speckle(R, trans, labs, real, rlabs, fan, args, M):
    from sonospadepp.v2.evaldata import REAL_ORGANS
    for dr in args.dr_list:
        mw, lw = [], []
        for t in REAL_ORGANS:
            rw = [M.nakagami_windows(img, (lab == t) & fan, dr_db=dr, **NK) for img, lab in zip(real, rlabs)]
            fw = [M.nakagami_windows(img, (lab == t) & fan, dr_db=dr, **NK) for img, lab in zip(trans, labs)]
            rm = np.concatenate([w[0] for w in rw]); fm = np.concatenate([w[0] for w in fw])
            ro = np.concatenate([w[1] for w in rw]); fo = np.concatenate([w[1] for w in fw])
            n = f"{len(rm)}/{len(fm)}"
            if len(rm) < args.min_windows or len(fm) < args.min_windows:
                R.add("nakagami_m_w1", float("nan"), tissue=tname(t), n=n, dr_db=dr, protocol="skipped: few windows")
                continue
            wm = M.w1_1d(rm, fm)
            if dr == args.dr:
                def boot(idx_r, idx_f):
                    a = np.concatenate([rw[k][0] for k in idx_r]); b = np.concatenate([fw[k][0] for k in idx_f])
                    return M.w1_1d(a, b)
                rng = np.random.default_rng(4)
                bs = [boot(rng.integers(0, len(rw), len(rw)), rng.integers(0, len(fw), len(fw)))
                      for _ in range(args.n_boot)]
                ci = tuple(np.nanquantile(bs, [0.025, 0.975]))
                R.add("nakagami_m_w1", wm, tissue=tname(t), ci=ci, n=n, dr_db=dr)
                R.add("log_omega_w1", M.w1_1d(ro, fo), tissue=tname(t), n=n, dr_db=dr)
                R.add("nakagami_m_mean_real", float(np.mean(rm)), tissue=tname(t), n=len(rm), dr_db=dr)
                R.add("nakagami_m_mean_fake", float(np.mean(fm)), tissue=tname(t), n=len(fm), dr_db=dr)
                R.add("nakagami_m_median_real", float(np.median(rm)), tissue=tname(t), n=len(rm), dr_db=dr)
                R.add("nakagami_m_median_fake", float(np.median(fm)), tissue=tname(t), n=len(fm), dr_db=dr)
                R.add("nakagami_logm_w1", M.w1_1d(np.log(rm), np.log(fm)), tissue=tname(t), n=n, dr_db=dr,
                      protocol="")
            else:
                R.add("nakagami_m_w1", wm, tissue=tname(t), n=n, dr_db=dr, protocol="DR sensitivity")
                R.add("nakagami_logm_w1", M.w1_1d(np.log(rm), np.log(fm)), tissue=tname(t), n=n, dr_db=dr,
                      protocol="DR sensitivity")
            mw.append(wm)
            lw.append(M.w1_1d(np.log(rm), np.log(fm)))
        if mw:
            R.add("nakagami_m_w1", np.mean(mw), tissue="mean", n=len(mw), dr_db=dr,
                  protocol="" if dr == args.dr else "DR sensitivity")
            R.add("nakagami_logm_w1", np.mean(lw), tissue="mean", n=len(lw), dr_db=dr,
                  protocol="" if dr == args.dr else "DR sensitivity")

    # speckle size (DR = args.dr)
    errs = []
    for t in REAL_ORGANS:
        ra = [M.speckle_acov_sum(img, (lab == t) & fan, dr_db=args.dr, **SZ) for img, lab in zip(real, rlabs)]
        fa = [M.speckle_acov_sum(img, (lab == t) & fan, dr_db=args.dr, **SZ) for img, lab in zip(trans, labs)]
        rs, rcnt = sum(a[0] for a in ra), sum(a[1] for a in ra)
        fs, fcnt = sum(a[0] for a in fa), sum(a[1] for a in fa)
        te = []
        for b in range(rs.shape[0]):
            if rcnt[b] < args.min_patches or fcnt[b] < args.min_patches:
                continue
            rax, rlat = M.fwhm_from_acov(rs[b]); fax, flat = M.fwhm_from_acov(fs[b])
            for ax, r_, f_ in [("axial", rax, fax), ("lateral", rlat, flat)]:
                R.add(f"speckle_fwhm_real_{ax}", r_, tissue=tname(t), n=int(rcnt[b]), protocol=f"depth_bin={b}")
                R.add(f"speckle_fwhm_fake_{ax}", f_, tissue=tname(t), n=int(fcnt[b]), protocol=f"depth_bin={b}")
                if np.isfinite(r_) and np.isfinite(f_):
                    te.append(abs(f_ - r_))
        if te:
            R.add("speckle_size_err", float(np.mean(te)), tissue=tname(t), n=len(te), dr_db=args.dr)
            errs.append(float(np.mean(te)))
    if errs:
        R.add("speckle_size_err", float(np.mean(errs)), tissue="mean", n=len(errs), dr_db=args.dr)


# ------------------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="row label, e.g. v1, v1_paper, physics, cut")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kind", required=True,
                    choices=["physics", "sonospade", "cut", "cyclegan", "cyclegan_sc", "sonospade_v2", "paired_lab"])
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--pairs", default="data/dataset/pairs_med.npz")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--real-pairs", default="data/eval/kaggle_vitale_test.npz")
    ap.add_argument("--pool", default="data/real/kaggle_vitale/pool")
    ap.add_argument("--extractors", default="usfm,dinov2")
    ap.add_argument("--dr", type=float, default=60.0)
    ap.add_argument("--dr-list", type=float, nargs="+", default=[40.0, 50.0, 60.0, 70.0])
    ap.add_argument("--band", type=int, default=8)
    ap.add_argument("--n-locality", type=int, default=40)
    ap.add_argument("--crop", type=int, default=16,
                    help="primary crop size; fixed in WP0 from real-frame coverage alone (all 5 organs >= 43 crops)")
    ap.add_argument("--crop-sens", type=int, nargs="*", default=[32])
    ap.add_argument("--crop-cover", type=float, default=0.5)
    ap.add_argument("--crop-erode", type=int, default=1)
    ap.add_argument("--crops-per-frame", type=int, default=8)
    ap.add_argument("--min-crops", type=int, default=20)
    ap.add_argument("--min-windows", type=int, default=30)
    ap.add_argument("--min-patches", type=int, default=3)
    ap.add_argument("--tstr-iters", type=int, default=500)
    ap.add_argument("--tstr-seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--skip-tstr", action="store_true")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-boot-heavy", type=int, default=200)
    ap.add_argument("--n-boot-kid", type=int, default=100)
    ap.add_argument("--out-dir", default="results/v2/rows")
    ap.add_argument("--style-mode", choices=["mean", "sample"], default="mean",
                    help="style-coded models: mean = each tissue's bank mean (E2a); sample = every tissue's code "
                         "drawn from the bank per frame (E2b realism of sampled styles)")
    ap.add_argument("--only", choices=["", "speckle"], default="",
                    help="speckle: recompute only the speckle rows and replace them in the existing rows file")
    args = ap.parse_args(argv)

    from usbg.render_lotus import get_device
    from sonospadepp.v2 import metrics as M
    from sonospadepp.v2 import features as F
    from sonospadepp.v2.evaldata import Stage, fan_mask, load_pairs, load_pool

    device = get_device()
    phys, labs, _ = load_pairs(args.pairs, args.limit)
    real, rlabs, _ = load_pairs(args.real_pairs, None)
    fan = fan_mask(load_pool(args.pool))
    stage = Stage(args.kind, args.ckpt, device=device)
    if args.style_mode == "sample":
        tr = stage.translator
        if not tr.has_style:
            raise SystemExit("--style-mode sample needs a style-coded checkpoint")
        srng = np.random.default_rng(12345)

        def sampled(img, lab, tr=tr, srng=srng):
            gs = sorted({int(g) for g in np.unique(tr.group_map[np.asarray(lab, np.int64)])})
            return tr.translate_aligned(img, lab, style={g: tr.sample_style(g, srng) for g in gs})
        stage._f = sampled
    R = Rows(args.model, args.seed, sha256(args.ckpt))

    trans = [stage(p, l) for p, l in zip(phys, labs)]
    out = os.path.join(args.out_dir, f"{args.model}_s{args.seed}.csv")
    if args.only == "speckle":
        with open(out) as fh:
            keep = [r for r in csv.DictReader(fh) if not r["metric"].startswith(SPECKLE_METRICS)]
        eval_speckle(R, trans, labs, real, rlabs, fan, args, M)
        R.rows = keep + R.rows
        R.write(out)
        print(f"replaced speckle rows in {out}")
        return 0
    print(f"[{args.model} s{args.seed}] translated {len(trans)} frames; real {len(real)}; fan {fan.mean():.2f}")

    # metric hygiene: never evaluate realism with the encoder used in the training loss
    from sonospadepp.v2.hygiene import trained_extractors
    trained = trained_extractors(args.ckpt) if args.kind == "sonospade_v2" else set()   # fm_disc and S_US init
    names = [e for e in args.extractors.split(",") if e and e not in trained]
    for bad in ("echocare", "medsam", "biomedparse"):        # never evaluation extractors
        if bad in names:
            raise SystemExit(f"{bad} may not be used as an evaluation extractor (pretraining overlap / license)")
    for e in sorted(trained):
        print(f"  excluded extractor {e!r} (its weights enter this model's training loss)")
        R.add("excluded_extractor", float("nan"), extractor=e, protocol="metric hygiene: in the training loss")
    encoders = {n: F.FMEncoder(n, device) for n in names}

    eval_v1_metrics(R, stage, phys, labs, trans, real, rlabs, args, M)
    eval_realism(R, trans, labs, real, rlabs, fan, args, M, F, encoders)
    eval_speckle(R, trans, labs, real, rlabs, fan, args, M)
    if args.kind == "sonospade_v2" and getattr(stage.translator, "has_style", False):
        from sonospadepp.v2.diversity import eval_diversity
        eval_diversity(R, stage.translator, phys, labs, args)
    else:
        R.add("diversity_lpips", 0.0, protocol="deterministic stage: one output per label")

    R.write(out)
    print(f"wrote {len(R.rows)} rows -> {out}")
    for r in R.rows:
        if r["tissue"] in ("all", "mean") and (r["extractor"] in ("", "usfm")) and r["dr_db"] in ("", args.dr):
            print(f"  {r['metric']:18s} {r['tissue']:5s} {r['extractor']:7s} {r['value']:9.4f}  "
                  f"[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]  {r['protocol']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
