"""Invariants every v2 model must keep.

* v1 is untouched: the pinned `sonospade` package is v0.2.0, the vendored v1 scripts are byte-identical to
  the v0.2.0 release, and the released paper checkpoint loads and translates unchanged.
* Label-swap locality: v2's harness reproduces v1's definition on the real v1 checkpoint.
* Style-swap locality (WP2) and the latency budget (<= 1.25x v1 on CPU) for a v2 checkpoint given in
  SONOSPADE_V2_CKPT; skipped while none exists.
"""
import hashlib
import importlib.metadata as md
import json
import os

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V1_PAPER_CKPT = os.path.join(ROOT, "data/v1_ckpt/exp_real_sonospade.pt")
V1_PAPER_SHA = "489b488552f1fdff133f21812495067051333857e13ab985cb5bcb949fcbf04b"   # MANIFEST v0.2.0
PAIRS = os.path.join(ROOT, "data/dataset/pairs_med.npz")
V2_CKPT = os.environ.get("SONOSPADE_V2_CKPT", "")
V1_SCRIPTS_SHA = {   # scripts/v1/* == `git show v0.2.0:scripts/<name>` of tohnperfect/SonoSpade
    "bench_latency.py": "3991d66eb67f0c82368f8c6f6a3db2c474458f10d50fb63cbb8aa45e0da77e48",
    "eval_fid_kid.py": "78dbcfe6efae64da1e0c96b13604a95e36fd612e615eb521d5c760e83db1d10d",
    "eval_texture.py": "0e240643afb0d16bbc3f435ccd305209fbc885c0f575b75b862bedfdcd0ee71d",
    "ingest_kaggle_us.py": "b96519bacce01b849c2965e2e1f68535258a7d517149b203d6e1ba7745cfcc39",
    "run_sonospade_experiment_real.sh": "ba62835ec741c3f0678e79c8d969f54956a17e07cb56a5c037b5969068410652",
    "train_texture_gan.py": "79bf93f3da1de67c5d3d4461a469e3f773e1c5d8b36bc186aa0e615565666f54",
}


def _sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def test_v1_package_is_pinned_release():
    assert md.version("sonospade") == "0.2.0"
    du = md.distribution("sonospade").read_text("direct_url.json")
    if du:                                   # installed from git: must be the v0.2.0 commit
        info = json.loads(du)
        commit = info.get("vcs_info", {}).get("commit_id", "")
        assert commit.startswith("94b590686c8f"), commit


def test_vendored_v1_scripts_are_byte_identical():
    for name, sha in V1_SCRIPTS_SHA.items():
        assert _sha(os.path.join(ROOT, "scripts/v1", name)) == sha, name


@pytest.mark.skipif(not os.path.exists(V1_PAPER_CKPT), reason="v1 paper checkpoint not linked into data/")
def test_released_v1_checkpoint_unchanged_and_loads():
    from usbg import texture_gan as TG
    assert _sha(V1_PAPER_CKPT) == V1_PAPER_SHA
    tr = TG.load_translator(V1_PAPER_CKPT, device="cpu")
    assert isinstance(tr, TG.SonoSPADETranslator) and tr.n_classes == 13
    out = tr.translate_aligned(np.full((128, 128), 0.5, np.float32), np.ones((128, 128), np.int64))
    assert out.shape == (128, 128) and np.isfinite(out).all()


@pytest.mark.skipif(not (os.path.exists(V1_PAPER_CKPT) and os.path.exists(PAIRS)), reason="data not linked")
def test_label_swap_locality_harness_matches_v1_on_v1_checkpoint():
    from usbg import texture_eval as TE
    from usbg import texture_gan as TG
    from sonospadepp.v2 import metrics as M
    from sonospadepp.v2.evaldata import load_pairs
    tr = TG.SonoSPADETranslator(V1_PAPER_CKPT, device="cpu")
    imgs, labs, _ = load_pairs(PAIRS, 3)
    for img, lab in zip(imgs, labs):
        v1 = TE.label_swap_sensitivity(tr.translate_aligned, img, lab)
        v2 = M.label_swap_locality(tr.translate_aligned, img, lab, band=0)
        assert v1.keys() == v2.keys() and len(v1) > 0
        for c in v1:
            assert v2[c]["locality"] == pytest.approx(v1[c]["locality"], rel=1e-5)
        # the 8 px band only removes near-boundary pixels from the outside region
        v2b = M.label_swap_locality(tr.translate_aligned, img, lab, band=8)
        for c in v1:
            if np.isfinite(v2b[c]["locality"]):
                assert v2b[c]["out_region"] <= v1[c]["out_region"] * 1.5


def _v2_translator():
    from sonospadepp.v2.translator import SonoSPADEv2Translator
    return SonoSPADEv2Translator(V2_CKPT, device="cpu")


@pytest.mark.skipif(not V2_CKPT, reason="set SONOSPADE_V2_CKPT to a v2 checkpoint")
def test_v2_label_swap_locality_noninferior_on_sample():
    from sonospadepp.v2 import metrics as M
    from sonospadepp.v2.evaldata import load_pairs
    from usbg import texture_gan as TG
    imgs, labs, _ = load_pairs(PAIRS, 5)
    v1 = TG.SonoSPADETranslator(V1_PAPER_CKPT, device="cpu")
    v2 = _v2_translator()

    def mean_loc(f):
        vals = [d["locality"] for img, lab in zip(imgs, labs)
                for d in M.label_swap_locality(f, img, lab, band=8).values()]
        return float(np.nanmean(vals))
    assert mean_loc(v2.translate_aligned) >= 0.5 * mean_loc(v1.translate_aligned)   # smoke bound; the
    # real 95% non-inferiority gate runs on the full protocol over seeds (eval_v2.py, make_v2_tables.py)


@pytest.mark.skipif(not V2_CKPT, reason="set SONOSPADE_V2_CKPT to a v2 checkpoint with style codes")
def test_v2_style_swap_locality():
    from sonospadepp.v2 import metrics as M
    from sonospadepp.v2.evaldata import load_pairs
    v2 = _v2_translator()
    if not getattr(v2, "has_style", False):
        pytest.skip("checkpoint has no style codes")
    imgs, labs, _ = load_pairs(PAIRS, 5)
    rng = np.random.default_rng(0)
    ratios = []
    for img, lab in zip(imgs, labs):
        for t in np.unique(lab):
            a = {}
            b = {int(t): v2.sample_style(int(t), rng)}
            r = M.style_swap_locality(lambda im, lb, s: v2.translate_aligned(im, lb, style=s), img, lab, a, b,
                                      tissue=int(t))
            if np.isfinite(r["locality"]):
                ratios.append(r["locality"])
    assert ratios and np.mean(ratios) > 1.0


@pytest.mark.skipif(not (V2_CKPT and os.path.exists(V1_PAPER_CKPT)), reason="needs a v2 and the v1 checkpoint")
def test_latency_budget_cpu():
    import time
    import torch
    from usbg import texture_gan as TG
    torch.set_num_threads(4)
    img = np.random.default_rng(0).uniform(0, 1, (128, 128)).astype(np.float32)
    lab = np.random.default_rng(1).integers(0, 13, (128, 128))

    def bench(f, n=40, warm=8):
        for _ in range(warm):
            f(img, lab)
        t = time.perf_counter()
        for _ in range(n):
            f(img, lab)
        return (time.perf_counter() - t) / n

    v1 = TG.SonoSPADETranslator(V1_PAPER_CKPT, device="cpu")
    v2 = _v2_translator()
    t1 = min(bench(v1.translate_aligned) for _ in range(3))
    t2 = min(bench(v2.translate_aligned) for _ in range(3))
    assert t2 <= 1.25 * t1, (t1, t2)
