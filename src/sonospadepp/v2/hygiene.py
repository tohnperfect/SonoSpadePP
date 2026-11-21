"""Metric hygiene (leakage control): an encoder whose weights enter a model's training loss is
never used to evaluate that model. Two routes exist: the foundation-model discriminator / feature matching (A1,
`fm_disc`) and the segmenter S_US (A3), whose encoder may be initialized from a foundation model; S_US drives the
consistency loss and the OASIS pseudo-labels, so a USFM-initialized S_US makes USFM features part of the training
signal. (Found 2026-11-17: the USFM-initialized S_US cut the USFM tissue KID from 374 to 177 with no change on
DINOv2, and the DINOv2-initialized one did the mirror image.)

Evaluation rows carry the exclusion as `excluded_extractor` rows (value nan, extractor = the excluded encoder), so
gates and tables read it from the data."""
from __future__ import annotations

EVAL_EXTRACTORS = ("usfm", "dinov2")


def trained_extractors(ckpt_path) -> set:
    """Encoders in a checkpoint's training loss: its fm_disc and its S_US's initialization."""
    import os
    import torch
    out = set()
    if not ckpt_path or not os.path.exists(ckpt_path):
        return out
    st = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    fm = (st.get("v2_config") or {}).get("fm_disc", "none")
    if fm and fm != "none":
        out.add(fm)
    s_us = (st.get("run_config") or {}).get("s_us")
    if s_us and os.path.exists(s_us):
        seg = torch.load(s_us, map_location="cpu", weights_only=True)
        if isinstance(seg, dict) and seg.get("kind") == "fm_segmenter":
            out.add(seg["enc_name"])
    return out


def excluded_from_rows(rows, model) -> set:
    return {r["extractor"] for r in rows if r["model"] == model and r["metric"] == "excluded_extractor"}


def realism_extractor(rows, *models) -> str | None:
    """The primary extractor (USFM) unless any of the compared models was trained with it; then DINOv2; None if
    both are excluded for some model."""
    bad = set().union(*(excluded_from_rows(rows, m) for m in models))
    for e in EVAL_EXTRACTORS:
        if e not in bad:
            return e
    return None
