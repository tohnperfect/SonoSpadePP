#!/usr/bin/env python3
"""Add `excluded_extractor` rows (metric hygiene, src/sonospadepp/v2/hygiene.py) to evaluation rows written before
eval_v2.py recorded them (2026-11-17). Each rows file is matched to its checkpoint by the SHA-256 in its rows; the
excluded encoders are read from that checkpoint (fm_disc and the S_US's initialization). Idempotent.

  python -I scripts/v2/mark_excluded.py [--rows-dir results/v2/rows] [--runs runs/v2 data]
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(argv=None):
    from sonospadepp.v2.hygiene import trained_extractors
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows-dir", default="results/v2/rows")
    ap.add_argument("--runs", nargs="+", default=["runs/v2"])
    a = ap.parse_args(argv)
    by_sha = {}
    for root in a.runs:
        for p in glob.glob(os.path.join(root, "**", "*.pt"), recursive=True):
            by_sha.setdefault(sha256(p), p)
    for f in sorted(glob.glob(os.path.join(a.rows_dir, "*.csv"))):
        rows = list(csv.DictReader(open(f)))
        if not rows or any(r["metric"] == "excluded_extractor" for r in rows):
            continue
        shas = {r["ckpt_sha256"] for r in rows if r.get("ckpt_sha256")}
        ck = [by_sha[s] for s in shas if s in by_sha]
        if len(ck) != 1:
            print(f"{os.path.basename(f)}: checkpoint {'not found' if not ck else 'ambiguous'}; skipped")
            continue
        ex = trained_extractors(ck[0])
        if not ex:
            continue
        r0 = rows[0]
        new = [dict(r0, metric="excluded_extractor", tissue="all", value="nan", ci_lo="nan", ci_hi="nan", n="",
                    extractor=e, dr_db="", protocol="metric hygiene: in the training loss (added by mark_excluded.py)")
               for e in sorted(ex)]
        with open(f, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(r0.keys()))
            for r in new:
                w.writerow(r)
        print(f"{os.path.basename(f)}: {ck[0]} -> excluded {sorted(ex)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
