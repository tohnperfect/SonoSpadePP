"""The paper's numbers rebuild from the tracked result files, byte for byte.

scripts/v2/paper_macros_v2.py reads only files in this repository (results/v2, results/trusted, outputs/*.json) and
writes every number the manuscript prints as LaTeX macros, plus the selection table of the supplement. The hashes
below are those of the two files the manuscript was built with. If a result file changes on purpose, rerun the
script and update the hashes in the same commit.
"""
import hashlib
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MACROS_SHA256 = "77598f3e89f756bbdcb57d9735d47ac5abd802157b695e11523ee32375e83a97"
SEL_TABLE_SHA256 = "7ae65a5dab7d9c52b18d04a823fac05fbf72e9f0267c20932a7616d6aee2799e"
N_MACROS = 2280


def _sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def test_paper_macros_rebuild_byte_for_byte(tmp_path):
    out, tab = tmp_path / "results_macros.tex", tmp_path / "tab_selection.tex"
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "v2", "paper_macros_v2.py"),
                        "--out", str(out), "--sel-table", str(tab)], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert f"{N_MACROS} macros" in r.stdout
    assert _sha(out) == MACROS_SHA256
    assert _sha(tab) == SEL_TABLE_SHA256

def test_summary_tables_match_rows(tmp_path):
    """results/v2/metrics.csv and summary.csv are what make_v2_tables.py writes from results/v2/rows."""
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "v2", "make_v2_tables.py"),
                        "--out-dir", str(tmp_path)], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    for name in ("metrics.csv", "summary.csv"):
        assert _sha(tmp_path / name) == _sha(os.path.join(ROOT, "results", "v2", name)), name
