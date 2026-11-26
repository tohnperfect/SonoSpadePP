# SonoSpadePP

**Patient-faithful, real-time ultrasound synthesis from CT for teaching simulators.**

Code, result files and table scripts for the manuscript *Patient-Faithful Real-Time Ultrasound
Synthesis From CT for Teaching Simulators* (T. Intharah, Z. Gao, Y. Liu, H. Dong; under review).

[SonoSPADE](https://github.com/tohnperfect/SonoSpade) (MICCAI 2026 Workshops, SASHIMI) turns a
physics B-mode render computed from a patient's CT into ultrasound-like frames, conditioned per
tissue on the CT label map. This repository extends it and tests it in three ways that the
workshop paper did not: tissue by tissue, in the speckle, and against real ultrasound of the same
patient.

## What is in the study

The translator is trained at one of three levels of supervision.

| Tier | Training data | Scripts |
|---|---|---|
| 0 | Unlabeled real frames of other people (unpaired). Four additions to SonoSPADE are tested, each alone: a foundation-model critic with per-tissue feature matching, per-region style codes, a segmenter initialized from an ultrasound foundation model (USFM), and a per-tissue speckle term. | `scripts/v2/train_v2.py`, `scripts/v2/train_sus_v2.py`, `scripts/v2/run_wp*.sh` |
| 1 | Registered CT and ultrasound planes of other patients (TRUSTED, kidney), 5-fold by patient. | `scripts/train_m7.py --mode lab` |
| 2 | Tier 1 plus one registered image of the patient being simulated (one-shot fine-tune). | `scripts/adapt_m9a.py --mode lab` |

Endpoints:

* **Per-tissue realism**: kernel inception distance on 16 x 16 tissue crops in the feature space of
  a frozen encoder (USFM, DINOv2), against the 60 annotated real frames (`src/sonospadepp/v2/metrics.py`).
* **Speckle**: Wasserstein distance of the window-level log Nakagami shape parameter per tissue,
  and speckle size (`src/sonospadepp/v2/speckle.py`).
* **Per-tissue control**: edge IoU and label-swap locality, with and without an 8 px band.
* **Same-patient test**: correlation between the simulated plane and the real ultrasound plane of
  the same patient on TRUSTED (354 scored planes of 59 kidneys of 39 patients), with fixed
  registration and with an alignment tolerance, per-kidney Wilcoxon tests
  (`scripts/eval_trusted.py`, `scripts/eval_aligned.py`, `scripts/summarize_merge.py`).
* **Speed**: a whole frame, from CT label slice to translated image, at 128 x 128 pixels.
* **Closed loop** (supplement): the translators as the texture stage of
  [USB Gym](https://github.com/tohnperfect/USBGym).

Findings, as the result files in this repository give them:

* Tier 1 raised the detrended kidney correlation with the patient's own ultrasound from 0.277
  (an unpaired control trained on the same images) to 0.491, higher in 51 of 59 kidneys, and passed
  the criterion fixed before training. The cost is about half of the per-tissue control.
* Tier 2 did not beat the simpler baseline of warping the given image onto the other planes
  through the label maps (criterion failed).
* Without paired data, the physics render stayed closer to real ultrasound than SonoSPADE's output
  when judged by USFM features (tissue KID x 1000: 292 against 374) and by speckle statistics
  (W1 of log m: 0.75 against 1.34). None of the four tier-0 additions made the texture
  consistently more realistic.
* A whole frame runs at 43 frames per second on a laptop's integrated GPU and at 28 on its CPU
  through ONNX Runtime.
* An encoder that enters a model's training loss is never used to score that model
  (`src/sonospadepp/v2/hygiene.py`); two medical foundation models were excluded as evaluators
  because their training data include the evaluation frames.

For research simulation only, not for clinical use.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"          # pulls SonoSPADE v0.2.0 from GitHub; it provides the usbg package
pytest -q
```

Python 3.10 or newer. Everything runs on CPU or Apple-Silicon MPS (`export PYTORCH_ENABLE_MPS_FALLBACK=1`).
The checkpoint-dependent tests skip when no data or weights are present.

## Reproduce

There are three levels. Only the first works from a clone alone.

### 1. From the tracked result files (no data, no weights, seconds)

```bash
python -I scripts/v2/make_v2_tables.py      # results/v2/metrics.csv, results/v2/summary.csv
python -I scripts/v2/paper_macros_v2.py     # results/paper_macros/results_macros.tex, tab_selection.tex
```

`paper_macros_v2.py` writes every number the manuscript prints (2280 LaTeX macros) and the
selection table of the supplement. `tests/test_paper_macros.py` checks that the output is byte for
byte the file the manuscript was built with. Which files feed which table:

| Manuscript item | Tracked inputs |
|---|---|
| Table II (per-tissue realism and speckle, 60 annotated frames) | `results/v2/rows/*.csv`, `results/v2/sus.csv` |
| Table I (same-patient agreement on TRUSTED) | `outputs/m9_summary.json`, `outputs/merge_summary.json`, `results/trusted/*` |
| Table III (tier-0 components) | `results/v2/rows/*.csv`, `results/v2/gate_*.json`, `results/v2/selection*.csv` |
| Speed | `results/v2/latency.csv`, `results/v2/latency_endtoend.csv` |
| Supplement: selection table, closed loop | `results/v2/selection.csv`, `results/v2/usbgym*.json`, `results/v2/gate_t1g.json` |
| Fig. 3 (speckle distributions on the fan-geometry content, with tier 1), summary numbers | `results/v2/fig_speckle_fan.json` (`results/v2/fig_speckle.json`: the same plot on the content of Table II) |

`docs/TRUSTED_MERGE.md` and `docs/TRUSTED_M9.md` are the same-patient results as readable tables.

### 2. With the datasets and the released weights (metrics, figures, summaries)

Put the data in place first (see "Data"). Checkpoints are named in `WEIGHTS.md` with the paths the
scripts expect.

```bash
# Table II and III rows for one model (repeat per model and seed; the run_wp*.sh chains list them all)
python -I scripts/v2/eval_v2.py --model v1 --seed 0 --kind sonospade --ckpt runs/v2/v1_repro/sonospade_s0.pt --tstr-seeds 0 1 2
python -I scripts/v2/eval_sus_v2.py --name usfm --ckpt runs/v2/wp3/s_us_usfm.pt
python -I scripts/v2/bench_latency_v2.py --help          # per-model latency
python -I scripts/v2/bench_endtoend.py --help            # whole-frame time; needs a USB Gym checkout

# Table I: score on TRUSTED, then summarize (needs TRUSTED; see scripts/v2/merge/*.sh for every row)
python -I scripts/eval_trusted.py --pairs data/trusted_pairs_def.npz --ckpt b0_s0=runs/b0_s0.pt --append --out outputs/trusted_def.json
python -I scripts/summarize_m9.py           # outputs/m9_summary.json, docs/TRUSTED_M9.md
python -I scripts/summarize_merge.py        # outputs/merge_summary.json, docs/TRUSTED_MERGE.md

# Figures (written to figures/, CPU)
python -I scripts/v2/fig_speckle_v2.py --pairs data/dataset/pairs_spp_eval200.npz --summary results/v2/fig_speckle_fan.json \
    --stage "Physics render=physics:" \
    --stage "SonoSPADE=sonospade:runs/b0_s0.pt,runs/b0_s1.pt,runs/b0_s2.pt" \
    --stage "Tier 1=paired_lab:runs/b0_s0.pt::runs/m7/lab_s0_f0.pt,runs/b0_s1.pt::runs/m7/lab_s1_f0.pt,runs/b0_s2.pt::runs/m7/lab_s2_f0.pt"    # Fig. 3
python -I scripts/v2/fig_trusted_v2.py      # Fig. 2, same-patient examples (TRUSTED images: never commit the output)
python -I scripts/v2/fig_variants_v2.py     # Fig. 4, figures/qual.pdf
python -I scripts/v2/fig_control_v2.py --only-v1    # Fig. 5, figures/control.pdf
python -I scripts/v2/fig_overview_assets.py  # the image thumbnails of Fig. 1 (public CT only)
```

The per-plane score files that `eval_trusted.py` and `eval_aligned.py` write
(`outputs/trusted*.json`) are not in this repository; `summarize_m9.py`, `summarize_merge.py` and
`fig_trusted_v2.py` need them and the TRUSTED planes. Fig. 1 of the manuscript is a diagram; `fig_overview_assets.py`
writes its image thumbnails.

### 3. Full retraining

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
bash scripts/v2/run_wp0.sh                  # SonoSPADE retrained, three seeds, and the evaluation harness
bash scripts/v2/run_queue.sh                # the tier-0 components, selection, gates (run_wp1..wp5, wpfinal)
bash scripts/run_compare.sh                 # fan-geometry pairs, segmenter and SonoSPADE for the same-patient study
bash scripts/trusted_totalseg.sh            # TotalSegmentator labels for every TRUSTED CT
bash scripts/run_trusted.sh                 # register, cut the native planes, render, first scores
bash scripts/run_hires_prepare.sh           # registration (rigid, then deformable) and the 128 px planes
bash scripts/run_dense_prepare.sh           # dense training planes
bash scripts/run_b0t.sh                     # unpaired control on the TRUSTED planes
bash scripts/run_m7.sh                      # tier 1 (and its CT-intensity variant), 5 folds x 3 seeds
bash scripts/run_m9.sh; bash scripts/run_m9_eval.sh     # tier 2 and the label-warp baseline, scoring
bash scripts/v2/merge/run_merge_v2.sh       # tiers 0 to 2 started from the USFM-segmenter model
bash scripts/v2/merge/run_paired_checks.sh  # pix2pix baseline; tier 1 on the 60 annotated frames
bash scripts/v2/merge/run_tier1_fan.sh      # tier 1 side effects on its own render
bash scripts/v2/usbgym/run_usbgym_v2.sh     # closed loop (needs USBG_GYM_ROOT and USBG_SEGTRAIN_ROOT)
bash scripts/v2/usbgym/run_usbgym_t1g.sh    # tier 1 in the closed loop
```

The chains are the ones the study ran; they are resumable and run one heavy job at a time. They
were run on Apple-Silicon hardware and have not been rerun from this export. The study used two
working copies of this repository, one for tier 0 and one for the same-patient work; the chains
default to a single copy and accept `SONOSPADE_V2_ROOT` and `SONOSPADEPP_ROOT` to split them.

Environment variables: `SONOSPADE_FM_DIR` (foundation-model weights, default `data/fm`),
`USSIM_SRC` (the Kaggle download, read by `scripts/v2/make_splits.py`; `scripts/v1/ingest_kaggle_us.py` takes `--src`), `USBG_GYM_ROOT` (a USB Gym checkout), `USBG_SEGTRAIN_ROOT`
(USB Gym's segmenter-training companion), `SONOSPADE_V2_CKPT` (a tier-0 checkpoint for the
invariant tests).

## Data

Nothing below is redistributed here. Each set keeps its own terms.

| Data | Used for | Source | Terms |
|---|---|---|---|
| TotalSegmentator CTs with segmentations | simulator content (label maps, physics render) | Wasserthal et al., Radiol. Artif. Intell. 2023, doi:10.1148/ryai.230024 | CC BY 4.0 |
| Abdominal ultrasound set *ussimandsegm* | unlabeled real pool (404 frames) and the 60 annotated evaluation frames | Vitale et al., Int. J. Comput. Assist. Radiol. Surg. 2020, doi:10.1007/s11548-019-02046-5; distributed on Kaggle | no license stated by its authors |
| TRUSTED | paired kidney CT and 3D ultrasound: tiers 1 and 2, the same-patient test | Ndzimbong et al., Sci. Data 2025, doi:10.1038/s41597-025-04467-1 | data usage agreement with IRCAD France; request access from the dataset's authors |
| USFM weights | evaluation features; segmenter initialization | Jiao et al., Med. Image Anal. 2024, doi:10.1016/j.media.2024.103202; `USFM_latest.pth` from the openmedlab/USFM repository | CC BY-NC 4.0 |
| DINOv2 ViT-B/14 weights | evaluation features; critic | Oquab et al., TMLR 2024; timm `vit_base_patch14_dinov2.lvd142m` | Apache-2.0 |
| EchoCare weights | training only, never evaluation | Zhang et al., arXiv:2509.11752; CAIR-HKISI/EchoCare | no license stated |

This work uses the TRUSTED dataset and acknowledges it as its agreement requires:
W. Ndzimbong, C. Fourniol, L. Themyr, N. Thome, Y. Keeza, B. Sauer, P.-T. Piechaud, A. Mejean,
J. Marescaux, D. George, D. Mutter, A. Hostettler, and T. Collins, "TRUSTED: The Paired 3D
Transabdominal Ultrasound and CT Human Data for Kidney Segmentation and Registration Research,"
*Scientific Data*, 2025, doi:10.1038/s41597-025-04467-1. No TRUSTED volume, plane or image derived
from one is in this repository, and none may be added (`.gitignore` blocks the usual paths).

Layout the scripts expect:

```
data/raw/totalseg_small/<case>/{ct.nii.gz,segmentations/}   TotalSegmentator cases
data/raw/ussimandsegm/                                     the Kaggle download (or pass its location, see below)
data/real/kaggle_vitale/pool/, data/eval/kaggle_vitale_test.npz   built by scripts/v1/ingest_kaggle_us.py
data/dataset/pairs_med.npz                                 simulator pairs, scripts/v1/make_dataset.py
data/v1_ckpt/                                              the workshop paper's SonoSPADE generator (exp_real_sonospade.pt,
                                                           the v0.2.0 release's sonospade_sashimi_linear.pt) and the label-free
                                                           baselines (exp_real_cut.pt, exp_real_cyclegan.pt, exp_real_cyclegan_sc.pt)
data/dataset/s_us_grouped.pt                               the v0.2.0 release's s_us_sashimi_linear.pt
data/fm/USFM_latest.pth, data/fm/echocare/echocare_encoder.pth    foundation-model weights
data/trusted/raw/                                          TRUSTED as delivered
data/splits_v2.json                                        the fixed real-frame split (tracked)
```

`scripts/link_data.sh` links an existing SonoSPADE data folder into the names the same-patient
scripts use (`SRC=/path/to/data bash scripts/link_data.sh`).

## Weights

Trained checkpoints are release assets of this repository, not files in git. `WEIGHTS.md` lists
each one with its SHA-256, the path the scripts expect, and its terms: CC BY-NC 4.0, with
attribution to this work, to USFM and to TRUSTED. Models trained with EchoCare features are not
released because EchoCare's weights carry no license. Tier 2 has no checkpoint: it is a one-shot
fine-tune run per patient at evaluation time (`scripts/adapt_m9a.py`).

## Figures with real ultrasound frames

`figures/qual.pdf`, `figures/qual_baselines.pdf` and `outputs/compare_panel.png` show real frames
of the abdominal ultrasound set of Vitale et al. beside simulated ones, for comparison. Credit and
details are in `figures/README.md`. All other figures are rebuilt locally by the scripts above.

## Layout

```
src/sonospadepp/        physics render driven by per-pixel maps, CT volumes and slicing, registration,
                        TRUSTED geometry, paired translator, metrics of the same-patient study
src/sonospadepp/v2/     tier-0 components, per-tissue and speckle metrics, foundation-model encoders,
                        metric hygiene, evaluation data
scripts/                same-patient study: data preparation, trainers (train_b0t, train_m7, adapt_m9a,
                        train_m9), scoring and summaries; the earlier CT-intensity study (M1 to M6)
scripts/v1/             scripts of the SonoSPADE v0.2.0 release, for data preparation and the retrained baseline
                        (six of them are pinned byte for byte to that release by a test)
scripts/v2/             tier-0 training, evaluation, selection, gates, tables, macros and figures
scripts/v2/merge/       chains that combine the tiers; scripts/v2/usbgym/ the closed loop
results/v2/             per-seed metric rows, summaries, gates, selection runs, latency, closed loop
results/trusted/        generated number files of the same-patient study that the macro script reads
outputs/                same-patient summaries; results of the earlier CT-intensity study
docs/                   result tables as Markdown; RESULTS.md and TRUSTED_*.md cover the earlier
                        CT-intensity study (HU variants M1 to M6), whose pipeline tiers 1 and 2 reuse
tests/                  unit tests and the macro regression test
```

## Citing

See `CITATION.cff`. The releases are archived on Zenodo:
[doi:10.5281/zenodo.24548095](https://doi.org/10.5281/zenodo.24548095) for all versions, and
[doi:10.5281/zenodo.24548096](https://doi.org/10.5281/zenodo.24548096) for version 1.0.0. Please also cite SonoSPADE,
and the datasets and models you use through this code (table above).

## License

Code: Apache-2.0 (`LICENSE`). Built on SonoSPADE v0.2.0 (Apache-2.0). Released weights: CC BY-NC 4.0
(`WEIGHTS.md`). Result files: Apache-2.0 with the code.
