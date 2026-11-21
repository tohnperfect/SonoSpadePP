# Figures

Figures are rebuilt locally by the scripts in `scripts/v2/` (see the main README) and are ignored by
git, with two exceptions that are tracked so the qualitative results can be seen without the data:

| File | Script | What it shows |
|---|---|---|
| `qual.pdf` | `scripts/v2/fig_variants_v2.py` | Two held-out simulator frames from different CTs: label map, physics render, SonoSPADE, tier 0, tier 1 (seed 0 each), and in the last column a real evaluation frame of another person, not paired with the content. |
| `qual_baselines.pdf` | `scripts/v2/fig_qual_v2.py` | Held-out simulator frames through the label-free baselines (CUT, CycleGAN, S-CycleGAN), SonoSPADE and the USFM-segmenter model, with a last row of real evaluation frames of other people. |

`outputs/compare_panel.png` (`scripts/fig_compare.py`) belongs to the earlier CT-intensity study: the
same held-out poses through its variants, with a last row of real frames of other patients.

## Credit for the real ultrasound frames

The real B-mode frames in these three figures come from the abdominal ultrasound set of

> S. Vitale, J. I. Orlando, E. Iarussi, and I. Larrabide, "Improving Realism in Patient-Specific
> Abdominal Ultrasound Simulation Using CycleGANs," *International Journal of Computer Assisted
> Radiology and Surgery*, vol. 15, 2020, doi:10.1007/s11548-019-02046-5,

distributed by its authors on Kaggle as *ussimandsegm*. They are shown at 128 x 128 pixels for
comparison only. The set carries no stated license; the frames remain the property of their
authors and are not covered by this repository's Apache-2.0 license. If you reuse these figures,
keep this credit.

The same-patient figure of the manuscript (`trusted.pdf`, `scripts/v2/fig_trusted_v2.py`) shows
TRUSTED planes and is never committed.
