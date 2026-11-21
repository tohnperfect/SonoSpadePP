# One real US image given (M9)

For every kidney the central native plane (4th of 7 by sweep angle) is the one real US image given; the other 354 native planes are scored, for every row. Agreement: depth-detrended Pearson r of the 1.5 px smoothed images; per-kidney means, mean over the three seeds; comparisons by Wilcoxon signed-rank over kidneys. Angle bins: distance of the scored plane from the given plane.

## fixed registration

| row | seeds | r kidney | r kidney+band | kidney, 0-7 deg | kidney, 7-13 deg | kidney, 13-90 deg |
|---|---|---|---|---|---|---|
| SonoSPADE B0 (unpaired, Kaggle) | 3 | +0.247 | +0.280 | +0.278 | +0.312 | +0.180 |
| B0t: SonoSPADE recipe fine-tuned on TRUSTED | 3 | +0.277 | +0.283 | +0.328 | +0.338 | +0.218 |
| Approach 1 (M7c: paired training) | 3 | +0.491 | +0.533 | +0.586 | +0.562 | +0.410 |
| Approach 2 (M9ac: + one-shot fine-tune on the image) | 3 | +0.583 | +0.646 | +0.731 | +0.662 | +0.474 |
| aux: given image copied | 1 | +0.372 | +0.458 | +0.684 | +0.444 | +0.231 |
| aux: given image warped by labels | 1 | +0.473 | +0.518 | +0.718 | +0.561 | +0.325 |
| aux: approach 1 + HU (M7) | 3 | +0.493 | +0.538 | +0.588 | +0.573 | +0.405 |
| aux: approach 2 + HU (M9a) | 3 | +0.586 | +0.649 | +0.739 | +0.669 | +0.473 |
| aux: per-patient, half sweep (M8c) | 3 | +0.564 | +0.629 | +0.701 | +0.652 | +0.454 |
| aux: M8c + HU (M8) | 3 | +0.576 | +0.639 | +0.708 | +0.663 | +0.476 |
| aux: exemplar warp alone (M9c W) | 3 | +0.410 | +0.469 | +0.651 | +0.504 | +0.277 |
| aux: exemplar warp alone + HU (M9 W) | 3 | +0.407 | +0.466 | +0.646 | +0.502 | +0.274 |
| aux: exemplar network (M9c) | 3 | +0.523 | +0.579 | +0.630 | +0.599 | +0.434 |
| aux: exemplar network + HU (M9) | 3 | +0.527 | +0.583 | +0.632 | +0.611 | +0.434 |

| comparison | what it tests | median per-kidney diff (kidney) | p | kidneys up | diff (band) | p |
|---|---|---|---|---|---|---|
| m7c - b0t | approach 1 vs domain-matched SonoSPADE | +0.232 | 1.6e-09 | 51/59 | +0.279 | 2.2e-10 |
| b0t - b0 | scanner domain only | +0.010 | 0.27 | 31/59 | -0.020 | 0.75 |
| m7c - b0 | approach 1 vs SonoSPADE | +0.245 | 4.3e-10 | 53/59 | +0.253 | 2.7e-10 |
| m9ac - m7c | approach 2 vs approach 1 | +0.067 | 2.6e-09 | 50/59 | +0.087 | 5.1e-11 |
| m9ac - labwarp | approach 2 vs the image warped by labels | +0.110 | 4.6e-07 | 48/59 | +0.121 | 4.7e-09 |
| m9ac - b0t | approach 2 vs domain-matched SonoSPADE | +0.320 | 2.6e-11 | 58/59 | +0.372 | 2.4e-11 |
| m7 - m7c | HU, approach 1 | +0.008 | 0.47 | 34/59 | +0.007 | 0.21 |
| m9a - m9ac | HU, approach 2 | +0.007 | 0.22 | 34/59 | +0.006 | 0.18 |
| m8 - m8c | HU, per-patient | +0.008 | 0.014 | 39/59 | +0.008 | 0.0081 |
| m9 - m9c | HU, exemplar network | +0.007 | 0.37 | 35/59 | +0.007 | 0.17 |
| m9c - m7c | exemplar network vs approach 1 | +0.028 | 0.0005 | 38/59 | +0.029 | 1.3e-07 |
| m9ac - m9c | one-shot fine-tune vs exemplar network | +0.029 | 4.3e-06 | 45/59 | +0.049 | 3.3e-08 |
| m9ac - m8c | one image vs half the sweep | +0.015 | 0.013 | 36/59 | +0.014 | 0.004 |

## alignment-tolerant (smooth warp of at most 5 mm)

| row | seeds | r kidney | r kidney+band | kidney, 0-7 deg | kidney, 7-13 deg | kidney, 13-90 deg |
|---|---|---|---|---|---|---|
| SonoSPADE B0 (unpaired, Kaggle) | 3 | +0.629 | +0.615 | +0.614 | +0.661 | +0.612 |
| B0t: SonoSPADE recipe fine-tuned on TRUSTED | 3 | +0.628 | +0.601 | +0.636 | +0.650 | +0.617 |
| Approach 1 (M7c: paired training) | 3 | +0.736 | +0.723 | +0.755 | +0.766 | +0.711 |
| Approach 2 (M9ac: + one-shot fine-tune on the image) | 3 | +0.781 | +0.786 | +0.835 | +0.809 | +0.743 |
| aux: given image copied | 1 | +0.679 | +0.697 | +0.823 | +0.723 | +0.609 |
| aux: given image warped by labels | 1 | +0.729 | +0.725 | +0.839 | +0.767 | +0.660 |
| aux: approach 1 + HU (M7) | 3 | +0.737 | +0.729 | +0.762 | +0.773 | +0.705 |
| aux: approach 2 + HU (M9a) | 3 | +0.784 | +0.790 | +0.842 | +0.814 | +0.741 |
| aux: per-patient, half sweep (M8c) | 3 | +0.772 | +0.777 | +0.823 | +0.803 | +0.733 |
| aux: M8c + HU (M8) | 3 | +0.781 | +0.785 | +0.833 | +0.810 | +0.745 |
| aux: exemplar warp alone (M9c W) | 3 | +0.695 | +0.684 | +0.806 | +0.736 | +0.629 |
| aux: exemplar warp alone + HU (M9 W) | 3 | +0.696 | +0.683 | +0.806 | +0.735 | +0.633 |
| aux: exemplar network (M9c) | 3 | +0.755 | +0.753 | +0.789 | +0.783 | +0.722 |
| aux: exemplar network + HU (M9) | 3 | +0.755 | +0.755 | +0.789 | +0.792 | +0.716 |

| comparison | what it tests | median per-kidney diff (kidney) | p | kidneys up | diff (band) | p |
|---|---|---|---|---|---|---|
| m7c - b0t | approach 1 vs domain-matched SonoSPADE | +0.110 | 4.7e-09 | 52/59 | +0.123 | 1.5e-09 |
| b0t - b0 | scanner domain only | -0.011 | 0.65 | 26/59 | -0.014 | 0.092 |
| m7c - b0 | approach 1 vs SonoSPADE | +0.107 | 6.5e-09 | 52/59 | +0.089 | 5.9e-09 |
| m9ac - m7c | approach 2 vs approach 1 | +0.031 | 3.7e-08 | 49/59 | +0.046 | 1.5e-10 |
| m9ac - labwarp | approach 2 vs the image warped by labels | +0.049 | 9.4e-06 | 43/59 | +0.057 | 1.1e-07 |
| m9ac - b0t | approach 2 vs domain-matched SonoSPADE | +0.145 | 2.8e-11 | 58/59 | +0.177 | 2.5e-11 |
| m7 - m7c | HU, approach 1 | +0.000 | 0.96 | 30/59 | +0.003 | 0.11 |
| m9a - m9ac | HU, approach 2 | +0.004 | 0.41 | 35/59 | +0.004 | 0.1 |
| m8 - m8c | HU, per-patient | +0.011 | 0.011 | 37/59 | +0.008 | 0.0084 |
| m9 - m9c | HU, exemplar network | -0.001 | 0.8 | 29/59 | +0.005 | 0.26 |
| m9c - m7c | exemplar network vs approach 1 | +0.015 | 0.0016 | 39/59 | +0.024 | 1.8e-06 |
| m9ac - m9c | one-shot fine-tune vs exemplar network | +0.018 | 0.00049 | 42/59 | +0.030 | 2.2e-07 |
| m9ac - m8c | one image vs half the sweep | +0.009 | 0.02 | 38/59 | +0.006 | 0.017 |
