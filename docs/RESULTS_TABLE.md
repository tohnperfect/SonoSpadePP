Held-out CT content: 330 frames from 11 CTs (s0519, s0522, s0591, s0649, s0703, s0720, s0727, s0762, s0777, s1086, s1373); real reference: 60 labeled Kaggle ussimandsegm frames.
Trained rows: mean ± SD over GAN seeds (n per row in the last column). TSTR per row is itself a mean over 3 segmenter seeds.

| Row | organ W1 ↓ | heterogeneity gap ↓ | TSTR Dice ↑ | edge IoU ↑ | locality ↑ | HU sensitivity | n |
|---|---|---|---|---|---|---|---|
| physics render, base labels | 0.056 | 0.0657 | 0.101 | 1.000 | n/a | n/a | 1 |
| physics render, M1 labels | 0.054 | 0.0622 | 0.112 | 1.000 | n/a | n/a | 1 |
| physics render, HU-modulated | 0.055 | 0.0688 | 0.103 | 1.000 | n/a | n/a | 1 |
| B0 SonoSPADE (labels + physics) | 0.089 ± 0.008 | 0.0514 ± 0.0020 | 0.109 ± 0.002 | 0.552 ± 0.006 | 25.7 ± 0.6 | 0.0000 ± 0.0000 | 3 |
| M1 CT sub-labels | 0.125 ± 0.007 | 0.0589 ± 0.0097 | 0.107 ± 0.008 | 0.554 ± 0.016 | 25.4 ± 2.6 | 0.0000 ± 0.0000 | 3 |
| M2 HU-modulated physics | 0.099 ± 0.003 | 0.0574 ± 0.0005 | 0.106 ± 0.008 | 0.526 ± 0.002 | 29.0 ± 3.3 | 0.0000 ± 0.0000 | 3 |
| M3 HU as a SPADE channel | 0.102 ± 0.022 | 0.0530 ± 0.0053 | 0.113 ± 0.000 | 0.541 ± 0.022 | 22.6 ± 3.1 | 0.0030 ± 0.0003 | 3 |

Per-organ intensity W1 to real (lower is better):

| Row | liver | gallbladder | vessel | kidney | spleen |
|---|---|---|---|---|---|
| physics render, base labels | 0.022 | 0.041 | 0.098 | 0.064 | n/a |
| physics render, M1 labels | 0.023 | 0.040 | 0.099 | 0.054 | n/a |
| physics render, HU-modulated | 0.025 | 0.040 | 0.091 | 0.066 | n/a |
| B0 SonoSPADE (labels + physics) | 0.047 ± 0.005 | 0.059 ± 0.007 | 0.116 ± 0.012 | 0.133 ± 0.023 | n/a |
| M1 CT sub-labels | 0.090 ± 0.006 | 0.088 ± 0.009 | 0.168 ± 0.011 | 0.152 ± 0.006 | n/a |
| M2 HU-modulated physics | 0.069 ± 0.001 | 0.047 ± 0.022 | 0.133 ± 0.006 | 0.149 ± 0.027 | n/a |
| M3 HU as a SPADE channel | 0.057 ± 0.017 | 0.075 ± 0.014 | 0.134 ± 0.030 | 0.142 ± 0.029 | n/a |

Low-frequency heterogeneity (SD of 2-px-smoothed intensity inside the organ; real in brackets):

| Row | liver | kidney | spleen |
|---|---|---|---|
| physics render, base labels | 0.1084 [0.1758] | 0.0291 [0.0932] | n/a |
| physics render, M1 labels | 0.1104 [0.1758] | 0.0343 [0.0932] | n/a |
| physics render, HU-modulated | 0.1033 [0.1758] | 0.0280 [0.0932] | n/a |
| B0 SonoSPADE (labels + physics) | 0.1127 ± 0.0014 [0.1758] | 0.0534 ± 0.0028 [0.0932] | n/a |
| M1 CT sub-labels | 0.1011 ± 0.0116 [0.1758] | 0.0500 ± 0.0080 [0.0932] | n/a |
| M2 HU-modulated physics | 0.1010 ± 0.0017 [0.1758] | 0.0531 ± 0.0013 [0.0932] | n/a |
| M3 HU as a SPADE channel | 0.1139 ± 0.0087 [0.1758] | 0.0490 ± 0.0020 [0.0932] | n/a |

TSTR per-class Dice on the real frames:

| Row | liver | gallbladder | vessel | kidney | spleen |
|---|---|---|---|---|---|
| physics render, base labels | 0.395 | 0.036 | 0.075 | 0.000 | 0.000 |
| physics render, M1 labels | 0.427 | 0.018 | 0.115 | 0.000 | 0.000 |
| physics render, HU-modulated | 0.428 | 0.020 | 0.065 | 0.000 | 0.000 |
| B0 SonoSPADE (labels + physics) | 0.416 ± 0.013 | 0.060 ± 0.017 | 0.037 ± 0.010 | 0.033 ± 0.002 | 0.000 ± 0.000 |
| M1 CT sub-labels | 0.401 ± 0.041 | 0.073 ± 0.011 | 0.036 ± 0.010 | 0.025 ± 0.001 | 0.000 ± 0.000 |
| M2 HU-modulated physics | 0.402 ± 0.036 | 0.056 ± 0.004 | 0.045 ± 0.011 | 0.027 ± 0.004 | 0.000 ± 0.000 |
| M3 HU as a SPADE channel | 0.412 ± 0.014 | 0.069 ± 0.006 | 0.042 ± 0.004 | 0.039 ± 0.009 | 0.000 ± 0.000 |
