# Pass criteria of the same-patient test on TRUSTED

Per-kidney means over the 354 non-given native planes, mean over seeds; Wilcoxon over kidneys; pass = p < 0.05 with a positive median difference (for W1, 'not worse' = not significantly higher).

## approach 1 vs B0t (m7c - b0t): **PASS**

| measure | bin | median diff | p | kidneys up | pass |
|---|---|---|---|---|---|
| kidney dt | all | +0.232 | 1.6e-09 | 51/59 | yes |
| band dt | all | +0.279 | 2.2e-10 | 53/59 | yes |
| kidney | all | +0.489 | 4e-11 | 56/59 | yes |
| fan | all | +0.167 | 1.4e-09 | 52/59 | yes |
| kidney al | all | +0.110 | 4.7e-09 | 52/59 | yes |
| band al | all | +0.123 | 1.5e-09 | 52/59 | yes |
| kidney W1 (lower is better) | all | -0.0452 | 2.4e-09 | | not worse |

## approach 2 vs the label-warped given image (m9ac - labwarp): **FAIL**

| measure | bin | median diff | p | kidneys up | pass |
|---|---|---|---|---|---|
| kidney dt | 0-7 deg | +0.046 | 0.3 | 19/31 | no |
| kidney | 0-7 deg | -0.189 | 2.4e-07 | 4/31 | no |
| fan | 0-7 deg | -0.458 | 9.3e-10 | 0/31 | no |
| kidney al | 0-7 deg | +0.012 | 0.92 | 16/31 | no |
| kidney dt | 7-13 deg | +0.087 | 1.4e-07 | 49/58 | yes |
| kidney | 7-13 deg | -0.146 | 5.4e-06 | 12/58 | no |
| fan | 7-13 deg | -0.454 | 3.5e-11 | 0/58 | no |
| kidney al | 7-13 deg | +0.044 | 1.8e-05 | 45/58 | yes |
| kidney dt | 13-90 deg | +0.148 | 3.2e-05 | 40/52 | yes |
| kidney | 13-90 deg | -0.079 | 0.094 | 21/52 | no |
| fan | 13-90 deg | -0.191 | 6.6e-09 | 4/52 | no |
| kidney al | 13-90 deg | +0.056 | 5.1e-05 | 40/52 | yes |

## approach 2 vs approach 1 (m9ac - m7c): **FAIL**

| measure | bin | median diff | p | kidneys up | pass |
|---|---|---|---|---|---|
| kidney dt | 0-7 deg | +0.116 | 9.3e-10 | 31/31 | yes |
| kidney | 0-7 deg | +0.068 | 0.00076 | 24/31 | yes |
| fan | 0-7 deg | +0.028 | 1.6e-06 | 26/31 | yes |
| kidney al | 0-7 deg | +0.054 | 9.3e-10 | 31/31 | yes |
| kidney dt | 7-13 deg | +0.083 | 1.7e-08 | 49/58 | yes |
| kidney | 7-13 deg | +0.027 | 0.019 | 35/58 | yes |
| fan | 7-13 deg | +0.024 | 1.9e-08 | 51/58 | yes |
| kidney al | 7-13 deg | +0.035 | 6e-07 | 46/58 | yes |
| kidney dt | 13-90 deg | +0.046 | 0.00047 | 36/52 | yes |
| kidney | 13-90 deg | +0.022 | 0.077 | 32/52 | no |
| fan | 13-90 deg | +0.021 | 5.3e-06 | 41/52 | yes |
| kidney al | 13-90 deg | +0.018 | 0.0012 | 35/52 | yes |

## v2 vs v1 on TRUSTED, no fine-tuning (reported either way) (b0_v2 - b0): **reported**

| measure | bin | median diff | p | kidneys up | pass |
|---|---|---|---|---|---|
| kidney dt | all | -0.022 | 0.039 | 24/59 | no |
| band dt | all | -0.019 | 0.0069 | 20/59 | no |
| kidney | all | -0.015 | 0.0056 | 22/59 | no |
| fan | all | +0.019 | 0.00019 | 42/59 | yes |
| kidney al | all | -0.009 | 0.056 | 25/59 | no |
| band al | all | -0.010 | 0.017 | 23/59 | no |

## approach 1 from v2 vs B0t-v2 (m7c_v2 - b0t_v2): **PASS**

| measure | bin | median diff | p | kidneys up | pass |
|---|---|---|---|---|---|
| kidney dt | all | +0.326 | 2.1e-10 | 56/59 | yes |
| band dt | all | +0.312 | 8.5e-11 | 56/59 | yes |
| kidney | all | +0.600 | 3.3e-11 | 57/59 | yes |
| fan | all | +0.223 | 4e-11 | 57/59 | yes |
| kidney al | all | +0.105 | 7.7e-08 | 50/59 | yes |
| band al | all | +0.127 | 8.4e-08 | 47/59 | yes |
| kidney W1 (lower is better) | all | -0.0985 | 3.2e-10 | | not worse |

## approach 1 from v2 vs approach 1 from v1 (m7c_v2 - m7c): **FAIL**

| measure | bin | median diff | p | kidneys up | pass |
|---|---|---|---|---|---|
| kidney dt | all | -0.001 | 0.8 | 29/59 | no |
| band dt | all | -0.006 | 0.12 | 24/59 | no |
| kidney | all | +0.008 | 0.043 | 34/59 | yes |
| fan | all | +0.016 | 0.0052 | 41/59 | yes |
| kidney al | all | -0.003 | 0.42 | 27/59 | no |
| band al | all | -0.005 | 0.15 | 26/59 | no |
| kidney W1 (lower is better) | all | +0.0058 | 0.0079 | | worse |

## approach 2 from v2 vs the label-warped given image (m9ac_v2 - labwarp): **FAIL**

| measure | bin | median diff | p | kidneys up | pass |
|---|---|---|---|---|---|
| kidney dt | 0-7 deg | +0.040 | 0.35 | 19/31 | no |
| kidney | 0-7 deg | -0.177 | 2.4e-07 | 4/31 | no |
| fan | 0-7 deg | -0.463 | 9.3e-10 | 0/31 | no |
| kidney al | 0-7 deg | +0.000 | 0.98 | 16/31 | no |
| kidney dt | 7-13 deg | +0.095 | 5.4e-07 | 48/58 | yes |
| kidney | 7-13 deg | -0.120 | 1.2e-05 | 12/58 | no |
| fan | 7-13 deg | -0.443 | 3.5e-11 | 0/58 | no |
| kidney al | 7-13 deg | +0.038 | 8.7e-06 | 45/58 | yes |
| kidney dt | 13-90 deg | +0.159 | 0.00011 | 40/52 | yes |
| kidney | 13-90 deg | -0.085 | 0.17 | 22/52 | no |
| fan | 13-90 deg | -0.174 | 1.1e-08 | 3/52 | no |
| kidney al | 13-90 deg | +0.068 | 6.4e-05 | 39/52 | yes |

## approach 2 from v2 vs approach 1 from v2 (m9ac_v2 - m7c_v2): **FAIL**

| measure | bin | median diff | p | kidneys up | pass |
|---|---|---|---|---|---|
| kidney dt | 0-7 deg | +0.107 | 9.3e-10 | 31/31 | yes |
| kidney | 0-7 deg | +0.056 | 0.012 | 23/31 | yes |
| fan | 0-7 deg | +0.025 | 1.4e-05 | 25/31 | yes |
| kidney al | 0-7 deg | +0.065 | 4.7e-09 | 29/31 | yes |
| kidney dt | 7-13 deg | +0.074 | 1e-08 | 48/58 | yes |
| kidney | 7-13 deg | +0.028 | 0.046 | 37/58 | yes |
| fan | 7-13 deg | +0.026 | 2.9e-08 | 51/58 | yes |
| kidney al | 7-13 deg | +0.037 | 1.1e-06 | 46/58 | yes |
| kidney dt | 13-90 deg | +0.057 | 0.00012 | 38/52 | yes |
| kidney | 13-90 deg | +0.022 | 0.13 | 34/52 | no |
| fan | 13-90 deg | +0.016 | 3.4e-06 | 41/52 | yes |
| kidney al | 13-90 deg | +0.022 | 0.0017 | 36/52 | yes |

## approach 1 vs the pix2pix baseline (m7c - p2p): **FAIL**

| measure | bin | median diff | p | kidneys up | pass |
|---|---|---|---|---|---|
| kidney dt | all | +0.084 | 2.8e-09 | 55/59 | yes |
| band dt | all | +0.104 | 9.7e-10 | 55/59 | yes |
| kidney | all | -0.064 | 0.0053 | 21/59 | no |
| fan | all | -0.184 | 1.2e-07 | 13/59 | no |
| kidney al | all | +0.043 | 5.8e-10 | 52/59 | yes |
| band al | all | +0.070 | 9.9e-11 | 55/59 | yes |
| kidney W1 (lower is better) | all | +0.0133 | 0.0025 | | worse |

## pix2pix baseline vs B0t (reported either way) (p2p - b0t): **reported**

| measure | bin | median diff | p | kidneys up | pass |
|---|---|---|---|---|---|
| kidney dt | all | +0.148 | 5e-06 | 46/59 | yes |
| band dt | all | +0.159 | 1.9e-06 | 46/59 | yes |
| kidney | all | +0.589 | 3.1e-10 | 55/59 | yes |
| fan | all | +0.314 | 7.7e-11 | 55/59 | yes |
| kidney al | all | +0.056 | 0.00047 | 42/59 | yes |
| band al | all | +0.042 | 0.0064 | 41/59 | yes |

## approach 1 vs the untranslated physics render (secondary) (m7c - phys_b0): **reported**

| measure | bin | median diff | p | kidneys up | pass |
|---|---|---|---|---|---|
| kidney dt | all | +0.403 | 3.3e-11 | 57/59 | yes |
| band dt | all | +0.413 | 3.6e-11 | 57/59 | yes |
| kidney | all | +0.554 | 2.9e-11 | 58/59 | yes |
| fan | all | +0.179 | 2.1e-10 | 54/59 | yes |
| kidney al | all |  |  |  | no |
| band al | all |  |  |  | no |

## retrieved image vs approach 1 (retrieve - m7c): **FAIL**

| measure | bin | median diff | p | kidneys up | pass |
|---|---|---|---|---|---|
| kidney dt | all | -0.257 | 6.3e-11 | 3/59 | no |
| band dt | all | -0.290 | 2.9e-11 | 2/59 | no |
| kidney | all | -0.074 | 0.0081 | 19/59 | no |
| fan | all | +0.078 | 0.0045 | 39/59 | yes |
| kidney al | all | -0.152 | 1.3e-10 | 3/59 | no |
| band al | all | -0.182 | 3.6e-11 | 2/59 | no |
| kidney W1 (lower is better) | all | -0.0010 | 0.95 | | not worse |

## retrieved image vs SonoSPADE (reported) (retrieve - b0): **reported**

| measure | bin | median diff | p | kidneys up | pass |
|---|---|---|---|---|---|
| kidney dt | all | +0.034 | 0.49 | 31/59 | no |
| band dt | all | -0.006 | 0.28 | 27/59 | no |
| kidney | all | +0.434 | 4.5e-09 | 52/59 | yes |
| fan | all | +0.238 | 6.2e-08 | 50/59 | yes |
| kidney al | all | -0.044 | 0.0081 | 21/59 | no |
| band al | all | -0.061 | 0.00058 | 19/59 | no |

## retrieved image vs the untranslated render (reported) (retrieve - phys_b0): **reported**

| measure | bin | median diff | p | kidneys up | pass |
|---|---|---|---|---|---|
| kidney dt | all | +0.194 | 2.9e-05 | 40/59 | yes |
| band dt | all | +0.132 | 0.0023 | 42/59 | yes |
| kidney | all | +0.497 | 3.7e-10 | 54/59 | yes |
| fan | all | +0.267 | 4.1e-08 | 50/59 | yes |
| kidney al | all |  |  |  | no |
| band al | all |  |  |  | no |

## retrieved image vs the patient's own label-warped image (reported) (retrieve - labwarp): **reported**

| measure | bin | median diff | p | kidneys up | pass |
|---|---|---|---|---|---|
| kidney dt | all | -0.224 | 1.1e-09 | 8/59 | no |
| band dt | all | -0.266 | 6e-11 | 3/59 | no |
| kidney | all | -0.251 | 5.8e-10 | 8/59 | no |
| fan | all | -0.288 | 2.8e-11 | 1/59 | no |
| kidney al | all | -0.133 | 2.5e-09 | 7/59 | no |
| band al | all | -0.170 | 7e-11 | 4/59 | no |

## hybrid (tier 1 + retrieved image) vs approach 1 (m7x - m7c): **FAIL**

| measure | bin | median diff | p | kidneys up | pass |
|---|---|---|---|---|---|
| kidney dt | all | -0.007 | 0.12 | 25/59 | no |
| band dt | all | -0.007 | 0.13 | 24/59 | no |
| kidney | all | -0.000 | 0.25 | 29/59 | no |
| fan | all | -0.001 | 0.61 | 26/59 | no |
| kidney al | all | -0.001 | 0.36 | 27/59 | no |
| band al | all | +0.000 | 0.41 | 30/59 | no |
| kidney W1 (lower is better) | all | -0.0020 | 0.002 | | not worse |

## hybrid vs the retrieved image alone (reported) (m7x - retrieve): **reported**

| measure | bin | median diff | p | kidneys up | pass |
|---|---|---|---|---|---|
| kidney dt | all | +0.257 | 8.9e-11 | 54/59 | yes |
| band dt | all | +0.275 | 4.2e-11 | 55/59 | yes |
| kidney | all | +0.058 | 0.022 | 37/59 | yes |
| fan | all | -0.093 | 0.0038 | 20/59 | no |
| kidney al | all | +0.143 | 1.7e-10 | 55/59 | yes |
| band al | all | +0.183 | 4.4e-11 | 57/59 | yes |

## hybrid vs the patient's own label-warped image (reported) (m7x - labwarp): **reported**

| measure | bin | median diff | p | kidneys up | pass |
|---|---|---|---|---|---|
| kidney dt | all | +0.024 | 0.42 | 35/59 | no |
| band dt | all | +0.038 | 0.4 | 35/59 | no |
| kidney | all | -0.187 | 8.6e-07 | 14/59 | no |
| fan | all | -0.346 | 2.5e-11 | 1/59 | no |
| kidney al | all | +0.018 | 0.39 | 37/59 | no |
| band al | all | -0.001 | 0.95 | 29/59 | no |

## Row means (all 354 planes)

| row | kidney dt | band dt | kidney | fan | kidney al | band al | kidney W1 |
|---|---|---|---|---|---|---|---|
| b0 | +0.247 | +0.280 | -0.087 | +0.153 | +0.629 | +0.615 |  |
| b0_v2 | +0.222 | +0.252 | -0.105 | +0.169 | +0.614 | +0.592 |  |
| b0t | +0.277 | +0.283 | -0.061 | +0.133 | +0.628 | +0.601 | +0.130 |
| b0t_v2 | +0.177 | +0.208 | -0.185 | +0.101 | +0.621 | +0.598 | +0.176 |
| copy | +0.372 | +0.458 | +0.482 | +0.782 | +0.679 | +0.697 |  |
| ct_hu | -0.370 | -0.393 | -0.364 | -0.118 |  |  |  |
| labwarp | +0.473 | +0.518 | +0.546 | +0.670 | +0.729 | +0.725 | +0.031 |
| m1 | +0.211 | +0.249 | -0.096 | +0.161 |  |  |  |
| m2 | +0.269 | +0.296 | -0.038 | +0.174 |  |  |  |
| m3 | +0.249 | +0.288 | -0.075 | +0.156 |  |  |  |
| m3_hu0 | +0.254 | +0.293 | -0.070 | +0.156 |  |  |  |
| m4 | +0.352 | +0.370 | +0.117 | +0.211 |  |  |  |
| m4c | +0.351 | +0.370 | +0.115 | +0.210 |  |  |  |
| m5 | +0.354 | +0.374 | +0.108 | +0.216 |  |  |  |
| m5c | +0.355 | +0.374 | +0.113 | +0.215 |  |  |  |
| m5k | +0.355 | +0.383 | +0.078 | +0.222 |  |  |  |
| m5kc | +0.356 | +0.384 | +0.082 | +0.220 |  |  |  |
| m6 | +0.424 | +0.446 | +0.133 | +0.220 | +0.696 | +0.674 |  |
| m6c | +0.423 | +0.448 | +0.133 | +0.218 | +0.695 | +0.675 |  |
| m7 | +0.493 | +0.538 | +0.377 | +0.291 | +0.737 | +0.729 |  |
| m7c | +0.491 | +0.533 | +0.376 | +0.294 | +0.736 | +0.723 | +0.081 |
| m7c_v2 | +0.491 | +0.526 | +0.390 | +0.309 | +0.735 | +0.718 | +0.086 |
| m7r | +0.500 | +0.541 | +0.386 | +0.293 | +0.741 | +0.732 |  |
| m7rc | +0.491 | +0.532 | +0.375 | +0.297 | +0.736 | +0.723 |  |
| m7x | +0.484 | +0.528 | +0.366 | +0.292 | +0.733 | +0.721 | +0.079 |
| m8 | +0.576 | +0.639 | +0.443 | +0.320 | +0.781 | +0.785 |  |
| m8c | +0.564 | +0.629 | +0.430 | +0.325 | +0.772 | +0.777 |  |
| m9 | +0.527 | +0.583 | +0.448 | +0.406 | +0.755 | +0.755 |  |
| m9a | +0.586 | +0.649 | +0.431 | +0.315 | +0.784 | +0.790 |  |
| m9ac | +0.583 | +0.646 | +0.425 | +0.320 | +0.781 | +0.786 | +0.077 |
| m9ac_v2 | +0.580 | +0.637 | +0.429 | +0.332 | +0.780 | +0.782 | +0.082 |
| m9c | +0.523 | +0.579 | +0.450 | +0.420 | +0.755 | +0.753 |  |
| m9cw | +0.410 | +0.469 | +0.501 | +0.628 | +0.695 | +0.684 |  |
| m9w | +0.407 | +0.466 | +0.499 | +0.627 | +0.696 | +0.683 |  |
| p2p | +0.397 | +0.422 | +0.452 | +0.468 | +0.677 | +0.646 | +0.065 |
| phys_b0 | +0.064 | +0.125 | -0.157 | +0.130 |  |  | +0.164 |
| phys_b0t | +0.064 | +0.125 | -0.157 | +0.130 |  |  |  |
| phys_copy | +0.064 | +0.125 | -0.157 | +0.130 |  |  |  |
| phys_labwarp | +0.064 | +0.125 | -0.157 | +0.130 |  |  |  |
| phys_m1 | +0.064 | +0.123 | -0.151 | +0.130 |  |  |  |
| phys_m2 | +0.120 | +0.167 | -0.121 | +0.140 |  |  |  |
| phys_m4 | +0.321 | +0.336 | +0.228 | +0.194 |  |  |  |
| phys_m4c | +0.320 | +0.335 | +0.226 | +0.194 |  |  |  |
| phys_m5 | +0.316 | +0.335 | +0.208 | +0.198 |  |  |  |
| phys_m5c | +0.321 | +0.338 | +0.220 | +0.198 |  |  |  |
| phys_m5k | +0.327 | +0.342 | +0.208 | +0.206 |  |  |  |
| phys_m5kc | +0.327 | +0.342 | +0.214 | +0.204 |  |  |  |
| phys_m6 | +0.333 | +0.356 | +0.222 | +0.192 |  |  |  |
| phys_m6c | +0.343 | +0.367 | +0.226 | +0.193 |  |  |  |
| phys_m7 | +0.064 | +0.125 | -0.157 | +0.130 |  |  |  |
| phys_m7c | +0.064 | +0.125 | -0.157 | +0.130 |  |  |  |
| phys_m7r | +0.064 | +0.125 | -0.157 | +0.130 |  |  |  |
| phys_m7rc | +0.064 | +0.125 | -0.157 | +0.130 |  |  |  |
| phys_m7x | +0.064 | +0.125 | -0.157 | +0.130 |  |  |  |
| phys_m8 | +0.064 | +0.125 | -0.157 | +0.130 |  |  |  |
| phys_m8c | +0.064 | +0.125 | -0.157 | +0.130 |  |  |  |
| phys_m9 | +0.064 | +0.125 | -0.157 | +0.130 |  |  |  |
| phys_m9a | +0.064 | +0.125 | -0.157 | +0.130 |  |  |  |
| phys_m9ac | +0.064 | +0.125 | -0.157 | +0.130 |  |  |  |
| phys_m9c | +0.064 | +0.125 | -0.157 | +0.130 |  |  |  |
| phys_m9cw | +0.064 | +0.125 | -0.157 | +0.130 |  |  |  |
| phys_m9w | +0.064 | +0.125 | -0.157 | +0.130 |  |  |  |
| phys_p2p | +0.064 | +0.125 | -0.157 | +0.130 |  |  |  |
| phys_retrieve | +0.064 | +0.125 | -0.157 | +0.130 |  |  |  |
| retrieve | +0.223 | +0.233 | +0.297 | +0.371 | +0.574 | +0.537 | +0.081 |
