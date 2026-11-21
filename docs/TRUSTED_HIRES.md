# Same-patient test at full resolution and with deformable registration

## Registration

59 kidneys. Kidney Dice rigid 0.858 (min 0.613) -> deformable 0.969 (min 0.940). Mean landmark residual per kidney, median over kidneys: rigid 6.6 mm -> deformable 5.5 mm (renal pelvis 4.1 -> 4.3 mm); the landmarks are not used by the deformable step. Min Jacobian determinant over all fields -0.05; median max displacement 9.1 mm.

## Full resolution (0.3 mm, renders only)

Mean over kidneys of the per-kidney mean r (depth-detrended kidney interior / kidney + band; raw fan).

| row | reg | sigma | r kidney | r kidney+band | r fan |
|---|---|---|---|---|---|
| CT slice itself | rig | 1 mm | -0.296 | -0.347 | -0.206 |
| CT slice itself | rig | 2 mm | -0.338 | -0.384 | -0.216 |
| CT slice itself | def | 1 mm | -0.332 | -0.421 | -0.217 |
| CT slice itself | def | 2 mm | -0.387 | -0.447 | -0.225 |
| render, base labels | rig | 1 mm | +0.038 | +0.083 | +0.091 |
| render, base labels | rig | 2 mm | +0.048 | +0.090 | +0.086 |
| render, base labels | def | 1 mm | +0.040 | +0.088 | +0.094 |
| render, base labels | def | 2 mm | +0.051 | +0.093 | +0.089 |
| render, M2 | rig | 1 mm | +0.087 | +0.127 | +0.103 |
| render, M2 | rig | 2 mm | +0.105 | +0.137 | +0.098 |
| render, M2 | def | 1 mm | +0.095 | +0.135 | +0.106 |
| render, M2 | def | 2 mm | +0.111 | +0.141 | +0.101 |
| render, M2 backscatter only | rig | 1 mm | +0.088 | +0.130 | +0.105 |
| render, M2 backscatter only | rig | 2 mm | +0.107 | +0.140 | +0.101 |
| render, M2 backscatter only | def | 1 mm | +0.096 | +0.138 | +0.109 |
| render, M2 backscatter only | def | 2 mm | +0.113 | +0.145 | +0.104 |
| M4c at full resolution (no HU) | rig | 1 mm | +0.249 | +0.268 | +0.169 |
| M4c at full resolution (no HU) | rig | 2 mm | +0.297 | +0.304 | +0.167 |
| M4c at full resolution (no HU) | def | 1 mm | +0.264 | +0.312 | +0.184 |
| M4c at full resolution (no HU) | def | 2 mm | +0.316 | +0.336 | +0.180 |
| M4 at full resolution | rig | 1 mm | +0.249 | +0.268 | +0.169 |
| M4 at full resolution | rig | 2 mm | +0.297 | +0.304 | +0.167 |
| M4 at full resolution | def | 1 mm | +0.264 | +0.313 | +0.184 |
| M4 at full resolution | def | 2 mm | +0.317 | +0.337 | +0.180 |
| M5c at full resolution, sweep halves (no HU) | rig | 1 mm | +0.252 | +0.274 | +0.172 |
| M5c at full resolution, sweep halves (no HU) | rig | 2 mm | +0.301 | +0.310 | +0.170 |
| M5c at full resolution, sweep halves (no HU) | def | 1 mm | +0.268 | +0.316 | +0.189 |
| M5c at full resolution, sweep halves (no HU) | def | 2 mm | +0.322 | +0.341 | +0.186 |
| M5 at full resolution, sweep halves | rig | 1 mm | +0.252 | +0.275 | +0.172 |
| M5 at full resolution, sweep halves | rig | 2 mm | +0.299 | +0.310 | +0.170 |
| M5 at full resolution, sweep halves | def | 1 mm | +0.269 | +0.314 | +0.187 |
| M5 at full resolution, sweep halves | def | 2 mm | +0.321 | +0.339 | +0.184 |
| M5c at full resolution, other kidney (no HU) | rig | 1 mm | +0.249 | +0.265 | +0.170 |
| M5c at full resolution, other kidney (no HU) | rig | 2 mm | +0.301 | +0.302 | +0.168 |
| M5c at full resolution, other kidney (no HU) | def | 1 mm | +0.270 | +0.318 | +0.187 |
| M5c at full resolution, other kidney (no HU) | def | 2 mm | +0.327 | +0.344 | +0.184 |
| M5 at full resolution, other kidney | rig | 1 mm | +0.248 | +0.264 | +0.168 |
| M5 at full resolution, other kidney | rig | 2 mm | +0.299 | +0.300 | +0.166 |
| M5 at full resolution, other kidney | def | 1 mm | +0.269 | +0.315 | +0.186 |
| M5 at full resolution, other kidney | def | 2 mm | +0.324 | +0.342 | +0.183 |

Paired differences (median over kidneys, Wilcoxon p, kidneys improved):

| comparison | reg | sigma | kidney | kidney+band |
|---|---|---|---|---|
| render, M2 - render, base labels | rig | 1 mm | +0.022 (p=1.8e-10, 55/59) | +0.020 (p=3.3e-11, 57/59) |
| render, M2 - render, base labels | rig | 2 mm | +0.025 (p=1.5e-10, 54/59) | +0.027 (p=3.6e-11, 57/59) |
| render, M2 - render, base labels | def | 1 mm | +0.029 (p=3.3e-11, 56/59) | +0.028 (p=2.6e-11, 58/59) |
| render, M2 - render, base labels | def | 2 mm | +0.028 (p=3.8e-11, 56/59) | +0.028 (p=2.6e-11, 58/59) |
| render, M2 backscatter only - render, base labels | rig | 1 mm | +0.022 (p=2.9e-10, 54/59) | +0.023 (p=2.9e-11, 57/59) |
| render, M2 backscatter only - render, base labels | rig | 2 mm | +0.030 (p=1.8e-10, 54/59) | +0.029 (p=2.4e-11, 59/59) |
| render, M2 backscatter only - render, base labels | def | 1 mm | +0.034 (p=3.1e-11, 57/59) | +0.033 (p=2.5e-11, 58/59) |
| render, M2 backscatter only - render, base labels | def | 2 mm | +0.038 (p=2.8e-11, 58/59) | +0.032 (p=2.4e-11, 59/59) |
| M4 at full resolution - M4c at full resolution (no HU) | rig | 1 mm | +0.000 (p=0.17, 34/59) | +0.000 (p=0.54, 31/59) |
| M4 at full resolution - M4c at full resolution (no HU) | rig | 2 mm | +0.000 (p=0.44, 30/59) | -0.000 (p=0.65, 29/59) |
| M4 at full resolution - M4c at full resolution (no HU) | def | 1 mm | +0.000 (p=0.36, 31/59) | +0.000 (p=0.004, 39/59) |
| M4 at full resolution - M4c at full resolution (no HU) | def | 2 mm | +0.000 (p=0.17, 34/59) | +0.000 (p=0.043, 37/59) |
| M5 at full resolution, sweep halves - M5c at full resolution, sweep halves (no HU) | rig | 1 mm | -0.000 (p=0.63, 29/59) | +0.000 (p=0.17, 33/59) |
| M5 at full resolution, sweep halves - M5c at full resolution, sweep halves (no HU) | rig | 2 mm | -0.000 (p=0.41, 29/59) | +0.000 (p=0.25, 34/59) |
| M5 at full resolution, sweep halves - M5c at full resolution, sweep halves (no HU) | def | 1 mm | +0.000 (p=0.11, 36/59) | +0.000 (p=0.16, 36/59) |
| M5 at full resolution, sweep halves - M5c at full resolution, sweep halves (no HU) | def | 2 mm | +0.000 (p=0.21, 34/59) | +0.000 (p=0.46, 35/59) |
| M5 at full resolution, other kidney - M5c at full resolution, other kidney (no HU) | rig | 1 mm | +0.000 (p=0.83, 22/40) | -0.000 (p=0.48, 20/40) |
| M5 at full resolution, other kidney - M5c at full resolution, other kidney (no HU) | rig | 2 mm | +0.000 (p=0.88, 20/40) | -0.000 (p=0.38, 19/40) |
| M5 at full resolution, other kidney - M5c at full resolution, other kidney (no HU) | def | 1 mm | +0.001 (p=0.15, 24/40) | +0.000 (p=0.58, 21/40) |
| M5 at full resolution, other kidney - M5c at full resolution, other kidney (no HU) | def | 2 mm | +0.000 (p=0.61, 22/40) | -0.000 (p=0.27, 18/40) |
| M4c at full resolution (no HU) - render, base labels | rig | 1 mm | +0.224 (p=2.1e-10, 52/59) | +0.164 (p=1.7e-10, 54/59) |
| M4c at full resolution (no HU) - render, base labels | rig | 2 mm | +0.256 (p=2.2e-10, 53/59) | +0.187 (p=1.3e-10, 53/59) |
| M4c at full resolution (no HU) - render, base labels | def | 1 mm | +0.234 (p=1e-10, 54/59) | +0.217 (p=4.9e-11, 56/59) |
| M4c at full resolution (no HU) - render, base labels | def | 2 mm | +0.262 (p=1.1e-10, 55/59) | +0.224 (p=6.3e-11, 56/59) |
| M5c at full resolution, sweep halves (no HU) - M4c at full resolution (no HU) | rig | 1 mm | +0.002 (p=0.38, 33/59) | +0.002 (p=0.037, 35/59) |
| M5c at full resolution, sweep halves (no HU) - M4c at full resolution (no HU) | rig | 2 mm | +0.004 (p=0.36, 33/59) | +0.002 (p=0.039, 38/59) |
| M5c at full resolution, sweep halves (no HU) - M4c at full resolution (no HU) | def | 1 mm | +0.002 (p=0.012, 40/59) | +0.002 (p=0.0004, 42/59) |
| M5c at full resolution, sweep halves (no HU) - M4c at full resolution (no HU) | def | 2 mm | +0.002 (p=0.0039, 38/59) | +0.003 (p=0.00031, 43/59) |
| M5 at full resolution, sweep halves - M4 at full resolution | rig | 1 mm | +0.002 (p=0.39, 36/59) | +0.003 (p=0.015, 40/59) |
| M5 at full resolution, sweep halves - M4 at full resolution | rig | 2 mm | +0.003 (p=0.45, 34/59) | +0.003 (p=0.017, 40/59) |
| M5 at full resolution, sweep halves - M4 at full resolution | def | 1 mm | +0.003 (p=0.006, 41/59) | +0.002 (p=0.0021, 42/59) |
| M5 at full resolution, sweep halves - M4 at full resolution | def | 2 mm | +0.003 (p=0.0069, 41/59) | +0.002 (p=0.0018, 42/59) |

Deformable minus rigid, same row (sigma 1 mm):

| row | kidney | kidney+band |
|---|---|---|
| CT slice itself | -0.022 (p=0.0014, 20/59) | -0.065 (p=1.3e-07, 12/59) |
| render, base labels | +0.007 (p=0.45, 34/59) | +0.004 (p=0.37, 31/59) |
| render, M2 | +0.013 (p=0.085, 36/59) | +0.002 (p=0.23, 33/59) |
| render, M2 backscatter only | +0.012 (p=0.074, 36/59) | +0.002 (p=0.2, 31/59) |
| M4c at full resolution (no HU) | +0.021 (p=0.064, 36/59) | +0.034 (p=1.2e-05, 43/59) |
| M4 at full resolution | +0.020 (p=0.069, 35/59) | +0.035 (p=1.1e-05, 44/59) |
| M5c at full resolution, sweep halves (no HU) | +0.017 (p=0.066, 37/59) | +0.039 (p=4.5e-06, 45/59) |
| M5 at full resolution, sweep halves | +0.017 (p=0.063, 36/59) | +0.037 (p=3.1e-05, 43/59) |
| M5c at full resolution, other kidney (no HU) | +0.020 (p=0.032, 26/40) | +0.040 (p=7.5e-06, 31/40) |
| M5 at full resolution, other kidney | +0.019 (p=0.042, 27/40) | +0.040 (p=6.9e-06, 32/40) |

## 128 px pipeline with deformable registration

| row | r kidney | r kidney+band | vs B0 (kidney) |
|---|---|---|---|
| b0 | +0.250\,\pm\,0.016 | +0.274\,\pm\,0.016 |  |
| b0t | +0.285\,\pm\,0.006 | +0.282\,\pm\,0.005 | +0.014 (p=0.12, 34/59) |
| m1 | +0.215\,\pm\,0.024 | +0.246\,\pm\,0.013 | -0.031 (p=4.4e-05, 13/59) |
| m2 | +0.272\,\pm\,0.005 | +0.292\,\pm\,0.010 | +0.020 (p=0.017, 37/59) |
| m3 | +0.252\,\pm\,0.008 | +0.282\,\pm\,0.010 | +0.001 (p=0.49, 30/59) |
| m3_hu0 | +0.258\,\pm\,0.006 | +0.288\,\pm\,0.008 | +0.007 (p=0.0063, 36/59) |
| m4c | +0.354\,\pm\,0.005 | +0.366\,\pm\,0.005 | +0.094 (p=3.9e-10, 54/59) |
| m4 | +0.355\,\pm\,0.005 | +0.367\,\pm\,0.005 | +0.097 (p=4.1e-10, 54/59) |
| m5c | +0.358\,\pm\,0.005 | +0.372\,\pm\,0.005 | +0.097 (p=4.5e-10, 52/59) |
| m5 | +0.358\,\pm\,0.006 | +0.373\,\pm\,0.005 | +0.099 (p=5.8e-10, 52/59) |
| m5kc | +0.362\,\pm\,0.006 | +0.382\,\pm\,0.003 | +0.095 (p=3.8e-10, 38/40) |
| m5k | +0.360\,\pm\,0.006 | +0.381\,\pm\,0.002 | +0.092 (p=3.8e-10, 38/40) |
| m6c | +0.432\,\pm\,0.003 | +0.451\,\pm\,0.003 | +0.178 (p=3.4e-11, 58/59) |
| m6 | +0.432\,\pm\,0.005 | +0.449\,\pm\,0.008 | +0.182 (p=4.4e-11, 57/59) |
| m7c | +0.502\,\pm\,0.010 | +0.538\,\pm\,0.007 | +0.261 (p=2.9e-10, 54/59) |
| m7 | +0.503\,\pm\,0.007 | +0.543\,\pm\,0.002 | +0.233 (p=1.3e-10, 53/59) |
| m8c | +0.592\,\pm\,0.003 | +0.653\,\pm\,0.001 | +0.338 (p=2.4e-11, 59/59) |
| m8 | +0.603\,\pm\,0.006 | +0.662\,\pm\,0.003 | +0.334 (p=2.4e-11, 59/59) |
| m7rc | +0.502\,\pm\,0.009 | +0.537\,\pm\,0.005 | +0.257 (p=2.8e-10, 54/59) |
| m7r | +0.509\,\pm\,0.008 | +0.545\,\pm\,0.004 | +0.225 (p=8.9e-11, 55/59) |
| ct_hu | -0.378 | -0.392 |  |
| phys_b0 | +0.065 | +0.120 |  |
| phys_m2 | +0.124 | +0.163 | +0.031 (p=8.5e-11, 54/59) |

M4 - M4c (KidD): +0.001 (p=0.00022, 41/59)

M4 - M4c (BandD): +0.000 (p=0.0029, 38/59)

m6 - m6c (KidD): -0.0004 (p=0.65, 28/59)

m6 - m6c (BandD): -0.0019 (p=0.42, 27/59)

m6c - m4c (KidD): +0.0679 (p=6e-10, 52/59)

m6c - m4c (BandD): +0.0752 (p=1.5e-09, 53/59)

m6 - m4 (KidD): +0.0685 (p=1.1e-09, 50/59)

m6 - m4 (BandD): +0.0648 (p=2.6e-09, 51/59)

m5 - m5c (KidD): +0.0001 (p=0.96, 31/59)

m5 - m5c (BandD): -0.0001 (p=0.72, 29/59)

m5k - m5kc (KidD): -0.0003 (p=0.45, 19/40)

m5k - m5kc (BandD): -0.0003 (p=0.35, 19/40)

m7 - m7c (KidD): +0.0080 (p=0.54, 33/59)

m7 - m7c (BandD): +0.0093 (p=0.19, 35/59)

m7c - m6c (KidD): +0.0614 (p=5.4e-05, 45/59)

m7c - m6c (BandD): +0.0876 (p=7.3e-06, 48/59)

m7 - m6 (KidD): +0.0710 (p=6.1e-05, 46/59)

m7 - m6 (BandD): +0.0858 (p=2.8e-06, 47/59)

m8 - m8c (KidD): +0.0085 (p=0.017, 36/59)

m8 - m8c (BandD): +0.0057 (p=0.0088, 39/59)

m8c - m7c (KidD): +0.0671 (p=1.4e-08, 50/59)

m8c - m7c (BandD): +0.0998 (p=8.1e-11, 57/59)

m8 - m7 (KidD): +0.0811 (p=3.7e-09, 53/59)

m8 - m7 (BandD): +0.0937 (p=1.4e-10, 56/59)

m7r - m7rc (KidD): +0.0066 (p=0.25, 34/59)

m7r - m7rc (BandD): +0.0038 (p=0.13, 35/59)

m7rc - m7c (KidD): +0.0005 (p=0.93, 31/59)

m7rc - m7c (BandD): +0.0010 (p=0.53, 33/59)

m7c - b0t (KidD): +0.2336 (p=1.2e-09, 51/59)

m7c - b0t (BandD): +0.2758 (p=2e-10, 53/59)

b0t - b0 (KidD): +0.0143 (p=0.12, 34/59)

b0t - b0 (BandD): -0.0240 (p=0.98, 26/59)

M6 HU use: mean |change of log factor| when z and HU are flattened per class 0.108 (spread of log factor: M6 1.22, M6c 1.17); training r M6 0.466, M6c 0.468

## Alignment-tolerant agreement (smooth warp of each output, 4x4 grid, at most 4 px)

| row | kidney before | kidney after | band before | band after |
|---|---|---|---|---|
| b0 | +0.250 | +0.626 | +0.274 | +0.608 |
| m6 | +0.432 | +0.695 | +0.449 | +0.671 |
| m6c | +0.432 | +0.695 | +0.451 | +0.673 |
| m7 | +0.503 | +0.739 | +0.543 | +0.730 |
| m7c | +0.502 | +0.738 | +0.538 | +0.724 |
| m8 | +0.603 | +0.791 | +0.662 | +0.796 |
| m8c | +0.592 | +0.783 | +0.653 | +0.788 |
| m7r | +0.509 | +0.743 | +0.545 | +0.733 |
| m7rc | +0.502 | +0.738 | +0.537 | +0.724 |
| b0t | +0.285 | +0.629 | +0.282 | +0.599 |

aligned m6 - m6c (Kid): -0.0009 (p=0.4, 26/59)

aligned m6 - m6c (Band): -0.0024 (p=0.29, 25/59)

aligned m7 - m7c (Kid): -0.0007 (p=0.91, 28/59)

aligned m7 - m7c (Band): +0.0037 (p=0.098, 33/59)

aligned m8 - m8c (Kid): +0.0069 (p=0.017, 35/59)

aligned m8 - m8c (Band): +0.0070 (p=0.0066, 39/59)

aligned m7c - b0 (Kid): +0.1046 (p=4.7e-09, 52/59)

aligned m7c - b0 (Band): +0.0920 (p=1.9e-09, 54/59)

aligned m8c - m7c (Kid): +0.0244 (p=1.3e-07, 47/59)

aligned m8c - m7c (Band): +0.0514 (p=7.3e-10, 53/59)

aligned m7r - m7rc (Kid): +0.0022 (p=0.42, 32/59)

aligned m7r - m7rc (Band): +0.0041 (p=0.031, 36/59)

aligned m7rc - m7c (Kid): -0.0002 (p=0.82, 28/59)

aligned m7rc - m7c (Band): +0.0001 (p=0.9, 30/59)

aligned m7c - b0t (Kid): +0.1082 (p=4.1e-09, 52/59)

aligned m7c - b0t (Band): +0.1165 (p=8.8e-10, 53/59)

aligned b0t - b0 (Kid): -0.0259 (p=0.8, 26/59)

aligned b0t - b0 (Band): -0.0187 (p=0.25, 23/59)
