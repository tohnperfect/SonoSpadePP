TRUSTED same-patient planes: 413 frames from 59 kidneys. Pearson r between the smoothed synthetic and the smoothed real US of the same plane (sigma 1.5 px), averaged over each kidney's planes, then over kidneys. Trained rows: mean ± SD across GAN seeds.

Kidney and band correlations are computed after removing a quadratic trend in depth inside the region (gain and attenuation along the beam); the last column is the raw kidney value.

| Row | r kidney interior ↑ | r kidney + band ↑ | r whole fan ↑ | raw r kidney |
|---|---|---|---|---|
| CT intensity itself (reference) | -0.326 | -0.333 | -0.108 | -0.349 |
| physics render, base labels | +0.063 | +0.121 | +0.119 | -0.165 |
| physics render, M1 labels | +0.070 | +0.124 | +0.120 | -0.153 |
| physics render, HU-modulated | +0.119 | +0.164 | +0.130 | -0.128 |
| physics render, M2 impedance only | +0.062 ± 0.000 | +0.119 ± 0.000 | +0.117 ± 0.000 | -0.165 ± 0.000 |
| physics render, M2 backscatter only | +0.120 ± 0.000 | +0.166 ± 0.000 | +0.132 ± 0.000 | -0.127 ± 0.000 |
| physics render, M2 backscatter as a per-tissue mean | +0.092 ± 0.000 | +0.147 ± 0.000 | +0.127 ± 0.000 | -0.149 ± 0.000 |
| physics render, learned per-tissue echo (no HU) | +0.310 ± 0.002 | +0.314 ± 0.001 | +0.179 ± 0.001 | +0.227 ± 0.009 |
| physics render, learned HU-to-echo | +0.310 ± 0.002 | +0.315 ± 0.001 | +0.179 ± 0.002 | +0.230 ± 0.005 |
| physics render, M4 with 100x weaker penalties | +0.310 ± 0.002 | +0.315 ± 0.002 | +0.172 ± 0.002 | +0.251 ± 0.005 |
| physics render, M4 started at M2's rule | +0.310 ± 0.002 | +0.315 ± 0.001 | +0.173 ± 0.002 | +0.251 ± 0.005 |
| B0 SonoSPADE (labels + physics) | +0.228 ± 0.017 | +0.260 ± 0.016 | +0.140 ± 0.002 | -0.077 ± 0.022 |
| M1 CT sub-labels | +0.200 ± 0.019 | +0.239 ± 0.010 | +0.148 ± 0.008 | -0.078 ± 0.020 |
| M2 HU-modulated physics | +0.248 ± 0.006 | +0.280 ± 0.009 | +0.161 ± 0.006 | -0.021 ± 0.013 |
| M2 ablation: impedance only (M2 generator) | +0.231 ± 0.008 | +0.267 ± 0.010 | +0.152 ± 0.006 | -0.045 ± 0.014 |
| M2 ablation: backscatter only (M2 generator) | +0.249 ± 0.006 | +0.281 ± 0.009 | +0.162 ± 0.006 | -0.020 ± 0.013 |
| M2 ablation: per-tissue mean backscatter only (M2 generator) | +0.241 ± 0.008 | +0.276 ± 0.009 | +0.159 ± 0.006 | -0.033 ± 0.014 |
| M3 HU as a SPADE channel | +0.230 ± 0.006 | +0.268 ± 0.008 | +0.143 ± 0.006 | -0.064 ± 0.013 |
| M3 with its HU channel zeroed (ablation) | +0.233 ± 0.005 | +0.271 ± 0.007 | +0.144 ± 0.005 | -0.060 ± 0.012 |
| M4 control: learned per-tissue echo, no HU | +0.325 ± 0.005 | +0.343 ± 0.004 | +0.195 ± 0.001 | +0.140 ± 0.023 |
| M4 learned HU-to-echo (same-patient training, CV) | +0.325 ± 0.005 | +0.343 ± 0.004 | +0.195 ± 0.001 | +0.142 ± 0.022 |
| M4 check: 100x weaker penalties | +0.326 ± 0.005 | +0.345 ± 0.003 | +0.190 ± 0.000 | +0.155 ± 0.022 |
| M4 check: started at M2's backscatter rule, weak penalties | +0.326 ± 0.005 | +0.345 ± 0.003 | +0.190 ± 0.001 | +0.155 ± 0.021 |
| physics render, M5c patient-specific echo, no HU (sweep halves) | +0.307 ± 0.004 | +0.318 ± 0.001 | +0.183 ± 0.003 | +0.214 ± 0.014 |
| physics render, M5 patient-specific HU-to-echo (sweep halves) | +0.303 ± 0.005 | +0.319 ± 0.001 | +0.185 ± 0.005 | +0.206 ± 0.015 |
| physics render, M5c fitted on the other kidney | +0.313 ± 0.003 | +0.314 ± 0.002 | +0.189 ± 0.002 | +0.207 ± 0.015 |
| physics render, M5 fitted on the other kidney | +0.311 ± 0.003 | +0.313 ± 0.003 | +0.189 ± 0.002 | +0.203 ± 0.014 |
| M5c patient-specific echo, no HU (sweep halves) | +0.326 ± 0.005 | +0.348 ± 0.004 | +0.200 ± 0.001 | +0.134 ± 0.027 |
| M5 patient-specific HU-to-echo (fit on half the sweep, test on the other half) | +0.326 ± 0.005 | +0.350 ± 0.004 | +0.202 ± 0.003 | +0.131 ± 0.025 |
| M5c patient-specific echo, no HU (other kidney) | +0.330 ± 0.005 | +0.351 ± 0.004 | +0.205 ± 0.001 | +0.114 ± 0.027 |
| M5 patient-specific HU-to-echo (fit on the other kidney) | +0.329 ± 0.005 | +0.350 ± 0.003 | +0.205 ± 0.002 | +0.112 ± 0.027 |

Paired difference against B0 (trained rows) or the base-label physics render (physics rows): median over kidneys of the per-kidney difference, Wilcoxon signed-rank p, and how many kidneys improved.

| Row | r kidney interior | r kidney + band | r whole fan | raw r kidney |
|---|---|---|---|---|
| physics render, M1 labels | +0.000 (p=0.802, 16/59 up) | +0.000 (p=0.308, 21/59 up) | -0.000 (p=0.546, 24/59 up) | +0.000 (p=0.521, 19/59 up) |
| physics render, HU-modulated | +0.028 (p=1.33e-10, 55/59 up) | +0.024 (p=3.25e-11, 57/59 up) | +0.006 (p=5.52e-06, 46/59 up) | +0.014 (p=2.94e-11, 57/59 up) |
| physics render, M2 impedance only | -0.001 (p=0.0736, 25/59 up) | -0.002 (p=0.000115, 19/59 up) | -0.001 (p=1.63e-05, 18/59 up) | -0.001 (p=0.0209, 24/59 up) |
| physics render, M2 backscatter only | +0.031 (p=1.15e-10, 54/59 up) | +0.025 (p=2.79e-11, 57/59 up) | +0.007 (p=1.01e-08, 52/59 up) | +0.017 (p=3.79e-11, 58/59 up) |
| physics render, M2 backscatter as a per-tissue mean | +0.014 (p=4.53e-10, 55/59 up) | +0.016 (p=1.4e-10, 56/59 up) | +0.005 (p=2.48e-09, 54/59 up) | +0.008 (p=3.73e-10, 55/59 up) |
| physics render, learned per-tissue echo (no HU) | +0.216 (p=9.89e-11, 54/59 up) | +0.179 (p=8.44e-10, 51/59 up) | +0.048 (p=0.00253, 34/59 up) | +0.365 (p=2.94e-11, 58/59 up) |
| physics render, learned HU-to-echo | +0.216 (p=1.04e-10, 54/59 up) | +0.181 (p=8.44e-10, 51/59 up) | +0.050 (p=0.00273, 33/59 up) | +0.367 (p=2.94e-11, 58/59 up) |
| physics render, M4 with 100x weaker penalties | +0.224 (p=1.47e-10, 54/59 up) | +0.205 (p=1.07e-09, 51/59 up) | +0.059 (p=0.0339, 31/59 up) | +0.384 (p=3.09e-11, 57/59 up) |
| physics render, M4 started at M2's rule | +0.223 (p=1.47e-10, 54/59 up) | +0.202 (p=1.07e-09, 51/59 up) | +0.064 (p=0.0286, 31/59 up) | +0.384 (p=3.09e-11, 57/59 up) |
| M1 CT sub-labels | -0.018 (p=0.000611, 17/59 up) | -0.021 (p=0.00587, 21/59 up) | +0.003 (p=0.216, 32/59 up) | -0.013 (p=0.0275, 21/59 up) |
| M2 HU-modulated physics | +0.021 (p=0.00881, 37/59 up) | +0.024 (p=0.00384, 39/59 up) | +0.022 (p=8.67e-05, 41/59 up) | +0.065 (p=4.26e-08, 50/59 up) |
| M2 ablation: impedance only (M2 generator) | +0.008 (p=0.402, 32/59 up) | +0.013 (p=0.167, 38/59 up) | +0.010 (p=0.00393, 37/59 up) | +0.037 (p=2.06e-06, 48/59 up) |
| M2 ablation: backscatter only (M2 generator) | +0.022 (p=0.00643, 38/59 up) | +0.024 (p=0.00247, 39/59 up) | +0.022 (p=3.53e-05, 42/59 up) | +0.064 (p=2.89e-08, 50/59 up) |
| M2 ablation: per-tissue mean backscatter only (M2 generator) | +0.016 (p=0.0333, 37/59 up) | +0.019 (p=0.013, 39/59 up) | +0.021 (p=0.000176, 40/59 up) | +0.052 (p=1.43e-07, 50/59 up) |
| M3 HU as a SPADE channel | +0.004 (p=0.274, 34/59 up) | +0.010 (p=0.000516, 44/59 up) | +0.004 (p=0.0167, 38/59 up) | +0.010 (p=7.88e-06, 44/59 up) |
| M3 with its HU channel zeroed (ablation) | +0.007 (p=0.0145, 39/59 up) | +0.015 (p=5.39e-05, 48/59 up) | +0.006 (p=0.00942, 40/59 up) | +0.016 (p=8.02e-08, 49/59 up) |
| M4 control: learned per-tissue echo, no HU | +0.087 (p=2.29e-10, 53/59 up) | +0.076 (p=6.76e-09, 51/59 up) | +0.038 (p=0.00103, 41/59 up) | +0.197 (p=2.65e-11, 58/59 up) |
| M4 learned HU-to-echo (same-patient training, CV) | +0.084 (p=2.53e-10, 53/59 up) | +0.075 (p=7.73e-09, 50/59 up) | +0.038 (p=0.00124, 40/59 up) | +0.200 (p=2.65e-11, 58/59 up) |
| M4 check: 100x weaker penalties | +0.081 (p=4.32e-10, 53/59 up) | +0.075 (p=2.23e-08, 50/59 up) | +0.028 (p=0.0201, 36/59 up) | +0.210 (p=2.52e-11, 58/59 up) |
| M4 check: started at M2's backscatter rule, weak penalties | +0.081 (p=4.32e-10, 53/59 up) | +0.076 (p=2.14e-08, 50/59 up) | +0.029 (p=0.0164, 36/59 up) | +0.210 (p=2.52e-11, 58/59 up) |
| physics render, M5c patient-specific echo, no HU (sweep halves) | +0.209 (p=8.51e-11, 54/59 up) | +0.177 (p=6.04e-10, 51/59 up) | +0.046 (p=0.00173, 35/59 up) | +0.328 (p=3.25e-11, 57/59 up) |
| physics render, M5 patient-specific HU-to-echo (sweep halves) | +0.203 (p=8.1e-11, 54/59 up) | +0.177 (p=4.53e-10, 50/59 up) | +0.047 (p=0.00127, 36/59 up) | +0.315 (p=3.6e-11, 57/59 up) |
| physics render, M5c fitted on the other kidney | +0.259 (p=1.27e-10, 37/40 up) | +0.194 (p=1.6e-10, 36/40 up) | +0.050 (p=0.00186, 24/40 up) | +0.416 (p=9.09e-12, 39/40 up) |
| physics render, M5 fitted on the other kidney | +0.259 (p=1.6e-10, 37/40 up) | +0.193 (p=1.6e-10, 36/40 up) | +0.055 (p=0.0013, 24/40 up) | +0.414 (p=9.09e-12, 39/40 up) |
| M5c patient-specific echo, no HU (sweep halves) | +0.086 (p=3.56e-10, 53/59 up) | +0.077 (p=5.15e-09, 49/59 up) | +0.043 (p=0.000921, 40/59 up) | +0.185 (p=3.42e-11, 57/59 up) |
| M5 patient-specific HU-to-echo (fit on half the sweep, test on the other half) | +0.082 (p=4.99e-10, 53/59 up) | +0.078 (p=2.37e-09, 51/59 up) | +0.044 (p=0.000461, 41/59 up) | +0.180 (p=3.6e-11, 57/59 up) |
| M5c patient-specific echo, no HU (other kidney) | +0.093 (p=1.27e-10, 36/40 up) | +0.073 (p=3.16e-09, 35/40 up) | +0.047 (p=0.00105, 27/40 up) | +0.217 (p=3.64e-12, 39/40 up) |
| M5 patient-specific HU-to-echo (fit on the other kidney) | +0.089 (p=2.49e-10, 36/40 up) | +0.074 (p=3.7e-09, 35/40 up) | +0.053 (p=0.000941, 27/40 up) | +0.214 (p=3.64e-12, 39/40 up) |
