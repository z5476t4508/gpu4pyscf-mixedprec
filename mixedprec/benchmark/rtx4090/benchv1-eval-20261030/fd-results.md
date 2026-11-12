# SNT-2 finite-difference force diagnostics (benchv1 card)

Diagnostic only: FD-vs-analytic agreement on the auto lane criterion 3x nominal floor 1e-04 Eh/Bohr; never the 3e-5 gate.

## cytosine_b3lyp_def2-svp

FD atoms: #8 (H; H on the ring N (N1) that bears exactly one H and >=2 C), #12 (H; ammonium-class (NH2) H with largest |g| atom-norm on the auto-lane gradient)

| step (Bohr) | atom | axis | g_fd | g_analytic | |delta| |
|---|---|---|---|---|---|
| 0.0010 | 8H | x | -8.574669e-03 | -8.586842e-03 | 1.217e-05 |
| 0.0010 | 8H | y | -5.169828e-04 | -5.162523e-04 | 7.305e-07 |
| 0.0010 | 8H | z | +8.597851e-07 | +8.620169e-07 | 2.232e-09 |
| 0.0010 | 12H | x | +6.208793e-03 | +6.214328e-03 | 5.535e-06 |
| 0.0010 | 12H | y | +1.167584e-03 | +1.165545e-03 | 2.039e-06 |
| 0.0010 | 12H | z | +2.621789e-06 | +2.617774e-06 | 4.015e-09 |
| — | — | max | | | 1.217e-05 |

step 0.0010: measured noise floor 0.000e+00, threshold 3.0e-04 -> within=True
| 0.0020 | 8H | x | -8.574608e-03 | -8.586842e-03 | 1.223e-05 |
| 0.0020 | 8H | y | -5.177558e-04 | -5.162523e-04 | 1.504e-06 |
| 0.0020 | 8H | z | +8.601262e-07 | +8.620169e-07 | 1.891e-09 |
| 0.0020 | 12H | x | +6.208736e-03 | +6.214328e-03 | 5.592e-06 |
| 0.0020 | 12H | y | +1.166769e-03 | +1.165545e-03 | 1.224e-06 |
| 0.0020 | 12H | z | +2.621633e-06 | +2.617774e-06 | 3.858e-09 |
| — | — | max | | | 1.223e-05 |

step 0.0020: measured noise floor 0.000e+00, threshold 3.0e-04 -> within=True

## acetaminophen_wb97mv_def2-svp

FD atoms: #19 (H; H on the single hydroxyl O (phenolic H)), #2 (N; the single N of acetaminophen (the amide N))

| step (Bohr) | atom | axis | g_fd | g_analytic | |delta| |
|---|---|---|---|---|---|
| 0.0010 | 19H | x | +1.854347e-03 | +1.830670e-03 | 2.368e-05 |
| 0.0010 | 19H | y | +9.418593e-03 | +9.415632e-03 | 2.961e-06 |
| 0.0010 | 19H | z | +8.565223e-06 | +8.564809e-06 | 4.140e-10 |
| 0.0010 | 2N | x | +2.265942e-02 | +2.266741e-02 | 7.991e-06 |
| 0.0010 | 2N | y | -7.832799e-03 | -7.853227e-03 | 2.043e-05 |
| 0.0010 | 2N | z | -7.636174e-05 | -7.616215e-05 | 1.996e-07 |
| — | — | max | | | 2.368e-05 |

step 0.0010: measured noise floor 0.000e+00, threshold 3.0e-04 -> within=True
| 0.0020 | 19H | x | +1.854350e-03 | +1.830670e-03 | 2.368e-05 |
| 0.0020 | 19H | y | +9.417586e-03 | +9.415632e-03 | 1.954e-06 |
| 0.0020 | 19H | z | +8.564228e-06 | +8.564809e-06 | 5.808e-10 |
| 0.0020 | 2N | x | +2.265957e-02 | +2.266741e-02 | 7.841e-06 |
| 0.0020 | 2N | y | -7.833337e-03 | -7.853227e-03 | 1.989e-05 |
| 0.0020 | 2N | z | -7.636439e-05 | -7.616215e-05 | 2.022e-07 |
| — | — | max | | | 2.368e-05 |

step 0.0020: measured noise floor 0.000e+00, threshold 3.0e-04 -> within=True

Overall worst |delta|: 2.368e-05 Eh/Bohr; all within 3x nominal floor: True
