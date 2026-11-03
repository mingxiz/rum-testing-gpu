# Pricing backends on the pairwise design

Same instances (`data/gen_instances.py --feas`, fingerprint-checked), same OPT reference, same stopping rules; one fresh process per run on an otherwise idle RTX 5090 (32 GB). Every CGBFQP iteration adds one column, so the number in parentheses is the iteration count. `--` = does not fit in device memory; blank = not run.

## Race: first column whose feasibility-pushed UB satisfies (UB - OPT)/OPT <= 1 %

| n | mp | OPT | mask (today): s (columns) | two layers, float32: s (columns) | two layers, uint16 block-scaled: s (columns) | u16 margin | device memory GiB mask / f32 / u16 |
|---|---|---|---|---|---|---|---|
| 20 | 380 | 1.876125 | 0.05 (63) | 0.09 (63) | 0.11 (63) | 9.6e-04 | <0.01 / <0.01 / <0.01 |
| 25 | 600 | 2.806456 | 0.29 (74) | 0.38 (74) | 0.41 (74) | 1.7e-03 | 0.28 / 0.05 / 0.03 |
| 30 | 870 | 3.344219 | 14.6 (94) | 14.5 (94) | 15.0 (94) | 2.2e-03 | 5.6 / 1.2 / 0.63 |
| 32 | 992 | 3.612884 | 67.0 (96) | 67.6 (96) | 69.4 (96) | 2.7e-03 | 22.2 / 4.6 / 2.4 |
| 33 | 1,056 | 3.899904 | -- | 146.0 (90) | 154.5 (90) | 2.7e-03 | -- / 8.9 / 4.5 |
| 34 | 1,122 | 3.842021 | -- | 325.0 (103) | 334.0 (103) | 3.1e-03 | -- / 17.7 / 9.0 |
| 35 | 1,190 | 4.092598 | -- | -- | 708.2 (103) | 3.2e-03 | -- / -- / 17.4 |

## Certification: (UB - LB)/LB <= 1e-3 with the Frank-Wolfe lower bound

| n | mp | OPT | mask (today): s (columns) | two layers, float32: s (columns) | two layers, uint16 block-scaled: s (columns) | u16 margin | device memory GiB mask / f32 / u16 | certified [LB, UB] mask / f32 / u16 |
|---|---|---|---|---|---|---|---|---|
| 20 | 380 | 1.876125 | 0.21 (181) |  |  |  | <0.01 / -- / -- | [1.87432, 1.87613] / - / - |
| 25 | 600 | 2.806456 | 1.20 (226) |  |  |  | 0.28 / -- / -- | [2.80392, 2.80646] / - / - |
| 30 | 870 | 3.344219 | 58.2 (321) |  |  |  | 5.6 / -- / -- | [3.34091, 3.34422] / - / - |
| 32 | 992 | 3.612884 | 245.7 (341) |  |  |  | 22.2 / -- / -- | [3.60933, 3.61288] / - / - |
| 33 | 1,056 | 3.899904 | -- | 561.3 (345) | 609.3 (355) | 2.7e-03 | -- / 8.9 / 4.5 | - / [3.89604, 3.89990] / [3.89626, 3.89986] |
| 34 | 1,122 | 3.842021 | -- | 1214.8 (387) | 1294.5 (400) | 3.1e-03 | -- / 17.7 / 9.0 | - / [3.83829, 3.84202] / [3.83816, 3.84200] |
| 35 | 1,190 | 4.092598 | -- | -- | 2602.0 (379) | 3.2e-03 | -- / -- / 17.4 | - / - / [4.08859, 4.09260] |

Device memory: mask = 5 B per subset (V float32 + pred uint8) + one layer buffer of C(n, n/2) masks; two layers = 2 x C(n, n/2) x (4 or 2) B + uint8 predecessors for the top 10 layers + the 2^(n-10) sub-lattice scaffolding. The uint16 margin (0.5 x sum over layers of the largest block step) is added to Vmax before the Frank-Wolfe gap, so the uint16 lower bound is valid, only looser by the margin; the columns generated can differ from the float32 run only where two rankings tie within the margin.
