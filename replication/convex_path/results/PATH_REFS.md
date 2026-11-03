# OPT references along the convex path (CGBFQP-GPU certificates)

p(lambda) = lambda * p_inside + (1 - lambda) * p_hat on the pairwise design; p_inside = 1/|D| on every observed menu (0.5 / 0.5). lambda = 0 is the stored pairwise reference n{n}_sparse. Certificate: (UB - LB)/LB <= 1e-3, or UB - LB <= 1e-5 (absolute) where the distance is ~0. OPT = UB of the certified feasible point.

| n | lambda | key | OPT (UB) | OPT LB | gap rel | gap abs | criterion | columns | wall s | method |
|---|---|---|---|---|---|---|---|---|---|---|
| 25 | 0.0 | n25_sparse | 2.806456 | 2.803916 | 9.1e-04 | 2.5e-03 | gap_rel | 226 | 1 | CB-GPU gap_rel |
| 25 | 0.1 | n25_sparse_lam1 | 2.257018 | 2.254828 | 9.7e-04 | 2.2e-03 | gap_rel | 275 | 1 | CB-GPU gap_rel |
| 25 | 0.2 | n25_sparse_lam2 | 1.715725 | 1.714184 | 9.0e-04 | 1.5e-03 | gap_rel | 322 | 1 | CB-GPU gap_rel |
| 25 | 0.3 | n25_sparse_lam3 | 1.189463 | 1.188293 | 9.8e-04 | 1.2e-03 | gap_rel | 448 | 2 | CB-GPU gap_rel |
| 25 | 0.4 | n25_sparse_lam4 | 0.699681 | 0.698995 | 9.8e-04 | 6.9e-04 | gap_rel | 767 | 7 | CB-GPU gap_rel |
| 25 | 0.5 | n25_sparse_lam5 | 0.306190 | 0.305923 | 8.7e-04 | 2.7e-04 | gap_rel | 358 | 2 | CB-GPU gap_rel |
| 25 | 0.6 | n25_sparse_lam6 | 0.043123 | 0.043091 | 7.5e-04 | 3.2e-05 | gap_rel | 286 | 2 | CB-GPU gap_rel |
| 25 | 0.7 | n25_sparse_lam7 | 0.000009 | 0.000000 | - | 9.0e-06 | gap_abs<=1e-05 | 114 | 0 | CB-GPU gap_abs |
| 25 | 0.8 | n25_sparse_lam8 | 0.000010 | 0.000000 | - | 9.7e-06 | gap_abs<=1e-05 | 92 | 0 | CB-GPU gap_abs |
| 25 | 0.9 | n25_sparse_lam9 | 0.000009 | 0.000000 | - | 9.1e-06 | gap_abs<=1e-05 | 88 | 0 | CB-GPU gap_abs |
| 25 | 1.0 | n25_sparse_lam10 | 0.000000 | 0.000000 | - | 0.0e+00 | gap_abs<=1e-05 | 2 | 0 | CB-GPU gap_abs |
| 30 | 0.0 | n30_sparse | 3.344219 | 3.340907 | 9.9e-04 | 3.3e-03 | gap_rel | 321 | 58 | CB-GPU gap_rel |
| 30 | 0.1 | n30_sparse_lam1 | 2.700336 | 2.697640 | 1.0e-03 | 2.7e-03 | gap_rel | 387 | 62 | CB-GPU gap_rel |
| 30 | 0.2 | n30_sparse_lam2 | 2.063945 | 2.062148 | 8.7e-04 | 1.8e-03 | gap_rel | 483 | 79 | CB-GPU gap_rel |
| 30 | 0.3 | n30_sparse_lam3 | 1.440452 | 1.439053 | 9.7e-04 | 1.4e-03 | gap_rel | 650 | 107 | CB-GPU gap_rel |
| 30 | 0.4 | n30_sparse_lam4 | 0.846616 | 0.845874 | 8.8e-04 | 7.4e-04 | gap_rel | 990 | 165 | CB-GPU gap_rel |
| 30 | 0.5 | n30_sparse_lam5 | 0.358373 | 0.358100 | 7.6e-04 | 2.7e-04 | gap_rel | 686 | 116 | CB-GPU gap_rel |
| 30 | 0.6 | n30_sparse_lam6 | 0.040900 | 0.040861 | 9.8e-04 | 4.0e-05 | gap_rel | 398 | 65 | CB-GPU gap_rel |
| 30 | 0.7 | n30_sparse_lam7 | 0.000009 | 0.000000 | - | 9.3e-06 | gap_abs<=1e-05 | 146 | 24 | CB-GPU gap_abs |
| 30 | 0.8 | n30_sparse_lam8 | 0.000009 | 0.000000 | - | 9.4e-06 | gap_abs<=1e-05 | 111 | 18 | CB-GPU gap_abs |
| 30 | 0.9 | n30_sparse_lam9 | 0.000009 | 0.000000 | - | 9.4e-06 | gap_abs<=1e-05 | 106 | 17 | CB-GPU gap_abs |
| 30 | 1.0 | n30_sparse_lam10 | 0.000000 | 0.000000 | - | 0.0e+00 | gap_abs<=1e-05 | 2 | 0 | CB-GPU gap_abs |
| 32 | 0.0 | n32_sparse | 3.612884 | 3.609332 | 9.8e-04 | 3.6e-03 | gap_rel | 341 | 246 | CB-GPU gap_rel |
| 32 | 0.1 | n32_sparse_lam1 | 2.918740 | 2.915893 | 9.8e-04 | 2.8e-03 | gap_rel | 412 | 295 | CB-GPU gap_rel |
| 32 | 0.2 | n32_sparse_lam2 | 2.231999 | 2.229959 | 9.1e-04 | 2.0e-03 | gap_rel | 535 | 384 | CB-GPU gap_rel |
| 32 | 0.3 | n32_sparse_lam3 | 1.557759 | 1.556270 | 9.6e-04 | 1.5e-03 | gap_rel | 664 | 479 | CB-GPU gap_rel |
| 32 | 0.4 | n32_sparse_lam4 | 0.910000 | 0.909168 | 9.1e-04 | 8.3e-04 | gap_rel | 1027 | 748 | CB-GPU gap_rel |
| 32 | 0.5 | n32_sparse_lam5 | 0.364491 | 0.364127 | 1.0e-03 | 3.6e-04 | gap_rel | 1945 | 1559 | CB-GPU gap_rel |
| 32 | 0.6 | n32_sparse_lam6 | 0.072665 | 0.072596 | 9.5e-04 | 6.9e-05 | gap_rel | 389 | 281 | CB-GPU gap_rel |
| 32 | 0.7 | n32_sparse_lam7 | 0.000009 | 0.000000 | - | 9.2e-06 | gap_abs<=1e-05 | 154 | 111 | CB-GPU gap_abs |
| 32 | 0.8 | n32_sparse_lam8 | 0.000009 | 0.000000 | - | 9.3e-06 | gap_abs<=1e-05 | 116 | 83 | CB-GPU gap_abs |
| 32 | 0.9 | n32_sparse_lam9 | 0.000009 | 0.000000 | - | 9.3e-06 | gap_abs<=1e-05 | 110 | 79 | CB-GPU gap_abs |
| 32 | 1.0 | n32_sparse_lam10 | 0.000000 | 0.000000 | - | 0.0e+00 | gap_abs<=1e-05 | 2 | 1 | CB-GPU gap_abs |
| 33 | 0.0 | n33_sparse | 3.899904 | 3.896044 | 9.9e-04 | 3.9e-03 | gap_rel | 345 | 561 | CB-GPU gap_rel [stream_f32] |
| 33 | 0.1 | n33_sparse_lam1 | 3.171690 | 3.168833 | 9.0e-04 | 2.9e-03 | gap_rel | 427 | 1495 | CB-GPU gap_rel [stream_f32] |
| 33 | 0.2 | n33_sparse_lam2 | 2.450467 | 2.448088 | 9.7e-04 | 2.4e-03 | gap_rel | 508 | 1797 | CB-GPU gap_rel [stream_f32] |
| 33 | 0.3 | n33_sparse_lam3 | 1.740740 | 1.739068 | 9.6e-04 | 1.7e-03 | gap_rel | 688 | 3555 | CB-GPU gap_rel [stream_f32] |
| 33 | 0.4 | n33_sparse_lam4 | 1.055034 | 1.054033 | 9.5e-04 | 1.0e-03 | gap_rel | 1012 | 1657 | CB-GPU gap_rel [stream_f32] |
| 33 | 0.5 | n33_sparse_lam5 | 0.449975 | 0.449537 | 9.7e-04 | 4.4e-04 | gap_rel | 2279 | 3862 | CB-GPU gap_rel [stream_f32] |
| 33 | 0.6 | n33_sparse_lam6 | 0.077970 | 0.077895 | 9.6e-04 | 7.5e-05 | gap_rel | 503 | 821 | CB-GPU gap_rel [stream_f32] |
| 33 | 0.7 | n33_sparse_lam7 | 0.000009 | 0.000000 | - | 9.4e-06 | gap_abs<=1e-05 | 213 | 346 | CB-GPU gap_abs [stream_f32] |
| 33 | 0.8 | n33_sparse_lam8 | 0.000010 | 0.000000 | - | 9.6e-06 | gap_abs<=1e-05 | 121 | 196 | CB-GPU gap_abs [stream_f32] |
| 33 | 0.9 | n33_sparse_lam9 | 0.000009 | 0.000000 | - | 8.8e-06 | gap_abs<=1e-05 | 116 | 188 | CB-GPU gap_abs [stream_f32] |
| 33 | 1.0 | n33_sparse_lam10 | 0.000000 | 0.000000 | - | 0.0e+00 | gap_abs<=1e-05 | 2 | 3 | CB-GPU gap_abs [stream_f32] |
| 34 | 0.0 | n34_sparse | 3.842021 | 3.838290 | 9.7e-04 | 3.7e-03 | gap_rel | 387 | 1215 | CB-GPU gap_rel [stream_f32] |
| 34 | 0.1 | n34_sparse_lam1 | 3.105332 | 3.102262 | 9.9e-04 | 3.1e-03 | gap_rel | 436 | 1375 | CB-GPU gap_rel [stream_f32] |
| 34 | 0.2 | n34_sparse_lam2 | 2.375673 | 2.373362 | 9.7e-04 | 2.3e-03 | gap_rel | 524 | 1652 | CB-GPU gap_rel [stream_f32] |
| 34 | 0.3 | n34_sparse_lam3 | 1.658676 | 1.657068 | 9.7e-04 | 1.6e-03 | gap_rel | 712 | 2248 | CB-GPU gap_rel [stream_f32] |
| 34 | 0.4 | n34_sparse_lam4 | 0.971225 | 0.970326 | 9.3e-04 | 9.0e-04 | gap_rel | 1099 | 3477 | CB-GPU gap_rel [stream_f32] |
| 34 | 0.5 | n34_sparse_lam5 | 0.403238 | 0.402895 | 8.5e-04 | 3.4e-04 | gap_rel | 1150 | 3660 | CB-GPU gap_rel [stream_f32] |
| 34 | 0.6 | n34_sparse_lam6 | 0.072429 | 0.072366 | 8.7e-04 | 6.3e-05 | gap_rel | 499 | 1574 | CB-GPU gap_rel [stream_f32] |
| 34 | 0.7 | n34_sparse_lam7 | 0.000009 | 0.000000 | - | 9.0e-06 | gap_abs<=1e-05 | 166 | 521 | CB-GPU gap_abs [stream_f32] |
| 34 | 0.8 | n34_sparse_lam8 | 0.000010 | 0.000000 | - | 9.7e-06 | gap_abs<=1e-05 | 119 | 376 | CB-GPU gap_abs [stream_f32] |
| 34 | 0.9 | n34_sparse_lam9 | 0.000010 | 0.000000 | - | 9.9e-06 | gap_abs<=1e-05 | 113 | 355 | CB-GPU gap_abs [stream_f32] |
| 34 | 1.0 | n34_sparse_lam10 | 0.000000 | 0.000000 | - | 0.0e+00 | gap_abs<=1e-05 | 2 | 6 | CB-GPU gap_abs [stream_f32] |
| 35 | 0.0 | n35_sparse | 4.092598 | 4.088588 | 9.8e-04 | 4.0e-03 | gap_rel | 379 | 2602 | CB-GPU gap_rel [stream_u16] |
| 35 | 0.1 | n35_sparse_lam1 | 3.321841 | 3.318524 | 1.0e-03 | 3.3e-03 | gap_rel | 434 | 3042 | CB-GPU gap_rel [stream_u24] |
| 35 | 0.2 | n35_sparse_lam2 | 2.557365 | 2.554895 | 9.7e-04 | 2.5e-03 | gap_rel | 564 | 3949 | CB-GPU gap_rel [stream_u24] |
| 35 | 0.3 | n35_sparse_lam3 | 1.803198 | 1.801424 | 9.8e-04 | 1.8e-03 | gap_rel | 730 | 5109 | CB-GPU gap_rel [stream_u24] |
| 35 | 0.4 | n35_sparse_lam4 | 1.068344 | 1.067300 | 9.8e-04 | 1.0e-03 | gap_rel | 1080 | 7578 | CB-GPU gap_rel [stream_u24] |
| 35 | 0.5 | n35_sparse_lam5 | 0.395131 | 0.394738 | 1.0e-03 | 3.9e-04 | gap_rel | 2316 | 16368 | CB-GPU gap_rel [stream_u24] |
| 35 | 0.6 | n35_sparse_lam6 | 0.038466 | 0.038430 | 9.5e-04 | 3.7e-05 | gap_rel | 559 | 3919 | CB-GPU gap_rel [stream_u24] |
| 35 | 0.7 | n35_sparse_lam7 | 0.000010 | 0.000000 | - | 9.7e-06 | gap_abs<=1e-05 | 197 | 1379 | CB-GPU gap_abs [stream_u24] |
| 35 | 0.8 | n35_sparse_lam8 | 0.000009 | 0.000000 | - | 9.1e-06 | gap_abs<=1e-05 | 122 | 853 | CB-GPU gap_abs [stream_u24] |
| 35 | 0.9 | n35_sparse_lam9 | 0.000010 | 0.000000 | - | 9.7e-06 | gap_abs<=1e-05 | 116 | 814 | CB-GPU gap_abs [stream_u24] |
| 35 | 1.0 | n35_sparse_lam10 | 0.000000 | 0.000000 | - | 0.0e+00 | gap_abs<=1e-05 | 2 | 14 | CB-GPU gap_abs [stream_u24] |

## Columns to (UB - OPT)/OPT <= 1e-2 along the path (from the certificate trajectories)

| lambda | n = 25 | n = 30 | n = 32 | n = 33 | n = 34 | n = 35 |
|---|---|---|---|---|---|---|
| 0.0 | 74 (0.4 s) | 94 (16.6 s) | 96 (72.8 s) | - | - | - |
| 0.1 | 87 (0.3 s) | 111 (17.7 s) | 113 (80.8 s) | - | - | - |
| 0.2 | 120 (0.5 s) | 150 (24.2 s) | 163 (116.7 s) | - | - | - |
| 0.3 | 171 (0.7 s) | 222 (35.9 s) | 237 (170.3 s) | - | - | - |
| 0.4 | 250 (1.0 s) | 369 (60.1 s) | 389 (280.1 s) | - | - | - |
| 0.5 | 195 (0.8 s) | 322 (52.3 s) | 579 (421.2 s) | - | - | - |
| 0.6 | 179 (0.8 s) | 241 (39.1 s) | 225 (162.1 s) | - | - | - |

For lambda >= 0.7 the optimum is 0 and the relative criterion is undefined. The 1 % target is relative, so it is ten times tighter in absolute terms at lambda = 0.5 than at lambda = 0.
