# OPT references along the convex path, n = 20 density axis

p(lambda) = lambda * p_inside + (1 - lambda) * p_hat on each density cell's menus; p_inside = 1/|D| on every observed cell. lambda = 0 is the stored reference n20_{cat}. Certificate: (UB - LB)/LB <= 1e-3, or UB - LB <= 1e-5 * OPT(lambda = 0) (absolute) where the distance is ~0. Certifier: CGBFQP-GPU at 1 %, BM-ADMM at 5-100 % (iterations = ADMM iterations, certificate every 200); a cell certified by the fallback is marked.

| density | lambda | key | OPT (UB) | OPT LB | gap rel | gap abs | criterion | certifier | iterations / columns | wall s | other certifier (check) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 % | 0.0 | n20_d1pct | 18.621558 | 18.602976 | 1.0e-03 | 1.9e-02 | gap_rel | CB-GPU gap_rel | 1387 | 130 | bm_admm: [18.60354, 18.62179] 7800 its |
| 1 % | 0.1 | n20_d1pct_lam1 | 16.714936 | 16.698352 | 9.9e-04 | 1.7e-02 | gap_rel | CB-GPU gap_rel | 1555 | 192 | bm_admm: [16.69828, 16.71495] 9000 its |
| 1 % | 0.2 | n20_d1pct_lam2 | 14.809731 | 14.795059 | 9.9e-04 | 1.5e-02 | gap_rel | CB-GPU gap_rel | 1796 | 250 | bm_admm: [14.79486, 14.80953] 11000 its |
| 1 % | 0.3 | n20_d1pct_lam3 | 12.906375 | 12.893496 | 1.0e-03 | 1.3e-02 | gap_rel | CB-GPU gap_rel | 2064 | 338 | bm_admm: [12.89334, 12.90619] 12400 its |
| 1 % | 0.4 | n20_d1pct_lam4 | 11.005434 | 10.994469 | 1.0e-03 | 1.1e-02 | gap_rel | CB-GPU gap_rel | 2491 | 499 | bm_admm: [10.99427, 11.00516] 16600 its |
| 1 % | 0.5 | n20_d1pct_lam5 | 9.107835 | 9.098761 | 1.0e-03 | 9.1e-03 | gap_rel | CB-GPU gap_rel | 3017 | 779 | bm_admm: [9.09841, 9.10748] 24000 its |
| 1 % | 0.6 | n20_d1pct_lam6 | 7.214896 | 7.207692 | 1.0e-03 | 7.2e-03 | gap_rel | CB-GPU gap_rel | 3786 | 1288 | bm_admm: [7.20738, 7.21457] 36000 its |
| 1 % | 0.7 | n20_d1pct_lam7 | 5.328930 | 5.323612 | 1.0e-03 | 5.3e-03 | gap_rel | K-ADMM cert (beta=11.8) | 56400 | 333 | cgbfqp: [5.32386, 5.32918] 5133 cols |
| 1 % | 0.8 | n20_d1pct_lam8 | 3.455506 | 3.452059 | 1.0e-03 | 3.4e-03 | gap_rel | K-ADMM cert (beta=11.8) | 110200 | 649 |  |
| 1 % | 0.9 | n20_d1pct_lam9 | 1.609241 | 1.607633 | 1.0e-03 | 1.6e-03 | gap_rel | K-ADMM cert (beta=11.8) | 288400 | 1695 |  |
| 1 % | 1.0 | n20_d1pct_lam10 | 0.000036 | 0.000000 | - | 3.6e-05 | gap_abs<=0.000186 | K-ADMM cert (beta=11.8) | 200 | 1 |  |
| 5 % | 0.0 | n20_m1380 | 42.362644 | 42.321384 | 9.7e-04 | 4.1e-02 | gap_rel | K-ADMM cert (beta=26.6) | 2800 | 16 |  |
| 5 % | 0.1 | n20_m1380_lam1 | 38.087441 | 38.049980 | 9.8e-04 | 3.7e-02 | gap_rel | K-ADMM cert (beta=26.6) | 3200 | 32 |  |
| 5 % | 0.2 | n20_m1380_lam2 | 33.813783 | 33.781250 | 9.6e-04 | 3.3e-02 | gap_rel | K-ADMM cert (beta=26.6) | 3800 | 40 |  |
| 5 % | 0.3 | n20_m1380_lam3 | 29.541896 | 29.513558 | 9.6e-04 | 2.8e-02 | gap_rel | K-ADMM cert (beta=26.6) | 4600 | 49 |  |
| 5 % | 0.4 | n20_m1380_lam4 | 25.272289 | 25.247838 | 9.7e-04 | 2.4e-02 | gap_rel | K-ADMM cert (beta=26.6) | 5400 | 54 |  |
| 5 % | 0.5 | n20_m1380_lam5 | 21.005383 | 20.984899 | 9.8e-04 | 2.0e-02 | gap_rel | K-ADMM cert (beta=26.6) | 6600 | 67 |  |
| 5 % | 0.6 | n20_m1380_lam6 | 16.742396 | 16.725698 | 1.0e-03 | 1.7e-02 | gap_rel | K-ADMM cert (beta=26.6) | 7600 | 77 |  |
| 5 % | 0.7 | n20_m1380_lam7 | 12.485054 | 12.472692 | 9.9e-04 | 1.2e-02 | gap_rel | K-ADMM cert (beta=26.6) | 10800 | 112 |  |
| 5 % | 0.8 | n20_m1380_lam8 | 8.237697 | 8.229503 | 1.0e-03 | 8.2e-03 | gap_rel | K-ADMM cert (beta=26.6) | 20200 | 197 |  |
| 5 % | 0.9 | n20_m1380_lam9 | 4.013523 | 4.009515 | 1.0e-03 | 4.0e-03 | gap_rel | K-ADMM cert (beta=26.6) | 69200 | 708 |  |
| 5 % | 1.0 | n20_m1380_lam10 | 0.000080 | 0.000000 | - | 8.0e-05 | gap_abs<=0.000423626 | K-ADMM cert (beta=26.6) | 200 | 3 |  |
| 10 % | 0.0 | n20_m2759 | 60.176513 | 60.119300 | 9.5e-04 | 5.7e-02 | gap_rel | K-ADMM cert (beta=100) | None | 103 | bm_admm: [60.07769, 60.13360] 1600 its |
| 10 % | 0.1 | n20_m2759_lam1 | 54.085777 | 54.031875 | 1.0e-03 | 5.4e-02 | gap_rel | K-ADMM cert (beta=37.9) | 1600 | 17 |  |
| 10 % | 0.2 | n20_m2759_lam2 | 48.035736 | 47.990037 | 9.5e-04 | 4.6e-02 | gap_rel | K-ADMM cert (beta=37.9) | 1800 | 18 |  |
| 10 % | 0.3 | n20_m2759_lam3 | 41.985758 | 41.946125 | 9.4e-04 | 4.0e-02 | gap_rel | K-ADMM cert (beta=37.9) | 2000 | 21 |  |
| 10 % | 0.4 | n20_m2759_lam4 | 35.941150 | 35.905857 | 9.8e-04 | 3.5e-02 | gap_rel | K-ADMM cert (beta=37.9) | 2200 | 23 |  |
| 10 % | 0.5 | n20_m2759_lam5 | 29.898288 | 29.870091 | 9.4e-04 | 2.8e-02 | gap_rel | K-ADMM cert (beta=37.9) | 2800 | 28 |  |
| 10 % | 0.6 | n20_m2759_lam6 | 23.859918 | 23.836487 | 9.8e-04 | 2.3e-02 | gap_rel | K-ADMM cert (beta=37.9) | 4000 | 42 |  |
| 10 % | 0.7 | n20_m2759_lam7 | 17.827268 | 17.809573 | 9.9e-04 | 1.8e-02 | gap_rel | K-ADMM cert (beta=37.9) | 5800 | 58 |  |
| 10 % | 0.8 | n20_m2759_lam8 | 11.804470 | 11.792728 | 1.0e-03 | 1.2e-02 | gap_rel | K-ADMM cert (beta=37.9) | 10800 | 107 |  |
| 10 % | 0.9 | n20_m2759_lam9 | 5.803902 | 5.798125 | 1.0e-03 | 5.8e-03 | gap_rel | K-ADMM cert (beta=37.9) | 34800 | 361 |  |
| 10 % | 1.0 | n20_m2759_lam10 | 0.000114 | 0.000000 | - | 1.1e-04 | gap_abs<=0.000601765 | K-ADMM cert (beta=37.9) | 200 | 3 |  |
| 25 % | 0.0 | n20_m6899 | 95.313962 | 95.230742 | 8.7e-04 | 8.3e-02 | gap_rel | K-ADMM cert (beta=60.3) | 1200 | 7 |  |
| 25 % | 0.1 | n20_m6899_lam1 | 85.720486 | 85.673602 | 5.5e-04 | 4.7e-02 | gap_rel | K-ADMM cert (beta=60.3) | 1400 | 14 |  |
| 25 % | 0.2 | n20_m6899_lam2 | 76.172742 | 76.114761 | 7.6e-04 | 5.8e-02 | gap_rel | K-ADMM cert (beta=60.3) | 1400 | 15 |  |
| 25 % | 0.3 | n20_m6899_lam3 | 66.622438 | 66.556895 | 9.8e-04 | 6.6e-02 | gap_rel | K-ADMM cert (beta=60.3) | 1400 | 16 |  |
| 25 % | 0.4 | n20_m6899_lam4 | 57.052558 | 57.006432 | 8.1e-04 | 4.6e-02 | gap_rel | K-ADMM cert (beta=60.3) | 1600 | 17 |  |
| 25 % | 0.5 | n20_m6899_lam5 | 47.495364 | 47.455313 | 8.4e-04 | 4.0e-02 | gap_rel | K-ADMM cert (beta=60.3) | 1800 | 19 |  |
| 25 % | 0.6 | n20_m6899_lam6 | 37.940254 | 37.908144 | 8.5e-04 | 3.2e-02 | gap_rel | K-ADMM cert (beta=60.3) | 2400 | 26 |  |
| 25 % | 0.7 | n20_m6899_lam7 | 28.392840 | 28.366697 | 9.2e-04 | 2.6e-02 | gap_rel | K-ADMM cert (beta=60.3) | 3200 | 35 |  |
| 25 % | 0.8 | n20_m6899_lam8 | 18.856386 | 18.837949 | 9.8e-04 | 1.8e-02 | gap_rel | K-ADMM cert (beta=60.3) | 5600 | 56 |  |
| 25 % | 0.9 | n20_m6899_lam9 | 9.340488 | 9.331194 | 1.0e-03 | 9.3e-03 | gap_rel | K-ADMM cert (beta=60.3) | 12800 | 115 |  |
| 25 % | 1.0 | n20_m6899_lam10 | 0.000180 | 0.000000 | - | 1.8e-04 | gap_abs<=0.00095314 | K-ADMM cert (beta=60.3) | 200 | 2 |  |
| 50 % | 0.0 | n20_m13797 | 134.959895 | 134.892420 | 5.0e-04 | 6.7e-02 | gap_rel | K-ADMM cert (beta=85.8) | 1400 | 9 |  |
| 50 % | 0.1 | n20_m13797_lam1 | 121.438915 | 121.370632 | 5.6e-04 | 6.8e-02 | gap_rel | K-ADMM cert (beta=85.8) | 1400 | 16 |  |
| 50 % | 0.2 | n20_m13797_lam2 | 107.919607 | 107.848077 | 6.6e-04 | 7.2e-02 | gap_rel | K-ADMM cert (beta=85.8) | 1400 | 15 |  |
| 50 % | 0.3 | n20_m13797_lam3 | 94.415511 | 94.323800 | 9.7e-04 | 9.2e-02 | gap_rel | K-ADMM cert (beta=85.8) | 1200 | 13 |  |
| 50 % | 0.4 | n20_m13797_lam4 | 80.868340 | 80.813622 | 6.8e-04 | 5.5e-02 | gap_rel | K-ADMM cert (beta=85.8) | 1600 | 18 |  |
| 50 % | 0.5 | n20_m13797_lam5 | 67.339670 | 67.299111 | 6.0e-04 | 4.1e-02 | gap_rel | K-ADMM cert (beta=85.8) | 1800 | 20 |  |
| 50 % | 0.6 | n20_m13797_lam6 | 53.826631 | 53.786020 | 7.6e-04 | 4.1e-02 | gap_rel | K-ADMM cert (beta=85.8) | 2000 | 22 |  |
| 50 % | 0.7 | n20_m13797_lam7 | 40.317116 | 40.276865 | 1.0e-03 | 4.0e-02 | gap_rel | K-ADMM cert (beta=85.8) | 2200 | 16 |  |
| 50 % | 0.8 | n20_m13797_lam8 | 26.809400 | 26.783311 | 9.7e-04 | 2.6e-02 | gap_rel | K-ADMM cert (beta=85.8) | 3200 | 21 |  |
| 50 % | 0.9 | n20_m13797_lam9 | 13.322977 | 13.309789 | 9.9e-04 | 1.3e-02 | gap_rel | K-ADMM cert (beta=85.8) | 5800 | 38 |  |
| 50 % | 1.0 | n20_m13797_lam10 | 0.000254 | 0.000000 | - | 2.5e-04 | gap_abs<=0.0013496 | K-ADMM cert (beta=85.8) | 200 | 2 |  |
| 100 % | 0.0 | n20_dense | 191.218322 | 191.028601 | 9.9e-04 | 1.9e-01 | gap_rel | K-ADMM cert (beta=100) | None | 244 | bm_admm: [191.00472, 191.17166] 1400 its |
| 100 % | 0.1 | n20_dense_lam1 | 172.019458 | 171.875022 | 8.4e-04 | 1.4e-01 | gap_rel | K-ADMM cert (beta=100) | 1400 | 10 |  |
| 100 % | 0.2 | n20_dense_lam2 | 152.871860 | 152.746093 | 8.2e-04 | 1.3e-01 | gap_rel | K-ADMM cert (beta=100) | 1400 | 10 |  |
| 100 % | 0.3 | n20_dense_lam3 | 133.714988 | 133.617246 | 7.3e-04 | 9.8e-02 | gap_rel | K-ADMM cert (beta=100) | 1400 | 10 |  |
| 100 % | 0.4 | n20_dense_lam4 | 114.584407 | 114.492423 | 8.0e-04 | 9.2e-02 | gap_rel | K-ADMM cert (beta=100) | 1400 | 10 |  |
| 100 % | 0.5 | n20_dense_lam5 | 95.438352 | 95.368658 | 7.3e-04 | 7.0e-02 | gap_rel | K-ADMM cert (beta=100) | 1600 | 11 |  |
| 100 % | 0.6 | n20_dense_lam6 | 76.318368 | 76.245213 | 9.6e-04 | 7.3e-02 | gap_rel | K-ADMM cert (beta=100) | 1600 | 11 |  |
| 100 % | 0.7 | n20_dense_lam7 | 57.185335 | 57.132393 | 9.3e-04 | 5.3e-02 | gap_rel | K-ADMM cert (beta=100) | 2000 | 14 |  |
| 100 % | 0.8 | n20_dense_lam8 | 38.064567 | 38.028074 | 9.6e-04 | 3.6e-02 | gap_rel | K-ADMM cert (beta=100) | 2600 | 18 |  |
| 100 % | 0.9 | n20_dense_lam9 | 18.953487 | 18.937757 | 8.3e-04 | 1.6e-02 | gap_rel | K-ADMM cert (beta=100) | 4200 | 30 |  |
| 100 % | 1.0 | n20_dense_lam10 | 0.000360 | 0.000000 | - | 3.6e-04 | gap_abs<=0.00191 | K-ADMM cert (beta=100) | 200 | 2 |  |

## Iterations (BM-ADMM, 200-iteration checks) or columns (CGBFQP) to (UB - OPT)/OPT <= 1e-2, from the certificate trajectories

| lambda | 1 % | 5 % | 10 % | 25 % | 50 % | 100 % |
|---|---|---|---|---|---|---|
| 0.0 | 600 its | 400 its | 200 its | 200 its | 200 its | 200 its |
| 0.1 | 217 cols; 600 its | 400 its | 400 its | 200 its | 200 its | 200 its |
| 0.2 | 256 cols; 800 its | 400 its | 200 its | 200 its | 200 its | 200 its |
| 0.3 | 315 cols; 800 its | 400 its | 400 its | 200 its | 200 its | 200 its |
| 0.4 | 381 cols; 1000 its | 600 its | 400 its | 200 its | 200 its | 200 its |
| 0.5 | 495 cols; 1000 its | 600 its | 600 its | 600 its | 600 its | 200 its |
| 0.6 | 658 cols; 1400 its | 800 its | 1000 its | 400 its | 200 its | 600 its |
| 0.7 | 960 cols; 1800 its | 1000 its | 1000 its | 600 its | 600 its | 600 its |
| 0.8 | 1622 cols; 1600 its | 1200 its | 1200 its | 1200 its | 600 its | 400 its |
| 0.9 | 3974 cols; 3800 its | 1800 its | 1800 its | 1600 its | 1400 its | 400 its |
| 1.0 | cgbfqp: -; 200 its | 200 its | 200 its | 200 its | 200 its | 200 its |

Cells with OPT = 0 (inside the polytope) have no relative target; the 1 % target is relative and therefore tighter in absolute terms as lambda grows.
