# Feasibility study

Race criterion (UB-OPT)/OPT <= 1%; OPT certified to (UB-LB)/LB <= 1e-3 (data/opt_refs.json). Times in seconds; in parentheses columns (CG methods) or ADMM iterations. One process per run, GPU idle before each.


## (1) Pairwise design, growing n

| n | mp | 2^n | OPT | OPT certified by (cols, s) | CGBFQP-GPU race | CGIP | CGIP-heur | CGBFQP-GPU in human units | device memory |
|---|---|---|---|---|---|---|---|---|---|
| 20 | 380 | 2^20 = 1,048,576 | 1.876125 | CB-GPU gap_rel (181, 0 s) | 0.05 (63 cols) |  |  | 0.051 s | 0.0 GB |
| 25 | 600 | 2^25 = 33,554,432 | 2.806456 | CB-GPU gap_rel (226, 1 s) | 0.29 (74 cols) |  |  | 0.29 s | 0.3 GB |
| 30 | 870 | 2^30 = 1,073,741,824 | 3.344219 | CB-GPU gap_rel (321, 58 s) | 14.6 (94 cols) |  |  | 15 s | 5.6 GB |
| 32 | 992 | 2^32 = 4,294,967,296 | 3.612884 | CB-GPU gap_rel (341, 246 s) | 67.0 (96 cols) |  |  | 1.1 min | 22.2 GB |
| 33 | 1,056 | 2^33 = 8,589,934,592 | 3.899904 | CB-GPU gap_rel [stream_f32] (345, 561 s) | 146.0 (90 cols) [stream_f32] |  |  | 2.4 min | 8.9 GB |
| 34 | 1,122 | 2^34 = 17,179,869,184 | 3.842021 | CB-GPU gap_rel [stream_f32] (387, 1215 s) | 325.0 (103 cols) [stream_f32] |  |  | 5.4 min | 17.7 GB |
| 35 | 1,190 | 2^35 = 34,359,738,368 | 4.092598 | CB-GPU gap_rel [stream_u16] (379, 2602 s) | 708.2 (103 cols) [stream_u16] |  |  | 11.8 min | 17.4 GB |

## (2) n = 20, growing density

| density | mp | OPT | OPT certified by (its, s) | BM-ADMM race | CGIP | CGIP-heur | BM-ADMM in human units | device memory |
|---|---|---|---|---|---|---|---|---|
| 1.0% | 104,885 | 18.621558 | CB-GPU gap_rel (1387, 130 s) | 1.87 (700 it) |  |  | 1.9 s | 2.0 GB |
| 5.0% | 524,403 | 42.362644 | K-ADMM cert (beta=26.6) (2800, 16 s) | 0.82 (350 it) |  |  | 0.82 s | 1.0 GB |
| 10.0% | 1,048,422 | 60.176513 | K-ADMM cert (beta=100) (-, 103 s) | 0.37 (100 it) |  |  | 0.37 s | 1.0 GB |
| 25.0% | 2,621,622 | 95.313962 | K-ADMM cert (beta=60.3) (1200, 7 s) | 0.40 (100 it) |  |  | 0.4 s | 1.1 GB |
| 50.0% | 5,242,865 | 134.959895 | K-ADMM cert (beta=85.8) (1400, 9 s) | 0.45 (100 it) |  |  | 0.45 s | 1.2 GB |
| 100.0% | 10,485,740 | 191.218322 | K-ADMM cert (beta=100) (-, 244 s) | 0.59 (100 it) |  |  | 0.59 s | 1.5 GB |

## Trajectories

Every race JSON carries `trace`: per column (CGBFQP) or per 50-iteration check (BM-ADMM): [step, wall_s, UB, LB]. First/last entries:

- n20_d1pct bm_admm: 14 entries; first [50, 0.1478, 18.9818, 19.2522]; last [700, 1.612, 18.6441, 18.766]
- n20_dense bm_admm: 2 entries; first [50, 0.1552, 190.2975, 193.4331]; last [100, 0.2541, 191.5424, 191.4769]
- n20_m13797 bm_admm: 2 entries; first [50, 0.1495, 134.0891, 137.6274]; last [100, 0.2437, 135.4985, 135.6741]
- n20_m1380 bm_admm: 7 entries; first [50, 0.147, 42.4401, 43.2129]; last [350, 0.6964, 42.4032, 42.7861]
- n20_m2759 bm_admm: 2 entries; first [50, 0.1474, 59.9904, 60.8414]; last [100, 0.2392, 60.5147, 60.7197]
- n20_m6899 bm_admm: 2 entries; first [50, 0.148, 94.7467, 96.6113]; last [100, 0.2405, 95.7606, 95.7844]
- n20_sparse bm_admm: 261 entries; first [50, 0.0877, 0.2033, None]; last [13050, 41.1529, 1.8798, 1.8948]
- n20_sparse cgbfqp: 63 entries; first [1, 0.0007, 10.9824, 0.0, 1]; last [63, 0.0508, 1.8947, 1.7691, 63]
- n25_sparse cgbfqp: 74 entries; first [1, 0.0038, 13.4641, 0.0, 1]; last [74, 0.2861, 2.8336, 2.6673, 74]
- n25_sparse_lam1 cgbfqp: 87 entries; first [1, 0.0038, 13.229, 0.0, 1]; last [87, 0.3364, 2.2791, 2.1569, 87]
- n25_sparse_lam10 cgbfqp: 2 entries; first [1, 0.0038, 12.2474, 0.0, 1]; last [2, 0.0075, 0.0, 0.0, 2]
- n25_sparse_lam2 cgbfqp: 120 entries; first [1, 0.0038, 13.0166, 0.0, 1]; last [120, 0.4698, 1.7324, 1.6264, 120]
- n25_sparse_lam3 cgbfqp: 171 entries; first [1, 0.0038, 12.8279, 0.0, 1]; last [171, 0.6819, 1.2013, 1.1251, 171]
- n25_sparse_lam4 cgbfqp: 250 entries; first [1, 0.0038, 12.6641, 0.0, 1]; last [250, 1.0258, 0.7067, 0.6582, 250]
- n25_sparse_lam5 cgbfqp: 195 entries; first [1, 0.0037, 12.526, 0.0, 1]; last [195, 0.7864, 0.3092, 0.1737, 195]
- n25_sparse_lam6 cgbfqp: 179 entries; first [1, 0.0038, 12.4146, 0.0, 1]; last [179, 0.7383, 0.0435, 0.0, 179]
- n25_sparse_lam7 cgbfqp: 114 entries; first [1, 0.0038, 12.3306, 0.0, 1]; last [114, 0.4441, 0.0, 0.0, 114]
- n25_sparse_lam8 cgbfqp: 92 entries; first [1, 0.0038, 12.2745, 0.0, 1]; last [92, 0.3556, 0.0, 0.0, 92]
- n25_sparse_lam9 cgbfqp: 88 entries; first [1, 0.0038, 12.2467, 0.0, 1]; last [88, 0.3397, 0.0, 0.0, 88]
- n30_sparse cgbfqp: 94 entries; first [1, 0.1499, 16.2105, 0.0, 1]; last [94, 14.6434, 3.3776, 3.2053, 94]
- n30_sparse_lam1 cgbfqp: 111 entries; first [1, 0.1577, 15.9374, 0.0, 1]; last [111, 17.2262, 2.7271, 2.6075, 111]
- n30_sparse_lam10 cgbfqp: 2 entries; first [1, 0.1484, 14.7479, 0.0, 1]; last [2, 0.2976, 0.0, 0.0, 2]
- n30_sparse_lam2 cgbfqp: 150 entries; first [1, 0.1498, 15.6897, 0.0, 1]; last [150, 23.4394, 2.0845, 1.9741, 150]
- n30_sparse_lam3 cgbfqp: 222 entries; first [1, 0.1504, 15.4686, 0.0, 1]; last [222, 34.907, 1.4547, 1.3755, 222]
- n30_sparse_lam4 cgbfqp: 369 entries; first [1, 0.1503, 15.2753, 0.0, 1]; last [369, 58.4586, 0.855, 0.8095, 369]
- n30_sparse_lam5 cgbfqp: 322 entries; first [1, 0.1503, 15.1107, 0.0, 1]; last [322, 50.9877, 0.3619, 0.2681, 322]
- n30_sparse_lam6 cgbfqp: 241 entries; first [1, 0.1504, 14.9759, 0.0, 1]; last [241, 38.0983, 0.0413, 0.0, 241]
- n30_sparse_lam7 cgbfqp: 146 entries; first [1, 0.1504, 14.8717, 0.0, 1]; last [146, 22.9592, 0.0, 0.0, 146]
- n30_sparse_lam8 cgbfqp: 111 entries; first [1, 0.1505, 14.7987, 0.0, 1]; last [111, 17.4099, 0.0, 0.0, 111]
- n30_sparse_lam9 cgbfqp: 106 entries; first [1, 0.1503, 14.7573, 0.0, 1]; last [106, 16.6295, 0.0, 0.0, 106]
- n32_sparse cgbfqp: 96 entries; first [1, 0.6847, 17.4757, 0.0, 1]; last [96, 67.035, 3.6489, 3.4707, 96]
- n32_sparse_lam1 cgbfqp: 113 entries; first [1, 0.6841, 17.1659, 0.0, 1]; last [113, 78.9651, 2.9473, 2.809, 113]
- n32_sparse_lam10 cgbfqp: 2 entries; first [1, 0.672, 15.748, 0.0, 1]; last [2, 1.3382, 0.0, 0.0, 2]
- n32_sparse_lam2 cgbfqp: 163 entries; first [1, 0.6863, 16.8833, 0.0, 1]; last [163, 114.2431, 2.2542, 2.1532, 163]
- n32_sparse_lam3 cgbfqp: 237 entries; first [1, 0.6868, 16.6293, 0.0, 1]; last [237, 166.3522, 1.5732, 1.4885, 237]
- n32_sparse_lam4 cgbfqp: 389 entries; first [1, 0.687, 16.4051, 0.0, 1]; last [389, 273.8104, 0.9191, 0.8696, 389]
- n32_sparse_lam5 cgbfqp: 579 entries; first [1, 0.6875, 16.2122, 0.0, 1]; last [579, 408.4563, 0.3681, 0.3298, 579]
- n32_sparse_lam6 cgbfqp: 225 entries; first [1, 0.6885, 16.0514, 0.0, 1]; last [225, 158.1141, 0.0734, 0.0, 225]
- n32_sparse_lam7 cgbfqp: 154 entries; first [1, 0.6886, 15.924, 0.0, 1]; last [154, 108.0506, 0.0, 0.0, 154]
- n32_sparse_lam8 cgbfqp: 116 entries; first [1, 0.6867, 15.8305, 0.0, 1]; last [116, 81.2637, 0.0, 0.0, 116]
- n32_sparse_lam9 cgbfqp: 110 entries; first [1, 0.6861, 15.7718, 0.0, 1]; last [110, 77.0286, 0.0, 0.0, 110]
- n33_sparse cgbfqp: 90 entries; first [1, 1.6109, 18.6449, 0.0, 1]; last [90, 145.9909, 3.9388, 3.7502, 90]
- n33_sparse_lam1 cgbfqp: 114 entries; first [1, 1.6107, 18.2707, 0.0, 1]; last [114, 184.8909, 3.2031, 3.0476, 114]
- n33_sparse_lam10 cgbfqp: 2 entries; first [1, 1.6024, 16.2481, 0.0, 1]; last [2, 3.1866, 0.0, 0.0, 2]
- n33_sparse_lam2 cgbfqp: 157 entries; first [1, 1.6169, 17.9226, 0.0, 1]; last [157, 254.6889, 2.4749, 2.3406, 157]
- n33_sparse_lam3 cgbfqp: 240 entries; first [1, 1.6166, 17.602, 0.0, 1]; last [240, 389.1439, 1.7581, 1.6769, 240]
- n33_sparse_lam4 cgbfqp: 390 entries; first [1, 1.6134, 17.3104, 0.0, 1]; last [390, 633.7599, 1.0655, 1.0161, 390]
- n33_sparse_lam5 cgbfqp: 703 entries; first [1, 1.6169, 17.0494, 0.0, 1]; last [703, 1144.8015, 0.4545, 0.4239, 703]
- n33_sparse_lam6 cgbfqp: 299 entries; first [1, 1.6144, 16.8204, 0.0, 1]; last [299, 484.2113, 0.0787, 0.0, 299]
- n33_sparse_lam7 cgbfqp: 213 entries; first [1, 1.6088, 16.6246, 0.0, 1]; last [213, 345.689, 0.0, 0.0, 213]
- n33_sparse_lam8 cgbfqp: 121 entries; first [1, 1.6173, 16.4633, 0.0, 1]; last [121, 196.0976, 0.0, 0.0, 121]
- n33_sparse_lam9 cgbfqp: 116 entries; first [1, 1.6146, 16.3375, 0.0, 1]; last [116, 188.0011, 0.0, 0.0, 116]
- n34_sparse cgbfqp: 103 entries; first [1, 3.1383, 19.0114, 0.0, 1]; last [103, 324.9788, 3.8802, 3.7111, 103]
- n34_sparse_lam1 cgbfqp: 124 entries; first [1, 3.127, 18.6461, 0.0, 1]; last [124, 389.0875, 3.136, 2.9817, 124]
- n34_sparse_lam10 cgbfqp: 2 entries; first [1, 3.0856, 16.7481, 0.0, 1]; last [2, 6.1437, 0.0, 0.0, 2]
- n34_sparse_lam2 cgbfqp: 169 entries; first [1, 3.1162, 18.3079, 0.0, 1]; last [169, 529.4512, 2.3993, 2.2937, 169]
- n34_sparse_lam3 cgbfqp: 246 entries; first [1, 3.1199, 17.9983, 0.0, 1]; last [246, 771.5206, 1.6751, 1.5831, 246]
- n34_sparse_lam4 cgbfqp: 417 entries; first [1, 3.1196, 17.7189, 0.0, 1]; last [417, 1308.4443, 0.9809, 0.9299, 417]
- n34_sparse_lam5 cgbfqp: 473 entries; first [1, 3.1183, 17.471, 0.0, 1]; last [473, 1486.321, 0.4073, 0.3347, 473]
- n34_sparse_lam6 cgbfqp: 259 entries; first [1, 3.1334, 17.2561, 0.0, 1]; last [259, 812.2624, 0.0731, 0.0, 259]
- n34_sparse_lam7 cgbfqp: 166 entries; first [1, 3.1186, 17.0753, 0.0, 1]; last [166, 520.1431, 0.0, 0.0, 166]
- n34_sparse_lam8 cgbfqp: 119 entries; first [1, 3.1211, 16.9298, 0.0, 1]; last [119, 373.3771, 0.0, 0.0, 119]
- n34_sparse_lam9 cgbfqp: 113 entries; first [1, 3.1262, 16.8205, 0.0, 1]; last [113, 354.2811, 0.0, 0.0, 113]
- n35_sparse cgbfqp: 103 entries; first [1, 6.8351, 19.6258, 0.0, 1]; last [103, 708.2481, 4.133, 3.9605, 103]
- n35_sparse_lam1 cgbfqp: 122 entries; first [1, 6.9847, 19.2439, 0.0, 1]; last [122, 853.18, 3.3547, 3.1896, 122]
- n35_sparse_lam10 cgbfqp: 2 entries; first [1, 6.8486, 17.2482, 0.0, 1]; last [2, 13.6496, 0.0, 0.0, 2]
- n35_sparse_lam2 cgbfqp: 167 entries; first [1, 6.9708, 18.8901, 0.0, 1]; last [167, 1168.0248, 2.5827, 2.4487, 167]
- n35_sparse_lam3 cgbfqp: 245 entries; first [1, 6.9673, 18.5659, 0.0, 1]; last [245, 1710.2317, 1.8211, 1.7255, 245]
- n35_sparse_lam4 cgbfqp: 410 entries; first [1, 6.9948, 18.273, 0.0, 1]; last [410, 2866.4493, 1.079, 1.0162, 410]
- n35_sparse_lam5 cgbfqp: 1042 entries; first [1, 6.9776, 18.0127, 0.0, 1]; last [1042, 7304.4743, 0.3991, 0.3751, 1042]
- n35_sparse_lam6 cgbfqp: 340 entries; first [1, 6.9609, 17.7867, 0.0, 1]; last [340, 2375.7481, 0.0388, 0.0, 340]
- n35_sparse_lam7 cgbfqp: 197 entries; first [1, 6.9908, 17.5962, 0.0, 1]; last [197, 1376.2071, 0.0, 0.0, 197]
- n35_sparse_lam8 cgbfqp: 122 entries; first [1, 6.987, 17.4423, 0.0, 1]; last [122, 855.7226, 0.0, 0.0, 122]
- n35_sparse_lam9 cgbfqp: 116 entries; first [1, 6.9863, 17.326, 0.0, 1]; last [116, 811.5703, 0.0, 0.0, 116]

Device memory: for CGBFQP-GPU the lattice footprint of the layout (`device_bytes`); for BM-ADMM the pool high-water mark.
