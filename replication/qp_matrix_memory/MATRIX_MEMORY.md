# Memory to hand the RUM projection to a QP solver as explicit sparse matrices

Measured with scipy CSR (float64 values, int32 indices: 12 B per nonzero plus row pointers); the closed forms are checked against the measured nonzero counts and used above the largest n built here. A QP solver holds at least one such copy, typically two or three (presolved model, factorization workspace), so these are floors. This machine has 64 GB of host memory.

## Block-Marschak form: K p~ >= 0 (nnz = n 3^(n-1)), S p~ = 1 (nnz = n 2^(n-1))

| n | m2 = n 2^(n-1) | nnz(K) measured | nnz(K) = n 3^(n-1) | K bytes (CSR, int32) | K bytes (int64 idx) | S bytes | build s |
|---|---|---|---|---|---|---|---|
| 8 | 1,024 | 17,496 | 17,496 | 209.0 KB | 281.4 KB | 13.0 KB | 0.0 |
| 9 | 2,304 | 59,049 | 59,049 | 701.0 KB | 940.6 KB | 29.0 KB | 0.0 |
| 10 | 5,120 | 196,830 | 196,830 | 2.3 MB | 3.0 MB | 64.0 KB | 0.0 |
| 11 | 11,264 | 649,539 | 649,539 | 7.5 MB | 10.0 MB | 140.0 KB | 0.1 |
| 12 | 24,576 | 2,125,764 | 2,125,764 | 24.4 MB | 32.6 MB | 304.0 KB | 0.2 |
| 13 | 53,248 | 6,908,733 | 6,908,733 | 79.3 MB | 105.8 MB | 656.0 KB | 0.4 |
| 14 | 114,688 | 22,320,522 | 22,320,522 | 255.9 MB | 341.5 MB | 1.4 MB | 1.1 |
| 15 | 245,760 | 71,744,535 | 71,744,535 | 822.0 MB | 1.1 GB | 2.9 MB | 3.1 |
| 16 | 524,288 | 229,582,512 | 229,582,512 | 2.6 GB | 3.4 GB | 6.3 MB | 11.0 |
| 17 | 1,114,112 | (formula) | 731,794,257 | 8.2 GB | 10.9 GB | 13.3 MB | - |
| 18 | 2,359,296 | (formula) | 2,324,522,934 | 26.0 GB | 34.7 GB | 28.0 MB | - |
| 19 | 4,980,736 | (formula) | 7,360,989,291 | 82.3 GB | 109.7 GB | 59.0 MB | - |
| 20 | 10,485,760 | (formula) | 23,245,229,340 | 259.8 GB | 346.5 GB | 124.0 MB | - |

## Arc-flow form: A r = b (nnz = n 2^n), 2^n rows, n 2^(n-1) columns

| n | rows 2^n | cols m2 | nnz(A) measured | nnz = n 2^n | A bytes (CSR, int32) | build s |
|---|---|---|---|---|---|---|
| 8 | 256 | 1,024 | 2,048 | 2,048 | 25.0 KB | 0.0 |
| 9 | 512 | 2,304 | 4,608 | 4,608 | 56.0 KB | 0.0 |
| 10 | 1,024 | 5,120 | 10,240 | 10,240 | 124.0 KB | 0.0 |
| 11 | 2,048 | 11,264 | 22,528 | 22,528 | 272.0 KB | 0.0 |
| 12 | 4,096 | 24,576 | 49,152 | 49,152 | 592.0 KB | 0.0 |
| 13 | 8,192 | 53,248 | 106,496 | 106,496 | 1.3 MB | 0.0 |
| 14 | 16,384 | 114,688 | 229,376 | 229,376 | 2.7 MB | 0.0 |
| 15 | 32,768 | 245,760 | 491,520 | 491,520 | 5.8 MB | 0.0 |
| 16 | 65,536 | 524,288 | 1,048,576 | 1,048,576 | 12.3 MB | 0.0 |
| 17 | 131,072 | 1,114,112 | 2,228,224 | 2,228,224 | 26.0 MB | 0.0 |
| 18 | 262,144 | 2,359,296 | 4,718,592 | 4,718,592 | 55.0 MB | 0.1 |
| 19 | 524,288 | 4,980,736 | 9,961,472 | 9,961,472 | 116.0 MB | 0.2 |
| 20 | 1,048,576 | 10,485,760 | 20,971,520 | 20,971,520 | 244.0 MB | 0.5 |
