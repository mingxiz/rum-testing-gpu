# tab:ai: LLM choice data, n = 18

The Hong-Li bootstrap test of RUM and the two-sided 95 % interval for dist on the LLM shopping data, through
`rum.bootstrap` (`solver/inference/`). The IP-CG and IP-H-CG entries of tab:ai are cluster runs; see
[`../cgip/`](../cgip/).

## The data

18 Apple Watch configurations; all 153 binary and 816 ternary menus, a 3 % sample of every cardinality 4-17,
and the grand menu: **8,807 menus**. Each menu was put to a model many times, every time in a separate API
call with its own random display order. One cell per (menu, member alternative): **M = 73,492** (3.1 % of all
cells).

| data set | `--model` | file in `final_data/Application_AI/raw_data/` | replies per menu | replies kept | unparseable, dropped |
|---|---|---|---|---|---|
| gpt-6 luna without reasoning (tab:ai: GPT-6 Luna) | `gpt6luna_noreason` | `gpt6luna_noreason.npz` | 2,000 | 17,614,000 | 0 |
| GMI Cloud (tab:ai: DeepSeek V4 Flash) | `gmicloud` | `gmicloud.npz` | 2,000 | 17,611,586 | 2,414 |
| gpt-5.6-luna | `luna` | `gpt5.6-luna.npz` | 100 | 880,700 | 0 |
| deepseek-v4-flash | `deepseek` | `deepseekv4-flash.npz` | 100 | 880,535 | 165 |

Each `.npz` is the table `replies.py` reads from the model's jsonl replies (one row per reply: the replicate,
the menu shown as a bitmask, the product picked; a reply that does not parse to a position on the menu is
dropped), saved by `replies.save_table`; `replies.load_table` reads it back. The two 100-reply sets are the
earlier data; the cluster's IP-CG and IP-H-CG runs on the AI application used them.

## Run it

    python replication/empirical_ai_n18/run_bootstrap.py --model gpt6luna_noreason --no-ci
    python replication/empirical_ai_n18/run_bootstrap.py --model gpt6luna_noreason --method flow_cg --no-ci --tag _flow_cg
    python replication/empirical_ai_n18/run_bootstrap.py --model gpt6luna_noreason --input replies --no-ci --distance-interval

and the same with `--model gmicloud`, `luna` or `deepseek`.

- **The p-value** (the first two lines): the replies as a panel, the replicate as the respondent; B = 10,000
  multinomial_reject draws, seed 0; dist(p&#770;) to abs 5e-4 (rel 0); n = the replicates (2,000 or 100), so
  scale = 4.5495 or 3.1544. BM-ADMM with beta = 5, an upper bound every 25 iterations and a lower bound every
  100 (the default `--method published`), or Flow-CG (`--method flow_cg`).
- **The two-sided interval** (the third line): `--input replies` resamples each menu's replies on their own, as
  the data were made, and `--distance-interval` reports [max(0, LB - c), UB + c], c the 95 % quantile of
  ||p&#770;<sub>b</sub> - p&#770;|| over the draws. It solves no draw, because |dist(p) - dist(p&#770;)| <= ||p - p&#770;||.
  This is the interval the paper reports.
- Without `--no-ci` a run also computes the one-sided Hong-Li interval, which has to solve about 630 draws to
  tolerance here (about 1.3 h on the 100-reply data).

A run writes `results/bootstrap_ai_<model>[_replies]<tag>.json`.

## Results

RTX 5090, B = 10,000, seed 0, abs 5e-4; `CGBFQP-GPU` is the records' name for Flow-CG.

| data | solver | dist(p&#770;) in [LB, UB] | below / above / undecided | p-value | time |
|---|---|---|---|---|---|
| gpt-6 luna, no reasoning | BM-ADMM | [13.389228, 13.389728] | 10,000 / 0 / 0 | 0 | 110 s |
| gpt-6 luna, no reasoning | CGBFQP-GPU | [13.388845, 13.389308] | 10,000 / 0 / 0 | 0 | 106 s |
| GMI Cloud | BM-ADMM | [7.220187, 7.220685] | 10,000 / 0 / 0 | 0 | 131 s |
| GMI Cloud | CGBFQP-GPU | [7.220069, 7.220562] | 10,000 / 0 / 0 | 0 | 545 s |
| gpt-5.6-luna | BM-ADMM | [13.510657, 13.511156] | 10,000 / 0 / 0 | 0 | 21.9 s |
| gpt-5.6-luna | CGBFQP-GPU | [13.510625, 13.511076] | 10,000 / 0 / 0 | 0 | 35.8 s |
| deepseek-v4-flash | BM-ADMM | [11.151396, 11.151895] | 10,000 / 0 / 0 | 0 | 21.5 s |
| deepseek-v4-flash | CGBFQP-GPU | [11.151184, 11.151677] | 10,000 / 0 / 0 | 0 | 105 s |

Every draw is decided below by the screen, so no draw is solved: p = 0 and RUM is rejected on all four data
sets. The two-sided interval (BM-ADMM, menu-by-menu draws):

| data | dist(p&#770;) in [LB, UB] | c | 95 % interval | time |
|---|---|---|---|---|
| gpt-6 luna, no reasoning | [13.389228, 13.389728] | 1.4472 | [11.942, 14.837] | 45.2 s |
| GMI Cloud | [7.220187, 7.220685] | 1.7673 | [5.453, 8.988] | 68.7 s |
| gpt-5.6-luna | [13.510657, 13.511156] | 6.9326 | [6.578, 20.444] | 42.0 s |
| deepseek-v4-flash | [11.151396, 11.151895] | 7.6143 | [3.537, 18.766] | 52.6 s |

c shrinks roughly as 1/sqrt(n), n the replies per menu, so the 100-reply sets have the widest intervals.
`final_results/Application_AI/` lists these runs with every column.

## Files

| file | what it is |
|---|---|
| `run_bootstrap.py` | the test through `rum.bootstrap`: panel or replies input, the p-value and both intervals |
| `replies.py` | jsonl replies -> the table (replicate, menu, chosen); `save_table` and `load_table` for the `.npz` form |
| `results/bootstrap_ai_<model>.json` | BM-ADMM, panel input |
| `results/bootstrap_ai_<model>_flow_cg.json` | Flow-CG, panel input |
| `results/bootstrap_ai_<model>_replies.json` | BM-ADMM, replies input, with the two-sided interval |
