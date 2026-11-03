"""The AI choice data as the bootstrap's panel: one row per API reply.

    from replies import read_replies, load_table, save_table
    panel, dropped = read_replies("replies.jsonl")       # one API reply per line
    save_table("table.npz", panel, dropped)             # the same table, compact
    panel, dropped = load_table("table.npz")            # a .jsonl or such an .npz

Each line of the jsonl is one reply, {"j", "id", "reply", ...}. The id carries the display order
and the replicate: "09v06_r007" is product 9 shown first and product 6 second, replicate 7. The
rules are those of the paper's runs: respondent = the replicate; menu = the
set shown, as a bitmask; chosen = the product at the reply's position (1-based). A reply that does
not parse to 1..size is dropped and counted, and a repeated job j keeps its first reply.
"""
import json

import numpy as np
import pandas as pd


def read_replies(path):
    """(panel, number of dropped replies): the panel has the columns respondent, menu and chosen."""
    respondent, menu, chosen, dropped, jobs = [], [], [], 0, set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["j"] in jobs:
                continue
            jobs.add(r["j"])
            order_txt, rep_txt = r["id"].split("_r")
            order = [int(t) for t in order_txt.split("v")]
            try:
                k = int(str(r["reply"]).strip())
            except ValueError:
                dropped += 1
                continue
            if not 1 <= k <= len(order):
                dropped += 1
                continue
            respondent.append(int(rep_txt))
            menu.append(sum(1 << i for i in sorted(order)))
            chosen.append(order[k - 1])
    panel = pd.DataFrame({"respondent": np.array(respondent, dtype=np.int64),
                          "menu": np.array(menu, dtype=np.int64), "chosen": np.array(chosen, dtype=np.int64)})
    return panel, dropped


def save_table(path, table, dropped):
    """The table as a compressed .npz (replicate int16, menu int32, chosen int8, in row order) with the
    number of dropped replies: the 2.1 GB gpt-6 luna file becomes a small file with the same rows."""
    np.savez_compressed(path, respondent=table["respondent"].to_numpy().astype(np.int16),
                        menu=table["menu"].to_numpy().astype(np.int32),
                        chosen=table["chosen"].to_numpy().astype(np.int8), dropped=np.int64(dropped))


def load_table(path):
    """(table, number of dropped replies) from a .jsonl of replies or an .npz written by save_table."""
    if not path.endswith(".npz"):
        return read_replies(path)
    z = np.load(path)
    table = pd.DataFrame({c: z[c].astype(np.int64) for c in ("respondent", "menu", "chosen")})
    return table, int(z["dropped"])
