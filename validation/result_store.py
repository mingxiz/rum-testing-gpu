"""Where stored results live.

Every result the paper reports is kept under replication/<paper object>/results/, next to
the README and reproduce.py of the object it belongs to. Fresh runs write to a working
directory (validation/results/...); the table scripts read the working directory first and
fall back to the stores, so a fresh clone regenerates every table from the committed results.
"""
import glob
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPLICATION = os.path.join(ROOT, "replication")
DENSITY_GRID = os.path.join(REPLICATION, "density_grid", "results")


FEASIBILITY = ("feasibility_in_n", "feasibility_in_density", "pricing_backends", "convex_path")


def stores(folders=None):
    """replication/*/results/ directories, optionally only the named folders."""
    out = sorted(glob.glob(os.path.join(REPLICATION, "*", "results")))
    if folders is not None:
        out = [d for d in out if os.path.basename(os.path.dirname(d)) in folders]
    return out


def find(name, first=()):
    """Path of a stored result file by basename: the `first` directories, then every store."""
    for d in list(first) + stores():
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


def glob_all(pattern, first=(), folders=None):
    """Every file matching `pattern` (a basename glob) across the stores, then `first`
    (listed last so a caller that keys by name lets the working directory override)."""
    out = []
    for d in stores(folders) + list(first):
        out += glob.glob(os.path.join(d, pattern))
    return out
