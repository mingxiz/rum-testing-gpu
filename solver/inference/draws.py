"""The choices behind p_hat, and the bootstrap draws made from them.

The choices come in one of two tables, one row per choice (a pandas DataFrame or a dict of
equal-length columns; other columns are ignored and row order does not matter):

    panel     DSR style: respondent, menu, chosen. People who each answered several menus;
              respondents are resampled, so a respondent's answers move together (Draws).
    replies   AI style: menu, chosen. Every reply was collected on its own, as when an LLM is
              asked each menu many times; each menu's replies are resampled on their own (MenuDraws).

    respondent   who chose: an integer or string id. Rows with the same id are one unit and are
                 resampled together; the sample size n is the number of distinct ids.
    menu         the menu offered, an integer bitmask (bit a set = alternative a offered): the
                 encoding of RUM(menu, ...).
    chosen       the alternative picked, an integer from 0, a member of its menu.

A respondent may answer a menu more than once (each row counts). rum's rows must be counted from the
table: for a row (D, a), p_hat = #{rows with menu D and chosen a} / #{rows with menu D}.
cells_from_panel builds rum's rows that way, from either table:

    panel = pd.read_csv("final_data/Application_DSR/raw_data/overload_data_25_long.csv")
    menu, alt, p_hat = cells_from_panel(panel, 25, alternatives=[0])     # DSR alt0: M = 301
    rum = RUM(menu, alt, p_hat, num_alternatives=25)

The panel's draws. Draw b gives respondent i a weight w_bi and counts again,

    p_hat_b(D, a) = sum_i w_bi #{rows of i: menu D, chosen a} / sum_i w_bi #{rows of i: menu D},

so a respondent's answers move together. The weights are drawn as in the paper's runs, seed
for seed:

    multinomial_reject   how often i is picked in R draws with replacement; a resample that leaves a
                         menu of rum unseen is drawn again (the default)
    multinomial          the same without the redraw; a menu left unseen keeps p_hat
    exponential, poisson multiplier weights, Exp(1) and Poisson(1)
    mammen               the wild bootstrap, p_hat_b = clip(p_hat + sum_i w_bi (chose_i - offered_i
                         p_hat) / N_D, 0, 1), N_D the rows with menu D

The replies' draws. Draw b draws N_D replies with replacement from menu D's N_D replies, for every
menu on its own, and counts again: menu D's counts are Multinomial(N_D, the shares of D's replies).
"""
import hashlib

import numpy as np
import scipy.sparse as sp

WEIGHTS = ("multinomial_reject", "multinomial", "exponential", "poisson", "mammen")
COLUMNS = ("respondent", "menu", "chosen")
P_HAT_TOL = 1e-9                # rum's p_hat against the table's counts
BLOCK_BYTES = 1 << 27           # draws are built at most 128 MB at a time
DENSE_ENTRIES = 1 << 25         # respondent-by-row counts stay dense up to 256 MB
REDRAW_ROUNDS = 10000           # multinomial_reject gives up after this many rounds (as the drivers)


def read_panel(panel, num_alternatives=None):
    """The panel's columns, checked: (respondent codes 0..R-1 in sorted id order, R, menu, chosen)."""
    resp, menu, chosen = _read_choices(panel, "panel", num_alternatives, respondent=True)
    if resp.dtype.kind == "f" and not np.isfinite(resp).all():
        raise ValueError("panel row %d: missing respondent" % np.flatnonzero(~np.isfinite(resp))[0])
    if resp.dtype.kind == "O":
        bad = [i for i, r in enumerate(resp) if r is None or r != r]
        if bad:
            raise ValueError("panel row %d: missing respondent" % bad[0])
    try:
        ids, codes = np.unique(resp, return_inverse=True)
    except TypeError:
        raise ValueError("respondent ids must be all numbers or all strings") from None
    return codes.astype(np.int64), int(ids.size), menu, chosen


def _read_choices(table, what, num_alternatives=None, respondent=False):
    """The table's menu and chosen columns, checked, after its respondent column if asked for;
    what names the table in the errors."""
    names = COLUMNS if respondent else COLUMNS[1:]
    missing = [c for c in names if c not in table]
    if missing:
        raise ValueError("the %s needs the columns %s and %s; missing: %s"
                         % (what, ", ".join(names[:-1]), names[-1], ", ".join(missing)))
    cols = [np.asarray(table[c]) for c in names]
    if not (all(c.ndim == 1 for c in cols) and len({c.size for c in cols}) == 1):
        raise ValueError("the %s's columns must be 1-D and of equal length" % what)
    if cols[0].size == 0:
        raise ValueError("the %s has no rows" % what)
    menu, chosen = _int_column(cols[-2], "menu", what), _int_column(cols[-1], "chosen", what)
    bad = np.flatnonzero((menu <= 0) | (menu >= 1 << 62))
    if bad.size:
        raise ValueError("%s row %d: menu mask %d is not a nonempty set of at most 62 alternatives"
                         % (what, bad[0], menu[bad[0]]))
    bad = np.flatnonzero((chosen < 0) | (chosen > 61))
    if bad.size:
        raise ValueError("%s row %d: chosen %d is not an alternative (0 to 61)" % (what, bad[0], chosen[bad[0]]))
    bad = np.flatnonzero(((menu >> chosen) & 1) == 0)
    if bad.size:
        raise ValueError("%s row %d: chosen %d is not in its menu (mask %d)"
                         % (what, bad[0], chosen[bad[0]], menu[bad[0]]))
    if num_alternatives is not None:
        bad = np.flatnonzero(menu >= 1 << int(num_alternatives))
        if bad.size:
            raise ValueError("%s row %d: menu mask %d has alternatives beyond num_alternatives = %d"
                             % (what, bad[0], menu[bad[0]], num_alternatives))
    return (cols[0], menu, chosen) if respondent else (menu, chosen)


def cells_from_panel(table, num_alternatives=None, alternatives=None):
    """(menu, alternative, p_hat) counted from the panel or the replies, for RUM(menu, alternative,
    p_hat, ...); a table with a respondent column is checked as a panel.

    One row per menu of the table and member alternative, or only those in alternatives (DSR
    alt0: [0]); menus in increasing mask order and alternatives increasing within a menu, as the
    published drivers built them. p_hat is the share of the menu's rows that chose the
    alternative; one offered but never chosen is a row with p_hat = 0."""
    if COLUMNS[0] in table:
        _codes, _R, menu, chosen = read_panel(table, num_alternatives)
    else:
        menu, chosen = _read_choices(table, "replies table", num_alternatives)
    n = (int(num_alternatives) if num_alternatives is not None
         else max(int(menu.max()).bit_length(), int(chosen.max()) + 1))
    menus, inv, den = np.unique(menu, return_inverse=True, return_counts=True)
    alts = np.arange(n) if alternatives is None else np.unique(np.asarray(alternatives, dtype=np.int64))
    if alts.size == 0 or alts[0] < 0 or alts[-1] >= n:
        raise ValueError("alternatives must be a nonempty list of alternatives from 0 to %d" % (n - 1))
    j, k = np.nonzero((menus[:, None] >> alts[None, :]) & 1)       # menu-major, alternatives increasing
    if j.size == 0:
        raise ValueError("no menu of the table offers any of the alternatives %s" % alts.tolist())
    num = np.bincount(inv * n + chosen, minlength=menus.size * n)
    return menus[j], alts[k], num[j * n + alts[k]] / den[j]


class _Blocks:
    """B draws kept as what rebuilds them, a block at a time: block(k) is draws [k c, (k + 1) c)
    for c = rows_per_block. A subclass's _build(k) returns (the block, how many of its draws left a
    menu unseen). A full pass of blocks() also sets fingerprint, the first 16 hex digits of the
    sha256 of the B x M float64 draws in row order (the drivers' draws_fingerprint)."""

    def block(self, k):
        """Draws [k c, (k + 1) c) as a (rows, M) array (the last block built is kept)."""
        if self._cache[0] != k:
            self._cache = (k,) + self._build(k)
        return self._cache[1]

    def blocks(self):
        """(first, last + 1, draws) for every block in order; a full pass sets fingerprint and n_empty."""
        h, n_empty = hashlib.sha256(), 0
        for k in range(self.n_blocks):
            out = self.block(k)
            n_empty += self._cache[2]
            h.update(np.ascontiguousarray(out).tobytes())
            yield k * self.rows_per_block, k * self.rows_per_block + out.shape[0], out
        self.fingerprint, self.n_empty = h.hexdigest()[:16], n_empty

    def rows(self, idx):
        """The draws idx, as a (len(idx), M) array."""
        idx = np.asarray(idx, dtype=np.int64)
        out = np.empty((idx.size, self.M))
        for k in np.unique(idx // self.rows_per_block):
            sel = idx // self.rows_per_block == k
            out[sel] = self.block(int(k))[idx[sel] - k * self.rows_per_block]
        return out


class Draws(_Blocks):
    """B draws of p_hat for rum's rows (menu, alternative, p_hat), made from the panel.

    The weights W (B x R) are drawn once, in the drivers' RNG order. The draws are rebuilt from W
    a block at a time (BLOCK_BYTES), so the B x M matrix is never held. With integer weights
    (multinomial, multinomial_reject, poisson) every count is exact, so the draws do not depend on
    the block size, and a row's weighted count of views is its menu's: one product per menu, not
    per row (the AI panel with 2,000 replicates: 8,807 menus against 73,492 rows)."""

    input = "panel"

    def __init__(self, panel, menu, alternative, p_hat, B=10000, weights="multinomial_reject", seed=0):
        if weights not in WEIGHTS:
            raise ValueError("unknown weights %r; choose from %s" % (weights, WEIGHTS))
        if int(B) != B or B < 1:
            raise ValueError("B must be a positive integer")
        codes, R, pm, pc = read_panel(panel)
        self.input_fingerprint = hashlib.sha256(codes.tobytes() + pm.tobytes() + pc.tobytes()).hexdigest()[:16]
        menu, alternative = np.asarray(menu, dtype=np.int64), np.asarray(alternative, dtype=np.int64)
        p_hat = np.asarray(p_hat, dtype=np.float64)
        M = menu.size
        # panel rows -> rum's menus, and -> rum's rows by (menu, chosen)
        rum_menus, row_menu = np.unique(menu, return_inverse=True)
        m = np.minimum(np.searchsorted(rum_menus, pm), rum_menus.size - 1)
        used = rum_menus[m] == pm
        row_key = row_menu * 64 + alternative
        order = np.argsort(row_key)
        key = m * 64 + pc
        pos = np.minimum(np.searchsorted(row_key[order], key), M - 1)
        hit = used & (row_key[order][pos] == key)
        offered = sp.csr_array((np.ones(int(used.sum())), (codes[used], m[used])), shape=(R, rum_menus.size))
        chose = sp.csr_array((np.ones(int(hit.sum())), (codes[hit], order[pos[hit]])), shape=(R, M))

        N = np.asarray(offered.sum(axis=0)).ravel()[row_menu]
        if (N == 0).any():
            k = int(np.flatnonzero(N == 0)[0])
            raise ValueError("rum row %d (menu %d): the menu is not in the panel" % (k, menu[k]))
        c = np.asarray(chose.sum(axis=0)).ravel()
        gap = np.abs(c / N - p_hat)
        if gap.max() > P_HAT_TOL:
            k = int(np.argmax(gap))
            raise ValueError("rum row %d (menu %d, alternative %d): p_hat = %.6g, but the panel gives %.6g (%d of %d "
                             "rows); build rum with cells_from_panel(panel, ...)"
                             % (k, menu[k], alternative[k], p_hat[k], c[k] / N[k], c[k], N[k]))

        dense = R * M <= DENSE_ENTRIES
        self._offered = offered.toarray() if R * rum_menus.size <= DENSE_ENTRIES else offered
        self._row_menu, self._seen = row_menu, None
        if weights in ("mammen", "exponential"):        # real weights: per-row counts, summed as the drivers did
            seen = offered @ sp.csr_array((np.ones(M), (row_menu, np.arange(M))), shape=(rum_menus.size, M))
        if weights == "mammen":
            resid = (chose - seen.multiply(p_hat[None, :])).tocsr()
            self._resid, self._N = (resid.toarray() if dense else resid), np.maximum(N, 1.0)
        else:
            self._chose = chose.toarray() if dense else chose
            if weights == "exponential":
                self._seen = seen.toarray() if dense else seen
        self.B, self.R, self.M, self.weights, self.seed = int(B), R, M, weights, seed
        self.p_hat = p_hat
        self.rows_unused = int((~used).sum())           # panel rows on menus that have no rum row
        self.rows_per_block = max(1, min(self.B, BLOCK_BYTES // (8 * M)))
        self.n_blocks = -(-self.B // self.rows_per_block)
        self.n_redraws = self.n_reject_rounds = 0
        self.n_empty = None                             # draws that left a menu unseen (after a full pass)
        self.fingerprint = None
        self._cache = (None, None, 0)
        self.W = self._weights(np.random.default_rng(seed))

    @property
    def default_n(self):
        """n when none is given: the number of respondents."""
        return self.R

    def check_n(self, n):
        if self.R != n:
            raise ValueError("n = %d, but the panel has %d respondents" % (n, self.R))

    def summary(self):
        """The record's account of the panel and its draws."""
        return dict(respondents=self.R, rows_unused=self.rows_unused, redraws=self.n_redraws,
                    emptied_menu_draws=self.n_empty)

    @property
    def describe(self):
        return "%s draws over respondents" % self.weights

    def _weights(self, rng):
        B, R = self.B, self.R
        if self.weights == "mammen":
            s5 = np.sqrt(5.0)
            return np.where(rng.random((B, R)) < (s5 + 1) / (2 * s5), -(s5 - 1) / 2, (s5 + 1) / 2)
        if self.weights == "exponential":
            return rng.exponential(1.0, size=(B, R))
        if self.weights == "poisson":
            return rng.poisson(1.0, size=(B, R)).astype(np.float64)
        W = rng.multinomial(R, np.ones(R) / R, size=B).astype(np.float64)
        if self.weights == "multinomial_reject":
            for _ in range(REDRAW_ROUNDS):
                bad = self._unseen(W)
                if not bad.any():
                    break
                self.n_redraws += int(bad.sum())
                self.n_reject_rounds += 1
                W[bad] = rng.multinomial(R, np.ones(R) / R, size=int(bad.sum())).astype(np.float64)
        return W

    def _unseen(self, W):
        """Which weight rows leave some menu of rum with no weight."""
        step = max(1, BLOCK_BYTES // (8 * self._offered.shape[1]))
        return np.concatenate([np.any(W[lo:lo + step] @ self._offered <= 0, axis=1)
                               for lo in range(0, W.shape[0], step)])

    def _build(self, k):
        """(draws of block k, how many of them left a menu unseen)."""
        Wb = self.W[k * self.rows_per_block:(k + 1) * self.rows_per_block]
        if self.weights == "mammen":
            return np.clip(self.p_hat + np.asarray(Wb @ self._resid) / self._N, 0.0, 1.0), 0
        seen = (np.asarray(Wb @ self._offered)[:, self._row_menu] if self._seen is None
                else np.asarray(Wb @ self._seen))
        chose = np.asarray(Wb @ self._chose)
        empty = seen <= 0
        # the drivers divide by max(seen, 1), the same for integer weights; with real weights a
        # menu's total weight can fall below 1, and the weighted share divides by the weight itself
        return (np.where(empty, self.p_hat, chose / np.where(empty, 1.0, seen)),
                int(np.any(empty, axis=1).sum()))

    def __repr__(self):
        return ("Draws(B=%d, %s, seed %s: %d respondents, %d rows, %d per block%s)"
                % (self.B, self.weights, self.seed, self.R, self.M, self.rows_per_block,
                   "" if self.fingerprint is None else ", fingerprint " + self.fingerprint))


class MenuDraws(_Blocks):
    """B draws of p_hat for rum's rows (menu, alternative, p_hat), made from the replies: each
    menu's replies resampled on their own.

    Draw b draws N_D replies with replacement from menu D's N_D replies, for every menu on its
    own: menu D's counts are Multinomial(N_D, the shares of D's replies over every alternative
    chosen there), independent across menus and draws. Block k (rows_per_block draws, set by M) is
    drawn from numpy.random.default_rng([seed, k]), menus in increasing mask order, so any block
    can be rebuilt alone; the draws are fixed by the replies, rum's rows, B and the seed."""

    input, weights = "replies", "multinomial"

    def __init__(self, replies, menu, alternative, p_hat, B=10000, seed=0):
        if int(B) != B or B < 1:
            raise ValueError("B must be a positive integer")
        pm, pc = _read_choices(replies, "replies table")
        self.input_fingerprint = hashlib.sha256(pm.tobytes() + pc.tobytes()).hexdigest()[:16]
        menu, alternative = np.asarray(menu, dtype=np.int64), np.asarray(alternative, dtype=np.int64)
        p_hat = np.asarray(p_hat, dtype=np.float64)
        M = menu.size
        rum_menus, row_menu = np.unique(menu, return_inverse=True)
        m = np.minimum(np.searchsorted(rum_menus, pm), rum_menus.size - 1)
        used = rum_menus[m] == pm
        N = np.bincount(m[used], minlength=rum_menus.size)
        if (N == 0).any():
            k = int(np.flatnonzero(N[row_menu] == 0)[0])
            raise ValueError("rum row %d (menu %d): the menu is not in the replies" % (k, menu[k]))
        # every (menu, alternative) chosen at least once, menu-major, and how often
        support, count = np.unique(m[used] * 64 + pc[used], return_counts=True)
        row_key = row_menu * 64 + alternative
        pos = np.minimum(np.searchsorted(support, row_key), support.size - 1)
        hit = support[pos] == row_key
        c, den = np.where(hit, count[pos], 0), N[row_menu]
        gap = np.abs(c / den - p_hat)
        if gap.max() > P_HAT_TOL:
            k = int(np.argmax(gap))
            raise ValueError("rum row %d (menu %d, alternative %d): p_hat = %.6g, but the replies give %.6g (%d of "
                             "%d rows); build rum with cells_from_panel(replies, ...)"
                             % (k, menu[k], alternative[k], p_hat[k], c[k] / den[k], c[k], den[k]))
        self.N = N                                      # replies per menu of rum, in increasing mask order
        self._shares = count / N[support // 64]
        self._bounds = np.r_[np.searchsorted(support // 64, np.arange(N.size)), support.size]
        self._col = np.where(hit, pos, -1)              # each row's entry of the support; -1: never chosen
        self._den = den.astype(np.float64)
        self.B, self.M, self.seed, self.p_hat = int(B), M, seed, p_hat
        self.rows_unused = int((~used).sum())           # replies on menus that have no rum row
        self.rows_per_block = max(1, min(self.B, BLOCK_BYTES // (8 * max(M, support.size))))
        self.n_blocks = -(-self.B // self.rows_per_block)
        self.n_redraws = 0
        self.n_empty = None
        self.fingerprint = None
        self._cache = (None, None, 0)

    @property
    def default_n(self):
        """n when none is given: the most replies on any menu (every menu's, when all have as many)."""
        return int(self.N.max())

    def check_n(self, n):
        if not self.N.min() <= n <= self.N.max():
            raise ValueError("n = %d, but the menus have %d to %d replies; with replies, n is the number of replies "
                             "per menu" % (n, self.N.min(), self.N.max()))

    def summary(self):
        """The record's account of the replies and their draws."""
        return dict(replies=int(self.N.sum()), replies_per_menu=[int(self.N.min()), int(self.N.max())],
                    rows_unused=self.rows_unused)

    @property
    def describe(self):
        return "draws menu by menu"

    def _build(self, k):
        """(draws of block k, 0): no menu is ever left unseen."""
        rows = min(self.rows_per_block, self.B - k * self.rows_per_block)
        rng = np.random.default_rng([self.seed, k])
        counts = np.empty((rows, self._shares.size))
        for j in range(self.N.size):
            lo, hi = self._bounds[j], self._bounds[j + 1]
            counts[:, lo:hi] = rng.multinomial(self.N[j], self._shares[lo:hi], size=rows)
        return np.where(self._col >= 0, counts[:, np.maximum(self._col, 0)], 0.0) / self._den, 0

    def __repr__(self):
        return ("MenuDraws(B=%d, seed %s: %d replies on %d menus, %d to %d each; %d rows, %d per block%s)"
                % (self.B, self.seed, self.N.sum(), self.N.size, self.N.min(), self.N.max(), self.M,
                   self.rows_per_block, "" if self.fingerprint is None else ", fingerprint " + self.fingerprint))


def _int_column(x, column, what):
    if x.dtype.kind == "f":
        if not np.isfinite(x).all():
            raise ValueError("%s row %d: missing %s" % (what, np.flatnonzero(~np.isfinite(x))[0], column))
        if not (x == np.floor(x)).all():
            raise ValueError("%s row %d: %s %r is not an integer" % (what, np.flatnonzero(x != np.floor(x))[0], column,
                                                                     x[np.flatnonzero(x != np.floor(x))[0]]))
    elif x.dtype.kind not in "iu":
        raise ValueError("the %s's %s column must hold integers" % (what, column))
    return x.astype(np.int64)
