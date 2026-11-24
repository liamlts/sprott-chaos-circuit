"""The two-level jumps of u = x - integral(x') in state A (fig_x_minus_int). Diagnostic only, nothing fitted.

Per run: u per capture (raw, line removed as in fig_x_minus_int), median-filtered, split into two levels by the
midpoint of its two histogram peaks (k-means with 2 centres). Flips = level changes that last > MIN_DWELL.
Printed: step size, flips per second, dwell times per level, and Ch2 (x') and x just before each up / down flip.
Circuit or probe? D_off = -(x''' + A x'' + x) from Ch1 where the diode is off (Ch2 < 0.25 V), mean per level.
If only the Ch1 reading jumps, D_off differs by the step between levels; if the circuit's x jumps, it does not.
"""

import numpy as np
from scipy.ndimage import median_filter

import fig_diode_inferred as fd
import sprott_common as sc
from fig_diode_inferred import RUNS
from fig_x_minus_int import u_of

MED = 25  # samples (1 ms) median filter on u
MIN_DWELL = 25  # samples
PRE = 5  # samples before the flip where x, x' are read


def levels(u):
    c = np.percentile(u, [20, 80])
    for _ in range(30):
        lab = np.abs(u[:, None] - c[None, :]).argmin(1)
        c = np.array(
            [u[lab == i].mean() if (lab == i).any() else c[i] for i in range(2)]
        )
    return np.sort(c)


def run(name):
    r = sc.load(sc.DATA / RUNS[name])
    n_cap, n = r["x"].shape
    us = np.array([median_filter(u_of(r, c)[2], MED) for c in range(n_cap)])
    lo, hi = levels(us.ravel())
    mid = 0.5 * (lo + hi)
    _, xs, _, D = fd.terms(r["x"], r["xd"], r["dt"])
    s = np.s_[:, fd.W3 : -fd.W3]
    state = (us > mid).astype(int)
    st = state[s].ravel()
    y = fd.savgol_filter(r["xd"], fd.W, fd.PO, axis=1)[s].ravel()
    off = y < fd.OFF
    up, dn, dwell = [], [], {0: [], 1: []}
    for c in range(n_cap):
        sc_ = state[c]
        edges = np.flatnonzero(np.diff(sc_) != 0) + 1
        seg = np.split(np.arange(n), edges)
        for k, g in enumerate(seg):
            if 0 < k < len(seg) - 1:
                dwell[sc_[g[0]]].append(g.size * r["dt"] * 1e3)
        for e in edges:
            if (
                e - PRE < 0
                or (np.diff(np.r_[e, edges[edges > e]])[:1].tolist() or [n])[0]
                < MIN_DWELL
            ):
                continue
            jump = np.median(us[c, e : e + MIN_DWELL]) - np.median(
                us[c, max(e - MIN_DWELL, 0) : e]
            )
            (up if sc_[e] == 1 else dn).append(
                (r["xd"][c, e - PRE], r["x"][c, e - PRE], jump)
            )
    up, dn = np.array(up), np.array(dn)
    T = n_cap * n * r["dt"]
    print(
        f"{name}: step {(hi - lo) * 1e3:.0f} mV (levels {lo * 1e3:+.0f}, {hi * 1e3:+.0f}), "
        f"time at high level {state.mean():.2f}, flips {(len(up) + len(dn)) / T:.0f} /s; per-flip jump median "
        f"up {np.median(up[:, 2]) * 1e3:+.0f} mV, down {np.median(dn[:, 2]) * 1e3:+.0f} mV"
    )
    for lv in (0, 1):
        print(
            f"   dwell at {'low ' if lv == 0 else 'high'}: median {np.median(dwell[lv]):.1f} ms, "
            f"10-90 % {np.percentile(dwell[lv], 10):.1f}-{np.percentile(dwell[lv], 90):.1f} ms (n {len(dwell[lv])})"
        )
    for lab, a in (("up  ", up), ("down", dn)):
        if len(a):
            print(
                f"   before {lab} flips (n {len(a)}): x' median {np.median(a[:, 0]):+.2f} V (10-90 % {np.percentile(a[:, 0], 10):+.2f} "
                f"to {np.percentile(a[:, 0], 90):+.2f}), x median {np.median(a[:, 1]):+.2f} V "
                f"({np.percentile(a[:, 1], 10):+.2f} to {np.percentile(a[:, 1], 90):+.2f})"
            )
    for lv in (0, 1):
        m = off & (st == lv)
        print(
            f"   D_off at {'low ' if lv == 0 else 'high'} level: {D[m].mean() * 1e3:+.0f} mV (n {m.sum()})"
        )


def main():
    for k in (
        "A (D1, 13:46)",
        "A (D1, rep1)",
        "A (1N4004)",
        "A (1N4148 #2)",
        "Ge (0.2 V/div)",
        "B (D1 reseated)",
        "C (reseated again)",
    ):
        run(k)


if __name__ == "__main__":
    main()
