"""State A: what sets off the jumps of u = x - integral(x')? Diagnostic only, nothing fitted.

Levels and flips as in check_telegraph (u median-filtered, two levels, flips that last > MIN_DWELL).
Hazard = flips per ms while in a level, binned by the circuit state just before the flip (PRE samples):
x (Ch1), true x' (dx/dt from Ch1, smoothed) and Ch2. A threshold shows up as a hazard that jumps from ~0 to large
at one value of one variable. Printed as tables, one row per variable, for up flips (from low) and down flips.
"""

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter

import fig_diode_inferred as fd
import sprott_common as sc
from check_telegraph import MED, MIN_DWELL, PRE, levels
from fig_diode_inferred import RUNS
from fig_x_minus_int import u_of

BINS = np.arange(-1.6, 1.01, 0.2)
RUN = "A (D1, 13:46)"


def collect(name):
    r = sc.load(sc.DATA / RUNS[name])
    n_cap, n = r["x"].shape
    h = r["dt"] / sc.TAU
    us = np.array([median_filter(u_of(r, c)[2], MED) for c in range(n_cap)])
    lo, hi = levels(us.ravel())
    state = (us > 0.5 * (lo + hi)).astype(int)
    var = {
        "x": savgol_filter(r["x"], fd.W, fd.PO, axis=1),
        "true x'": savgol_filter(r["x"], fd.W, fd.PO, deriv=1, delta=h, axis=1),
        "Ch2 x'": savgol_filter(r["xd"], fd.W, fd.PO, axis=1),
    }
    flips = {0: [], 1: []}  # keyed by the level they leave
    keep = np.zeros_like(state, bool)
    keep[:, fd.W3 : -fd.W3] = True
    for c in range(n_cap):
        e_all = np.flatnonzero(np.diff(state[c]) != 0) + 1
        for i, e in enumerate(e_all):
            nxt = e_all[i + 1] if i + 1 < len(e_all) else n
            if e - PRE < fd.W3 or nxt - e < MIN_DWELL or e > n - fd.W3:
                continue
            flips[state[c, e - 1]].append((c, e - PRE))
    return r["dt"], state, var, flips, keep


def main():
    dt, state, var, flips, keep = collect(RUN)
    c = 0.5 * (BINS[1:] + BINS[:-1])
    print(
        f"{RUN}; hazard in flips per ms, bins (V): " + " ".join(f"{q:+5.1f}" for q in c)
    )
    for lv, lab in ((0, "up (leaving low)"), (1, "down (leaving high)")):
        idx = np.array(flips[lv])
        print(f"--- {lab}: {len(idx)} flips")
        for k, v in var.items():
            occ = (
                np.histogram(v[(state == lv) & keep], BINS)[0] * dt * 1e3
            )  # ms spent in each bin
            hit = np.histogram(v[idx[:, 0], idx[:, 1]], BINS)[0]
            hz = np.where(occ > 20, hit / np.maximum(occ, 1e-9), np.nan)
            print(
                f"{k:8s} "
                + " ".join(f"{q:5.2f}" if np.isfinite(q) else "   . " for q in hz)
            )
    # 2-D: up flips in (x, true x') to see if it is a line crossing
    idx = np.array(flips[0])
    xb, yb = np.arange(-1.4, 0.61, 0.2), np.arange(-0.8, 0.81, 0.2)
    occ = (
        np.histogram2d(
            var["x"][(state == 0) & keep], var["true x'"][(state == 0) & keep], [xb, yb]
        )[0]
        * dt
        * 1e3
    )
    hit = np.histogram2d(
        var["x"][idx[:, 0], idx[:, 1]], var["true x'"][idx[:, 0], idx[:, 1]], [xb, yb]
    )[0]
    hz = np.where(occ > 10, hit / np.maximum(occ, 1e-9), np.nan)
    print(
        "up-flip hazard (per ms), rows x, cols true x' "
        + " ".join(f"{q:+.1f}" for q in 0.5 * (yb[1:] + yb[:-1]))
    )
    for i, xc in enumerate(0.5 * (xb[1:] + xb[:-1])):
        print(
            f"x {xc:+.1f}  "
            + " ".join(f"{q:4.2f}" if np.isfinite(q) else "  . " for q in hz[i])
        )


if __name__ == "__main__":
    main()
