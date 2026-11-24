"""Is Ch2 (x') really the time derivative of Ch1 (x) in every run? Diagnostic only, model untouched.

Per run, with the same Savitzky-Golay derivatives as fig_diode_inferred:
  k_ols   slope of dx/dt (tau units) on x'      (integrator 3: should be 1 / G2-ish, ~1)
  k_rev   1 / slope of x' on dx/dt              (equal to k_ols if noise dilution is negligible)
  r       correlation of the two
  lag     shift of x' (ms) that maximises the correlation with dx/dt
  |H|     cross-spectral gain dx/dt / x' at 60-250 Hz (median over 5 bands)
  a, b, c diagnostic fit where the diode is off (x' < OFF): x''' = -a x'' - b x + c. Expect a ~ 0.98, b ~ 1.
"""

import numpy as np
from scipy.signal import csd, savgol_filter, welch

import sprott_common as sc
from fig_diode_inferred import OFF, PO, RUNS, W

BANDS = [(60, 90), (90, 120), (120, 160), (160, 200), (200, 250)]  # Hz
MAXLAG = 40  # samples (1.6 ms)


def derivs(x, xd, dt):
    h = dt / sc.TAU
    y = savgol_filter(xd, W, PO, axis=1)
    y1 = savgol_filter(xd, W, PO, deriv=1, delta=h, axis=1)
    y2 = savgol_filter(xd, W, PO, deriv=2, delta=h, axis=1)
    xs = savgol_filter(x, W, PO, axis=1)
    dxdt = savgol_filter(x, W, PO, deriv=1, delta=h, axis=1)
    return y, y1, y2, xs, dxdt


def best_lag(y, dxdt):
    s = slice(MAXLAG, -MAXLAG)
    cc = [
        np.corrcoef(np.roll(y, L, axis=1)[:, s].ravel(), dxdt[:, s].ravel())[0, 1]
        for L in range(-MAXLAG, MAXLAG + 1)
    ]
    return np.argmax(cc) - MAXLAG, max(cc)


def xfer(y, dxdt, dt):
    f, pxy = csd(y, dxdt, fs=1 / dt, nperseg=1250, axis=1)
    _, pyy = welch(y, fs=1 / dt, nperseg=1250, axis=1)
    H = pxy.mean(0) / pyy.mean(0)
    return [float(np.abs(H[(f >= lo) & (f < hi)]).mean()) for lo, hi in BANDS], [
        float(np.degrees(np.angle(H[(f >= lo) & (f < hi)]).mean())) for lo, hi in BANDS
    ]


def check(name, x, xd, dt):
    y, y1, y2, xs, dxdt = derivs(x, xd, dt)
    s = np.s_[:, W:-W]
    yv, dv = y[s].ravel(), dxdt[s].ravel()
    k_ols = np.polyfit(yv, dv, 1)[0]
    k_rev = 1 / np.polyfit(dv, yv, 1)[0]
    r = np.corrcoef(yv, dv)[0, 1]
    lag, rmax = best_lag(y, dxdt)
    mag, ph = xfer(y, dxdt, dt)
    off = yv < OFF
    M = np.column_stack([-y1[s].ravel()[off], -xs[s].ravel()[off], np.ones(off.sum())])
    a, b, c = np.linalg.lstsq(M, y2[s].ravel()[off], rcond=None)[0]
    print(
        f"{name:20s} k_ols {k_ols:.3f} k_rev {k_rev:.3f} r {r:.3f} | lag {lag * dt * 1e3:+.2f} ms (r {rmax:.3f}) | "
        f"|H| {' '.join(f'{m:.2f}' for m in mag)}  ph {' '.join(f'{p:+.0f}' for p in ph)} | a {a:.3f} b {b:.3f} c {c * 1e3:+.0f} mV"
    )


def clamps(name, x, xd, dt):
    """Episodes where the true x' (dx/dt from Ch1) runs > GAP above Ch2 (the inverter output, which drives D1).
    Current reaching integrator 1 from D = -(x''' + A x'' + x), all from Ch1 (fig_diode_inferred, SOURCE='x')."""
    import fig_diode_inferred as fd

    y, _, dxdt, D = fd.terms(x, xd, dt)
    n_cap = x.shape[0]
    on = (dxdt - y) > GAP
    starts = np.flatnonzero(np.diff(on.astype(int)) == 1)
    lens = np.diff(np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0]) != 0))[::2]
    T = n_cap * (x.shape[1] - 2 * fd.W3) * dt
    i_ua = D / sc.R_MEAN * 1e6
    top = y > np.percentile(y, 99.5)
    msg = (
        f"{name:20s} clamped {on.mean() * 100:4.1f} % of time, {starts.size / T:5.1f} /s, median {np.median(lens) * dt * 1e3 if lens.size else 0:.2f} ms | "
        f"Ch2 then {np.median(y[on]) if on.any() else np.nan:.3f} V, true x' {np.median(dxdt[on]) if on.any() else np.nan:.2f} | "
        f"I to N1 at Ch2 top 0.5 %: {np.median(i_ua[top]):4.0f} uA (Ch2 {np.median(y[top]):.3f} V)"
    )
    print(msg)


GAP = 0.2  # V per tau


def main():
    print(f"bands (Hz): {BANDS}")
    for k, f in RUNS.items():
        r = sc.load(sc.DATA / f)
        check(k, r["x"], r["xd"], r["dt"])
    for k, f in RUNS.items():
        r = sc.load(sc.DATA / f)
        clamps(k, r["x"], r["xd"], r["dt"])
    rb = sc.load(sc.DATA / RUNS["B (D1 reseated)"])
    mx, mxd = sc.scope_like_sim("simple", rb["x"], rb["xd"])
    check("model + noise", mx, mxd, sc.DT_SCOPE)
    clamps("model + noise", mx, mxd, sc.DT_SCOPE)


if __name__ == "__main__":
    main()
