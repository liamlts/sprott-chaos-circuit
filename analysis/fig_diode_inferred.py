"""The diode term read straight from the recordings, run by run (no fit, model unchanged).

The circuit obeys x''' = -A x'' - x - D, with D = R I_diode (volts, time in tau). Ch2 records x', so
x'' and x''' are its first and second time derivatives (Savitzky-Golay, window W, polyorder 4) and
    D = -(x''' + A x'' + x)
at every sample, with A = 0.982 from the measured parts. Binned against x' this traces the in-circuit diode curve
I(x') = D / R. Where the diode is off (x' < ~0.3 V) D should be 0; anything else there is not the diode.
Control: the model ('simple', 1N4148 SPICE) with the measured noise and 8-bit steps goes through the same code.
Also printed: the gain check slope of measured x' against dx/dt from Ch1 (should be ~1).
(a) D against x' (V), all runs, linear    (b) I = D / R against x' on a log axis, diode-on part
(c) D against x where the diode is off (x' < 0.25 V)   (d) state-A runs with three different diodes
"""

import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

import sprott_common as sc

W, PO = 21, 4  # Savitzky-Golay window (samples, 0.84 ms) and polyorder
W3 = 31  # window for the 2nd/3rd derivatives of x (1.24 ms)
# "x": x'' and x''' from Ch1 (x = integral of U1 pin 1, the true x'); Ch2 is then only the diode's anode voltage.
# "ch2": from Ch2, which assumes the inverter output (U2 pin 7) equals the true x'. Wrong in state A (check_x_vs_xd).
SOURCE = "x"
XD_BINS = np.arange(-1.0, 0.76, 0.025)
X_BINS = np.arange(-1.6, 0.81, 0.05)
OFF = 0.25  # V: diode treated as off below this x'
R, A = sc.R_MEAN, sc.A_EFF
RUNS = {
    "A (D1, 13:46)": "sprott_R4_10030_20261119_134608.h5",
    "A (D1, rep1)": "sprott_R4_10030_rep1_20261119_140223.h5",
    "A (1N4148 #2)": "sprott_diode_4148b_20261119_164709.h5",
    "A (1N4004)": "sprott_diode_4004_20261119_165526.h5",
    "B (D1 reseated)": "sprott_after_swaps_20261119_181028.h5",
    "C (reseated again)": "sprott_D1_reseated2_20261119_183303.h5",
    "Ge (0.2 V/div)": "sprott_diode_Ge_vdiv02_20261119_172031.h5",
}
COL = {
    "A (D1, 13:46)": sc.C_DATA,
    "B (D1 reseated)": "#2a9d8f",
    "C (reseated again)": "#8e44ad",
    "model + noise": sc.C_SIM,
    "Ge (0.2 V/div)": "#7f8c8d",
}


def terms(x, xd, dt):
    """Smoothed x' and x, dx/dt, and D = -(x''' + A x'' + x), all in tau units, edges trimmed."""
    h = dt / sc.TAU
    y = savgol_filter(xd, W, PO, axis=1)
    xs = savgol_filter(x, W, PO, axis=1)
    dxdt = savgol_filter(x, W, PO, deriv=1, delta=h, axis=1)
    if SOURCE == "ch2":
        y1 = savgol_filter(xd, W, PO, deriv=1, delta=h, axis=1)
        y2 = savgol_filter(xd, W, PO, deriv=2, delta=h, axis=1)
    else:  # Ch1 only: x'' and x''' as 2nd and 3rd derivatives of x (true x', independent of the inverter output)
        y1 = savgol_filter(x, W3, PO, deriv=2, delta=h, axis=1)
        y2 = savgol_filter(x, W3, PO, deriv=3, delta=h, axis=1)
    s = np.s_[:, W3:-W3]
    D = -(y2 + A * y1 + xs)
    return y[s].ravel(), xs[s].ravel(), dxdt[s].ravel(), D[s].ravel()


def binned(v, d, bins, nmin=200):
    idx = np.digitize(v, bins) - 1
    c = 0.5 * (bins[1:] + bins[:-1])
    m, e = np.full(c.size, np.nan), np.full(c.size, np.nan)
    for k in range(c.size):
        sel = d[idx == k]
        if sel.size >= nmin:
            m[k], e[k] = sel.mean(), sel.std() / np.sqrt(sel.size)
    return c, m, e


def summarize(name, y, xs, dxdt, D):
    gain = np.polyfit(dxdt, y, 1)[0]
    off = y < OFF
    slope_x, icpt = np.polyfit(xs[off], D[off], 1)
    c, m, _ = binned(y, D, XD_BINS)
    i_ua = m / R * 1e6

    def v_at(i_target):
        ok = np.isfinite(i_ua) & (c > 0.3)
        above = np.flatnonzero(ok & (i_ua >= i_target))
        return float(c[above[0]]) if above.size else float("nan")

    out = dict(
        gain_xd_vs_dxdt=float(gain),
        mean_D_V=float(D.mean()),
        mean_I_uA=float(D.mean() / R * 1e6),
        diode_off_mean_D_V=float(D[off].mean()),
        diode_off_D_vs_x_slope=float(slope_x),
        diode_off_D_at_x0_V=float(icpt),
        xd_at_100uA=v_at(100.0),
        xd_at_1mA=v_at(1000.0),
    )
    print(
        f"{name:20s} gain {gain:.3f}  mean I {out['mean_I_uA']:6.1f} uA  off: <D> {out['diode_off_mean_D_V'] * 1e3:+6.1f} mV, "
        f"dD/dx {slope_x:+.3f}  | x' at 0.1 mA {out['xd_at_100uA']:.3f} V, 1 mA {out['xd_at_1mA']:.3f} V"
    )
    return out


def main():
    data = {k: sc.load(sc.DATA / f) for k, f in RUNS.items()}
    rb = data["B (D1 reseated)"]
    mx, mxd = sc.scope_like_sim("simple", rb["x"], rb["xd"])
    sets = {k: terms(r["x"], r["xd"], r["dt"]) for k, r in data.items()}
    sets["model + noise"] = terms(mx, mxd, sc.DT_SCOPE)
    cx, cxd = sc.sim_windows("simple")
    sets["model, clean"] = terms(cx, cxd, sc.DT_SCOPE)
    res = {k: summarize(k, *v) for k, v in sets.items()}
    spice = lambda v: R * sc.IS * np.expm1(v / (sc.NID * sc.VT))  # noqa: E731
    for v in (0.48, 0.584):
        print(f"SPICE: I at x' = {v} V is {spice(v) / R * 1e6:.0f} uA")
    sc.save_results("diode_inferred", res)

    fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.6), layout="constrained")
    ax_a, ax_b, ax_c, ax_d = axs.ravel()
    vv = np.linspace(-1.0, 0.75, 300)
    main_sets = (
        "A (D1, 13:46)",
        "B (D1 reseated)",
        "C (reseated again)",
        "model + noise",
    )
    for k in main_sets:
        y, xs, _, D = sets[k]
        c, m, e = binned(y, D, XD_BINS)
        ls = "--" if k.startswith("model") else "-"
        ax_a.plot(c, m, ls, color=COL[k], lw=1.0, label=k)
        pos = m > 0
        ax_b.semilogy(c[pos], m[pos] / R * 1e6, ls, color=COL[k], lw=1.0, label=k)
        off = y < OFF
        c2, m2, _ = binned(xs[off], D[off], X_BINS)
        ax_c.plot(c2, m2 * 1e3, ls, color=COL[k], lw=1.0, label=k)
    ax_a.plot(
        vv,
        np.clip(spice(vv), None, 2.0),
        color="k",
        lw=0.6,
        ls=":",
        label="1N4148 SPICE",
    )
    ax_b.semilogy(
        vv[vv > 0.2],
        spice(vv[vv > 0.2]) / R * 1e6,
        color="k",
        lw=0.6,
        ls=":",
        label="1N4148 SPICE",
    )
    ax_a.set_xlim(-1.0, 0.75)
    ax_a.set_ylim(-0.4, 1.6)
    ax_a.axhline(0, color="grey", lw=0.5)
    ax_a.set_xlabel("x′ (V)")
    ax_a.set_ylabel("Diode term D (V)")
    ax_a.legend(frameon=False, fontsize=6.5, loc="upper left")
    ax_b.set_xlim(0.2, 0.75)
    ax_b.set_ylim(1, 3e4)
    ax_b.set_xlabel("x′ (V)")
    ax_b.set_ylabel("Diode current D / R (µA)")
    ax_c.axhline(0, color="grey", lw=0.5)
    ax_c.set_xlabel(f"x (V), only where x′ < {OFF} V")
    ax_c.set_ylabel("D where the diode is off (mV)")
    ax_c.set_xlim(-1.6, 0.8)
    for k in (
        "A (D1, 13:46)",
        "A (D1, rep1)",
        "A (1N4148 #2)",
        "A (1N4004)",
        "B (D1 reseated)",
        "Ge (0.2 V/div)",
    ):
        y, _, _, D = sets[k]
        c, m, _ = binned(y, D, XD_BINS)
        pos = (m > 0) & (c > 0.0)
        ax_d.semilogy(c[pos], m[pos] / R * 1e6, lw=1.0, color=COL.get(k), label=k)
    ax_d.semilogy(
        vv[vv > 0.2], spice(vv[vv > 0.2]) / R * 1e6, color="k", lw=0.6, ls=":"
    )
    ax_d.set_xlim(0.0, 0.75)
    ax_d.set_ylim(1, 3e4)
    ax_d.set_xlabel("x′ (V)")
    ax_d.set_ylabel("Diode current D / R (µA)")
    ax_d.legend(frameon=False, fontsize=6, loc="upper left")
    for axx, lab in zip(axs.ravel(), "abcd"):
        sc.panel(axx, lab, right=lab in "bd")
    sc.save(fig, "fig_diode_inferred")


if __name__ == "__main__":
    main()
