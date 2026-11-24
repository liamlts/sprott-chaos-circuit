"""State A: what the inverter output (Ch2, U2 pin 7, drives D1) does while the true x' (dx/dt from Ch1) runs ahead.
Diagnostic figure, nothing fitted. Episodes = true x' - Ch2 > GAP (check_x_vs_xd.clamps), aligned on their start.
(a) 8 episodes from the D1 13:46 run: true x' (black) and Ch2 (colour), +-1.5 ms
(b) mean over all episodes in that run, with the same for the 1N4004 and Ge runs (Ch2 only, dashed = true x')
(c) current reaching integrator 1 (Ch1-derived D / R) over the same window, mean over episodes
"""

import matplotlib.pyplot as plt
import numpy as np

import fig_diode_inferred as fd
import sprott_common as sc
from check_x_vs_xd import GAP
from fig_diode_inferred import COL, RUNS
from scipy.signal import savgol_filter

PRE, POST = 38, 38  # samples (1.5 ms) before / after the episode start
SHOW = ("A (D1, 13:46)", "A (1N4004)", "Ge (0.2 V/div)")
COL2 = {
    "A (D1, 13:46)": sc.C_DATA,
    "A (1N4004)": "#e67e22",
    "Ge (0.2 V/div)": "#7f8c8d",
}


def episodes(r):
    x, xd, dt = r["x"], r["xd"], r["dt"]
    h = dt / sc.TAU
    y = savgol_filter(xd, fd.W, fd.PO, axis=1)
    d = savgol_filter(x, fd.W, fd.PO, deriv=1, delta=h, axis=1)
    x2 = savgol_filter(x, fd.W3, fd.PO, deriv=2, delta=h, axis=1)
    x3 = savgol_filter(x, fd.W3, fd.PO, deriv=3, delta=h, axis=1)
    xs = savgol_filter(x, fd.W, fd.PO, axis=1)
    i_ua = -(x3 + fd.A * x2 + xs) / sc.R_MEAN * 1e6
    out = []
    for c in range(x.shape[0]):
        on = (d[c] - y[c]) > GAP
        st = np.flatnonzero(np.diff(on.astype(int)) == 1) + 1
        for s in st[(st > fd.W3 + PRE) & (st < x.shape[1] - fd.W3 - POST)]:
            out.append(
                (
                    d[c, s - PRE : s + POST],
                    y[c, s - PRE : s + POST],
                    i_ua[c, s - PRE : s + POST],
                )
            )
    return np.array(out), dt


def main():
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6), layout="constrained")
    for k in SHOW:
        ep, dt = episodes(sc.load(sc.DATA / RUNS[k]))
        t = (np.arange(-PRE, POST)) * dt * 1e3
        print(
            f"{k:16s} {len(ep)} episodes; Ch2 min in window {np.median(ep[:, 1].min(1)):.3f} V, "
            f"true x' max {np.median(ep[:, 0].max(1)):.2f}, Ch2 max {np.median(ep[:, 1].max(1)):.2f}, "
            f"I max {np.median(ep[:, 2].max(1)):.0f} uA"
        )
        if k == SHOW[0]:
            for j in np.linspace(0, len(ep) - 1, 8).astype(int):
                axs[0].plot(t, ep[j, 0], color="k", lw=0.5)
                axs[0].plot(t, ep[j, 1], color=COL[k], lw=0.7)
        m = ep.mean(0)
        axs[1].plot(t, m[1], color=COL2[k], lw=1.1, label=k)
        axs[1].plot(t, m[0], color=COL2[k], lw=0.8, ls="--")
        axs[2].plot(t, m[2], color=COL2[k], lw=1.1, label=k)
    for ax in axs:
        ax.axvline(0, color="grey", lw=0.5, ls="--")
        ax.set_xlabel("t from episode start (ms)")
        ax.set_xlim(-1.5, 1.5)
    axs[0].set_ylabel("x′ (V per τ)")
    axs[1].set_ylabel("x′, mean (V per τ)")
    axs[2].set_ylabel("Current into integrator 1 (µA)")
    axs[1].legend(frameon=False, fontsize=6, loc="upper left")
    for axx, lab in zip(axs, "abc"):
        sc.panel(axx, lab)
    sc.save(fig, "fig_clamp_episodes")


if __name__ == "__main__":
    main()
