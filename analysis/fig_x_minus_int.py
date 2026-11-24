"""Does x (Ch1) jump in state A? u(t) = x(t) - integral of x' (Ch2), raw data, no smoothing; a straight-line fit is
removed per capture (it takes out the constant Ch2 offset, ~+12 mV in state A). Integrator 3 makes x from -x' at
U1 pin 1, Ch2 is the inverter copy, so u is flat (noise only) if both copies agree. Diagnostic, nothing fitted.
(a) state A capture CAP: x and u      (b) state B, same capture index
"""

import matplotlib.pyplot as plt
import numpy as np

import sprott_common as sc
from fig_diode_inferred import COL, RUNS

CAP = 3
RUN_A, RUN_B = "A (D1, 13:46)", "B (D1 reseated)"


def u_of(r, c):
    x, xd, dt = r["x"][c], r["xd"][c], r["dt"]
    integ = np.concatenate([[0.0], np.cumsum(0.5 * (xd[1:] + xd[:-1]))]) * dt / sc.TAU
    u = x - integ
    t = np.arange(x.size) * dt
    u = u - np.polyval(np.polyfit(t, u, 1), t)
    return t * 1e3, x, u


def main():
    fig, axs = plt.subplots(2, 1, figsize=(7.0, 4.4), layout="constrained", sharex=True)
    for ax, k in zip(axs, (RUN_A, RUN_B)):
        r = sc.load(sc.DATA / RUNS[k])
        t, x, u = u_of(r, CAP)
        ax.plot(t, x, color=COL[k], lw=0.6, label="x (Ch1)")
        ax.plot(t, u * 3 - 1.6, color="k", lw=0.7, label="x − ∫x′, ×3, shifted down")
        ax.set_ylabel("V")
        ax.set_title(k, fontsize=8)
        # step statistic over all captures: 0.4 ms jumps in u
        jumps = []
        for c in range(r["x"].shape[0]):
            _, _, uu = u_of(r, c)
            jumps.append(uu[10:] - uu[:-10])
        j = np.concatenate(jumps)
        print(
            f"{k:16s} rms u {np.std(u) * 1e3:.0f} mV; 0.4 ms changes in u: 99.9th pct {np.percentile(np.abs(j), 99.9) * 1e3:.0f} mV, "
            f"frac > 100 mV {np.mean(np.abs(j) > 0.1):.4f} (+: {np.mean(j > 0.1):.4f}, -: {np.mean(j < -0.1):.4f})"
        )
    axs[0].legend(frameon=False, fontsize=7, loc="upper right", ncol=2)
    axs[1].set_xlabel("t (ms)")
    axs[1].set_xlim(0, 100)
    for axx, lab in zip(axs, "ab"):
        sc.panel(axx, lab)
    sc.save(fig, "fig_x_minus_int", png=True)


if __name__ == "__main__":
    main()
