"""Where Ch1 (x) stops being the integral of Ch2 (x') in state A. Diagnostic figure, model untouched.

e(t) = dx/dt - x' (tau units, Savitzky-Golay as in fig_diode_inferred): zero for an ideal integrator 3.
(a) 30 ms of state A (D1, 13:46): dx/dt and x'      (b) the same for state B
(c) power spectrum of e for A, B and the model with scope noise
(d) rms of e against |x|, i.e. where on the attractor it happens
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

import sprott_common as sc
from check_x_vs_xd import derivs
from fig_diode_inferred import COL, RUNS, W

RUN_A, RUN_B = "A (D1, 13:46)", "B (D1 reseated)"
CAP, T0, SPAN = 3, 200, 750  # capture, start sample, samples shown (30 ms)
X_BINS = np.arange(0, 1.61, 0.1)


def resid(x, xd, dt):
    y, _, _, xs, dxdt = derivs(x, xd, dt)
    s = np.s_[:, W:-W]
    return y[s], dxdt[s], xs[s], dxdt[s] - y[s]


def main():
    ra, rb = sc.load(sc.DATA / RUNS[RUN_A]), sc.load(sc.DATA / RUNS[RUN_B])
    mx, mxd = sc.scope_like_sim("simple", rb["x"], rb["xd"])
    sets = {
        RUN_A: resid(ra["x"], ra["xd"], ra["dt"]),
        RUN_B: resid(rb["x"], rb["xd"], rb["dt"]),
    }
    sets["model + noise"] = resid(mx, mxd, sc.DT_SCOPE)
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.4), layout="constrained")
    t = np.arange(SPAN) * ra["dt"] * 1e3
    for ax, k in ((axs[0, 0], RUN_A), (axs[0, 1], RUN_B)):
        y, d, _, _ = sets[k]
        ax.plot(t, d[CAP, T0 : T0 + SPAN], color="k", lw=0.7, label="dx/dt (from Ch1)")
        ax.plot(t, y[CAP, T0 : T0 + SPAN], color=COL[k], lw=0.9, label="x′ (Ch2)")
        ax.set_xlim(0, t[-1])
        ax.set_xlabel("t (ms)")
        ax.set_ylabel("V per τ")
        ax.set_title(k, fontsize=8)
    axs[0, 0].legend(frameon=False, fontsize=7, loc="lower left")
    for k, (_, _, xs, e) in sets.items():
        f, p = welch(e, fs=1 / sc.DT_SCOPE, nperseg=1024, axis=1)
        axs[1, 0].loglog(
            f[1:],
            p.mean(0)[1:],
            color=COL[k],
            lw=1.0,
            ls="--" if k.startswith("model") else "-",
            label=k,
        )
        idx = np.digitize(np.abs(xs).ravel(), X_BINS) - 1
        c = 0.5 * (X_BINS[1:] + X_BINS[:-1])
        rms = [
            np.sqrt(np.mean(e.ravel()[idx == i] ** 2))
            if (idx == i).sum() > 500
            else np.nan
            for i in range(c.size)
        ]
        axs[1, 1].plot(
            c,
            rms,
            "o-",
            ms=3,
            color=COL[k],
            lw=1.0,
            ls="--" if k.startswith("model") else "-",
            label=k,
        )
        print(
            f"{k:16s} rms e {np.sqrt(np.mean(e**2)):.3f} V/tau, rms x' {np.sqrt(np.mean(sets[k][0] ** 2)):.3f}"
        )
    axs[1, 0].set_xlabel("f (Hz)")
    axs[1, 0].set_ylabel("PSD of e")
    axs[1, 0].set_xlim(10, 12500)
    axs[1, 0].legend(frameon=False, fontsize=7)
    axs[1, 1].set_xlabel("|x| (V)")
    axs[1, 1].set_ylabel("rms of e (V per τ)")
    axs[1, 1].set_ylim(bottom=0)
    for axx, lab in zip(axs.ravel(), "abcd"):
        sc.panel(axx, lab)
    sc.save(fig, "fig_x_vs_xd")


if __name__ == "__main__":
    main()
