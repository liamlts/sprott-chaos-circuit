"""Run B (original D1 reseated, 18:10, 100 captures): the measured attractor against the model.

(a) x and x' against time, first capture of the run (no selection)
(b) density of (x, x') from all 100 captures
(c) the same for the model: 1N4148 model at the measured A = 0.982, plus the measured fast noise and the 8-bit
    scope steps (scope_like_sim, 400 windows); nothing fitted. Same colour scale as (b).
(d) power spectrum of x, Hann window per capture, averaged over captures, in dB relative to its maximum
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

import sprott_common as sc

RUN = sc.B_RUN
EDGES = [np.arange(-2.0, 1.2, 0.04) + 0.01, np.arange(-1.2, 1.0, 0.04) + 0.01]
XLIM, YLIM = (-1.8, 0.9), (-1.05, 1.1)  # headroom for the panel letter
FMAX = 800.0  # Hz shown in (d)
C_XD = "#2a9d8f"  # x' trace


def mean_spectrum_db(x, dt):
    f, p = sc.spectrum(x, dt)
    pm = p.mean(0)
    return f, 10 * np.log10(pm / pm.max())


def main():
    r = sc.load(RUN)
    x, xd, dt = r["x"], r["xd"], r["dt"]
    mx, mxd = sc.scope_like_sim("simple", x, xd)

    f_d, s_d = mean_spectrum_db(x, dt)
    f_m, s_m = mean_spectrum_db(mx, dt)
    hd = np.histogram2d(x.ravel(), xd.ravel(), EDGES, density=True)[0]
    hm = np.histogram2d(mx.ravel(), mxd.ravel(), EDGES, density=True)[0]
    vmax = max(hd.max(), hm.max())
    norm = LogNorm(vmin=vmax * 1e-4, vmax=vmax)

    fig = plt.figure(figsize=(7.0, 4.4), layout="constrained")
    sf_top, sf_bot = fig.subfigures(2, 1, height_ratios=[0.75, 1.25])
    fig.get_layout_engine().set(wspace=0.08)  # keeps the colourbar label clear of (d)
    ax_t = sf_top.subplots()
    ax_b, ax_c, ax_d = sf_bot.subplots(1, 3, width_ratios=[1, 1, 1.1])
    ax_c.sharex(ax_b)
    ax_c.sharey(ax_b)

    t_ms = np.arange(x.shape[1]) * dt * 1e3
    ax_t.plot(t_ms, x[0], color=sc.C_DATA, lw=0.7, label="x")
    ax_t.plot(t_ms, xd[0], color=C_XD, lw=0.7, label="x′")
    ax_t.set_xlim(0, t_ms[-1])
    ax_t.set_ylim(-1.9, 1.3)
    ax_t.set_xlabel("Time (ms)")
    ax_t.set_ylabel("Voltage (V)")
    ax_t.legend(frameon=False, ncol=2, loc="upper right")

    for axx, h in ((ax_b, hd), (ax_c, hm)):
        im = axx.pcolormesh(
            EDGES[0],
            EDGES[1],
            np.ma.masked_equal(h.T, 0),
            cmap=sc.CMAP,
            norm=norm,
            rasterized=True,
        )
        axx.set_xlim(*XLIM)
        axx.set_ylim(*YLIM)
        axx.set_xlabel("x (V)")
    ax_b.set_ylabel("x′ (V)")
    ax_b.set_title("Board", fontsize=8)
    ax_c.set_title("Model", fontsize=8)
    plt.setp(ax_c.get_yticklabels(), visible=False)
    sf_bot.colorbar(im, ax=[ax_b, ax_c], label="Density (1/V²)", pad=0.02)

    ax_d.plot(f_d, s_d, color=sc.C_DATA, lw=0.9, label="Board")
    ax_d.plot(f_m, s_m, color=sc.C_SIM, lw=0.9, ls="--", label="Model")
    ax_d.set_xlim(0, FMAX)
    ax_d.set_ylim(-60, 5)
    ax_d.set_xlabel("Frequency (Hz)")
    ax_d.set_ylabel("Power (dB)")
    ax_d.legend(frameon=False, loc="lower left")

    for axx, lab in zip((ax_t, ax_b, ax_c, ax_d), "abcd"):
        sc.panel(axx, lab, right=lab == "d")
    sc.save(fig, "fig_B_attractor", png=True)

    pk_d, pk_m = (
        sc.peak_freq(f_d, 10 ** (s_d / 10)),
        sc.peak_freq(f_m, 10 ** (s_m / 10)),
    )
    print(f"peak f: board {pk_d:.1f} Hz, model {pk_m:.1f} Hz")
    sc.save_results(
        "B_attractor",
        dict(run=RUN.name, peak_f_board=float(pk_d), peak_f_model=float(pk_m)),
    )


if __name__ == "__main__":
    main()
