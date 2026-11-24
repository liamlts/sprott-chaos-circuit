"""Germanium diode in D1 at two scope settings: 0.5 V/div (as the other runs, 20 mV steps) and
0.2 V/div (8 mV steps) for the smaller signal.
(a, b) density of (x, x') at each setting, zoomed to the Ge attractor
(c) heights of the x' maxima at both settings
(d) return map of successive x' maxima within a capture, M_{n+1} against M_n, 0.2 V/div
(e) the first 0.2 V/div capture against time
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

import sprott_common as sc

RUN_05 = sc.DATA / "sprott_diode_Ge_20261119_171152.h5"
RUN_02 = sc.DATA / "sprott_diode_Ge_vdiv02_20261119_172031.h5"
COL = {0.5: "#b8860b", 0.2: "#d35400"}
SEC_KW = dict(hmin=0.01, prom=0.02)  # same as the section in fig_returnmap
XLIM, YLIM = (-0.62, 0.32), (-0.36, 0.30)
LOW = 0.04  # the separate cluster of small maxima sits below this


def maxima(r):
    """Heights of the x' maxima, and successive pairs (M_n, M_n+1) inside each capture."""
    hs, pairs = [], []
    for a, b in zip(r["x"], r["xd"]):
        m = sc.xd_maxima(a, b, **SEC_KW)[2]
        hs.append(m)
        pairs.append(np.column_stack([m[:-1], m[1:]]))
    return np.concatenate(hs), np.concatenate(pairs)


def main():
    runs = {0.5: sc.load(RUN_05), 0.2: sc.load(RUN_02)}
    res = {}
    for v, r in runs.items():
        lsb = v / 25
        h, pr = maxima(r)
        res[v] = dict(
            step_mV=lsb * 1e3,
            levels_x=int(len(np.unique(np.round(r["x"] / lsb)))),
            levels_xd=int(len(np.unique(np.round(r["xd"] / lsb)))),
            hf_noise_mV=[sc.hf_noise(r["x"]) * 1e3, sc.hf_noise(r["xd"]) * 1e3],
            hf_noise_minus_steps_mV=[
                np.sqrt(sc.hf_noise(c) ** 2 - lsb**2 / 12) * 1e3
                for c in (r["x"], r["xd"])
            ],
            maxima_per_capture=len(h) / len(r["x"]),
            maxima_quantiles_5_25_50_75_95=np.percentile(
                h, [5, 25, 50, 75, 95]
            ).tolist(),
            frac_maxima_below_low=float(np.mean(h < LOW)),
            h=h,
            pairs=pr,
        )
        print(
            f"{v} V/div: step {lsb * 1e3:.0f} mV, levels x/x' {res[v]['levels_x']}/{res[v]['levels_xd']}, "
            f"hf noise {res[v]['hf_noise_mV'][0]:.1f}/{res[v]['hf_noise_mV'][1]:.1f} mV "
            f"(minus steps {res[v]['hf_noise_minus_steps_mV'][0]:.1f}/{res[v]['hf_noise_minus_steps_mV'][1]:.1f}), "
            f"maxima/capture {res[v]['maxima_per_capture']:.1f}, below {LOW} V: "
            f"{res[v]['frac_maxima_below_low'] * 100:.1f} %"
        )

    fig = plt.figure(figsize=(7.0, 8.0), layout="constrained")
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 0.65])
    a = fig.add_subplot(gs[0, 0])
    b = fig.add_subplot(gs[0, 1], sharey=a)
    c = fig.add_subplot(gs[1, 0])
    d = fig.add_subplot(gs[1, 1])
    e = fig.add_subplot(gs[2, :])

    # densities on each run's own grid: one 20 mV step per bin at 0.5 V/div, two 8 mV steps at 0.2 V/div,
    # bin edges halfway between steps so no bin catches an extra level
    grids = {
        0.5: np.arange(XLIM[0] - 0.05, XLIM[1] + 0.05, 0.02) + 0.01,
        0.2: np.arange(XLIM[0] - 0.05, XLIM[1] + 0.05, 0.016) + 0.004,
    }
    hs = {
        v: np.histogram2d(
            r["x"].ravel(), r["xd"].ravel(), [grids[v], grids[v]], density=True
        )[0]
        for v, r in runs.items()
    }
    vmax = max(h.max() for h in hs.values())
    norm = LogNorm(vmin=vmax * 1e-4, vmax=vmax)
    for axx, v in ((a, 0.5), (b, 0.2)):
        im = axx.pcolormesh(
            grids[v],
            grids[v],
            np.ma.masked_equal(hs[v].T, 0),
            cmap=sc.CMAP,
            norm=norm,
            rasterized=True,
        )
        axx.set_xlim(*XLIM)
        axx.set_ylim(*YLIM)
        axx.set_xlabel("x (V)")
        axx.set_title(
            f"{v} V/div ({v / 25 * 1e3:.0f} mV steps)", fontsize=8, color=COL[v]
        )
    a.set_ylabel("x′ (V)")
    plt.setp(b.get_yticklabels(), visible=False)
    fig.colorbar(im, ax=b, label="Density (1/V²)", pad=0.02)

    bins = np.arange(0, 0.30, 0.005)
    for v in (0.5, 0.2):
        c.hist(
            res[v]["h"],
            bins,
            density=True,
            histtype="step",
            color=COL[v],
            lw=1.0,
            label=f"{v} V/div",
        )
    c.axvline(LOW, color="grey", ls="--", lw=0.8)
    c.set_xlabel("x′ maximum $M_n$ (V)")
    c.set_ylabel("Density (1/V)")
    c.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.08, 1.0))

    pr = res[0.2]["pairs"]
    d.plot(pr[:, 0], pr[:, 1], ".", color=COL[0.2], ms=1.5, alpha=0.35, rasterized=True)
    d.plot([0, 0.27], [0, 0.27], color="grey", ls="--", lw=0.8)
    d.set_xlim(0, 0.27)
    d.set_ylim(0, 0.27)
    d.set_aspect("equal")
    d.set_xlabel("$M_n$ (V)")
    d.set_ylabel("$M_{n+1}$ (V)")

    r = runs[0.2]
    t_ms = (r["t"] - r["t"][0]) * 1e3
    e.plot(t_ms, r["x"][0], color=sc.C_DATA, lw=0.6, label="x")
    e.plot(t_ms, r["xd"][0], color=COL[0.2], lw=0.6, label="x′")
    e.axhline(0, color="grey", lw=0.5)
    e.set_xlim(t_ms[0], t_ms[-1])
    e.set_xlabel("Time (ms)")
    e.set_ylabel("Voltage (V)")
    lo, hi = e.get_ylim()
    e.set_ylim(lo, hi + 0.25 * (hi - lo))
    e.legend(frameon=False, ncol=2, loc="upper right")

    for axx, lab in zip((a, b, c, d, e), "abcde"):
        sc.panel(axx, lab, right=lab == "c")
    sc.save(fig, "fig_ge", png=True)
    sc.save_results(
        "ge",
        {
            f"{v} V/div": {k: val for k, val in d_.items() if k not in ("h", "pairs")}
            for v, d_ in res.items()
        },
    )


if __name__ == "__main__":
    main()
