"""One-page summary of the Sprott-circuit analysis (figures/sprott_summary.pdf).

Reads the numbers written by the other scripts (analysis/results/*.json), so run those first:
fig_peaks.py, fig_stats.py, fig_returnmap.py, fig_origin.py, fig_compare.py, fig_repeat.py.
(a) measured density of (x, x')  (b) model density   (c) time spent within r of the origin
(d) growth of the small loops near the origin   (e) divergence of neighbouring states (Lyapunov)
(f) repeatability: rms of x and time near the origin, 20-capture blocks over clock time
"""

import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

import sprott_common as sc

RES = sc.HERE / "results"


def J(name):
    return json.loads((RES / f"{name}.json").read_text())


def main():
    pk, st, rm, og, cp, rp = (
        J(n) for n in ("peaks", "stats", "returnmap", "origin", "compare", "repeat")
    )
    r = sc.load(sc.MAIN_RUN)
    x, xd = r["x"], r["xd"]
    sx, sxd = sc.scope_like_sim("simple", x, xd)
    D, M = st["data"], st["model"]
    sz = cp["size"]
    win = rm["measured"]["windows"]
    tab = rp["table"]
    big = [
        k for k, n in zip(rp["runs"], rp["n"]) if n >= 200 and rp["blocks"][k]["short"]
    ]
    rms_runs = [tab[k]["rms x (V)"][0] for k in big]
    near_runs = [tab[k][f"time frac r<{st['r_dwell']}"][0] for k in big]
    dr = rp["drift"]
    near_key = f"time frac r<{st['r_dwell']}"
    near_d, near_m = D["val"][near_key], M["val"][near_key]
    rx = sz["x range (max - min)"]["data"] / sz["x range (max - min)"]["simple"]
    rxd = sz["x' range (max - min)"]["data"] / sz["x' range (max - min)"]["simple"]
    rms_x = D["val"]["std x"] / M["val"]["std x"]
    rms_xd = D["val"]["std x'"] / M["val"]["std x'"]

    text = [
        (
            "Board",
            f"measured parts, A = {sc.A_EFF:.3f}, τ = {sc.TAU * 1e3:.3f} ms; 1N4148 model at the same values, "
            f"no fitted parameters. {sum(rp['n'])} captures in {len(rp['runs'])} runs: "
            f"{rp['minutes']:.0f} min of 10 ms/div runs plus one 25 ms/div run.",
        ),
        (
            "Chaotic, with the right linear part",
            f"λ from the data {win['2-12']:.2f}–{win['2-7']:.2f} per τ (fit-window dependent; model through the same "
            f"estimator {rm['sim. + noise, steps']['lam_flow']:.3f}, true {rm['lambda_model']:.3f}). "
            f"Growth of small loops near the origin {og['growth_low']:.3f} ± {og['growth_low_err']:.3f} per τ "
            f"vs {og['sigma_theory']:.3f} from linear theory.",
        ),
        (
            "Lingers at the origin",
            f"re-injected as close as {og['data_min'] * 1e3:.0f} mV (model never below {og['model_min']:.2f} V); "
            f"{near_d * 100:.0f} % "
            f"of the time within {st['r_dwell']} V of the origin vs "
            f"{near_m * 100:.0f} % "
            f"in the model; mean dwell {D['val']['mean dwell (ms)']:.1f} vs {M['val']['mean dwell (ms)']:.1f} ms. "
            f"The low Poincaré maxima ({pk['data_frac_low_nocut'] * 100:.0f} % below 0.25 V) are these real small loops, not noise.",
        ),
        (
            "Size",
            f"max−min range ×{rx:.2f} (x), "
            f"×{rxd:.2f} (x′) "
            f"of the model, but rms ×{rms_x:.2f} and ×{rms_xd:.2f}. "
            f"Half as many big loops ({sz['x minima < -0.5 V per 100 ms']['data']:.1f} vs "
            f"{sz['x minima < -0.5 V per 100 ms']['simple']:.1f} per 100 ms); mean diode current "
            f"{cp['mean_diode_uA']['data']:.0f} vs {cp['mean_diode_uA']['simple']:.0f} µA.",
        ),
        (
            "Frequency",
            f"{D['val']['peak f (Hz)']:.1f} ± {D['err']['peak f (Hz)']:.1f} Hz measured vs "
            f"{M['val']['peak f (Hz)']:.1f} Hz model ({st['f_lin']:.0f} Hz linear theory).",
        ),
        (
            "Repeatable",
            f"rms x {min(rms_runs):.3f}–{max(rms_runs):.3f} V and time near origin "
            f"{min(near_runs):.3f}–{max(near_runs):.3f} over the {len(big)} long runs; drift of rms x "
            f"{dr['rms x (V)']['slope_per_hour'] * 1e3:+.1f} ± {dr['rms x (V)']['err'] * 1e3:.1f} mV/h. Data–model "
            f"distance (Hellinger {cp['hellinger']['data vs simple']:.2f}) is "
            f"{cp['hellinger']['data vs simple'] / max(v for k, v in cp['hellinger'].items() if not k.startswith('data')):.0f}× "
            f"the largest run-to-run or split-half one.",
        ),
        (
            "Most likely cause (untested)",
            "the diode clamp: the real 1N4148 (and the op-amp driving it) is softer than the SPICE model. "
            "Not modelled; needs a DVM I–V of the diode.",
        ),
    ]

    fig = plt.figure(figsize=(8.5, 11.0))
    fig.text(
        0.06,
        0.965,
        "Sprott (2011) jerk circuit on the breadboard: measured vs 1N4148 model",
        fontsize=13,
        fontweight="bold",
        va="top",
    )
    y = 0.935
    import textwrap

    for head, body in text:
        lines = textwrap.wrap(f"{head}: {body}", 118)
        fig.text(
            0.06, y, lines[0].replace(head + ":", head + ":"), fontsize=8.5, va="top"
        )
        for ln in lines[1:]:
            y -= 0.0155
            fig.text(0.075, y, ln, fontsize=8.5, va="top")
        y -= 0.021
    top = y - 0.01

    gs = fig.add_gridspec(
        3, 2, left=0.09, right=0.97, bottom=0.05, top=top, hspace=0.42, wspace=0.28
    )
    a = fig.add_subplot(gs[0, 0])
    b = fig.add_subplot(gs[0, 1], sharey=a)
    c = fig.add_subplot(gs[1, 0])
    d = fig.add_subplot(gs[1, 1])
    e = fig.add_subplot(gs[2, 0])
    f = fig.add_subplot(gs[2, 1])

    edges = [np.arange(-2.0, 1.2, 0.04) + 0.01, np.arange(-1.2, 1.0, 0.04) + 0.01]
    hd = np.histogram2d(x.ravel(), xd.ravel(), edges, density=True)[0]
    hs = np.histogram2d(sx.ravel(), sxd.ravel(), edges, density=True)[0]
    vmax = max(hd.max(), hs.max())
    norm = LogNorm(vmin=vmax * 1e-4, vmax=vmax)
    for axx, h in ((a, hd), (b, hs)):
        im = axx.pcolormesh(
            edges[0],
            edges[1],
            np.ma.masked_equal(h.T, 0),
            cmap=sc.CMAP,
            norm=norm,
            rasterized=True,
        )
        axx.set_xlim(-1.8, 0.9)
        axx.set_ylim(-1.05, 0.9)
        axx.set_xlabel("x (V)")
    a.set_ylabel("x′ (V)")
    plt.setp(b.get_yticklabels(), visible=False)
    fig.colorbar(im, ax=b, label="Density (1/V²)", pad=0.02)

    rad = np.array(st["radii"])
    c.plot(rad, M["frac"], color=sc.C_SIM, lw=1.0, label="model")
    c.plot(rad, D["frac"], color=sc.C_DATA, lw=1.0, label="measured")
    c.axvline(st["r_dwell"], color="grey", ls="--", lw=0.8)
    c.set_xlabel("Radius r in (x, x′) (V)")
    c.set_ylabel("Fraction of time within r")
    c.legend(frameon=False, loc="lower right")

    mc, g, ge = np.array(og["mc"]), np.array(og["g"]), np.array(og["ge"])
    vg = np.array(og["vg"], dtype=float)
    d.plot(
        mc[mc < 0.2],
        vg[mc < 0.2],
        "o",
        mfc="none",
        color=sc.C_SIM,
        ms=4,
        label="model, started near 0",
    )
    d.errorbar(mc, g, ge, fmt="o", color=sc.C_DATA, ms=3, capsize=2, label="measured")
    d.axhline(og["sigma_theory"], color="grey", ls="--", lw=0.8)
    d.axhline(0, color="grey", lw=0.5)
    d.set_xlabel("x′ maximum $M_n$ (V)")
    d.set_ylabel(r"$\ln(M_{n+1}/M_n)/\Delta t$ (1/$\tau$)")
    d.legend(frameon=False, loc="lower left", fontsize=7)

    for nm, col in (
        ("simulation", sc.C_SIM),
        ("sim. + noise, steps", sc.C_SIM2),
        ("measured", sc.C_DATA),
    ):
        e.plot(rm[nm]["tk"], rm[nm]["curve"], color=col, lw=1.0, label=nm)
    t = np.linspace(0, 24, 10)
    e.plot(t, rm["lambda_model"] * t, color="k", ls=":", lw=0.8, label=f"slope {rm['lambda_model']:.3f}")
    e.set_xlabel(r"Time ($\tau$ = 1.006 ms)")
    e.set_ylabel(r"$\langle \ln d(t) \rangle - \langle \ln d(0) \rangle$")
    e.legend(frameon=False, loc="lower right", fontsize=7)

    cols = sc.CMAP(np.linspace(0.35, 0.95, len(rp["runs"])))
    ref = {
        q: np.mean(
            np.concatenate(
                [
                    np.array(rp["blocks"][k][q])[:, 1]
                    for k in rp["runs"]
                    if rp["blocks"][k]["short"] and len(rp["blocks"][k][q])
                ]
            )
        )
        for q in ("sx", "near")
    }
    for k, col in zip(rp["runs"], cols):
        mk = "o" if rp["blocks"][k]["short"] else "D"
        for q, off, fill in (("sx", -0.25, None), ("near", 0.25, "none")):
            bl = np.array(rp["blocks"][k][q])
            if len(bl):
                f.errorbar(
                    bl[:, 0] + off,
                    100 * (bl[:, 1] / ref[q] - 1),
                    100 * bl[:, 2] / ref[q],
                    fmt=mk,
                    color=col,
                    ms=3,
                    capsize=1.5,
                    mfc=fill,
                )
    f.axhline(0, color="grey", lw=0.5)
    f.plot([], [], "o", color="k", ms=3, label=f"rms x (mean {ref['sx']:.3f} V)")
    f.plot(
        [],
        [],
        "o",
        color="k",
        ms=3,
        mfc="none",
        label=f"time within 0.2 V (mean {ref['near']:.3f})",
    )
    f.legend(frameon=False, loc="lower left", fontsize=7)
    f.set_xlabel("Clock time (min)")
    f.set_ylabel("Change from mean (%)")
    lo, hi = f.get_ylim()
    f.set_ylim(lo - 0.35 * (hi - lo), hi)

    for axx, lab in zip((a, b, c, d, e, f), "abcdef"):
        sc.panel(axx, lab, right=lab in "d")
    out = sc.FIGS / "sprott_summary.pdf"
    fig.savefig(out)
    fig.savefig(sc.FIGS / "sprott_summary_preview.png", dpi=100)
    print(f"summary -> {out}")


if __name__ == "__main__":
    main()
