"""Statistics of one run with uncertainties from a bootstrap over captures (captures are 2.75 s
= 2700 tau apart, so they are independent samples of the attractor).

(a) histogram of x      (b) histogram of x'      lines: measured; band: 16-84 % bootstrap range
(c) mean power spectrum of x (dB), band = bootstrap range
(d) fraction of time within radius r of the origin in the (x, x') plane
(e) dwell times inside r = R_DWELL: complementary cumulative distribution
Simulation (1N4148 model, measured parts) with the scope's noise and 20 mV steps added, same pipeline.
Usage: python3 fig_stats.py [run.h5]   (default: the main run)
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import sprott_common as sc

R_DWELL = 0.2  # V
RADII = np.linspace(0.04, 0.5, 24)
N_BOOT = 1000


def run_stats(x, xd, dt, n_boot=N_BOOT, seed=0):
    """Point values and bootstrap arrays of the run statistics."""
    n = len(x)
    bx = np.arange(-2.0, 1.2, 0.04) + 0.01
    by = np.arange(-1.2, 1.0, 0.04) + 0.01
    hx = np.array([np.histogram(a, bx)[0] for a in x], float)
    hy = np.array([np.histogram(a, by)[0] for a in xd], float)
    f, p = sc.spectrum(x, dt)
    rad = np.array(
        [
            np.hypot(sc.boxcar(a, sc.SMOOTH), sc.boxcar(b, sc.SMOOTH))[
                sc.SMOOTH : -sc.SMOOTH
            ]
            for a, b in zip(x, xd)
        ]
    )
    frac = np.array([[np.mean(rr < r0) for r0 in RADII] for rr in rad])
    dw = sc.dwell_times(x, xd, R_DWELL, dt)

    def scal(idx):
        xx, yy = x[idx], xd[idx]
        pm = p[idx].mean(axis=0)
        d = np.concatenate([dw[i] for i in idx])
        return [
            xx.std(),
            yy.std(),
            np.percentile(xx, 1),
            np.percentile(xx, 99),
            np.percentile(yy, 1),
            np.percentile(yy, 99),
            sc.peak_freq(f, pm),
            frac[idx, np.argmin(abs(RADII - R_DWELL))].mean(),
            d.mean() * 1e3,
            len(d) / len(idx),
            xx.mean(),
            yy.mean(),
        ]

    names = [
        "std x",
        "std x'",
        "x 1%",
        "x 99%",
        "x' 1%",
        "x' 99%",
        "peak f (Hz)",
        f"time frac r<{R_DWELL}",
        "mean dwell (ms)",
        "dwells/capture",
        "mean x",
        "mean x'",
    ]
    rng = np.random.default_rng(seed)
    boots = [rng.integers(0, n, n) for _ in range(n_boot)]
    out = dict(
        names=names,
        val=np.array(scal(np.arange(n))),
        boot=np.array([scal(i) for i in boots]),
        bx=bx,
        by=by,
        hx=hx,
        hy=hy,
        f=f,
        p=p,
        frac=frac,
        dwell=np.concatenate(dw),
        hx_b=np.array([hx[i].sum(0) / hx[i].sum() / np.diff(bx) for i in boots]),
        hy_b=np.array([hy[i].sum(0) / hy[i].sum() / np.diff(by) for i in boots]),
        p_b=np.array([p[i].mean(0) for i in boots]),
        frac_b=np.array([frac[i].mean(0) for i in boots]),
    )
    return out


def band(ax, xc, b, color, step=False, **kw):
    lo, mid, hi = np.percentile(b, [16, 50, 84], axis=0)
    if step:
        ax.fill_between(xc, lo, hi, step="mid", color=color, alpha=0.3, lw=0)
        ax.step(xc, mid, where="mid", color=color, lw=0.9, **kw)
    else:
        ax.fill_between(xc, lo, hi, color=color, alpha=0.3, lw=0)
        ax.plot(xc, mid, color=color, lw=0.9, **kw)


def main(path):
    r = sc.load(path)
    x, xd, dt = r["x"], r["xd"], r["dt"]
    lsb = float(r.get("vdiv", 0.5)) / 25  # 8-bit step for this run's V/div
    sx, sxd = sc.scope_like_sim("simple", x, xd, dt_s=dt, lsb=lsb)
    D = run_stats(x, xd, dt)
    S = run_stats(sx, sxd, dt)

    print(f"{r['name']}: {len(x)} captures; simulation: {len(sx)} windows (scope-like)")
    print(f"{'quantity':22s} {'measured':>18s} {'simulation':>18s}  ratio")
    for i, nm in enumerate(D["names"]):
        dv, de = D["val"][i], D["boot"][:, i].std()
        sv, se = S["val"][i], S["boot"][:, i].std()
        print(
            f"{nm:22s} {dv:9.4f} +- {de:6.4f} {sv:9.4f} +- {se:6.4f}  {dv / sv if sv else np.nan:6.3f}"
        )
    for lab, d in (("measured", D), ("simulation", S)):
        dw = d["dwell"] * 1e3
        print(
            f"dwell r<{R_DWELL} V ({lab}): n={len(dw)}, median {np.median(dw):.2f} ms, "
            f"90% {np.percentile(dw, 90):.2f} ms, max {dw.max():.2f} ms"
        )

    fig, ax = plt.subplots(3, 2, figsize=(7.0, 7.6))
    a, b, c, d, e, g = ax.ravel()
    for S_or_D, col, lab in ((S, sc.C_SIM, "simulation"), (D, sc.C_DATA, "measured")):
        band(
            a,
            0.5 * (S_or_D["bx"][1:] + S_or_D["bx"][:-1]),
            S_or_D["hx_b"],
            col,
            step=True,
            label=lab,
        )
        band(
            b,
            0.5 * (S_or_D["by"][1:] + S_or_D["by"][:-1]),
            S_or_D["hy_b"],
            col,
            step=True,
            label=lab,
        )
        db = 10 * np.log10(S_or_D["p_b"] / np.median(S_or_D["p_b"], axis=0).max())
        band(c, S_or_D["f"], db, col, label=lab)
        band(d, RADII, S_or_D["frac_b"], col, label=lab)
        dw = np.sort(S_or_D["dwell"]) * 1e3
        e.step(
            dw,
            1 - np.arange(len(dw)) / len(dw),
            where="post",
            color=col,
            lw=0.9,
            label=lab,
        )
    a.set_xlabel("x (V)")
    a.set_ylabel("Density (1/V)")
    b.set_xlabel("x′ (V)")
    b.set_ylabel("Density (1/V)")
    a.legend(frameon=False, loc="upper right", fontsize=6)
    f_lin = 0.792552 / (2 * np.pi) / sc.TAU
    c.axvline(f_lin, color="grey", ls="--", lw=0.8)
    c.set_xscale("log")
    c.set_xlim(10, 2000)
    c.set_ylim(-50, 3)
    c.set_xlabel("Frequency (Hz)")
    c.set_ylabel("Power of x (dB)")
    d.axvline(R_DWELL, color="grey", ls="--", lw=0.8)
    d.set_xlabel("Radius r in (x, x′) (V)")
    d.set_ylabel("Fraction of time within r")
    e.set_yscale("log")
    e.set_xlabel(f"Dwell time inside r = {R_DWELL} V (ms)")
    e.set_ylabel("Fraction of dwells longer")
    # (f): per-capture dominant frequency
    fpk_d = np.array([sc.peak_freq(D["f"], pp) for pp in D["p"]])
    fpk_s = np.array([sc.peak_freq(S["f"], pp) for pp in S["p"]])
    bins = np.arange(40, 181, 5)
    g.hist(fpk_s, bins, density=True, histtype="step", color=sc.C_SIM)
    g.hist(fpk_d, bins, density=True, histtype="step", color=sc.C_DATA)
    g.axvline(f_lin, color="grey", ls="--", lw=0.8)
    g.set_xlabel("Dominant frequency per capture (Hz)")
    g.set_ylabel("Density (1/Hz)")
    for axx, lab in zip(ax.ravel(), "abcdef"):
        sc.panel(axx, lab, right=lab in "ce")
    fig.tight_layout()
    tag = "" if Path(path) == sc.MAIN_RUN else "_" + r["name"].replace("sprott_", "")
    sc.save(fig, "fig_stats" + tag)
    print(
        f"per-capture dominant f: measured {fpk_d.mean():.1f} (sd {fpk_d.std():.1f}) Hz, "
        f"sim {fpk_s.mean():.1f} (sd {fpk_s.std():.1f}) Hz; linear theory {f_lin:.1f} Hz"
    )

    def pack(Z):
        db = 10 * np.log10(
            np.median(Z["p_b"], axis=0) / np.median(Z["p_b"], axis=0).max()
        )
        return dict(
            val=dict(zip(Z["names"], Z["val"])),
            err=dict(zip(Z["names"], Z["boot"].std(axis=0))),
            f=Z["f"],
            db=db,
            db_lo=10
            * np.log10(
                np.percentile(Z["p_b"], 16, axis=0) / np.median(Z["p_b"], axis=0).max()
            ),
            db_hi=10
            * np.log10(
                np.percentile(Z["p_b"], 84, axis=0) / np.median(Z["p_b"], axis=0).max()
            ),
            frac=np.median(Z["frac_b"], axis=0),
            dwell_ms=np.sort(Z["dwell"]) * 1e3,
        )

    sc.save_results(
        "stats" + tag,
        dict(
            run=r["name"],
            radii=RADII,
            r_dwell=R_DWELL,
            f_lin=f_lin,
            data=pack(D),
            model=pack(S),
            fpk_data=fpk_d,
            fpk_model=fpk_s,
        ),
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else sc.MAIN_RUN)
