"""Data vs the 1N4148 model at the measured part values. No fitted parameters.

Models (see sprott_common): 'simple' (R = 9883, A = 0.982, g = 1) and 'exact' (R1, A = 0.982, Ch2 gain
g = 1.025 from R6/R5 and R3C3/tau). Both get the scope's noise and 20 mV steps added.
(a) measured density of (x, x')   (b) model density, same scale
(c) Q-Q of x: data quantile vs model quantile   (d) same for x'
(e) size ratio data/model of the central q..1-q range against q (how "big" depends on where you look)
(f) Wasserstein-1 distances between x (and x') distributions: data vs model, compared with
    split-half and run-to-run distances in the data
Writes the table of numbers to stdout (logs/fig_compare.log).
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.stats import wasserstein_distance

import sprott_common as sc

QS = np.array([0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25])
QQ = np.linspace(0.001, 0.999, 300)


def split_w1(v, n_rep=100, seed=0):
    rng = np.random.default_rng(seed)
    h = len(v) // 2
    out = []
    for _ in range(n_rep):
        k = rng.permutation(len(v))
        out.append(wasserstein_distance(v[k[:h]].ravel(), v[k[h:]].ravel()))
    return np.array(out)


def boot_w1(a, b, n_rep=100, seed=0):
    rng = np.random.default_rng(seed)
    return np.array(
        [
            wasserstein_distance(
                a[rng.integers(0, len(a), len(a))].ravel(),
                b[rng.integers(0, len(b), len(b))].ravel(),
            )
            for _ in range(n_rep)
        ]
    )


def hellinger2d(a, ad, b, bd, edges):
    ha = np.histogram2d(a.ravel(), ad.ravel(), edges)[0]
    hb = np.histogram2d(b.ravel(), bd.ravel(), edges)[0]
    pa, pb = ha / ha.sum(), hb / hb.sum()
    return np.sqrt(0.5 * np.sum((np.sqrt(pa) - np.sqrt(pb)) ** 2))


def local_extrema(v, sign, thresh):
    """Values of local maxima (sign=+1) or minima (sign=-1) of a smoothed record beyond thresh."""
    from scipy.signal import find_peaks

    out = []
    for row in v:
        s = sign * sc.boxcar(row, sc.SMOOTH)
        idx, _ = find_peaks(s, height=thresh, prominence=0.05, distance=sc.SMOOTH)
        idx = idx[(idx > sc.SMOOTH) & (idx < len(s) - sc.SMOOTH)]
        out.append(sign * s[idx])
    return np.concatenate(out)


def main():
    r = sc.load(sc.MAIN_RUN)
    x, xd = r["x"], r["xd"]
    models = {m: sc.scope_like_sim(m, x, xd) for m in ("simple", "exact")}
    reps = [sc.load(p) for p in sc.runs_matching("sprott_R4_10030_rep*.h5")]

    # ---- size table
    print(
        f"{'quantity':34s} {'data':>9s} {'simple':>9s} {'exact':>9s} {'d/simple':>9s} {'d/exact':>9s}"
    )

    size = {}

    def row(nm, f):
        dv, sv, ev = f(x, xd), f(*models["simple"]), f(*models["exact"])
        size[nm] = dict(data=dv, simple=sv, exact=ev)
        print(f"{nm:34s} {dv:9.4f} {sv:9.4f} {ev:9.4f} {dv / sv:9.3f} {dv / ev:9.3f}")

    row("x min", lambda a, b: a.min())
    row("x max", lambda a, b: a.max())
    row("x' min", lambda a, b: b.min())
    row("x' max", lambda a, b: b.max())
    row("x range (max - min)", lambda a, b: np.ptp(a))
    row("x' range (max - min)", lambda a, b: np.ptp(b))
    for q in (0.001, 0.01, 0.05):
        row(
            f"x range {q * 100:g}-{100 - q * 100:g} %",
            lambda a, b, q=q: np.diff(np.quantile(a, [q, 1 - q]))[0],
        )
        row(
            f"x' range {q * 100:g}-{100 - q * 100:g} %",
            lambda a, b, q=q: np.diff(np.quantile(b, [q, 1 - q]))[0],
        )
    row("rms x", lambda a, b: a.std())
    row("rms x'", lambda a, b: b.std())
    row("mean x  (= -<R I_D> + offsets)", lambda a, b: a.mean())
    row("mean x'", lambda a, b: b.mean())
    row(
        "median x minimum (< -0.5 V)", lambda a, b: np.median(local_extrema(a, -1, 0.5))
    )
    row(
        "x minima < -0.5 V per 100 ms",
        lambda a, b: local_extrema(a, -1, 0.5).size / len(a),
    )
    row(
        "median x' maximum (> 0.25 V)",
        lambda a, b: np.median(
            np.concatenate([sc.xd_maxima(u, v)[2] for u, v in zip(a, b)])
        ),
    )
    row(
        "99 % of x' maxima (> 0.25 V)",
        lambda a, b: np.percentile(
            np.concatenate([sc.xd_maxima(u, v)[2] for u, v in zip(a, b)]), 99
        ),
    )
    mean_id = -x.mean() / sc.R1
    print(
        f"mean diode current from <x> (offsets ignored): data {mean_id * 1e6:.1f} uA, "
        f"simple model {-models['simple'][0].mean() / sc.R_MEAN * 1e6:.1f} uA"
    )

    # ---- distances
    sw_x, sw_y = split_w1(x), split_w1(xd)
    print(
        f"W1 split-half in the data: x {np.median(sw_x) * 1e3:.1f} mV (95% {np.percentile(sw_x, 95) * 1e3:.1f}), "
        f"x' {np.median(sw_y) * 1e3:.1f} mV (95% {np.percentile(sw_y, 95) * 1e3:.1f})"
    )
    dist = {"split half": (sw_x, sw_y)}
    for rr in reps:
        nm = rr["name"].split("_")[3]
        dist[f"main vs {nm}"] = (boot_w1(x, rr["x"]), boot_w1(xd, rr["xd"]))
    for m, (mx, mxd) in models.items():
        dist[f"data vs {m}"] = (boot_w1(x, mx), boot_w1(xd, mxd))
    for k, (a, b) in dist.items():
        print(
            f"W1 {k:20s}: x {np.median(a) * 1e3:6.1f} +- {a.std() * 1e3:4.1f} mV, "
            f"x' {np.median(b) * 1e3:6.1f} +- {b.std() * 1e3:4.1f} mV"
        )
    edges = [np.arange(-2.0, 1.2, 0.04) + 0.01, np.arange(-1.2, 1.0, 0.04) + 0.01]
    hd = {"split half": hellinger2d(x[::2], xd[::2], x[1::2], xd[1::2], edges)}
    for rr in reps:
        hd[f"main vs {rr['name'].split('_')[3]}"] = hellinger2d(
            x, xd, rr["x"], rr["xd"], edges
        )
    for m, (mx, mxd) in models.items():
        hd[f"data vs {m}"] = hellinger2d(x, xd, mx, mxd, edges)
    for k, v in hd.items():
        print(f"Hellinger 2-D (x, x') {k:20s}: {v:.3f}")

    # ---- figure
    sx, sxd = models["simple"]
    fig, ax = plt.subplots(3, 2, figsize=(7.0, 8.0), layout="constrained")
    a, b, c, d, e, f = ax.ravel()
    hd_ = np.histogram2d(x.ravel(), xd.ravel(), edges, density=True)[0]
    hs_ = np.histogram2d(sx.ravel(), sxd.ravel(), edges, density=True)[0]
    vmax = max(hd_.max(), hs_.max())
    norm = LogNorm(vmin=vmax * 1e-4, vmax=vmax)
    for axx, h in ((a, hd_), (b, hs_)):
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
    fig.colorbar(im, ax=[a, b], label="Density (1/V²)", pad=0.02)

    for axx, dv, mv, lab in (
        (c, x, (sx, models["exact"][0]), "x"),
        (d, xd, (sxd, models["exact"][1]), "x′"),
    ):
        qd = np.quantile(dv, QQ)
        for mm, col, nm in zip(mv, (sc.C_SIM, sc.C_SIM2), ("simple", "exact")):
            axx.plot(np.quantile(mm, QQ), qd, color=col, lw=1.0, label=nm)
        lo, hi = qd.min() - 0.1, qd.max() + 0.1
        axx.plot([lo, hi], [lo, hi], color="grey", ls="--", lw=0.6)
        axx.set_xlabel(f"Model quantile of {lab} (V)")
        axx.set_ylabel(f"Measured quantile of {lab} (V)")
    c.legend(frameon=False, loc="lower right", fontsize=6)

    for v, m, mk, nm in ((x, sx, "o", "x"), (xd, sxd, "s", "x′")):
        ratio = [
            np.diff(np.quantile(v, [q, 1 - q]))[0]
            / np.diff(np.quantile(m, [q, 1 - q]))[0]
            for q in QS
        ]
        e.plot(
            QS * 100,
            ratio,
            mk + "-",
            color=sc.C_DATA,
            ms=3,
            lw=0.8,
            mfc="none" if nm == "x′" else None,
            label=nm,
        )
    e.axhline(1, color="grey", ls="--", lw=0.6)
    e.set_xscale("log")
    e.set_xlabel("Tail fraction q (%)")
    e.set_ylabel("Data / model width of q..(1−q) range")
    e.legend(frameon=False, loc="lower left")

    names = list(dist)
    yy = np.arange(len(names))
    for k, (mk, off, nm) in enumerate((("o", -0.12, "x"), ("s", 0.12, "x′"))):
        med = [np.median(dist[n][k]) * 1e3 for n in names]
        err = [dist[n][k].std() * 1e3 for n in names]
        f.errorbar(
            med,
            yy + off,
            xerr=err,
            fmt=mk,
            color=sc.C_DATA,
            ms=3,
            capsize=1.5,
            mfc="none" if k else None,
            label=nm,
        )
    f.set_yticks(yy, names, fontsize=6)
    f.minorticks_off()
    f.set_xscale("log")
    f.invert_yaxis()
    f.set_xlabel("W1 distance (mV)")
    f.legend(frameon=False, loc="upper center", fontsize=6)
    for axx, lab in zip(ax.ravel(), "abcdef"):
        sc.panel(axx, lab, right=lab in "ef")
    sc.save(fig, "fig_compare")
    sc.save_results("compare", dict(
        size=size, mean_diode_uA=dict(data=mean_id * 1e6, simple=-models["simple"][0].mean() / sc.R_MEAN * 1e6),
        w1={k: dict(x=np.median(a), x_err=a.std(), xd=np.median(b), xd_err=b.std()) for k, (a, b) in dist.items()},
        hellinger=hd))


if __name__ == "__main__":
    main()
