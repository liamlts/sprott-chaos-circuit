"""Repeatability and drift over all runs on the measured board (same parts, 10 ms/div; the 25 ms/div run is
added where the quantity does not depend on window length).

Per capture: rms of x and x', mean x and x' (offsets), dominant frequency, fraction of time within
r = 0.2 V of the origin, number of x' maxima below 0.1 V. Points: 20-capture block means with bootstrap
errors, against clock time.
(a) rms x and x'     (b) fraction of time near the origin     (c) dominant frequency
(d) mean x and x'    (e) x histograms of every run            (f) run-to-run distance of the x histogram
                                                                  vs the split-half distance inside one run
"""

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance

import sprott_common as sc

BLOCK = 20
R0 = 0.2
RUN_GLOBS = ("sprott_test_*.h5", "sprott_R4_10030_*.h5", "sprott_noisecheck_*.h5")


def per_capture(r):
    x, xd, dt = r["x"], r["xd"], r["dt"]
    f, p = sc.spectrum(x, dt)
    fpk = np.array([sc.peak_freq(f, pp) for pp in p])
    rad = [
        np.hypot(sc.boxcar(a, sc.SMOOTH), sc.boxcar(b, sc.SMOOTH))[
            sc.SMOOTH : -sc.SMOOTH
        ]
        for a, b in zip(x, xd)
    ]
    near = np.array([np.mean(rr < R0) for rr in rad])
    nlow = np.array(
        [
            np.sum(sc.xd_maxima(a, b, hmin=0.01, prom=0.02)[2] < 0.1)
            for a, b in zip(x, xd)
        ]
    )
    t0 = datetime.fromisoformat(r["started"])
    return dict(
        t=t0.timestamp() + r["tcap"],
        sx=x.std(1),
        sxd=xd.std(1),
        mx=x.mean(1),
        mxd=xd.mean(1),
        f=fpk,
        near=near,
        nlow=nlow / (x.shape[1] * dt) * 0.1,
    )  # low maxima per 100 ms


def blocks(t, v, n=BLOCK, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(0, len(v) - n + 1, n):
        vv = v[i : i + n]
        b = [vv[rng.integers(0, n, n)].mean() for _ in range(300)]
        out.append((t[i : i + n].mean(), vv.mean(), np.std(b)))
    return np.array(out)


def drift(t, v):
    """Slope per hour of a straight line through per-capture values, with a bootstrap error."""
    th = (t - t.min()) / 3600
    rng = np.random.default_rng(1)
    s = np.polyfit(th, v, 1)[0]
    bs = [
        np.polyfit(th[i], v[i], 1)[0]
        for i in (rng.integers(0, len(v), len(v)) for _ in range(500))
    ]
    return s, np.std(bs)


def split_half_w1(x, n_rep=200, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_rep):
        k = rng.permutation(len(x))
        h = len(x) // 2
        out.append(wasserstein_distance(x[k[:h]].ravel(), x[k[h:]].ravel()))
    return np.array(out)


def main():
    paths = sorted(
        {p for g in RUN_GLOBS for p in sc.runs_matching(g)},
        key=lambda p: p.stem.split("_")[-1],
    )
    paths = sorted(paths, key=lambda p: sc.load(p)["started"])
    runs = [sc.load(p) for p in paths]
    short = [r for r in runs if abs(r["tdiv"] - 0.01) < 1e-6]
    T0 = min(datetime.fromisoformat(r["started"]).timestamp() for r in runs)
    pc = {r["name"]: per_capture(r) for r in runs}
    cols = sc.CMAP(np.linspace(0.35, 0.95, len(runs)))
    lab = {}
    for r in runs:
        nm = r["name"].replace("sprott_", "").rsplit("_", 2)[0].replace("R4_10030_", "")
        lab[r["name"]] = "main" if nm == "R4_10030" else nm

    # ---- numbers
    keys = [
        ("sx", "rms x (V)"),
        ("sxd", "rms x' (V)"),
        ("mx", "mean x (V)"),
        ("mxd", "mean x' (V)"),
        ("near", f"time frac r<{R0}"),
        ("nlow", "maxima<0.1V /100ms"),
        ("f", "dominant f (Hz)"),
    ]
    print(f"{'run':16s} {'n':>4s} " + " ".join(f"{k[1]:>20s}" for k in keys))
    table = {}
    rng = np.random.default_rng(0)
    short_names = {r["name"] for r in short}
    for r in runs:
        d = pc[r["name"]]
        n = len(d["t"])
        row = []
        for k, _ in keys:
            v = d[k]
            e = np.std(
                [
                    v[i].mean()
                    for i in (
                        rng.integers(0, n, n) for _ in range(500)
                    )
                ]
            )
            row.append((v.mean(), e))
        table[r["name"]] = row
        print(
            f"{lab[r['name']]:16s} {n:4d} "
            + " ".join(f"{m:12.4f} +- {e:6.4f}" for m, e in row)
        )
    big = [r for r in short if len(r["x"]) >= 100]
    print(
        "consistency across the 200-capture 10 ms/div runs (chi2/dof of run means about their weighted mean):"
    )
    consist, drifts = {}, {}
    for j, (k, nm) in enumerate(keys):
        m = np.array([table[r["name"]][j][0] for r in big])
        e = np.array([table[r["name"]][j][1] for r in big])
        w = 1 / e**2
        mu = np.sum(w * m) / np.sum(w)
        chi = np.sum(w * (m - mu) ** 2) / max(len(m) - 1, 1)
        consist[nm] = dict(mean=mu, spread=m.max() - m.min(), chi2dof=chi, rel_spread=(m.max() - m.min()) / abs(mu))
        print(
            f"  {nm:20s} weighted mean {mu:.4f}, spread (max-min) {m.max() - m.min():.4f}, chi2/dof {chi:.2f}"
        )
    allt = np.concatenate([pc[r["name"]]["t"] for r in short])
    print(
        f"drift over all 10 ms/div captures ({(allt.max() - allt.min()) / 60:.0f} min):"
    )
    for k, nm in keys:
        v = np.concatenate([pc[r["name"]][k] for r in short])
        s, e = drift(allt, v)
        drifts[nm] = dict(slope_per_hour=s, err=e, mean=v.mean())
        print(f"  {nm:20s} slope {s:+.4f} +- {e:.4f} per hour (mean {v.mean():.4f})")

    # histogram distances
    ref = big[0] if big else short[0]
    null = split_half_w1(ref["x"])
    print(
        f"W1 distance of x histograms; split-half inside {lab[ref['name']]}: median {np.median(null) * 1e3:.1f} mV, "
        f"95% {np.percentile(null, 95) * 1e3:.1f} mV"
    )
    names = [r["name"] for r in runs]
    W = np.zeros((len(runs), len(runs)))
    for i, a in enumerate(runs):
        for j, b in enumerate(runs):
            if j > i:
                W[i, j] = W[j, i] = wasserstein_distance(a["x"].ravel(), b["x"].ravel())
    for i, nm in enumerate(names):
        print(
            f"  {lab[nm]:16s} "
            + " ".join(f"{W[i, j] * 1e3:6.1f}" for j in range(len(names)))
        )

    # ---- figure
    fig, ax = plt.subplots(3, 2, figsize=(7.0, 7.4))
    a, b, c, d, e, g = ax.ravel()
    for r, col in zip(runs, cols):
        v = pc[r["name"]]
        tm = (v["t"] - T0) / 60
        mk = "o" if r["name"] in short_names else "D"
        for axx, k, kw in (
            (a, "sx", {}),
            (a, "sxd", dict(mfc="none")),
            (b, "near", {}),
            (c, "f", {}),
            (d, "mx", {}),
            (d, "mxd", dict(mfc="none")),
        ):
            if k == "f" and r["name"] not in short_names:
                continue
            bl = blocks(tm, v[k])
            if len(bl):
                axx.errorbar(
                    bl[:, 0],
                    bl[:, 1],
                    bl[:, 2],
                    fmt=mk,
                    color=col,
                    ms=3,
                    capsize=1.5,
                    **kw,
                )
    a.set_ylabel("rms (V)")
    b.set_ylabel(f"Time fraction within r = {R0} V")
    c.set_ylabel("Dominant frequency (Hz)")
    d.set_ylabel("Mean (V)")
    for axx in (a, b, c, d):
        axx.set_xlabel("Clock time (min)")
    bx = np.arange(-2.0, 1.2, 0.04) + 0.01
    for r, col in zip(runs, cols):
        e.hist(
            r["x"].ravel(),
            bx,
            density=True,
            histtype="step",
            color=col,
            lw=0.8,
            label=lab[r["name"]],
        )
    e.set_xlabel("x (V)")
    e.set_ylabel("Density (1/V)")
    e.legend(frameon=False, fontsize=5.5, loc="upper left")
    im = g.imshow(W * 1e3, cmap=sc.CMAP, origin="upper")
    g.set_xticks(range(len(names)), [lab[n] for n in names], rotation=60, fontsize=6)
    g.set_yticks(range(len(names)), [lab[n] for n in names], fontsize=6)
    g.minorticks_off()
    fig.colorbar(im, ax=g, label="W1 distance of x (mV)")
    for axx, letter in zip(ax.ravel(), "abcdef"):
        sc.panel(axx, letter, right=letter == "e")
    fig.tight_layout()
    sc.save(fig, "fig_repeat")
    blk = {}
    for r in runs:
        v = pc[r["name"]]
        tm = (v["t"] - T0) / 60
        blk[lab[r["name"]]] = dict(short=r["name"] in short_names, sx=blocks(tm, v["sx"]), near=blocks(tm, v["near"]),
                                   f=blocks(tm, v["f"]))
    sc.save_results("repeat", dict(runs=[lab[n] for n in names], n=[len(r["x"]) for r in runs],
                                   table={lab[n]: dict(zip([k[1] for k in keys], table[n])) for n in names},
                                   consistency=consist, drift=drifts, minutes=(allt.max() - allt.min()) / 60,
                                   w1=W, w1_split=[np.median(null), np.percentile(null, 95)], blocks=blk))


if __name__ == "__main__":
    main()
