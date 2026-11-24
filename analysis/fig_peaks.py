"""Peak picking for the Poincare section: old vs new method, and how stable the result is.

(a) old method (5-point mean, prominence only, raw sample values): low false peaks and 20 mV stripes
(b) new method (9-point mean, x' max >= 0.25 V, parabolic sub-sample peak), with the simulation
    put through the same steps (scope-like: measured noise + 20 mV steps)
(c) heights of all x' maxima with no height cut: the data's low peaks vs the simulation's gap
(d) maxima per capture against the height cut, for several smoothing windows
(e) median x' maximum and median x at the maximum against the smoothing window
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

import sprott_common as sc

RUN = sc.MAIN_RUN
SMOOTHS = (3, 5, 9, 13, 17)
HMINS = np.arange(0.0, 0.46, 0.025)


def old_maxima(x, xd, smooth=5, prom=0.05):
    ys = np.convolve(xd, np.ones(smooth) / smooth, mode="same")
    idx, _ = find_peaks(ys, prominence=prom, distance=smooth)
    idx = idx[(idx > smooth) & (idx < len(xd) - smooth)]
    return x[idx], xd[idx]


def cat(pk, i):
    return np.concatenate([p[i] for p in pk])


def main():
    r = sc.load(RUN)
    x, xd = r["x"], r["xd"]
    n = len(x)
    sx, sxd = sc.scope_like_sim("simple", x, xd)
    cx, cxd = sc.sim_windows("simple")  # clean simulation
    ns = len(sx)

    old = [old_maxima(a, b) for a, b in zip(x, xd)]
    ox, oy = cat(old, 0), cat(old, 1)
    new = sc.all_maxima(x, xd)
    nx, ny = cat(new, 1), cat(new, 2)
    snew = sc.all_maxima(sx, sxd)
    snx, sny = cat(snew, 1), cat(snew, 2)
    cl = sc.all_maxima(cx, cxd, smooth=1, hmin=None, prom=0.01)
    clx, cly = cat(cl, 1), cat(cl, 2)

    # no-height-cut heights, 9-point mean
    nocut = cat(sc.all_maxima(x, xd, hmin=None), 2)
    snocut = cat(sc.all_maxima(sx, sxd, hmin=None), 2)

    # stability grid
    per_cap = np.zeros((len(SMOOTHS), len(HMINS)))
    sim_per_cap = np.zeros_like(per_cap)
    med = {}
    for i, w in enumerate(SMOOTHS):
        hts = [sc.xd_maxima(a, b, smooth=w, hmin=None)[2] for a, b in zip(x, xd)]
        shts = [sc.xd_maxima(a, b, smooth=w, hmin=None)[2] for a, b in zip(sx, sxd)]
        allh, sallh = np.concatenate(hts), np.concatenate(shts)
        per_cap[i] = [(allh >= h).sum() / n for h in HMINS]
        sim_per_cap[i] = [(sallh >= h).sum() / ns for h in HMINS]
        pk = sc.all_maxima(x, xd, smooth=w)
        ys_, xs_ = cat(pk, 2), cat(pk, 1)
        cuts = np.cumsum([0] + [len(p[2]) for p in pk])

        def stat(idx, ys_=ys_, xs_=xs_, cuts=cuts):
            sel = np.concatenate([np.arange(cuts[k], cuts[k + 1]) for k in idx])
            return [np.median(ys_[sel]), np.median(xs_[sel])]

        b = sc.bootstrap(stat, n, n_boot=300)
        spk = sc.all_maxima(sx, sxd, smooth=w)
        med[w] = (np.median(ys_), np.median(xs_), b.std(axis=0), np.median(cat(spk, 2)), np.median(cat(spk, 1)))

    # ---- figure
    fig = plt.figure(figsize=(7.0, 6.6))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.15, 1, 1])
    a = fig.add_subplot(gs[0, 0])
    b = fig.add_subplot(gs[0, 1], sharex=a, sharey=a)
    c = fig.add_subplot(gs[1, 0])
    d = fig.add_subplot(gs[1, 1])
    e = fig.add_subplot(gs[2, 0])
    f = fig.add_subplot(gs[2, 1])

    a.plot(ox, oy, ".", color=sc.C_DATA, ms=1.2, alpha=0.5)
    a.set_xlabel("x at x′ maximum (V)")
    a.set_ylabel("x′ maximum (V)")
    b.plot(clx, cly, ".", color=sc.C_SIM, ms=1.0, alpha=0.4, label="simulation")
    b.plot(nx, ny, ".", color=sc.C_DATA, ms=1.2, alpha=0.5, label="measured")
    b.set_xlabel("x at x′ maximum (V)")
    b.legend(frameon=False, loc="lower left", markerscale=5)
    plt.setp(b.get_yticklabels(), visible=False)

    bins = np.arange(-0.05, 0.95, 0.02)
    c.hist(
        nocut,
        bins=bins,
        histtype="step",
        color=sc.C_DATA,
        density=True,
        label="measured",
    )
    c.hist(
        snocut,
        bins=bins,
        histtype="step",
        color=sc.C_SIM,
        density=True,
        label="sim. + noise, steps",
    )
    c.axvline(sc.HMIN, color="grey", ls="--", lw=0.8)
    c.set_yscale("log")
    c.set_xlabel("x′ maximum (V), no height cut")
    c.set_ylabel("Density (1/V)")
    c.legend(frameon=False, loc="lower right", fontsize=6)

    cols = sc.CMAP(np.linspace(0.3, 0.95, len(SMOOTHS)))
    for i, w in enumerate(SMOOTHS):
        d.plot(HMINS, per_cap[i], color=cols[i], lw=1.0, label=f"{w} pts")
    d.plot(
        HMINS,
        sim_per_cap[SMOOTHS.index(sc.SMOOTH)],
        color=sc.C_SIM,
        ls=":",
        lw=1.2,
        label="sim., 9 pts",
    )
    d.axvline(sc.HMIN, color="grey", ls="--", lw=0.8)
    d.set_xlabel("Height cut (V)")
    d.set_ylabel("Maxima per capture")
    d.legend(frameon=False, loc="upper right", fontsize=6, ncol=2)

    ws = np.array(SMOOTHS)
    my = np.array([med[w][0] for w in SMOOTHS])
    mx = np.array([med[w][1] for w in SMOOTHS])
    ey = np.array([med[w][2][0] for w in SMOOTHS])
    ex = np.array([med[w][2][1] for w in SMOOTHS])
    e.errorbar(ws, my, ey, fmt="o", color=sc.C_DATA, ms=3, capsize=2)
    e.plot(ws, [med[w][3] for w in SMOOTHS], "o", mfc="none", color=sc.C_SIM, ms=3, label="sim. + noise, steps")
    e.set_xlabel("Smoothing window (samples)")
    e.set_ylabel("Median x′ maximum (V)")
    f.errorbar(ws, mx, ex, fmt="s", color=sc.C_DATA, ms=3, capsize=2, label="measured")
    f.plot(ws, [med[w][4] for w in SMOOTHS], "s", mfc="none", color=sc.C_SIM, ms=3, label="sim. + noise, steps")
    f.legend(frameon=False, loc="center", fontsize=6)
    f.set_xlabel("Smoothing window (samples)")
    f.set_ylabel("Median x at x′ maximum (V)")

    for ax, lab in zip((a, b, c, d, e, f), "abcdef"):
        sc.panel(ax, lab, right=lab in "abef")
    fig.tight_layout()
    sc.save(fig, "fig_peaks")
    sc.save_results("peaks", dict(
        old_per_capture=len(oy) / n, old_frac_low=np.mean(oy < 0.25), new_per_capture=len(ny) / n,
        sim_per_capture=len(sny) / ns, data_frac_low_nocut=np.mean(nocut < 0.25),
        sim_frac_low_nocut=np.mean(snocut < 0.25), clean_sim_min=cly.min(),
        per_cap_range_at_cut=[per_cap[:, np.argmin(abs(HMINS - sc.HMIN))].min(),
                              per_cap[:, np.argmin(abs(HMINS - sc.HMIN))].max()],
        median_xdmax={w: med[w][0] for w in SMOOTHS}, median_xdmax_sim={w: med[w][3] for w in SMOOTHS},
        px=nx, py=ny, sim_px=clx, sim_py=cly))

    # ---- numbers for the logbook
    print(
        f"old method: {len(oy) / n:.2f} maxima/capture, {np.mean(oy < 0.25) * 100:.1f}% below 0.25 V "
        f"({(oy < 0.25).sum() / n:.1f} per capture)"
    )
    print(
        f"new method: {len(ny) / n:.2f} maxima/capture; scope-like sim {len(sny) / ns:.2f}; "
        f"clean sim {len(cly) / len(cx):.2f}"
    )
    print(
        f"clean sim: lowest x' max {cly.min():.3f} V; 1st percentile {np.percentile(cly, 1):.3f} V"
    )
    print(
        f"scope-like sim, no cut: {np.mean(snocut < 0.25) * 100:.1f}% of maxima below 0.25 V; "
        f"data {np.mean(nocut < 0.25) * 100:.1f}%"
    )
    i9 = SMOOTHS.index(sc.SMOOTH)
    for h in (0.15, 0.2, 0.25, 0.3, 0.35):
        j = np.argmin(abs(HMINS - h))
        print(
            f"  cut {HMINS[j]:.3f} V: data {per_cap[i9, j]:.2f}/capture, sim {sim_per_cap[i9, j]:.2f}/capture; "
            f"across smoothing {per_cap[:, j].min():.2f}-{per_cap[:, j].max():.2f}"
        )
    for w in SMOOTHS:
        print(
            f"  smooth {w:2d}: median x' max {med[w][0]:.3f} +- {med[w][2][0]:.3f} V (sim {med[w][3]:.3f}), "
            f"median x at max {med[w][1]:+.3f} +- {med[w][2][1]:.3f} V (sim {med[w][4]:+.3f})"
        )
    print(
        f"clean sim medians: x' max {np.median(cly):.3f} V, x at max {np.median(clx):+.3f} V; "
        f"scope-like sim (9 pts, cut) {np.median(sny):.3f}, {np.median(snx):+.3f} V"
    )


if __name__ == "__main__":
    main()
