"""Where the board sits in the model: bifurcation diagram and a multi-start test, with no change to the model.

Model as everywhere else ('simple': R = R_MEAN, 1N4148 SPICE values, g = 1), only A varied, which is what
changing R4 does (A = tau / (R4 C1)). Time in units of tau. Fixed-step RK4 (dt = 0.004 tau), all trajectories
integrated together; checked against the cached LSODA simulation at A = 0.982.
Each trajectory: 1500 tau transient, then 500 tau recorded (x' maxima, rms x, time within 0.2 V of the origin).
(a) x' maxima against A over the R4-sweep range, 6 starting points per A
(b) zoom near the board's A = 0.982 (periodic window at A = 0.953-0.956)
(c) x' maxima measured on the board in its three states at A = 0.982 (A: lingering, B: model-like, C: period 3)
(d) 600 random starting points at A = 0.982: rms x against time near the origin, with the board's three states
"""

import numpy as np
import matplotlib.pyplot as plt

import sprott_common as sc

DT = 0.004
T_TRANS, T_REC = 1500.0, 500.0
RIS = sc.R_MEAN * sc.IS
NVT = sc.NID * sc.VT
R0 = 0.2
A_WIDE = np.round(np.arange(0.45, 1.3501, 0.005), 4)
A_FINE = np.round(np.arange(0.94, 1.0201, 0.0005), 5)
N_IC = 6
N_MULTI = 600
SWEEP_R4 = [20e3, 15e3, 12e3, 11e3, 10.03e3, 9.1e3, 8.2e3, 7.5e3]  # planned R4 sweep
STATES = {  # board runs at A = 0.982, same parts
    "A (lingering)": "sprott_R4_10030_20261119_134608.h5",
    "B (D1 reseated)": "sprott_after_swaps_20261119_181028.h5",
    "C (reseated again)": "sprott_D1_reseated2_20261119_183303.h5",
}
STATE_COL = {
    "A (lingering)": sc.C_DATA,
    "B (D1 reseated)": "#5d6d7e",
    "C (reseated again)": "#2a9d8f",
}
CACHE = sc.CACHE / "model_scan.npz"


def integrate(A, s0):
    """RK4 for many trajectories at once. A: (N,), s0: (3, N).
    Returns per-trajectory rms x, near-origin time fraction, divergence flag, and the x' maxima
    (trajectory index, height) found in the recorded part."""
    s = s0.astype(float).copy()
    bad = np.zeros(A.size, bool)

    def f(s):
        x, y, z = s
        return np.array([y, z, -A * z - x - RIS * np.expm1(np.minimum(y / NVT, 60.0))])

    def step(s):
        k1 = f(s)
        k2 = f(s + 0.5 * DT * k1)
        k3 = f(s + 0.5 * DT * k2)
        k4 = f(s + DT * k3)
        return s + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def check(s):
        b = ~np.isfinite(s).all(0) | (np.abs(s).max(0) > 50)
        bad[b] = True
        s[:, b] = 0.0
        return s

    for k in range(int(T_TRANS / DT)):
        s = step(s)
        if k % 2000 == 0:
            s = check(s)
    n_rec = int(T_REC / DT)
    sx = np.zeros(A.size)
    sx2 = np.zeros(A.size)
    near = np.zeros(A.size)
    y2, y1 = s[1].copy(), s[1].copy()
    mi, mh = [], []
    for k in range(n_rec):
        s = step(s)
        x, y = s[0], s[1]
        sx += x
        sx2 += x * x
        near += np.hypot(x, y) < R0
        pk = (y1 > y2) & (y1 >= y) & (y1 > 0.005)
        if pk.any():
            j = np.nonzero(pk)[0]
            mi.append(j)
            mh.append(y1[j])
        y2, y1 = y1, y.copy()
        if k % 2000 == 0:
            s = check(s)
    mean = sx / n_rec
    rms = np.sqrt(np.maximum(sx2 / n_rec - mean**2, 0))
    return dict(
        rms=rms,
        near=near / n_rec,
        bad=bad,
        mi=np.concatenate(mi) if mi else np.array([], int),
        mh=np.concatenate(mh) if mh else np.array([]),
    )


def period_of(h, tol=0.002, last=300):
    """Number of distinct x' maxima heights among the last `last` maxima (clusters closer than tol).
    A period-p orbit gives p (or a few more at rounding edges); chaos gives many."""
    h = np.sort(h[-last:])
    if h.size < 10:
        return 0
    return int(1 + np.sum(np.diff(h) > tol))


def starts(n, rng):
    return np.vstack(
        [
            rng.uniform(-1.5, 0.7, n),
            rng.uniform(-1.0, 0.6, n),
            rng.uniform(-1.0, 1.0, n),
        ]
    )


def run():
    if CACHE.exists():
        return dict(np.load(CACHE, allow_pickle=True))
    rng = np.random.default_rng(3)
    a_all = np.concatenate([A_WIDE, A_FINE])
    A = np.repeat(a_all, N_IC)
    r = integrate(A, starts(A.size, rng))
    Am = np.full(N_MULTI, sc.A_EFF)
    m = integrate(Am, starts(N_MULTI, rng))
    out = dict(
        A=A,
        rms=r["rms"],
        near=r["near"],
        bad=r["bad"],
        mi=r["mi"],
        mh=r["mh"],
        m_rms=m["rms"],
        m_near=m["near"],
        m_bad=m["bad"],
        m_mi=m["mi"],
        m_mh=m["mh"],
    )
    np.savez_compressed(CACHE, **out)
    return out


def classify(mi, mh, n):
    per = np.zeros(n, int)
    order = np.argsort(mi, kind="stable")
    mi, mh = mi[order], mh[order]
    cuts = np.searchsorted(mi, np.arange(n + 1))
    for j in range(n):
        per[j] = period_of(mh[cuts[j] : cuts[j + 1]])
    return per


def data_maxima(name):
    r = sc.load(sc.DATA / name)
    ys = np.concatenate(
        [sc.xd_maxima(a, b, hmin=0.01, prom=0.02)[2] for a, b in zip(r["x"], r["xd"])]
    )
    sm = (
        [sc.boxcar(c, sc.SMOOTH) for c in r["x"]],
        [sc.boxcar(c, sc.SMOOTH) for c in r["xd"]],
    )
    near = np.mean(
        [np.mean(np.hypot(a, b)[sc.SMOOTH : -sc.SMOOTH] < R0) for a, b in zip(*sm)]
    )
    return ys, float(r["x"].std()), float(near)


def main():
    # check the integrator against the cached LSODA run at the board's A
    xs, _ = sc.sim_windows("simple")
    d = run()
    A, n = d["A"], d["A"].size
    per = classify(d["mi"], d["mh"], n)
    m_per = classify(d["m_mi"], d["m_mh"], N_MULTI)
    ok = ~d["m_bad"]
    chaotic_m = ok & (m_per > 20)
    print(
        f"check at A = {sc.A_EFF:.4f}: LSODA rms x {xs.std():.4f} V; RK4 multi-start (chaotic ones) "
        f"rms x {np.median(d['m_rms'][chaotic_m]):.4f} V, time within {R0} V "
        f"{np.median(d['m_near'][chaotic_m]):.3f}"
    )

    # multi-start summary
    cls = {}
    for j in np.nonzero(ok)[0]:
        key = "chaotic" if m_per[j] > 20 else f"period {m_per[j]}"
        cls.setdefault(key, []).append(j)
    print(
        f"multi-start at A = {sc.A_EFF:.4f}: {N_MULTI} starts, {int(d['m_bad'].sum())} diverged"
    )
    for k, js in sorted(cls.items()):
        js = np.array(js)
        print(
            f"  {k:10s}: {len(js):4d} starts, rms x {d['m_rms'][js].mean():.3f} "
            f"(range {d['m_rms'][js].min():.3f}-{d['m_rms'][js].max():.3f}), "
            f"time within {R0} V {d['m_near'][js].mean():.3f}"
        )

    # classification along A
    tab = []
    for a in np.unique(A):
        j = np.nonzero((A == a) & ~d["bad"])[0]
        ps = per[j]
        lab = sorted(set("chaos" if p > 20 else f"P{p}" for p in ps))
        tab.append((float(a), lab, float(d["rms"][j].mean()) if len(j) else np.nan))
    print(
        "\nmodel near the board's A (6 starts each; P<n> = period n, by distinct x' maxima):"
    )
    for a, lab, rms in tab:
        if 0.955 <= a <= 1.005:
            print(f"  A = {a:.4f}: {', '.join(lab):20s} rms x {rms:.3f}")
    print("\nplanned R4 sweep:")
    sweep = []
    for r4 in SWEEP_R4:
        a = sc.TAU / (r4 * sc.C1)
        k = np.argmin(np.abs(np.unique(A) - a))
        a_near, lab, rms = tab[k]
        sweep.append(dict(R4=r4, A=a, model=lab, rms=rms))
        print(
            f"  R4 = {r4 / 1e3:6.2f} k: A = {a:.3f} (grid {a_near:.4f}) -> {', '.join(lab)}"
        )

    states = {k: data_maxima(v) for k, v in STATES.items()}

    # ------------------------------------------------------------------ figure
    fig = plt.figure(figsize=(7.2, 6.6), layout="constrained")
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 0.45])
    a_ax = fig.add_subplot(gs[0, :])
    b_ax = fig.add_subplot(gs[1, 0])
    c_ax = fig.add_subplot(gs[1, 1], sharey=b_ax)
    d_ax = fig.add_subplot(gs[1, 2])

    am = A[d["mi"]]
    keep = ~d["bad"][d["mi"]]
    for axx, lo, hi in ((a_ax, A_WIDE[0], A_WIDE[-1]), (b_ax, 0.94, 1.01)):
        sel = keep & (am >= lo - 1e-9) & (am <= hi + 1e-9)
        if axx is a_ax:  # wide panel: coarse grid only
            sel &= np.isin(am, A_WIDE)
        axx.plot(
            am[sel],
            d["mh"][sel],
            ".",
            ms=0.8,
            mew=0,
            color=sc.C_SIM,
            alpha=0.4,
            rasterized=True,
        )
        axx.axvline(sc.A_EFF, color="grey", ls="--", lw=0.8)
        axx.set_xlim(lo, hi)
        axx.set_xlabel("A = τ / (R4 C1)")
    a_ax.set_ylabel("x′ maxima (V)")
    b_ax.set_ylabel("x′ maxima (V)")
    for row in sweep:
        a_ax.plot(row["A"], -0.03, "^", color="grey", ms=3, clip_on=False)
    a_ax.set_ylim(-0.05, None)

    bins = np.arange(0.0, 0.75, 0.005)
    for k, (ys, _, _) in states.items():
        h, e = np.histogram(ys, bins, density=True)
        c_ax.plot(h, 0.5 * (e[1:] + e[:-1]), color=STATE_COL[k], lw=0.9, label=k)
    c_ax.set_xlabel("Density (1/V)")
    plt.setp(c_ax.get_yticklabels(), visible=False)
    c_ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    b_ax.set_ylim(0.0, 0.72)

    for key, js in sorted(cls.items()):
        js = np.array(js)
        d_ax.plot(
            d["m_near"][js],
            d["m_rms"][js],
            "o",
            ms=2.5,
            mfc="none",
            color=sc.C_SIM if key == "chaotic" else "#e08e45",
            label=f"model {key} ({len(js)})",
        )
    for k, (_, rms, near) in states.items():
        d_ax.plot(near, rms, "s", color=STATE_COL[k], ms=5)
    d_ax.set_xlabel("Time within 0.2 V")
    d_ax.set_ylabel("rms x (V)")
    d_ax.legend(frameon=False, fontsize=6, loc="center right")

    for axx, lab in zip((a_ax, b_ax, c_ax, d_ax), "abcd"):
        sc.panel(axx, lab, right=lab in "cd")
    sc.save(fig, "fig_model_scan")
    sc.save_results(
        "model_scan",
        dict(
            near_board=[
                dict(A=a, model=lab, rms=r) for a, lab, r in tab if 0.94 <= a <= 1.02
            ],
            sweep=sweep,
            multistart={k: len(v) for k, v in cls.items()},
            multistart_diverged=int(d["m_bad"].sum()),
            states={k: dict(rms=v[1], near=v[2]) for k, v in states.items()},
        ),
    )


if __name__ == "__main__":
    main()
