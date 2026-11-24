"""Return maps of the x' maxima and a Lyapunov exponent from the data.

(a) M(n+1) vs M(n): successive x' maxima in one capture
(b) X(n+1) vs X(n): x at those maxima
(c) divergence of neighbouring points in the section (X, M) over k returns -> lambda per return,
    divided by the mean time between maxima -> lambda per tau
(d) divergence of neighbouring states in the flow, state = (x, x', x(t + 2 ms)) from the smoothed
    traces (Rosenstein method) -> lambda per tau directly
Each method is run on the data, on the simulation with the scope's noise and 20 mV steps, and on the
clean simulation, so the estimator's bias can be read off against the model's lambda_1 = +0.0986 per tau
(sc.LAMBDA_SIM, from lyap_spectrum.py).
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

import sprott_common as sc

RUN = sc.MAIN_RUN
K_MAP = 5  # returns followed in the section
MAP_FIT = (0, 2)  # fit the map divergence over these returns
LAG = 50  # samples (2 ms) for the third coordinate of the flow state
FLOW_STEP = 5  # samples between flow points
FLOW_K = 600  # samples followed (24 ms = 24 tau)
FLOW_FIT = (
    2.0,
    12.0,
)  # tau: linear part of the flow divergence (read off the clean simulation)
N_REF = 4000
MAP_BAND = (
    0.03,
    0.06,
)  # V: starting distance of a neighbour pair in the section (well above the noise)
FLOW_BAND = (0.03, 0.06)  # V: same for the flow state
SEC_KW = dict(
    hmin=0.01, prom=0.02
)  # all maxima, so successive points are successive loops


def pick_neighbour(tree, pts, i, ok, band):
    """Closest point to pts[i] whose distance lies in `band` and that passes the mask `ok`."""
    j = np.asarray(tree.query_ball_point(pts[i], band[1]), dtype=int)
    if j.size == 0:
        return None
    j = j[ok[j]]
    d = np.linalg.norm(pts[j] - pts[i], axis=1)
    m = d >= band[0]
    if not m.any():
        return None
    return j[m][np.argmin(d[m])]


# ---------------------------------------------------------------- section (map) method
def section_points(x, xd):
    """Per capture: arrays X, M and times (tau) of the x' maxima."""
    out = []
    for p in sc.all_maxima(x, xd, **SEC_KW):
        out.append((p[1], p[2], p[0] * sc.DT_SCOPE / sc.TAU))
    return out


def map_divergence(sec, k_max=K_MAP, n_ref=N_REF, seed=0):
    """Rosenstein divergence in the 2-D section; neighbour from a different capture."""
    pts, cap, pos, nleft = [], [], [], []
    for c, (X, M, _) in enumerate(sec):
        for i in range(len(X)):
            pts.append((X[i], M[i]))
            cap.append(c)
            pos.append(i)
            nleft.append(len(X) - 1 - i)
    pts, cap, pos, nleft = map(np.asarray, (pts, cap, pos, nleft))
    tree = cKDTree(pts)
    rng = np.random.default_rng(seed)
    ok = np.flatnonzero(nleft >= k_max)
    refs = rng.choice(ok, min(n_ref, len(ok)), replace=False)
    logs, rcap = [], []
    for i in refs:
        j = pick_neighbour(tree, pts, i, (cap != cap[i]) & (nleft >= k_max), MAP_BAND)
        if j is None:
            continue
        d = np.array([np.hypot(*(pts[i + k] - pts[j + k])) for k in range(k_max + 1)])
        logs.append(np.log(np.maximum(d, 1e-4)))
        rcap.append(cap[i])
    return np.array(logs), np.array(rcap)


def mean_return_time(sec):
    return np.mean(np.concatenate([np.diff(t) for *_, t in sec]))


# ---------------------------------------------------------------- flow method
def flow_divergence(x, xd, n_ref=N_REF, seed=0):
    xs = np.array([sc.boxcar(a, sc.SMOOTH) for a in x])
    ys = np.array([sc.boxcar(b, sc.SMOOTH) for b in xd])
    n, npts = xs.shape
    i0, i1 = sc.SMOOTH, npts - sc.SMOOTH - LAG
    idx = np.arange(i0, i1, FLOW_STEP)
    pts = np.stack([xs[:, idx], ys[:, idx], xs[:, idx + LAG]], axis=-1).reshape(-1, 3)
    cap = np.repeat(np.arange(n), len(idx))
    pos = np.tile(idx, n)
    tree = cKDTree(pts)
    rng = np.random.default_rng(seed)
    ok = np.flatnonzero(pos + FLOW_K < i1)
    refs = rng.choice(ok, min(n_ref, len(ok)), replace=False)
    ks = np.arange(0, FLOW_K + 1, FLOW_STEP)
    logs, rcap = [], []
    for r in refs:
        j = pick_neighbour(
            tree, pts, r, (cap != cap[r]) & (pos + FLOW_K < i1), FLOW_BAND
        )
        if j is None:
            continue
        ci, pi, cj, pj = cap[r], pos[r], cap[j], pos[j]
        a = np.stack([xs[ci, pi + ks], ys[ci, pi + ks], xs[ci, pi + ks + LAG]], -1)
        b = np.stack([xs[cj, pj + ks], ys[cj, pj + ks], xs[cj, pj + ks + LAG]], -1)
        d = np.linalg.norm(a - b, axis=1)
        logs.append(np.log(np.maximum(d, 1e-4)))
        rcap.append(ci)
    return ks * sc.DT_SCOPE / sc.TAU, np.array(logs), np.array(rcap)


def slope_boot(tk, logs, rcap, window, n_boot=500, seed=0):
    """Slope of <ln d> vs t over `window`, with a bootstrap over reference captures."""
    m = (tk >= window[0]) & (tk <= window[1])
    caps = np.unique(rcap)
    groups = [np.flatnonzero(rcap == c) for c in caps]

    def fit(rows):
        return np.polyfit(tk[m], logs[rows].mean(axis=0)[m], 1)[0]

    rng = np.random.default_rng(seed)
    boot = [
        fit(
            np.concatenate(
                [groups[g] for g in rng.integers(0, len(groups), len(groups))]
            )
        )
        for _ in range(n_boot)
    ]
    return fit(np.arange(len(logs))), np.std(boot)


def main():
    r = sc.load(RUN)
    x, xd = r["x"], r["xd"]
    sets = {
        "measured": (x, xd),
        "sim. + noise, steps": sc.scope_like_sim("simple", x, xd),
        "simulation": sc.sim_windows("simple"),
    }
    colors = {
        "measured": sc.C_DATA,
        "sim. + noise, steps": sc.C_SIM2,
        "simulation": sc.C_SIM,
    }

    res = {}
    for name, (a, b) in sets.items():
        sec = section_points(a, b)
        trt = mean_return_time(sec)
        mlogs, mcap = map_divergence(sec)
        kk = np.arange(K_MAP + 1).astype(float)
        lam_k, lam_k_err = slope_boot(kk, mlogs, mcap, MAP_FIT)
        tk, flogs, fcap = flow_divergence(a, b)
        lam_f, lam_f_err = slope_boot(tk, flogs, fcap, FLOW_FIT)
        res[name] = dict(
            sec=sec,
            trt=trt,
            mlogs=mlogs,
            lam_k=lam_k,
            lam_k_err=lam_k_err,
            tk=tk,
            flogs=flogs,
            lam_f=lam_f,
            lam_f_err=lam_f_err,
        )
        print(
            f"{name:22s} mean time between maxima {trt:.2f} tau; "
            f"map: {lam_k:.3f} +- {lam_k_err:.3f} per return = {lam_k / trt:.3f} +- {lam_k_err / trt:.3f} per tau "
            f"({len(mlogs)} pairs); flow: {lam_f:.3f} +- {lam_f_err:.3f} per tau ({len(flogs)} pairs); "
            f"start distance map {np.exp(mlogs[:, 0].mean()) * 1e3:.1f} mV, flow {np.exp(flogs[:, 0].mean()) * 1e3:.1f} mV"
        )
    print("flow slope (per tau) for other fit windows:")
    for name, v in res.items():
        m = v["flogs"].mean(axis=0)
        row = []
        for w in ((1, 5), (2, 7), (3, 9), (2, 12), (5, 15)):
            k = (v["tk"] >= w[0]) & (v["tk"] <= w[1])
            row.append(f"{w[0]}-{w[1]} tau: {np.polyfit(v['tk'][k], m[k], 1)[0]:.3f}")
        print(f"  {name:22s} " + ", ".join(row))
    print(
        f"model lambda_1 (lyap_spectrum.py, A = {sc.A_EFF:.3f}): +{sc.LAMBDA_SIM:.4f} per tau"
    )

    # ---- figure
    fig, ax = plt.subplots(2, 2, figsize=(7.0, 6.2))
    a, b, c, d = ax.ravel()
    for name in ("simulation", "measured"):
        sec = res[name]["sec"]
        Mn = np.concatenate([M[:-1] for _, M, _ in sec])
        Mn1 = np.concatenate([M[1:] for _, M, _ in sec])
        Xn = np.concatenate([X[:-1] for X, _, _ in sec])
        Xn1 = np.concatenate([X[1:] for X, _, _ in sec])
        kw = dict(
            ls="",
            marker=".",
            color=colors[name],
            ms=1.2 if name == "measured" else 0.8,
            alpha=0.5 if name == "measured" else 0.35,
            label=name,
        )
        a.plot(Mn, Mn1, **kw)
        b.plot(Xn, Xn1, **kw)
    for axx in (a, b):
        lo, hi = axx.get_xlim()
        axx.plot([lo, hi], [lo, hi], color="grey", ls="--", lw=0.6)
        axx.set_xlim(lo, hi)
        axx.set_ylim(lo, hi)
        axx.set_aspect("equal")
    a.set_xlabel("x′ maximum $M_n$ (V)")
    a.set_ylabel("$M_{n+1}$ (V)")
    b.set_xlabel("x at maximum $X_n$ (V)")
    b.set_ylabel("$X_{n+1}$ (V)")
    a.legend(frameon=False, loc="lower right", markerscale=6)

    for name, v in res.items():
        kk = np.arange(K_MAP + 1)
        mean = v["mlogs"].mean(axis=0)
        c.plot(kk, mean - mean[0], "o-", color=colors[name], ms=3, lw=1.0, label=name)
        mean = v["flogs"].mean(axis=0)
        d.plot(v["tk"], mean - mean[0], color=colors[name], lw=1.0, label=name)
    for kx in MAP_FIT:
        c.axvline(kx, color="grey", ls="--", lw=0.6)
    for tx in FLOW_FIT:
        d.axvline(tx, color="grey", ls="--", lw=0.6)
    t = np.linspace(0, res["simulation"]["tk"][-1], 10)
    d.plot(t, sc.LAMBDA_SIM * t, color="k", ls=":", lw=0.8, label=f"slope {sc.LAMBDA_SIM:.3f}")
    c.set_xlabel("Returns k")
    c.set_ylabel(r"$\langle \ln d_k \rangle - \langle \ln d_0 \rangle$")
    d.set_xlabel(r"Time ($\tau$ = 1.006 ms)")
    d.set_ylabel(r"$\langle \ln d(t) \rangle - \langle \ln d(0) \rangle$")
    c.legend(frameon=False, loc="lower right", fontsize=6)
    d.legend(frameon=False, loc="lower right", fontsize=6)
    for axx, lab in zip(ax.ravel(), "abcd"):
        sc.panel(axx, lab)
    fig.tight_layout()
    sc.save(fig, "fig_returnmap")
    out = dict(lambda_model=sc.LAMBDA_SIM, flow_fit=FLOW_FIT, flow_band=FLOW_BAND)
    for name, v in res.items():
        m = v["flogs"].mean(axis=0)
        win = {}
        for w in ((1, 5), (2, 7), (3, 9), (2, 12), (5, 15)):
            k = (v["tk"] >= w[0]) & (v["tk"] <= w[1])
            win[f"{w[0]}-{w[1]}"] = np.polyfit(v["tk"][k], m[k], 1)[0]
        out[name] = dict(lam_flow=v["lam_f"], lam_flow_err=v["lam_f_err"], lam_map=v["lam_k"] / v["trt"],
                         lam_map_err=v["lam_k_err"] / v["trt"], windows=win, tk=v["tk"], curve=m - m[0])
    sc.save_results("returnmap", out)


if __name__ == "__main__":
    main()
