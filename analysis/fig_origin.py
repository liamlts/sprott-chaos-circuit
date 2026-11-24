"""Why the circuit lingers near the origin: linear growth rate vs re-injection depth.

Near the origin the model is linear, x''' = -A x'' - b x' - x (b = R Is g / n VT ~ 6e-4), with an
unstable spiral sigma +- i omega. Successive x' maxima there grow by exp(sigma * T) per turn.
(a) one capture with a slow spiral out of the origin
(b) growth rate ln(M[n+1] / M[n]) / dt against M[n], bootstrap over captures; dashed: sigma from theory
(c) time between successive maxima against M[n]; dashed: 2 pi / omega
(d) where the trajectory is re-injected: distribution of the x' maxima (no height cut), measured vs model
"""

import matplotlib.pyplot as plt
import numpy as np

import sprott_common as sc

RUN = sc.MAIN_RUN
EDGES = np.array(
    [0.02, 0.05, 0.08, 0.11, 0.14, 0.17, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7]
)
PROM_LOW = 0.02  # V: lower prominence so the small loops are kept


def pairs(x, xd):
    """Per capture: (M[n], ln(M[n+1]/M[n]), dt in tau)."""
    out = []
    for a, b in zip(x, xd):
        t, _, M = sc.xd_maxima(a, b, hmin=0.01, prom=PROM_LOW)
        dt = np.diff(t) * sc.DT_SCOPE / sc.TAU
        out.append((M[:-1], np.log(M[1:] / M[:-1]), dt))
    return out


def binned(pp, idx):
    M = np.concatenate([pp[i][0] for i in idx])
    L = np.concatenate([pp[i][1] for i in idx])
    D = np.concatenate([pp[i][2] for i in idx])
    k = np.digitize(M, EDGES) - 1
    g = [
        np.median(L[k == j] / D[k == j]) if np.sum(k == j) > 5 else np.nan
        for j in range(len(EDGES) - 1)
    ]
    t = [
        np.median(D[k == j]) if np.sum(k == j) > 5 else np.nan
        for j in range(len(EDGES) - 1)
    ]
    n = [np.sum(k == j) for j in range(len(EDGES) - 1)]
    return np.array(g), np.array(t), np.array(n)


def near_origin_sim(n_win=60, seed=5):
    """Model runs started close to the origin (0.005-0.03 V), sampled and noised like the scope,
    to test the growth-rate estimator where the model's own attractor never goes."""
    from scipy.integrate import solve_ivp

    rhs, g = sc.make_rhs("simple")
    rng = np.random.default_rng(seed)
    tt = np.arange(sc.NPTS) * sc.DT_SCOPE / sc.TAU
    xs, ys = np.empty((n_win, sc.NPTS)), np.empty((n_win, sc.NPTS))
    for k in range(n_win):
        a, ph = rng.uniform(0.005, 0.03), rng.uniform(0, 2 * np.pi)
        s0 = [a * np.cos(ph), a * np.sin(ph), 0.0]
        sol = solve_ivp(rhs, (0, tt[-1]), s0, method="LSODA", rtol=1e-8, atol=1e-11, t_eval=tt, max_step=0.05)
        xs[k], ys[k] = sol.y[0], g * sol.y[1]
    return xs, ys


def main():
    r = sc.load(RUN)
    x, xd = r["x"], r["xd"]
    b_lin = sc.R_MEAN * sc.IS / (sc.NID * sc.VT)
    roots = np.roots([1, sc.A_EFF, b_lin, 1])
    cpx = roots[np.argmax(roots.imag)]
    sigma, omega = cpx.real, cpx.imag
    print(
        f"linear theory at A = {sc.A_EFF:.4f}: roots {np.round(roots, 4)}; sigma {sigma:.4f} per tau, "
        f"period {2 * np.pi / omega:.2f} tau, growth per turn x{np.exp(sigma * 2 * np.pi / omega):.2f}"
    )

    pp = pairs(x, xd)
    n = len(pp)
    g, t, cnt = binned(pp, np.arange(n))
    boot = sc.bootstrap(lambda idx: np.concatenate(binned(pp, idx)[:2]), n, n_boot=500)
    nb = len(EDGES) - 1
    ge, te = np.nanstd(boot[:, :nb], axis=0), np.nanstd(boot[:, nb:], axis=0)
    mc = 0.5 * (EDGES[1:] + EDGES[:-1])
    print(f"{'M bin (V)':>12s} {'pairs':>6s} {'growth (1/tau)':>18s} {'dt (tau)':>14s}")
    for j in range(nb):
        print(
            f"{EDGES[j]:.2f}-{EDGES[j + 1]:.2f} {cnt[j]:6d} {g[j]:+8.3f} +- {ge[j]:.3f} {t[j]:7.2f} +- {te[j]:.2f}"
        )
    low = mc < 0.11
    w = 1 / ge[low] ** 2
    print(
        f"weighted mean growth for M < 0.11 V: {np.sum(w * g[low]) / np.sum(w):+.3f} +- {1 / np.sqrt(np.sum(w)):.3f} "
        f"per tau (theory {sigma:+.3f})"
    )

    # estimator check on model runs started near the origin (scope noise + steps added)
    vx, vxd = near_origin_sim()
    vx, vxd = sc.as_scope(vx, sc.scope_sigma(x), 7), sc.as_scope(vxd, sc.scope_sigma(xd), 8)
    vg, vt, vn = binned(pairs(vx, vxd), np.arange(len(vx)))
    for j in range(nb):
        if vn[j] > 5:
            print(f"  model from near origin, {EDGES[j]:.2f}-{EDGES[j + 1]:.2f} V: n={vn[j]:4d} "
                  f"growth {vg[j]:+.3f} per tau, dt {vt[j]:.2f} tau")

    # re-injection depth: all maxima, no height cut
    sx, sxd = sc.scope_like_sim("simple", x, xd)
    cx, cxd = sc.sim_windows("simple")
    md = np.concatenate(
        [sc.xd_maxima(a, b, hmin=0.01, prom=PROM_LOW)[2] for a, b in zip(x, xd)]
    )
    ms = np.concatenate(
        [sc.xd_maxima(a, b, hmin=0.01, prom=PROM_LOW)[2] for a, b in zip(sx, sxd)]
    )
    mcl = np.concatenate(
        [
            sc.xd_maxima(a, b, smooth=1, hmin=None, prom=0.005)[2]
            for a, b in zip(cx, cxd)
        ]
    )
    for lab, m in (("measured", md), ("sim. + noise, steps", ms), ("clean sim", mcl)):
        print(
            f"{lab:20s}: lowest x' max {m.min():.3f} V, 1% {np.percentile(m, 1):.3f} V, "
            f"fraction below 0.2 V {np.mean(m < 0.2):.3f}, below 0.1 V {np.mean(m < 0.1):.3f}"
        )

    # example capture: the one with the most maxima below 0.1 V
    nlow = [
        np.sum(sc.xd_maxima(a, b, hmin=0.01, prom=PROM_LOW)[2] < 0.1)
        for a, b in zip(x, xd)
    ]
    k = int(np.argmax(nlow))
    tt = r["t"] * 1e3

    fig, ax = plt.subplots(2, 2, figsize=(7.0, 5.6))
    a, b, c, d = ax.ravel()
    a.plot(tt, x[k], color=sc.C_DATA, lw=0.7, label="x")
    a.plot(tt, xd[k], color=sc.C_SIM2, lw=0.7, label="x′")
    a.set_xlim(0, tt[-1])
    a.set_xlabel("Time (ms)")
    a.set_ylabel("Voltage (V)")
    a.legend(frameon=False, loc="lower right", ncol=2)

    b.plot(mc[mc < 0.2], vg[mc < 0.2], "o", mfc="none", color=sc.C_SIM, ms=4, label="model, started near 0")
    b.errorbar(mc, g, ge, fmt="o", color=sc.C_DATA, ms=3, capsize=2, label="measured")
    b.legend(frameon=False, loc="lower left", fontsize=6)
    b.axhline(sigma, color="grey", ls="--", lw=0.8)
    b.axhline(0, color="grey", lw=0.5)
    b.set_xlabel("x′ maximum $M_n$ (V)")
    b.set_ylabel(r"$\ln(M_{n+1}/M_n)/\Delta t$ (1/$\tau$)")
    c.plot(mc[mc < 0.2], vt[mc < 0.2], "s", mfc="none", color=sc.C_SIM, ms=4)
    c.errorbar(mc, t, te, fmt="s", color=sc.C_DATA, ms=3, capsize=2)
    c.axhline(2 * np.pi / omega, color="grey", ls="--", lw=0.8)
    c.set_xlabel("x′ maximum $M_n$ (V)")
    c.set_ylabel(r"Time to next maximum ($\tau$)")

    bins = np.arange(0, 0.72, 0.02)
    d.hist(mcl, bins, density=True, histtype="step", color=sc.C_SIM, label="simulation")
    d.hist(md, bins, density=True, histtype="step", color=sc.C_DATA, label="measured")
    d.set_yscale("log")
    d.set_xlabel("x′ maximum (V), all maxima")
    d.set_ylabel("Density (1/V)")
    d.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.06, 0.0), fontsize=6)
    for axx, lab in zip(ax.ravel(), "abcd"):
        sc.panel(axx, lab, right=lab in "bc")
    fig.tight_layout()
    sc.save(fig, "fig_origin")
    w = 1 / ge[low] ** 2
    sc.save_results("origin", dict(
        sigma_theory=sigma, period_theory=2 * np.pi / omega,
        growth_low=np.sum(w * g[low]) / np.sum(w), growth_low_err=1 / np.sqrt(np.sum(w)),
        growth_model_estimator=[float(np.nanmin(vg[mc < 0.11])), float(np.nanmax(vg[mc < 0.11]))],
        dt_low=[float(np.nanmin(t[low])), float(np.nanmax(t[low]))],
        dt_model_estimator=[float(np.nanmin(vt[mc < 0.11])), float(np.nanmax(vt[mc < 0.11]))],
        data_min=md.min(), data_p1=np.percentile(md, 1), data_below02=np.mean(md < 0.2),
        data_below01=np.mean(md < 0.1), model_min=mcl.min(), model_below02=np.mean(mcl < 0.2),
        mc=mc, g=g, ge=ge, t=t, te=te, vg=vg, vt=vt, md=md, mcl=mcl))
    print(f"example capture {k}: {nlow[k]} maxima below 0.1 V")


if __name__ == "__main__":
    main()
