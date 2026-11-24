"""Run B: sensitivity to initial conditions and the largest Lyapunov exponent from the data.

Estimator (as fig_returnmap.py): states from the smoothed traces, here (x, x', x(t + lag)) ('flow') or
(x(t), x(t + lag), x(t + 2 lag)) ('delay'); for each of N_REF reference states the nearest state from a
different capture whose distance lies in `band`; <ln d(t)> over pairs; slope over a fit window. Errors are a
bootstrap over reference captures. The same code runs on three sets:
  board        run B, 100 captures
  model+noise  1N4148 model at A = 0.982 with the measured fast noise and 8-bit steps (400 windows)
  clean model  the same windows without noise or steps
The model's true lambda_1 comes from its full Lyapunov spectrum (lyap_spectrum.py, tangent equations).
Calibrated board value: lambda_board x lambda_1(true) / lambda_model+noise, setting by setting.

(a) x(t) of the closest pair of board states (at least D0_MIN apart) in two different captures, 60 ms
(b) ln of their separation; dotted line: slope lambda_1 of the model
(c) <ln d(t)> - <ln d(0)> for the three sets, baseline settings; dashed lines: fit window
(d) slope for every estimator setting, three sets; dotted line: lambda_1 of the model
"""

import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

import sprott_common as sc
from fig_returnmap import pick_neighbour, slope_boot

RUN = sc.B_RUN
CACHE_F = sc.CACHE / "B_lyap.npz"
N_REF = 4000
K_MAX = 600  # samples followed (24 ms)
STEP = 5
PAIR_SPAN = 1500  # samples shown for the example pair (60 ms)
D0_MIN = 0.01  # V: the example pair must start at least this far apart (8-bit steps make exact repeats)

# divergence runs: name -> (coords, lag in samples, band in V)
RUNS = {
    "base": ("flow", 50, (0.03, 0.06)),
    "band_lo": ("flow", 50, (0.02, 0.04)),
    "band_hi": ("flow", 50, (0.05, 0.10)),
    "lag1": ("flow", 25, (0.03, 0.06)),
    "lag3": ("flow", 75, (0.03, 0.06)),
    "delay": ("delay", 50, (0.03, 0.06)),
}
# estimator settings shown in (d): id -> (run, fit window in tau, axis label)
SETTINGS = {
    "baseline": ("base", (2.0, 12.0), "baseline"),
    "fit_2_7": ("base", (2.0, 7.0), "fit 2–7 τ"),
    "fit_3_9": ("base", (3.0, 9.0), "fit 3–9 τ"),
    "fit_5_15": ("base", (5.0, 15.0), "fit 5–15 τ"),
    "d0_20_40": ("band_lo", (2.0, 12.0), "$d_0$ 20–40 mV"),
    "d0_50_100": ("band_hi", (2.0, 12.0), "$d_0$ 50–100 mV"),
    "lag_1ms": ("lag1", (2.0, 12.0), "lag 1 ms"),
    "lag_3ms": ("lag3", (2.0, 12.0), "lag 3 ms"),
    "delay": ("delay", (2.0, 12.0), "delay coords."),
}
SETS = ("board", "model+noise", "clean model")
STYLE = {
    "board": dict(color=sc.C_DATA, marker="o", ls="-"),
    "model+noise": dict(color=sc.C_SIM, marker="s", ls="--"),
    "clean model": dict(color=sc.C_SIM2, marker="^", ls=":"),
}
C_PAIR2 = "#2a9d8f"  # not orange: orange is the clean model in (c, d)


def states(x, xd, coords, lag):
    xs = np.array([sc.boxcar(a, sc.SMOOTH) for a in x])
    ys = np.array([sc.boxcar(b, sc.SMOOTH) for b in xd])
    span = lag if coords == "flow" else 2 * lag
    i0, i1 = sc.SMOOTH, xs.shape[1] - sc.SMOOTH - span

    def at(c, i):
        if coords == "flow":
            return np.stack([xs[c, i], ys[c, i], xs[c, i + lag]], -1)
        return np.stack([xs[c, i], xs[c, i + lag], xs[c, i + 2 * lag]], -1)

    return at, i0, i1


def divergence(x, xd, coords, lag, band, seed=0):
    at, i0, i1 = states(x, xd, coords, lag)
    n = x.shape[0]
    idx = np.arange(i0, i1, STEP)
    cap = np.repeat(np.arange(n), len(idx))
    pos = np.tile(idx, n)
    pts = at(cap, pos)
    tree = cKDTree(pts)
    rng = np.random.default_rng(seed)
    room = pos + K_MAX < i1
    refs = rng.choice(np.flatnonzero(room), N_REF, replace=False)
    ks = np.arange(0, K_MAX + 1, STEP)
    logs, rcap = [], []
    for r in refs:
        j = pick_neighbour(tree, pts, r, (cap != cap[r]) & room, band)
        if j is None:
            continue
        d = np.linalg.norm(at(cap[r], pos[r] + ks) - at(cap[j], pos[j] + ks), axis=1)
        logs.append(np.log(np.maximum(d, 1e-4)))
        rcap.append(cap[r])
    return ks * sc.DT_SCOPE / sc.TAU, np.array(logs), np.array(rcap)


def closest_pair(x, xd, lag=50):
    """Closest flow states, at least D0_MIN apart, in two different captures with PAIR_SPAN samples
    left in both."""
    at, i0, i1 = states(x, xd, "flow", lag)
    idx = np.arange(i0, min(i1, x.shape[1] - PAIR_SPAN), 2)
    cap = np.repeat(np.arange(x.shape[0]), len(idx))
    pos = np.tile(idx, x.shape[0])
    pts = at(cap, pos)
    dist, nb = cKDTree(pts).query(pts, k=40)
    best = (np.inf, None)
    for i in range(len(pts)):
        ok = (cap[nb[i]] != cap[i]) & (dist[i] >= D0_MIN)
        if ok.any():
            k = np.argmax(ok)
            if dist[i, k] < best[0]:
                best = (dist[i, k], (cap[i], pos[i], cap[nb[i, k]], pos[nb[i, k]]))
    return best


def compute():
    if CACHE_F.exists():
        d = np.load(CACHE_F, allow_pickle=True)
        return d["res"].item()
    r = sc.load(RUN)
    data = {
        "board": (r["x"], r["xd"]),
        "model+noise": sc.scope_like_sim("simple", r["x"], r["xd"]),
        "clean model": sc.sim_windows("simple"),
    }
    res = {"curves": {}, "est": {}}
    for s, (x, xd) in data.items():
        for run, (coords, lag, band) in RUNS.items():
            tk, logs, rcap = divergence(x, xd, coords, lag, band)
            print(f"{s:12s} {run:8s} pairs {len(logs)}", flush=True)
            if run == "base":
                res["curves"][s] = (tk, logs.mean(0) - logs.mean(0)[0], len(logs))
            for lab, (rr, win, _) in SETTINGS.items():
                if rr == run:
                    res["est"][(s, lab)] = slope_boot(tk, logs, rcap, win, n_boot=200)
    np.savez(CACHE_F, res=np.array(res, dtype=object))
    return res


def main():
    res = compute()
    spec = json.loads((sc.HERE / "results" / "lyap_spectrum.json").read_text())
    lam1, lam1_err = spec["lam"][0], spec["lam_err"][0]
    r = sc.load(RUN)
    x, xd, dt = r["x"], r["xd"], r["dt"]
    d0, (ca, pa, cb, pb) = closest_pair(x, xd)
    p = dict(d0=d0, ca=ca, pa=pa, cb=cb, pb=pb)

    # numbers
    est = res["est"]
    cal = {
        lab: est[("board", lab)][0] * lam1 / est[("model+noise", lab)][0]
        for lab in SETTINGS
    }
    ratio = {
        lab: est[("board", lab)][0] / est[("model+noise", lab)][0] for lab in SETTINGS
    }
    for lab in SETTINGS:
        print(
            f"{lab:15s} "
            + "  ".join(
                f"{s} {est[(s, lab)][0]:.4f}±{est[(s, lab)][1]:.4f}" for s in SETS
            )
            + f"  ratio {ratio[lab]:.3f}  calibrated {cal[lab]:.4f}"
        )
    cv = np.array(list(cal.values()))
    print(
        f"model lambda_1 {lam1:.4f} ± {lam1_err:.4f}; board calibrated {np.mean(cv):.4f} "
        f"(settings {cv.min():.4f}-{cv.max():.4f}); closest pair d0 {p['d0'] * 1e3:.1f} mV"
    )

    # ------------------------------------------------------------------ figure
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.0), layout="constrained")
    ax_a, ax_b, ax_c, ax_d = axs.ravel()

    t_ms = np.arange(PAIR_SPAN) * dt * 1e3
    xa = x[p["ca"], p["pa"] : p["pa"] + PAIR_SPAN]
    xb = x[p["cb"], p["pb"] : p["pb"] + PAIR_SPAN]
    ax_a.plot(t_ms, xa, color=sc.C_DATA, lw=0.8, label=f"capture {p['ca'] + 1}")
    ax_a.plot(t_ms, xb, color=C_PAIR2, lw=0.8, label=f"capture {p['cb'] + 1}")
    ax_a.set_xlim(0, t_ms[-1])
    ax_a.set_ylim(-1.8, 1.1)
    ax_a.set_xlabel("Time (ms)")
    ax_a.set_ylabel("x (V)")
    ax_a.legend(frameon=False, ncol=2, loc="upper right")

    at, *_ = states(x, xd, "flow", 50)
    ii = np.arange(PAIR_SPAN)
    ok = p["pa"] + ii < x.shape[1] - sc.SMOOTH - 50
    ok &= p["pb"] + ii < x.shape[1] - sc.SMOOTH - 50
    sep = np.linalg.norm(
        at(p["ca"], p["pa"] + ii[ok]) - at(p["cb"], p["pb"] + ii[ok]), axis=1
    )
    ax_b.plot(t_ms[ok], np.log(np.maximum(sep, 1e-4)), color="k", lw=0.8)  # floor as in the estimator
    tt = np.array([0.0, 45.0])
    ax_b.plot(
        tt, np.log(p["d0"]) + lam1 * tt / (sc.TAU * 1e3), color="grey", ls=":", lw=1.0
    )
    ax_b.set_xlim(0, t_ms[-1])
    ax_b.set_xlabel("Time (ms)")
    ax_b.set_ylim(-7, 1.5)
    ax_b.set_ylabel("ln (separation / 1 V)")

    for s in SETS:
        tk, c, _ = res["curves"][s]
        ax_c.plot(
            tk,
            c,
            color=STYLE[s]["color"],
            ls=STYLE[s]["ls"],
            lw=1.0,
            label=s.capitalize(),
        )
    tl = np.linspace(0, 24, 10)
    ax_c.plot(tl, lam1 * tl, color="k", ls=":", lw=0.8)
    for edge in SETTINGS["baseline"][1]:
        ax_c.axvline(edge, color="grey", ls="--", lw=0.6)
    ax_c.set_xlim(0, 24)
    ax_c.set_xlabel(r"Time ($\tau$)")
    ax_c.set_ylabel(r"$\langle \ln d(t) \rangle - \langle \ln d(0) \rangle$")
    ax_c.legend(frameon=False, loc="lower right")

    pos = np.arange(len(SETTINGS))
    for k, s in enumerate(SETS):
        v = np.array([est[(s, lab)] for lab in SETTINGS])
        ax_d.errorbar(
            pos + (k - 1) * 0.2,
            v[:, 0],
            v[:, 1],
            fmt=STYLE[s]["marker"],
            color=STYLE[s]["color"],
            ms=3.5,
            capsize=1.5,
            lw=0.8,
            label=s.capitalize(),
        )
    ax_d.axhline(lam1, color="k", ls=":", lw=0.8)
    ax_d.set_xticks(pos, [v[2] for v in SETTINGS.values()], rotation=45, ha="right", fontsize=7)
    ax_d.set_xlim(-0.6, len(SETTINGS) - 0.4)
    ax_d.set_ylabel(r"Slope (1/$\tau$)")
    lo, hi = ax_d.get_ylim()
    ax_d.set_ylim(lo, hi + 0.25 * (hi - lo))
    ax_d.legend(frameon=False, ncol=3, loc="upper right", fontsize=7)

    for axx, lab in zip(axs.ravel(), "abcd"):
        sc.panel(axx, lab)
    sc.save(fig, "fig_B_lyapunov", png=True)

    sc.save_results(
        "B_lyapunov",
        dict(
            run=RUN.name,
            lambda1_model=lam1,
            lambda1_model_err=lam1_err,
            estimates={
                f"{s} | {lab}": list(map(float, est[(s, lab)]))
                for s in SETS
                for lab in SETTINGS
            },
            ratio_board_to_model_noise=ratio,
            board_calibrated=cal,
            board_calibrated_mean=float(np.mean(cv)),
            board_calibrated_range=[float(cv.min()), float(cv.max())],
            pairs_used={s: int(res["curves"][s][2]) for s in SETS},
            closest_pair={k: float(v) for k, v in p.items()},
        ),
    )


if __name__ == "__main__":
    main()
