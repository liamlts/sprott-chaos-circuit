"""Run B: correlation dimension of the attractor (Grassberger-Procaccia), against the model.

Delay vectors of the smoothed x, (x(t), x(t + L), ..., x(t + (m-1) L)), m = 1..6, every SUB samples. L is the
first zero of the autocorrelation of x (board, averaged over captures), used for every set. Max norm.
C(r) counts pairs closer than r between two disjoint halves of the captures (even against odd), so no two points
come from the same stretch of trajectory and no Theiler window is needed. Errors: bootstrap over captures.
Sets as in fig_B_lyapunov: board (run B), model + measured noise and 8-bit steps, clean model (100 windows each
for the models, like the 100 board captures). D2 = mean local slope d ln C / d ln r over R_FIT for m = 4-6, where the clean model's slope is
flat. Compared with the model's Kaplan-Yorke dimension from lyap_spectrum.py.

(a) C(r) for the board, m = 1..6
(b) local slope for the board, m = 1..6; dashed lines: R_FIT; dotted: D_KY of the model
(c) local slope at m = M_SHOW for the board and both models
"""

import json

import matplotlib.pyplot as plt
import numpy as np

import sprott_common as sc

RUN = sc.B_RUN
CACHE_F = sc.CACHE / "B_dimension.npz"
M_LIST = (1, 2, 3, 4, 5, 6)
M_SHOW = 5
SUB = 5  # samples between delay vectors
R = np.geomspace(0.004, 3.0, 50)  # V
R_FIT = (0.08, 0.3)  # V: scaling range (set from the clean model's flat local slope)
N_MODEL = 100  # model windows used (same number as board captures)
N_BOOT = 300
SETS = ("board", "model+noise", "clean model")
STYLE = {
    "board": dict(color=sc.C_DATA, ls="-"),
    "model+noise": dict(color=sc.C_SIM, ls="--"),
    "clean model": dict(color=sc.C_SIM2, ls=":"),
}


def first_zero_lag(x):
    xm = x - x.mean(1, keepdims=True)
    n = x.shape[1]
    f = np.fft.rfft(xm, 2 * n, axis=1)
    ac = np.fft.irfft(np.abs(f) ** 2, axis=1)[:, :n].mean(0)
    return int(np.argmax(ac <= 0))


def vectors(x, m, lag):
    xs = np.array([sc.boxcar(a, sc.SMOOTH)[sc.SMOOTH : -sc.SMOOTH] for a in x])
    idx = np.arange(0, xs.shape[1] - (m - 1) * lag, SUB)
    return [np.stack([c[idx + k * lag] for k in range(m)], -1) for c in xs]


def pair_counts(x, m, lag):
    """counts[i, j, k]: pairs (capture 2i, capture 2j+1) closer than R[k] (max norm); sizes of each set.
    Distances computed directly, one capture of the first half against the whole second half."""
    v = vectors(x, m, lag)
    v1, v2 = v[0::2], v[1::2]
    n1 = np.array([a.shape[0] for a in v1])
    n2 = np.array([b.shape[0] for b in v2])
    b_all = np.concatenate(v2).astype(np.float32)
    col_cap = np.repeat(np.arange(len(v2)), n2)
    nb = R.size + 1
    counts = np.zeros((len(v1), len(v2), R.size))
    for i, a in enumerate(v1):
        a = a.astype(np.float32)
        d = np.abs(a[:, None, 0] - b_all[None, :, 0])
        for k in range(1, m):
            np.maximum(d, np.abs(a[:, None, k] - b_all[None, :, k]), out=d)
        nless = np.searchsorted(R, d, side="right")  # pair is closer than R[k] for k >= nless
        h = np.bincount((col_cap[None, :] * nb + nless).ravel(), minlength=len(v2) * nb)
        counts[i] = np.cumsum(h.reshape(len(v2), nb), axis=1)[:, : R.size]
    return counts, n1, n2


def corr_sum(counts, n1, n2, i=None, j=None):
    i = np.arange(len(n1)) if i is None else i
    j = np.arange(len(n2)) if j is None else j
    return counts[np.ix_(i, j)].sum((0, 1)) / (n1[i].sum() * n2[j].sum())


def local_slope(c):
    return np.gradient(np.log(c), np.log(R))


def d2(slope):
    m = (R >= R_FIT[0]) & (R <= R_FIT[1])
    return float(slope[m].mean())


def compute():
    if CACHE_F.exists():
        d = np.load(CACHE_F, allow_pickle=True)
        return d["res"].item()
    r = sc.load(RUN)
    lag = first_zero_lag(r["x"])
    mx, mxd = sc.scope_like_sim("simple", r["x"], r["xd"])
    cx, _ = sc.sim_windows("simple")
    data = {"board": r["x"], "model+noise": mx[:N_MODEL], "clean model": cx[:N_MODEL]}
    res = {"lag": lag, "counts": {}}
    for s, x in data.items():
        for m in M_LIST:
            res["counts"][(s, m)] = pair_counts(x, m, lag)
            print(f"{s:12s} m = {m} done", flush=True)
    np.savez(CACHE_F, res=np.array(res, dtype=object))
    return res


def main():
    res = compute()
    lag = res["lag"]
    spec = json.loads((sc.HERE / "results" / "lyap_spectrum.json").read_text())
    d_ky = spec["d_ky"]

    c, sl = {}, {}
    for key, (cnt, n1, n2) in res["counts"].items():
        c[key] = corr_sum(cnt, n1, n2)
        sl[key] = local_slope(c[key])

    rng = np.random.default_rng(0)
    d2_val, d2_err = {}, {}
    for s in SETS:
        for m in M_LIST:
            cnt, n1, n2 = res["counts"][(s, m)]
            boot = [
                d2(
                    local_slope(
                        corr_sum(
                            cnt,
                            n1,
                            n2,
                            rng.integers(0, len(n1), len(n1)),
                            rng.integers(0, len(n2), len(n2)),
                        )
                    )
                )
                for _ in range(N_BOOT)
            ]
            d2_val[(s, m)], d2_err[(s, m)] = d2(sl[(s, m)]), float(np.std(boot))
    print(
        f"delay L = {lag} samples ({lag * sc.DT_SCOPE * 1e3:.2f} ms); model D_KY {d_ky:.3f}"
    )
    for s in SETS:
        print(
            f"{s:12s} D2: "
            + ", ".join(
                f"m{m} {d2_val[(s, m)]:.3f}±{d2_err[(s, m)]:.3f}" for m in M_LIST
            )
        )
    summary = {
        s: (
            float(np.mean([d2_val[(s, m)] for m in (4, 5, 6)])),
            float(np.mean([d2_err[(s, m)] for m in (4, 5, 6)])),
        )
        for s in SETS
    }
    print(
        "D2 (m = 4-6): "
        + ", ".join(f"{s} {v[0]:.3f}±{v[1]:.3f}" for s, v in summary.items())
    )

    # ------------------------------------------------------------------ figure
    fig, (ax_a, ax_b, ax_c) = plt.subplots(
        1, 3, figsize=(7.0, 2.6), layout="constrained"
    )
    cols = sc.CMAP(np.linspace(0.3, 0.95, len(M_LIST)))
    for m, col in zip(M_LIST, cols):
        ax_a.loglog(R, c[("board", m)], color=col, lw=0.9, label=f"m = {m}")
        ax_b.semilogx(R, sl[("board", m)], color=col, lw=0.9, label=f"m = {m}")
    ax_a.set_xlabel("r (V)")
    ax_a.set_ylabel("C(r)")
    ax_a.set_xlim(R[0], R[-1])
    ax_a.legend(frameon=False, fontsize=6.5, loc="lower right", ncol=2)

    for s in SETS:
        ax_c.semilogx(R, sl[(s, M_SHOW)], lw=1.0, label=s.capitalize(), **STYLE[s])
    for axx in (ax_b, ax_c):
        axx.axhline(d_ky, color="k", ls=":", lw=0.8)
        for edge in R_FIT:
            axx.axvline(edge, color="grey", ls="--", lw=0.6)
        axx.set_xlim(R[0], R[-1])
        axx.set_ylim(0, 4.5)
        axx.set_xlabel("r (V)")
        axx.set_ylabel(r"d ln C / d ln r")
    ax_c.legend(frameon=False, fontsize=7, loc="lower left")

    for axx, lab in zip((ax_a, ax_b, ax_c), "abc"):
        sc.panel(axx, lab, right=lab in "bc")
    sc.save(fig, "fig_B_dimension", png=True)

    sc.save_results(
        "B_dimension",
        dict(
            run=RUN.name,
            lag_samples=lag,
            lag_ms=lag * sc.DT_SCOPE * 1e3,
            r_fit=list(R_FIT),
            d2={
                f"{s} m{m}": [d2_val[(s, m)], d2_err[(s, m)]]
                for s in SETS
                for m in M_LIST
            },
            d2_m4_6=summary,
            d_ky_model=d_ky,
        ),
    )


if __name__ == "__main__":
    main()
