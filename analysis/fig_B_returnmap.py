"""Run B: where the board sits on the model's route to chaos, and its first-return map.

(a) model bifurcation diagram: x' maxima against A in steps of 0.001, 2 random starts per A, 1000 tau transient
    and 300 tau recorded (RK4 integrator from fig_model_scan.py, model unchanged; cached in
    cache/bif_dense.npz). Dashed line: the board's A = 0.982.
(b) distribution of the x' maxima at A = 0.982: board (run B, 100 captures) and model with the measured noise and
    8-bit steps, both through the same peak finder
(c) first-return map of the board: successive x' maxima within a capture, M_{n+1} against M_n
(d) the same for the model, same processing
"""

import matplotlib.pyplot as plt
import numpy as np

import sprott_common as sc
import fig_model_scan as fms
from fig_returnmap import SEC_KW

RUN = sc.B_RUN
YLIM = (0.0, 0.8)
MBINS = np.arange(0.0, 0.8, 0.01)
RMAP_LIM = (0.2, 0.7)
A_DENSE = np.round(np.arange(0.45, 1.3501, 0.001), 4)
N_START = 2
BIF_CACHE = sc.CACHE / "bif_dense.npz"


def bif_dense():
    """x' maxima on the dense A grid. Uses fms.integrate with shorter runs than fig_model_scan
    (its run lengths are module constants, set here and restored)."""
    if BIF_CACHE.exists():
        d = np.load(BIF_CACHE)
        return d["A"], d["mi"], d["mh"]
    rng = np.random.default_rng(11)
    A = np.repeat(A_DENSE, N_START)
    saved = fms.T_TRANS, fms.T_REC
    fms.T_TRANS, fms.T_REC = 1000.0, 300.0
    try:
        r = fms.integrate(A, fms.starts(A.size, rng))
    finally:
        fms.T_TRANS, fms.T_REC = saved
    print(f"dense scan: {A.size} trajectories, {int(r['bad'].sum())} diverged")
    np.savez_compressed(BIF_CACHE, A=A, mi=r["mi"], mh=r["mh"])
    return A, r["mi"], r["mh"]


def maxima_pairs(x, xd):
    """All x' maxima (V) and the pairs (M_n, M_n+1) within each capture."""
    heights, pairs = [], []
    for a, b in zip(x, xd):
        m = sc.xd_maxima(a, b, **SEC_KW)[2]
        heights.append(m)
        pairs.append(np.column_stack([m[:-1], m[1:]]))
    return np.concatenate(heights), np.concatenate(pairs)


def main():
    r = sc.load(RUN)
    x, xd = r["x"], r["xd"]
    mx, mxd = sc.scope_like_sim("simple", x, xd)
    h_b, p_b = maxima_pairs(x, xd)
    h_m, p_m = maxima_pairs(mx, mxd)

    A, mi, mh = bif_dense()
    a_pts, h_pts = A[mi], mh

    fig = plt.figure(figsize=(7.0, 6.1), layout="constrained")
    sf_top, sf_bot = fig.subfigures(2, 1, height_ratios=[1.0, 1.5])
    ax_a, ax_b = sf_top.subplots(1, 2, width_ratios=[3.2, 1], sharey=True)
    ax_c, ax_d = sf_bot.subplots(1, 2, sharex=True, sharey=True)

    ax_a.plot(a_pts, h_pts, ",", color="k", alpha=0.35, rasterized=True)
    ax_a.axvline(sc.A_EFF, color="grey", ls="--", lw=0.8)
    ax_a.set_xlim(A_DENSE[0], A_DENSE[-1])
    ax_a.set_ylim(*YLIM)
    ax_a.set_xlabel(r"A = $\tau$ / (R4 C1)")
    ax_a.set_ylabel("x′ maxima (V)")

    for h, col, ls, lab in (
        (h_b, sc.C_DATA, "-", "Board"),
        (h_m, sc.C_SIM, "--", "Model"),
    ):
        cnt, _ = np.histogram(h, MBINS, density=True)
        ax_b.stairs(
            cnt, MBINS, orientation="horizontal", color=col, ls=ls, lw=0.9, label=lab
        )
    ax_b.set_xlabel("Density (1/V)")
    ax_b.set_xlim(0, None)
    ax_b.legend(frameon=False, loc="lower right")
    plt.setp(ax_b.get_yticklabels(), visible=False)

    for axx, p, col, title in (
        (ax_c, p_b, sc.C_DATA, "Board"),
        (ax_d, p_m, sc.C_SIM, "Model"),
    ):
        axx.plot(
            p[:, 0], p[:, 1], ".", ms=1.5, mew=0, alpha=0.4, color=col, rasterized=True
        )
        axx.plot(RMAP_LIM, RMAP_LIM, color="grey", ls="--", lw=0.6)
        axx.set_xlim(*RMAP_LIM)
        axx.set_ylim(*RMAP_LIM)
        axx.set_aspect("equal")
        axx.set_xlabel("$M_n$ (V)")
        axx.set_title(title, fontsize=8)
    ax_c.set_ylabel("$M_{n+1}$ (V)")
    plt.setp(ax_d.get_yticklabels(), visible=False)

    for axx, lab in zip((ax_a, ax_b, ax_c, ax_d), "abcd"):
        sc.panel(axx, lab, right=lab == "b")
    sc.save(fig, "fig_B_returnmap", png=True)

    out = dict(
        run=RUN.name,
        n_maxima_board=int(h_b.size),
        n_maxima_model=int(h_m.size),
        maxima_per_100ms_board=float(h_b.size / len(x)),
        maxima_per_100ms_model=float(h_m.size / len(mx)),
        median_board=float(np.median(h_b)),
        median_model=float(np.median(h_m)),
        q01_q99_board=np.quantile(h_b, [0.01, 0.99]).tolist(),
        q01_q99_model=np.quantile(h_m, [0.01, 0.99]).tolist(),
    )
    print(out)
    sc.save_results("B_returnmap", out)


if __name__ == "__main__":
    main()
