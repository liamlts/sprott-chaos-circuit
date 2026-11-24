"""The three states the board showed on 2026-11-19 at A = 0.982 (same parts), against the 1N4148 model.
A: all runs 13:31-16:55 (main run here)   B: original D1 reseated, 18:10   C: reseated again, 18:33.
Model: 'simple' model plus each run's own noise and 8-bit steps (no fitted parameters).
(a-d) density of (x, x')   (e-h) return map of successive x' maxima, M_n+1 against M_n
(i) divergence of neighbouring states, same estimator as fig_returnmap (flow state x, x', x(t + 2 ms))
(j) 2-D Hellinger distance of each state's (x, x') density to the model, with the model-vs-model floor
    for the same number of captures
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

import sprott_common as sc
from fig_compare import hellinger2d
from fig_returnmap import FLOW_FIT, SEC_KW, flow_divergence, slope_boot

RUNS = {
    "A": sc.MAIN_RUN,
    "B": sc.DATA / "sprott_after_swaps_20261119_181028.h5",
    "C": sc.DATA / "sprott_D1_reseated2_20261119_183303.h5",
}
TITLE = {
    "A": "A: 13:31–16:55",
    "B": "B: D1 reseated",
    "C": "C: reseated again",
    "model": "model",
}
COL = {"A": sc.C_DATA, "B": "#5d6d7e", "C": "#2a9d8f", "model": sc.C_SIM}
N_SAME = 100  # captures per state used for the distances (B and C have 100)
WINDOWS = {
    "1-5": (1, 5),
    "2-7": (2, 7),
    "3-9": (3, 9),
    "2-12": (2, 12),
    "5-15": (5, 15),
}
EDGES = [np.arange(-2.0, 1.2, 0.04) + 0.01, np.arange(-1.2, 1.0, 0.04) + 0.01]


def return_pairs(x, xd):
    pairs = []
    for a, b in zip(x, xd):
        m = sc.xd_maxima(a, b, **SEC_KW)[2]
        pairs.append(np.column_stack([m[:-1], m[1:]]))
    return np.concatenate(pairs)


def main():
    data = {k: sc.load(p) for k, p in RUNS.items()}
    sx, sxd = sc.scope_like_sim("simple", data["A"]["x"], data["A"]["xd"])
    sets = {k: (r["x"], r["xd"]) for k, r in data.items()}
    sets["model"] = (sx, sxd)

    # distances to the model, all with N_SAME captures/windows
    rest = (sx[N_SAME:], sxd[N_SAME:])
    floor = [
        hellinger2d(
            sx[k * N_SAME : (k + 1) * N_SAME],
            sxd[k * N_SAME : (k + 1) * N_SAME],
            np.delete(sx, np.s_[k * N_SAME : (k + 1) * N_SAME], 0),
            np.delete(sxd, np.s_[k * N_SAME : (k + 1) * N_SAME], 0),
            EDGES,
        )
        for k in range(len(sx) // N_SAME)
    ]
    hel = {
        k: hellinger2d(data[k]["x"][:N_SAME], data[k]["xd"][:N_SAME], *rest, EDGES)
        for k in RUNS
    }
    print(
        f"Hellinger to the model ({N_SAME} captures each, model rest {len(rest[0])} windows): "
        + ", ".join(f"{k} {v:.3f}" for k, v in hel.items())
        + f"; model vs model {np.mean(floor):.3f} (range {min(floor):.3f}-{max(floor):.3f})"
    )

    # Lyapunov estimator, as in fig_returnmap
    lam = {}
    curves = {}
    for k, (x, xd) in sets.items():
        tk, logs, rcap = flow_divergence(x, xd)
        curves[k] = (tk, logs.mean(0) - logs.mean(0)[0])
        lam[k] = {
            w: slope_boot(tk, logs, rcap, win, n_boot=200) for w, win in WINDOWS.items()
        }
        print(
            f"{k:6s} flow divergence: "
            + ", ".join(f"{w} {v[0]:.3f}±{v[1]:.3f}" for w, v in lam[k].items())
        )

    pairs = {k: return_pairs(*v) for k, v in sets.items()}

    # ------------------------------------------------------------------ figure
    fig = plt.figure(figsize=(7.2, 6.8), layout="constrained")
    gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 0.8, 1.0])
    keys = ["A", "B", "C", "model"]
    dax = [fig.add_subplot(gs[0, i]) for i in range(4)]
    rax = [fig.add_subplot(gs[1, i]) for i in range(4)]
    iax = fig.add_subplot(gs[2, :2])
    jax = fig.add_subplot(gs[2, 2:])

    hs = {
        k: np.histogram2d(v[0].ravel(), v[1].ravel(), EDGES, density=True)[0]
        for k, v in sets.items()
    }
    vmax = max(h.max() for h in hs.values())
    norm = LogNorm(vmin=vmax * 1e-4, vmax=vmax)
    for i, (axx, k) in enumerate(zip(dax, keys)):
        im = axx.pcolormesh(
            EDGES[0],
            EDGES[1],
            np.ma.masked_equal(hs[k].T, 0),
            cmap=sc.CMAP,
            norm=norm,
            rasterized=True,
        )
        axx.set_xlim(-1.8, 0.9)
        axx.set_ylim(-1.05, 0.9)
        axx.set_xlabel("x (V)")
        axx.set_title(TITLE[k], fontsize=8, color=COL[k])
        if i:
            plt.setp(axx.get_yticklabels(), visible=False)
    dax[0].set_ylabel("x′ (V)")
    fig.colorbar(im, ax=dax, label="Density (1/V²)", pad=0.01, shrink=0.9)

    for i, (axx, k) in enumerate(zip(rax, keys)):
        p = pairs[k]
        axx.plot(
            p[:, 0],
            p[:, 1],
            ".",
            ms=1.2,
            mew=0,
            alpha=0.3,
            color=COL[k],
            rasterized=True,
        )
        axx.plot([0, 0.7], [0, 0.7], color="grey", ls="--", lw=0.6)
        axx.set_xlim(0, 0.7)
        axx.set_ylim(0, 0.7)
        axx.set_aspect("equal")
        axx.set_xlabel("$M_n$ (V)")
        if i:
            plt.setp(axx.get_yticklabels(), visible=False)
    rax[0].set_ylabel("$M_{n+1}$ (V)")

    for k in keys:
        tk, c = curves[k]
        iax.plot(tk, c, color=COL[k], lw=1.0, label=TITLE[k].split(":")[0])
    t = np.linspace(0, 24, 10)
    iax.plot(t, sc.LAMBDA_SIM * t, color="k", ls=":", lw=0.8, label=f"slope {sc.LAMBDA_SIM:.3f}")
    for edge in FLOW_FIT:
        iax.axvline(edge, color="grey", ls="--", lw=0.6)
    iax.set_xlabel(r"Time ($\tau$)")
    iax.set_ylabel(r"$\langle \ln d(t) \rangle - \langle \ln d(0) \rangle$")
    iax.legend(
        frameon=False, fontsize=7, loc="lower right", bbox_to_anchor=(1.0, 0.1), ncol=2
    )

    pos = np.arange(len(RUNS))
    for p_, k in zip(pos, RUNS):
        jax.plot(p_, hel[k], "s", color=COL[k], ms=6)
    jax.axhline(np.mean(floor), color="grey", ls="--", lw=0.8)
    jax.set_xticks(pos, [TITLE[k] for k in RUNS], fontsize=7)
    jax.set_xlim(-0.5, len(RUNS) - 0.5)
    jax.set_ylim(0, max(hel.values()) * 1.25)
    jax.set_ylabel("Hellinger distance to model")

    for axx, lab in zip(dax + rax + [iax, jax], "abcdefghij"):
        sc.panel(axx, lab, right=lab in "j")
    sc.save(fig, "fig_states", png=True)
    sc.save_results(
        "states",
        dict(
            hellinger_to_model=hel,
            hellinger_model_floor=[
                float(np.mean(floor)),
                float(min(floor)),
                float(max(floor)),
            ],
            n_same=N_SAME,
            lam_flow={
                k: {w: list(map(float, v)) for w, v in d.items()}
                for k, d in lam.items()
            },
            runs={k: str(p.name) for k, p in RUNS.items()},
        ),
    )


if __name__ == "__main__":
    main()
