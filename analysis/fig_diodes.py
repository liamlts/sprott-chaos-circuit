"""Diode swaps: the same board with a different diode in D1 (g14 -> g10), against the original D1.

Reference: the original 1N4148 at R4 = 10.03 kOhm, 10 ms/div (main run and rep1-3), and the 1N4148
model plus scope noise and 8-bit steps. Diode runs: data/sprott_diode_*.h5. A run with the original
D1 put back (sprott_R4_10030_D1back_*.h5) is included when it exists.
Per capture (as in fig_repeat): rms of x, time within 0.2 V of the origin, dominant frequency,
mean x (-<x>/R1 is the mean diode current, up to offsets), x' maxima below 0.1 V per 100 ms.
Per run: 0.1-99.9 % range of x. Points are run means with bootstrap errors over captures.
Top row: density of (x, x') for each run. Bottom: the numbers by run, model as a dashed line.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

import sprott_common as sc
from fig_repeat import per_capture

REF_RUNS = [sc.MAIN_RUN] + sc.runs_matching("sprott_R4_10030_rep*.h5")
DIODE_GLOB = "sprott_diode_*.h5"
BACK_GLOBS = (
    "sprott_R4_10030_D1back_*.h5",
    "sprott_after_swaps_*.h5",
    "sprott_D1_reseated2_*.h5",
)
NICE = {
    "R4_10030": "D1",
    "R4_10030_rep1": "D1 rep1",
    "R4_10030_rep2": "D1 rep2",
    "R4_10030_rep3": "D1 rep3",
    "R4_10030_D1back": "D1 back",
    "after_swaps": "D1 reseated",
    "D1_reseated2": "D1 reseated 2",
    "diode_4148b": "1N4148 #2",
    "diode_4004": "1N4004",
    "diode_Ge": "Ge",
    "diode_Ge_vdiv02": "Ge 0.2 V/div",
}
TEST_COLS = {
    "1N4148 #2": "#2a9d8f",
    "1N4004": "#8e44ad",
    "Ge": "#b8860b",
    "Ge 0.2 V/div": "#d35400",
    "D1 back": "#5d6d7e",
    "D1 reseated": "#5d6d7e",
    "D1 reseated 2": "#2c3e50",
}
OTHER_COL = "#7f8c8d"
QS = [
    ("sx", "rms x (V)"),
    ("near", "Time within 0.2 V"),
    ("f", "Dominant f (Hz)"),
    ("mx", "Mean x (V)"),
    ("nlow", "x′ maxima < 0.1 V\nper 100 ms"),
    ("range", "x range 0.1–99.9 % (V)"),
]
N_BOOT = 300


def label(r):
    stem = r["name"].replace("sprott_", "")
    key = "_".join(stem.split("_")[:-2])
    return NICE.get(key, key)


def x_range(x):
    lo, hi = np.percentile(x, [0.1, 99.9])
    return hi - lo


def run_numbers(r, seed=0):
    pc = per_capture(r)
    rng = np.random.default_rng(seed)
    n = len(r["x"])
    out = {}
    for q, _ in QS:
        if q == "range":
            val = x_range(r["x"])
            bs = [x_range(r["x"][rng.integers(0, n, n)]) for _ in range(N_BOOT)]
        else:
            v = pc[q]
            val = v.mean()
            bs = [v[rng.integers(0, n, n)].mean() for _ in range(N_BOOT)]
        out[q] = (float(val), float(np.std(bs)))
    return out


def main():
    refs = [sc.load(p) for p in REF_RUNS]
    tests = [sc.load(p) for p in sc.runs_matching(DIODE_GLOB)]
    tests += [sc.load(p) for g in BACK_GLOBS for p in sc.runs_matching(g)]
    tests.sort(key=lambda r: r["started"])  # in the order they were recorded
    main_run = refs[0]
    sx, sxd = sc.scope_like_sim("simple", main_run["x"], main_run["xd"])
    model = dict(
        x=sx,
        xd=sxd,
        dt=sc.DT_SCOPE,
        tcap=np.zeros(len(sx)),
        started="2026-01-01T00:00:00",
        name="sprott_model_x_x",
    )

    runs = refs + tests
    nums = {label(r): run_numbers(r) for r in runs}
    num_model = run_numbers(model)

    print(f"{'run':12s}" + "".join(f"{q:>18s}" for q, _ in QS))
    for lab, d in list(nums.items()) + [("model", num_model)]:
        print(f"{lab:12s}" + "".join(f"{d[q][0]:11.4f} ±{d[q][1]:6.4f}" for q, _ in QS))

    # ------------------------------------------------------------------ figure
    dens = [("D1 (orig.)", main_run, sc.C_DATA), ("model", model, sc.C_SIM)]
    dens += [(label(r), r, TEST_COLS.get(label(r), OTHER_COL)) for r in tests]
    nd = len(dens)
    fig = plt.figure(figsize=(7.2, 2.3 + 4.4), layout="constrained")
    sf_top, sf_bot = fig.subfigures(2, 1, height_ratios=[2.3, 4.4])
    axd = sf_top.subplots(1, nd, sharey=True, squeeze=False)[0]
    axs = sf_bot.subplots(2, 3).ravel()

    edges = [np.arange(-2.0, 1.2, 0.04) + 0.01, np.arange(-1.2, 1.0, 0.04) + 0.01]
    hs = [
        np.histogram2d(r["x"].ravel(), r["xd"].ravel(), edges, density=True)[0]
        for _, r, _ in dens
    ]
    vmax = max(h.max() for h in hs)
    norm = LogNorm(vmin=vmax * 1e-4, vmax=vmax)
    for axx, (lab, _, col), h in zip(axd, dens, hs):
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
        axx.set_title(lab, fontsize=8, color=col)
    axd[0].set_ylabel("x′ (V)")
    sf_top.colorbar(im, ax=axd.tolist(), label="Density (1/V²)", pad=0.01)

    labs = list(nums)
    pos = np.arange(len(labs))
    ref_labs = {label(r) for r in refs}
    test_col = {label(r): TEST_COLS.get(label(r), OTHER_COL) for r in tests}
    for axx, (q, ylab) in zip(axs, QS):
        for p, lab in zip(pos, labs):
            v, e = nums[lab][q]
            col = sc.C_DATA if lab in ref_labs else test_col[lab]
            mfc = "none" if lab.startswith(("D1 back", "D1 reseated")) else col
            axx.errorbar(p, v, e, fmt="o", color=col, mfc=mfc, ms=4, capsize=2)
        axx.axhline(num_model[q][0], color=sc.C_SIM, ls="--", lw=0.8)
        axx.set_xticks(pos, labs, rotation=60, ha="right", fontsize=7)
        axx.set_ylabel(ylab)
        axx.set_xlim(-0.6, len(labs) - 0.4)
        lo, hi = axx.get_ylim()
        axx.set_ylim(lo, hi + 0.2 * (hi - lo))  # room for the panel letter
    axs[0].plot([], [], color=sc.C_SIM, ls="--", lw=0.8, label="model")
    axs[0].legend(frameon=False, loc="best", fontsize=7)

    for axx, lab in zip(list(axd) + list(axs), "abcdefghijklmnop"):
        sc.panel(axx, lab)
    sc.save(fig, "fig_diodes", png=True)
    sc.save_results(
        "diodes",
        dict(
            runs={lab: {q: list(v) for q, v in d.items()} for lab, d in nums.items()},
            model={q: list(v) for q, v in num_model.items()},
        ),
    )


if __name__ == "__main__":
    main()
