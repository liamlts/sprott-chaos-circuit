"""State C (period-3 orbit, 18:33) against the model's periodic window at A = 0.9545-0.9555.
Uses the RK4 cache from fig_model_scan.py (no new integration, no change to the model).
Prints the four x' maxima levels, rms x and time within 0.2 V of the origin for both; saves results/window_C.json.
"""

import numpy as np

import sprott_common as sc
from fig_model_scan import CACHE, STATES, data_maxima

A_WIN = [
    0.9545,
    0.955,
    0.9555,
]  # all six starts give period 4 (distinct x' maxima) here
N_LEVELS = 4


def levels(h, n=N_LEVELS):
    """Each of the n levels occurs once per period, so split the sorted heights into n equal-count
    groups and take each group's median (robust to the ~13 mV noise smearing neighbouring levels)."""
    return [float(np.median(g)) for g in np.array_split(np.sort(h), n)]


def main():
    d = np.load(CACHE, allow_pickle=True)
    A, mi, mh = d["A"], d["mi"], d["mh"]
    model = {}
    for a in A_WIN:
        js = np.nonzero(np.isclose(A, a))[0]
        lv = [levels(mh[mi == j][-300:]) for j in js]
        model[a] = dict(
            levels=np.mean(lv, 0).tolist(),
            rms=float(d["rms"][js].mean()),
            near=float(d["near"][js].mean()),
        )
    ys, rms, near = data_maxima(STATES["C (reseated again)"])
    ys = ys[ys > 0.3]  # drops one stray low maximum
    board = dict(levels=levels(ys), rms=rms, near=near)

    fmt = lambda v: ", ".join(f"{x:.3f}" for x in v)  # noqa: E731
    print(
        f"board C       : x' maxima {fmt(board['levels'])} V, rms x {rms:.3f}, near {near:.3f}"
    )
    for a, m in model.items():
        print(
            f"model A={a:.4f}: x' maxima {fmt(m['levels'])} V, rms x {m['rms']:.3f}, "
            f"near {m['near']:.3f}"
        )
    sc.save_results(
        "window_C", dict(board=board, model={str(k): v for k, v in model.items()})
    )


if __name__ == "__main__":
    main()
