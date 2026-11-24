"""Talk figure: average growth of the gap between neighbouring states, exp(<ln d(t)> - <ln d(0)>), for run B,
the model with the measured noise and 8-bit steps, and the clean model (baseline settings of fig_B_lyapunov,
from cache/B_lyap.npz). Time in ms (tau = 1.006 ms)."""

import matplotlib.pyplot as plt
import numpy as np

import sprott_common as sc
import talk_common as tc
from fig_B_lyapunov import STYLE, compute

LABEL = {
    "board": tc.LAB_BOARD,
    "model+noise": tc.LAB_MODEL,
}
T_MAX = 24.0  # ms


def main():
    res = compute()
    fig, ax = plt.subplots(figsize=(9, 4.4), layout="constrained")
    for s in ("board", "model+noise"):
        tk, c, n = res["curves"][s]
        ax.semilogy(
            tk * sc.TAU * 1e3,
            np.exp(c),
            color=STYLE[s]["color"],
            ls=STYLE[s]["ls"],
            lw=2.0,
            label=LABEL[s],
        )
        print(
            f"{s}: {n} pairs, growth x{np.exp(c[-1]):.1f} after {tk[-1] * sc.TAU * 1e3:.1f} ms"
        )
    ax.set_xlim(0, T_MAX)
    ax.set_ylim(0.8, 15)
    ax.set_yticks([1, 2, 5, 10], labels=["1", "2", "5", "10"])
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Average growth of the gap (×)")
    ax.legend(frameon=False, loc="lower right")
    tc.save(fig, "fig_talk_divergence")


if __name__ == "__main__":
    main()
