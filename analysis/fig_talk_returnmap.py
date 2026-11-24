"""Talk figure: first-return map of the x' maxima, M_n+1 against M_n, run B next to the model (same peak
finder, model with the measured noise and 8-bit steps)."""

import matplotlib.pyplot as plt

import sprott_common as sc
import talk_common as tc
from fig_B_returnmap import RMAP_LIM, maxima_pairs


def main():
    x, xd, _ = tc.run_b()
    mx, mxd = tc.model_like_scope(x, xd)
    h_b, p_b = maxima_pairs(x, xd)
    _, p_m = maxima_pairs(mx, mxd)
    print(f"board peaks: {h_b.size}")
    fig, axs = plt.subplots(
        1, 2, figsize=(10, 5.2), layout="constrained", sharex=True, sharey=True
    )
    for ax, p, col, title in (
        (axs[0], p_b, sc.C_DATA, tc.LAB_BOARD),
        (axs[1], p_m, sc.C_SIM, tc.LAB_MODEL),
    ):
        ax.plot(
            p[:, 0], p[:, 1], ".", ms=4, mew=0, alpha=0.4, color=col, rasterized=True
        )
        ax.plot(RMAP_LIM, RMAP_LIM, color="grey", ls="--", lw=1.0)
        ax.set_xlim(*RMAP_LIM)
        ax.set_ylim(*RMAP_LIM)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("This peak (V)")
    axs[0].set_ylabel("Next peak (V)")
    tc.save(fig, "fig_talk_returnmap")


if __name__ == "__main__":
    main()
