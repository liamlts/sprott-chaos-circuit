"""Talk figure: model bifurcation diagram, x' maxima against A (cache/bif_dense.npz from fig_B_returnmap).
Dashed line: the circuit's A = 0.982 from the measured parts."""

import matplotlib.pyplot as plt

import sprott_common as sc
import talk_common as tc
from fig_B_returnmap import A_DENSE, YLIM, bif_dense


def main():
    A, mi, mh = bif_dense()
    fig, ax = plt.subplots(figsize=(10, 4.6), layout="constrained")
    ax.plot(A[mi], mh, ".", ms=0.8, mew=0, color="k", alpha=0.35, rasterized=True)
    ax.axvline(sc.A_EFF, color=sc.C_DATA, ls="--", lw=1.6)
    ax.text(
        sc.A_EFF + 0.01,
        YLIM[1] - 0.03,
        "my circuit",
        color=sc.C_DATA,
        va="top",
        fontsize=16,
    )
    ax.set_xlim(A_DENSE[0], A_DENSE[-1])
    ax.set_ylim(*YLIM)
    ax.set_xlabel("Friction setting A")
    ax.set_ylabel("Peak heights of x′ (V)")
    tc.save(fig, "fig_talk_bifurcation")


if __name__ == "__main__":
    main()
