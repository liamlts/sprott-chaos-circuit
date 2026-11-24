"""Talk figure: (x, x') density of the three states seen on 2026-11-19 (same parts, A = 0.982) and the model,
one shared colour scale. Runs as in fig_states.py; model with the measured noise and 8-bit steps."""

import matplotlib.pyplot as plt

import sprott_common as sc
import talk_common as tc
from fig_states import RUNS

TITLES = {
    "A": "A: all afternoon",
    "B": "B: diode reseated",
    "C": "C: reseated again",
    "model": "Model",
}


def main():
    data = {k: sc.load(p) for k, p in RUNS.items()}
    mx, mxd = tc.model_like_scope(data["B"]["x"], data["B"]["xd"])
    h = {k: tc.density(r["x"], r["xd"]) for k, r in data.items()}
    h["model"] = tc.density(mx, mxd)
    norm = tc.shared_norm(*h.values())
    fig, axs = plt.subplots(
        1, 4, figsize=(15, 4.4), layout="constrained", sharex=True, sharey=True
    )
    for ax, k in zip(axs, ("A", "B", "C", "model")):
        im = tc.show_density(ax, h[k], norm)
        ax.set_title(TITLES[k], color=sc.C_SIM if k == "model" else "k")
        ax.set_xlabel("x (V)")
    axs[0].set_ylabel("x′ (V)")
    tc.density_colorbar(fig, im, axs, norm)
    tc.save(fig, "fig_talk_states")


if __name__ == "__main__":
    main()
