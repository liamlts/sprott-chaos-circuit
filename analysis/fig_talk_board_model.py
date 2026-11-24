"""Talk figure: (x, x') density of run B next to the model (1N4148, A = 0.982, measured noise and 8-bit
steps, nothing fitted), same colour scale."""

import matplotlib.pyplot as plt

import talk_common as tc


def main():
    x, xd, _ = tc.run_b()
    mx, mxd = tc.model_like_scope(x, xd)
    hb, hm = tc.density(x, xd), tc.density(mx, mxd)
    norm = tc.shared_norm(hb, hm)
    fig, axs = plt.subplots(
        1, 2, figsize=(11, 4.8), layout="constrained", sharex=True, sharey=True
    )
    for ax, h, title in zip(axs, (hb, hm), (tc.LAB_BOARD, tc.LAB_MODEL)):
        im = tc.show_density(ax, h, norm)
        ax.set_title(title)
        ax.set_xlabel("x (V)")
    axs[0].set_ylabel("x′ (V)")
    tc.density_colorbar(fig, im, axs, norm)
    tc.save(fig, "fig_talk_board_model")


if __name__ == "__main__":
    main()
