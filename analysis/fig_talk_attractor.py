"""Talk figure: density of (x, x') from all 100 captures of run B (the measured attractor)."""

import matplotlib.pyplot as plt

import talk_common as tc


def main():
    x, xd, _ = tc.run_b()
    h = tc.density(x, xd)
    norm = tc.shared_norm(h)
    fig, ax = plt.subplots(figsize=(7.2, 5.4), layout="constrained")
    im = tc.show_density(ax, h, norm)
    ax.set_xlabel("x (V)")
    ax.set_ylabel("x′ (V)")
    tc.density_colorbar(fig, im, ax, norm)
    tc.save(fig, "fig_talk_attractor")


if __name__ == "__main__":
    main()
