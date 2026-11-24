"""Talk figure: power spectrum of x, run B against the model (Hann window per capture, averaged, dB re max)."""

import matplotlib.pyplot as plt

import sprott_common as sc
import talk_common as tc
from fig_B_attractor import mean_spectrum_db

FMAX = 800.0


def main():
    x, xd, dt = tc.run_b()
    mx, _ = tc.model_like_scope(x, xd)
    f_b, s_b = mean_spectrum_db(x, dt)
    f_m, s_m = mean_spectrum_db(mx, dt)
    fig, ax = plt.subplots(figsize=(9, 4.4), layout="constrained")
    ax.plot(f_b, s_b, color=sc.C_DATA, label=tc.LAB_BOARD)
    ax.plot(f_m, s_m, color=sc.C_SIM, ls="--", label=tc.LAB_MODEL)
    ax.set_xlim(0, FMAX)
    ax.set_ylim(-60, 5)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (dB)")
    ax.legend(frameon=False, loc="upper right")
    tc.save(fig, "fig_talk_spectrum")
    print(
        f"peak f: board {sc.peak_freq(f_b, 10 ** (s_b / 10)):.1f} Hz, "
        f"model {sc.peak_freq(f_m, 10 ** (s_m / 10)):.1f} Hz"
    )


if __name__ == "__main__":
    main()
