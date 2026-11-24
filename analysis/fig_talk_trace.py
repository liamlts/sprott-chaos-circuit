"""Talk figure: x and x' against time, first capture of run B (no selection), 100 ms."""

import matplotlib.pyplot as plt
import numpy as np

import sprott_common as sc
import talk_common as tc

CAPTURE = 0


def main():
    x, xd, dt = tc.run_b()
    t_ms = np.arange(x.shape[1]) * dt * 1e3
    fig, ax = plt.subplots(figsize=(10, 3.8), layout="constrained")
    ax.plot(t_ms, x[CAPTURE], color=sc.C_DATA, label="x")
    ax.plot(t_ms, xd[CAPTURE], color=tc.C_XD, label="x′")
    ax.set_xlim(0, t_ms[-1])
    ax.set_ylim(-1.9, 1.7)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Voltage (V)")
    ax.legend(frameon=False, ncol=2, loc="upper right")
    tc.save(fig, "fig_talk_trace")


if __name__ == "__main__":
    main()
