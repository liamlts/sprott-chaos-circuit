"""Talk figures: the closest pair of run-B states in two different captures (fig_B_lyapunov.closest_pair).
fig_talk_pair:       x(t) of both, 60 ms
fig_talk_separation: distance between the two states (x, x', x(t + 2 ms)) on a log axis; dotted line: the
                     model's lambda_1 (results/lyap_spectrum.json) from the same start
"""

import json

import matplotlib.pyplot as plt
import numpy as np

import sprott_common as sc
import talk_common as tc
from fig_B_lyapunov import C_PAIR2, PAIR_SPAN, closest_pair, states

LAG = 50


def main():
    x, xd, dt = tc.run_b()
    d0, (ca, pa, cb, pb) = closest_pair(x, xd, LAG)
    lam1 = json.loads((sc.HERE / "results" / "lyap_spectrum.json").read_text())["lam"][
        0
    ]
    print(f"pair: captures {ca + 1} and {cb + 1}, start {d0 * 1e3:.1f} mV apart")
    t_ms = np.arange(PAIR_SPAN) * dt * 1e3

    fig, ax = plt.subplots(figsize=(10, 3.8), layout="constrained")
    ax.plot(
        t_ms, x[ca, pa : pa + PAIR_SPAN], color=sc.C_DATA, label=f"Snapshot {ca + 1}"
    )
    ax.plot(t_ms, x[cb, pb : pb + PAIR_SPAN], color=C_PAIR2, label=f"Snapshot {cb + 1}")
    ax.set_xlim(0, t_ms[-1])
    ax.set_ylim(-1.8, 1.5)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("x (V)")
    ax.legend(frameon=False, ncol=2, loc="upper right")
    tc.save(fig, "fig_talk_pair")

    at, *_ = states(x, xd, "flow", LAG)
    ii = np.arange(PAIR_SPAN)
    ok = (pa + ii < x.shape[1] - sc.SMOOTH - LAG) & (
        pb + ii < x.shape[1] - sc.SMOOTH - LAG
    )
    sep = np.linalg.norm(at(ca, pa + ii[ok]) - at(cb, pb + ii[ok]), axis=1)
    fig, ax = plt.subplots(figsize=(9, 4.4), layout="constrained")
    ax.semilogy(t_ms[ok], np.maximum(sep, 1e-4), color="k", label="Measured gap")
    tt = np.array([0.0, 45.0])
    ax.semilogy(
        tt,
        d0 * np.exp(lam1 * tt / (sc.TAU * 1e3)),
        color="grey",
        ls=":",
        lw=2.0,
        label="Model’s growth rate",
    )
    ax.set_xlim(0, t_ms[-1])
    ax.set_ylim(1e-3, 5)
    ax.set_yticks([1e-3, 1e-2, 1e-1, 1], labels=["1 mV", "10 mV", "100 mV", "1 V"])
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Gap between the two")
    ax.legend(frameon=False, loc="lower right")
    tc.save(fig, "fig_talk_separation")


if __name__ == "__main__":
    main()
