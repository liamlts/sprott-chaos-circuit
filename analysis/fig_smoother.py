"""Cleaner x', x'' and x''' by combining the scope channels with a model-free smoother (rts_smoother.py).

The smoother only assumes the signal is smooth; it does not use the circuit equation, so the diode term
D = -(x''' + A x'' + x) it gives is still a test of the model. Ch2 is calibrated against dx/dt from Ch1
(gain, offset), and what is left over is compared with what noise alone would leave.

Validation: model windows ('simple') with the run-B noise, 8-bit steps and run-B Ch2 gain and offset, so the truth
is known. Methods: Savitzky-Golay on Ch1 (what fig_diode_inferred.py uses now), Savitzky-Golay on calibrated Ch2,
the smoother on Ch1, on Ch1 + Ch2, and (simulation only) with a third channel on x'' (U1 pin 7, same noise).
The diode current is binned against the Ch2 voltage (the diode's anode) in every case.
(a) diode current D / R against time, one pulse, simulation
(b) rms error of x', x'', x''' relative to their rms size, simulation
(c) binned diode current over the true binned current, simulation
(d) diode current against the Ch2 voltage on run B, with the 1N4148 SPICE curve
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

import rts_smoother as rts
import sprott_common as sc

N_SIM = 120  # model windows used for the validation
ORDER = 4  # states x, x', x'', x''' (x''' is needed for D)
# Savitzky-Golay windows and polyorder, as in fig_diode_inferred.py
W, W3, PO = 21, 31, 4
MID = np.s_[:, 100:-100]  # samples compared (edges dropped)
BINS = np.arange(-0.8, 0.76, 0.025)  # V, Ch2 voltage (as in fig_diode_inferred.py)
I_REF = 300e-6  # A: compare where each binned curve reaches this current
CHECK_RUNS = {
    "A": sc.MAIN_RUN,
    "B": sc.B_RUN,
    "C": sc.DATA / "sprott_D1_reseated2_20261119_183303.h5",
}
H = sc.DT_SCOPE / sc.TAU
A, R = sc.A_EFF, sc.R_MEAN
COL = {
    "SG, Ch1 (so far)": "#7f8c8d",
    "SG, Ch2": sc.C_SIM2,
    "smoother, Ch1": "#8e44ad",
    "smoother, Ch1+Ch2": sc.C_DATA,
    "smoother, +x″ channel": sc.C_SIM,
}


def spice(v):
    return R * sc.IS * np.expm1(np.minimum(v / (sc.NID * sc.VT), 60.0))


def sg_methods(x, xd, g, b):
    """[x, x', x'', x'''] and the Ch2 axis from Savitzky-Golay, Ch1 only and calibrated Ch2."""
    sg = lambda v, w, d: savgol_filter(v, w, PO, deriv=d, delta=H, axis=1)  # noqa: E731
    axis = sg(xd, W, 0)
    y = (xd - b) / g
    return {
        "SG, Ch1 (so far)": (
            [sg(x, W, 0), sg(x, W, 1), sg(x, W3, 2), sg(x, W3, 3)],
            axis,
        ),
        "SG, Ch2": ([sg(x, W, 0), sg(y, W, 0), sg(y, W, 1), sg(y, W, 2)], axis),
    }


def rts_estimates(chans, g, b):
    sig = {k: sc.hf_noise(v) for k, v in chans.items()}
    r = rts.run(chans, sig, H, order=ORDER, bins=BINS)
    out = {}
    for key, res in (("Ch1", r["x_only"]), ("all", r.get("all"))):
        if res is None:
            continue
        st = res["states"]
        out[key] = ([st[..., d] for d in range(ORDER)], g * st[..., 1] + b, res["q"])
    return out, r["cal"]


def diode(est):
    return -(est[3] + A * est[2] + est[0])


def binned(axis, d, nmin=300):
    idx = np.digitize(axis[MID].ravel(), BINS) - 1
    dd = d[MID].ravel()
    c = 0.5 * (BINS[1:] + BINS[:-1])
    m = np.array(
        [
            dd[idx == k].mean() if np.sum(idx == k) >= nmin else np.nan
            for k in range(c.size)
        ]
    )
    return c, m


def v_at(c, d, i_ref=I_REF):
    """Ch2 voltage where the binned current first reaches i_ref (log-linear between bins)."""
    i = d / R
    ok = np.flatnonzero(np.isfinite(i) & (c > 0.3) & (i >= i_ref))
    if not ok.size or ok[0] == 0 or not i[ok[0] - 1] > 0:
        return float("nan")
    k = ok[0]
    f = np.log(i_ref / i[k - 1]) / np.log(i[k] / i[k - 1])
    return float(c[k - 1] + f * (c[k] - c[k - 1]))


def main():
    # calibration check on three runs: is Ch2 a scaled dx/dt of Ch1?
    cal_runs, cal_b = {}, {}
    for k, path in CHECK_RUNS.items():
        r = sc.load(path)
        print(f"run {k} ({path.stem})")
        _, cal = rts_estimates({"x": r["x"], "xd": r["xd"]}, 1.0, 0.0)
        cal_runs[k] = {kk: v for kk, v in cal["xd"].items() if kk != "binned"}
        cal_runs[k]["binned_mV"] = (cal["xd"]["binned"][1] * 1e3).round(1)
        if k == "B":
            cal_b = cal["xd"]
    rb = sc.load(sc.B_RUN)
    g, b = cal_b["gain"], cal_b["offset_V"]

    # simulation with the run-B noise, steps and Ch2 calibration
    cx, cxd = sc.sim_windows("simple")
    cx, cxd = cx[:N_SIM], cxd[:N_SIM]
    cxdd = np.gradient(cxd, H, axis=1, edge_order=2)
    truth = [cx, cxd, cxdd, -A * cxdd - cx - spice(cxd)]
    t_axis = g * cxd + b
    sx, sxd = sc.scope_sigma(rb["x"]), sc.scope_sigma(rb["xd"])
    mx = sc.as_scope(cx, sx, seed=11)
    mxd = sc.as_scope(g * cxd + b, sxd, seed=21)
    mxdd = sc.as_scope(cxdd, sxd, seed=31)
    mmxd = sc.as_scope(-cxd, sxd, seed=41)

    print("simulation")
    sim = sg_methods(mx, mxd, g, b)
    r2, _ = rts_estimates({"x": mx, "xd": mxd}, g, b)
    r3, _ = rts_estimates({"x": mx, "xd": mxd, "xdd": mxdd}, g, b)
    r4, _ = rts_estimates({"x": mx, "xd": mxd, "xdd": mxdd, "mxd": mmxd}, g, b)
    sim["smoother, Ch1"] = r2["Ch1"][:2]
    sim["smoother, Ch1+Ch2"] = r2["all"][:2]
    sim["smoother, +x″ channel"] = r3["all"][:2]
    sim["smoother, 4 channels"] = r4["all"][:2]

    c, d_true = binned(t_axis, diode(truth))
    v_true = v_at(c, d_true)
    size = [np.sqrt(np.mean(t[MID] ** 2)) for t in truth[1:]]
    res: dict = {"calibration": cal_runs, "sim": {}}
    for k, (est, axis) in sim.items():
        err = [
            np.sqrt(np.mean((e[MID] - t[MID]) ** 2)) for e, t in zip(est[1:], truth[1:])
        ]
        _, d = binned(axis, diode(est))
        ratio = d / d_true
        k525 = np.argmin(abs(c - 0.525))
        res["sim"][k] = dict(
            err_rel=[e / s for e, s in zip(err, size)],
            err_abs=err,
            current_ratio_at_0p525V=float(ratio[k525]),
            dV_at_300uA_mV=(v_at(c, d) - v_true) * 1e3,
        )
        print(
            f"  {k:24s} rms error x' {err[0] * 1e3:5.1f} mV, x'' {err[1] * 1e3:6.1f} mV/tau, "
            f"x''' {err[2] * 1e3:5.0f} mV/tau^2 | binned current at 0.525 V: {ratio[k525]:.2f} of true, "
            f"reaches 300 uA {res['sim'][k]['dV_at_300uA_mV']:+.0f} mV from true"
        )

    # run B
    print("run B")
    data = sg_methods(rb["x"], rb["xd"], g, b)
    rB, _ = rts_estimates({"x": rb["x"], "xd": rb["xd"]}, g, b)
    data["smoother, Ch1+Ch2"] = rB["all"][:2]
    res["B_q"] = {"Ch1": rB["Ch1"][2], "Ch1+Ch2": rB["all"][2]}
    res["B_V_at_300uA"] = {}
    for k, (est, axis) in data.items():
        cb, d = binned(axis, diode(est))
        res["B_V_at_300uA"][k] = v_at(cb, d)
    res["SPICE_V_at_300uA"] = float(sc.NID * sc.VT * np.log1p(I_REF / sc.IS))
    print(
        "  Ch2 voltage at 300 uA:",
        {k: round(v, 3) for k, v in res["B_V_at_300uA"].items()},
        f"SPICE {res['SPICE_V_at_300uA']:.3f} V",
    )
    sc.save_results("smoother", res)

    # ---------------------------------------------------------------- figure
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.6), layout="constrained")
    ax_a, ax_b, ax_c, ax_d = axs.ravel()
    shown = [
        "SG, Ch1 (so far)",
        "SG, Ch2",
        "smoother, Ch1",
        "smoother, Ch1+Ch2",
        "smoother, +x″ channel",
    ]

    w = 0
    pk = 300 + np.argmax(cxd[w, 300:-300])
    sl = slice(pk - 150, pk + 150)
    tt = (np.arange(sl.start, sl.stop) - pk) * sc.DT_SCOPE * 1e3
    ax_a.plot(tt, diode(truth)[w, sl] / R * 1e3, color="k", lw=1.6, label="true")
    for k in ("SG, Ch1 (so far)", "smoother, Ch1+Ch2", "smoother, +x″ channel"):
        ax_a.plot(tt, diode(sim[k][0])[w, sl] / R * 1e3, color=COL[k], lw=1.0, label=k)
    ax_a.set_xlabel("Time from the x′ peak (ms)")
    ax_a.set_ylabel("Diode current D / R (mA)")
    ax_a.set_xlim(tt[0], tt[-1])
    ax_a.legend(frameon=False, fontsize=6.5, loc="upper right")

    xs = np.arange(len(shown))
    for j, (q, mk) in enumerate(zip(("x′", "x″", "x‴"), ("o", "s", "^"))):
        ax_b.semilogy(
            xs,
            [res["sim"][k]["err_rel"][j] for k in shown],
            mk,
            ms=4,
            mfc="none",
            color="k",
            label=q,
        )
    ax_b.set_xticks(
        xs,
        [
            "SG\nCh1 (so far)",
            "SG\nCh2",
            "smoother\nCh1",
            "smoother\nCh1+Ch2",
            "smoother\n+x″ ch.",
        ],
    )
    ax_b.tick_params(axis="x", labelsize=6.5, which="both", top=False, bottom=False)
    ax_b.set_ylabel("rms error / rms size")
    ax_b.set_ylim(top=3.0)
    ax_b.legend(frameon=False, fontsize=7, loc="lower left")

    on = c > 0.4
    for k in shown:
        _, d = binned(sim[k][1], diode(sim[k][0]))
        ax_c.plot(c[on], (d / d_true)[on], "o-", ms=3, lw=1.0, color=COL[k], label=k)
    ax_c.axhline(1.0, color="k", lw=0.6, ls="--")
    ax_c.set_xlabel("Ch2 voltage (V)")
    ax_c.set_ylabel("Binned current / true")
    ax_c.set_ylim(top=1.15 * ax_c.get_ylim()[1])
    ax_c.legend(frameon=False, fontsize=6, loc="upper right")

    for k in ("SG, Ch1 (so far)", "SG, Ch2", "smoother, Ch1+Ch2"):
        cb, d = binned(data[k][1], diode(data[k][0]))
        ok = (cb > 0.3) & (d > 0)
        ax_d.semilogy(
            cb[ok], d[ok] / R * 1e6, "o-", ms=3, lw=1.0, color=COL[k], label=k
        )
    vv = np.linspace(0.3, 0.66, 100)
    ax_d.semilogy(vv, spice(vv) / R * 1e6, "k--", lw=0.8, label="1N4148 SPICE")
    ax_d.set_xlabel("Ch2 voltage (V)")
    ax_d.set_ylabel("Diode current D / R (µA)")
    ax_d.set_xlim(0.3, 0.66)
    ax_d.set_ylim(1, 5e3)
    ax_d.legend(frameon=False, fontsize=6.5, loc="lower right")

    for ax, lab in zip(axs.ravel(), "abcd"):
        sc.panel(ax, lab)
    out = sc.save(fig, "fig_smoother")
    return out


if __name__ == "__main__":
    main()
