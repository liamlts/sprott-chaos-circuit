"""Which part of the diode I-V does state B use? Sets the current range for the D1 bench I-V (2026-11-20 plan).

The diode sits between the Ch2 node (V2) and the integrator's virtual ground, so its voltage is V2.
From run B's Ch2 trace: V2 percentiles, the SPICE 1N4148 current there, and which V2 range carries the
diode current (share of the time-integrated current, i.e. of the mean diode current, per V2 band).
Also the SPICE voltage at the bench test currents, to print on the record sheet.
Model values only (sc.IS, sc.NID, sc.VT); nothing fitted.
"""

import numpy as np

import sprott_common as sc

RUN = sc.B_RUN
TEST_I = [
    1e-6,
    3e-6,
    10e-6,
    30e-6,
    100e-6,
    300e-6,
    1e-3,
    2e-3,
]  # A, bench test currents
BANDS = [
    (-np.inf, 0.3),
    (0.3, 0.4),
    (0.4, 0.5),
    (0.5, 0.55),
    (0.55, 0.6),
    (0.6, np.inf),
]  # V


def i_spice(v):
    return sc.IS * np.expm1(np.minimum(v / (sc.NID * sc.VT), 60))


def v_spice(i):
    return sc.NID * sc.VT * np.log1p(i / sc.IS)


def main():
    v2 = sc.load(RUN)[
        "xd"
    ].ravel()  # raw Ch2 volts (no smoothing: the 20 mV steps are fine for percentiles)
    i = i_spice(v2)
    pct = [1, 50, 90, 99, 99.9, 100]
    vp = np.percentile(v2, pct)
    print(f"run {RUN.name}: {v2.size} samples")
    for p, v in zip(pct, vp):
        print(f"  V2 {p:5.1f} %: {v:+.3f} V  -> SPICE I = {i_spice(v) * 1e6:9.2f} uA")
    print(f"  mean SPICE current {i.mean() * 1e6:.1f} uA (mean x gave 26.5 uA)")
    share = {}
    for lo, hi in BANDS:
        m = (v2 >= lo) & (v2 < hi)
        share[f"{lo}-{hi}"] = (float(m.mean()), float(i[m].sum() / i.sum()))
        print(
            f"  V2 {lo:>5}-{hi:<5} V: time {m.mean():6.1%}, share of diode current {i[m].sum() / i.sum():6.1%}"
        )
    # control: the same calculation on the model with the measured noise and steps, and on the clean model.
    # If the board's V2 trace gives more SPICE current than the model's, the real diode (or the scope's Ch2 gain)
    # differs from SPICE by the voltage shift dv that brings the two means together.
    r = sc.load(RUN)
    _, mxd = sc.scope_like_sim("simple", r["x"], r["xd"])
    _, cxd = sc.sim_windows("simple")
    i_m, i_c = i_spice(mxd).mean(), i_spice(cxd).mean()
    print(f"  mean SPICE current: model+noise {i_m * 1e6:.1f} uA, clean model {i_c * 1e6:.1f} uA")
    dvs = np.linspace(0, 0.08, 801)
    dv = float(dvs[np.argmin([abs(i_spice(v2 - d).mean() - i_m) for d in dvs])])
    print(f"  board matches model+noise if V2 is shifted down by {dv * 1e3:.0f} mV")
    print("SPICE voltage at the bench currents (T = 27 C):")
    for t in TEST_I:
        print(f"  {t * 1e6:7.0f} uA -> {v_spice(t) * 1e3:6.1f} mV")
    sc.save_results(
        "diode_range",
        dict(
            run=RUN.name,
            v2_percentiles={str(p): float(v) for p, v in zip(pct, vp)},
            i_spice_at_percentiles_uA={
                str(p): float(i_spice(v) * 1e6) for p, v in zip(pct, vp)
            },
            mean_i_spice_uA=float(i.mean() * 1e6),
            mean_i_spice_model_noise_uA=float(i_m * 1e6),
            mean_i_spice_clean_model_uA=float(i_c * 1e6),
            v_shift_to_match_model_mV=dv * 1e3,
            bands_time_and_current_share=share,
            v_spice_at_test_currents_mV={
                f"{t * 1e6:g} uA": float(v_spice(t) * 1e3) for t in TEST_I
            },
        ),
    )


if __name__ == "__main__":
    main()
