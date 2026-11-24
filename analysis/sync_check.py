"""Two Sprott boards, one-way coupled on x': can the response lock to the drive?

Two places to inject (WHERE):
  "N1": into integrator 1 (U1 pin 6), described below. On the build layout that strip (column 10, f-j) is full.
  "N2": into integrator 2 (U1 pin 2, column 9 a-e, holes a9 and b9 free): one resistor Rc from the response's
        own -x' (U1 pin 1) to N2, i.e. across C2, and one from the DRIVE's x' (U2 pin 7) to N2. This adds
        -k (x'_r - x'_d) to the response's x'' equation, k = tau / (Rc C2). Linearised error:
        s^3 + (A + k) s^2 + (A k + f') s + 1, so k > ~0.6 is expected to be needed.

Coupling (no extra op-amp): two equal resistors Rc into the RESPONSE board's integrator-1 input N1 (U1 pin 6,
a virtual ground), one from the response's own x' (U2 pin 7), one from the DRIVE's -x' (U1 pin 1). This adds
-k (x'_r - x'_d) to the response's x''' with k = tau / (Rc C1). The drive only sees a load to ground through Rc,
so the coupling is one-way.
Linearised error: s^3 + A s^2 + (k + f') s + 1 with f' >= 0 the diode slope, so k > 1/A ~ 1 is expected to be
needed (Rc below ~10k). The scan checks this on the full model (1N4148 'simple' model, measured parts).
Sync error = rms(x_r - x_d) / rms(x_d) after the transient.
"""

import numpy as np
from scipy.integrate import solve_ivp

import sprott_common as sc

RC_VALUES = np.array(
    [2.2e3, 3.3e3, 4.7e3, 6.8e3, 8.2e3, 10e3, 12e3, 15e3, 22e3, 47e3, 1e6]
)
T_TRANS, T_REC = 400.0, 400.0  # tau
CASES = {
    "identical": dict(dA=0.0, is_fac=1.0),
    "A +3% (R4 or C1 off)": dict(dA=0.03, is_fac=1.0),
    "diode Is x2": dict(dA=0.0, is_fac=2.0),
    "A +3% and Is x2": dict(dA=0.03, is_fac=2.0),
}
R, A = sc.R_MEAN, sc.A_EFF
NVT = sc.NID * sc.VT


def f(y, is_fac=1.0):
    return R * sc.IS * is_fac * np.expm1(np.minimum(y / NVT, 60.0))


def rhs(_, s, k, dA, is_fac, where):
    xd, yd, zd, xr, yr, zr = s
    return [
        yd,
        zd,
        -A * zd - xd - f(yd),
        yr,
        zr - (k * (yr - yd) if where == "N2" else 0.0),
        -A * (1 + dA) * zr - xr - f(yr, is_fac) - (k * (yr - yd) if where == "N1" else 0.0),
    ]


def sync_error(rc, dA, is_fac, where):
    k = sc.TAU / (rc * (sc.C1 if where == "N1" else sc.C2))
    s0 = [0.1, 0.0, 0.0, -0.3, 0.2, 0.1]  # response starts somewhere else
    t_eval = np.linspace(T_TRANS, T_TRANS + T_REC, 20000)
    sol = solve_ivp(
        rhs,
        (0, T_TRANS + T_REC),
        s0,
        args=(k, dA, is_fac, where),
        t_eval=t_eval,
        method="LSODA",
        rtol=1e-8,
        atol=1e-10,
    )
    xd, xr = sol.y[0], sol.y[3]
    return k, np.sqrt(np.mean((xr - xd) ** 2)) / np.sqrt(np.mean(xd**2))


def main():
    print(f"A = {A:.3f}")
    print("Rc (ohm):".ljust(24) + "".join(f"{r:>8.0f}" for r in RC_VALUES))
    for where in ("N1", "N2"):
        print(f"--- inject at {where}")
        for name, c in CASES.items():
            errs = [sync_error(r, where=where, **c)[1] for r in RC_VALUES]
            print(name.ljust(24) + "".join(f"{e:>8.3f}" for e in errs), flush=True)


if __name__ == "__main__":
    main()
