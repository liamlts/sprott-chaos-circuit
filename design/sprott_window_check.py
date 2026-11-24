"""Lyapunov exponent vs damping A (= R / R4) for the paper diode model and a 1N4148, R = 1k and 10k."""

import numpy as np
from scipy.integrate import solve_ivp

VT = 0.02585
MODELS = {"paper (1e-12 A, n=1)": (1e-12, 1.0), "1N4148": (2.52e-9, 1.752)}


def make_rhs(R, Is, n, A):
    def rhs(t, s):
        x, y, z = s
        return [y, z, -A * z - x - R * Is * np.expm1(min(y / (n * VT), 60.0))]
    return rhs


def lyap(rhs, T=1500.0, dt=1.0, d0=1e-7):
    kw = dict(method="LSODA", rtol=1e-8, atol=1e-10)
    a = solve_ivp(rhs, (0, 200), [0.0, 0.4, 0.0], **kw).y[:, -1]
    b = a + [d0, 0, 0]
    s = 0.0
    for _ in range(int(T / dt)):
        a = solve_ivp(rhs, (0, dt), a, **kw).y[:, -1]
        b = solve_ivp(rhs, (0, dt), b, **kw).y[:, -1]
        d = np.linalg.norm(b - a)
        s += np.log(d / d0)
        b = a + (b - a) * d0 / d
    return s / T


for name, (Is, n) in MODELS.items():
    for R in (1e3, 1e4):
        row = []
        for A in (0.6, 0.8, 0.9, 1.0, 1.1, 1.2):
            row.append(f"{A:.1f}:{lyap(make_rhs(R, Is, n, A)):+.3f}")
        print(f"{name:22s} R={R:>6.0f}  " + "  ".join(row), flush=True)
