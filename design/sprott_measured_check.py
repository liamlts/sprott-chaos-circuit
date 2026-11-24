"""Lyapunov exponent near the A of the measured board (1N4148 model), to check it is not in a periodic window.

Measured 2026-11-19: R1 9.85k, R2 9.95k, R3 9.85k, R4 10.03k, R5 9.86k, R6 9.93k,
C1 102.1n, C2 99.4n, C3 103.9n  ->  A = (R1C1 R2C2 R3C3)^(1/3) / (R4 C1) = 0.982.
"""

import numpy as np
from scipy.integrate import solve_ivp

VT, IS, NID = 0.02585, 2.52e-9, 1.752
R_MEAN = 9883.0
A_BOARD = 0.9823
A_LIST = (0.95, 0.96, 0.97, 0.975, A_BOARD, 0.99, 1.0, 1.01, 1.02)


def make_rhs(R, A):
    def rhs(_, s):
        x, y, z = s
        return [y, z, -A * z - x - R * IS * np.expm1(min(y / (NID * VT), 60.0))]

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


if __name__ == "__main__":
    for A in A_LIST:
        tag = "  <- board" if A == A_BOARD else ""
        print(
            f"A = {A:.4f}   lambda = {lyap(make_rhs(R_MEAN, A)):+.3f}{tag}", flush=True
        )
