import cvxpy as cp
from scipy.signal import bode as scipy_bode
import numpy as np


def passivity_enforcement(A, B, C, weights=(0.5, 0.5), verbose=False):
    """
    LMI passivity enforcement (Faedo et al. 2021).
    Finds minimum ΔC such that (A, B, C+ΔC, 0) satisfies the PRL.

    Returns C_new, Cbar (the correction)
    """
    n = A.shape[0]
    p = B.shape[1]
    wb, wa = weights

    P = cp.Variable((n, n), symmetric=True)
    Cbar = cp.Variable((p, n))
    gamma = cp.Variable()
    beta = cp.Variable()

    diff = B.T @ P - C - Cbar  # PRL residual  (p × n)

    # N1: ||Cbar||_F ≤ γ  (Schur)
    M1 = cp.bmat([[cp.reshape(-gamma, (1, 1)), Cbar], [Cbar.T, -gamma * np.eye(n)]])
    # N2: Lyapunov
    M2 = P @ A + A.T @ P
    # N3: ||B'P - C - Cbar||_2 ≤ β  (Schur)
    M3 = cp.bmat([[cp.reshape(-beta, (1, 1)), diff], [diff.T, -beta * np.eye(n)]])

    constraints = [
        M1 << 0,
        M2 << 0,
        M3 << 0,
        P >> 1e-8 * np.eye(n),
        gamma >= 1e-10,
        beta >= 1e-10,
    ]

    prob = cp.Problem(cp.Minimize(wa * gamma + wb * beta), constraints)
    prob.solve(solver=cp.SCS, verbose=verbose)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        print(f"  Warning: solver status = {prob.status}")

    Cbar_opt = Cbar.value
    C_new = C + Cbar_opt
    print(
        f"  ||ΔC|| = {np.linalg.norm(Cbar_opt):.4e}   "
        f"β = {float(beta.value):.4e}   status = {prob.status}"
    )
    return C_new, Cbar_opt


def check_passivity(A, B, C, w_max=20.0, n_pts=2000):
    """Check Re[Y(jω)] ≥ 0. Returns (is_passive, min_Re)."""
    I = np.eye(A.shape[0])
    w = np.linspace(1e-3, w_max, n_pts)
    re_vals = np.array(
        [np.real((C @ np.linalg.solve(1j * wi * I - A, B))[0, 0]) for wi in w]
    )
    return bool(np.all(re_vals >= 0)), float(np.min(re_vals))
