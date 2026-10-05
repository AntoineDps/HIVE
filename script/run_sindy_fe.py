"""
Identify  fe(t) = f(x, xdot, eta_window)
using sparse linear regression (same STLSQ as SINDy, applied to a static mapping).

fe is NOT a differential equation — it is a direct algebraic output of the wave field.
This script finds which eta lags (past/future) and state terms best explain fe.
"""

# %% PACKAGES

import json
import pickle
import numpy as np
import matplotlib.pyplot as plt

from source.config import MODEL_DIR, OUT_DIR
from source.function.sea_state_utils import param2S, S2eta
from source.function.error_utils import metricError
from source.classes.SimRun import SimRun, get_solver
from source.classes.Model import Model
from source.classes.Plotter import Plotter

OUT = OUT_DIR / "sindy"
OUT.mkdir(parents=True, exist_ok=True)

# %% INPUTS

PKL_NAME = "r5"
REF_MODEL = "linear"
DT = 0.05
T_END = 300.0
T_WARMUP = 60.0
T_WAVE = 8.0
H_WAVE = 2.0

B_PTO = 50_000.0

N_PAST = 20  # eta lags:    eta(t-N_PAST·dt) … eta(t-dt)
N_FUT = 20  # eta advances: eta(t+dt) … eta(t+N_FUT·dt)

THRESHOLD = 1e-6  # STLSQ threshold for coefficient sparsity

# %% LOAD

with open(MODEL_DIR / "bem" / f"{PKL_NAME}.pkl", "rb") as f:
    hs = pickle.load(f)
with open(MODEL_DIR / REF_MODEL / "model.json") as f:
    mdef = json.load(f)

# %% DATA

print("Generating data…")
w_s = np.arange(0.01, 4.0, 0.01)
S_s = param2S(T_WAVE, H_WAVE, w_s, gamma=3.3, type="JONSWAP")
t_all = np.arange(0, T_END, DT)
t_all, eta_all, _, _ = S2eta(w_s, S_s, t_all)

sr = SimRun.simulate(
    Model.from_config(
        mdef,
        hs,
        eta_t=t_all,
        eta_values=eta_all,
        pto={"damping": B_PTO, "stiffness": 0},
    ),
    get_solver("RK4"),
    t0=t_all[0],
    t_end=t_all[-1],
    dt=DT,
    eta_t=t_all,
    eta_values=eta_all,
    label="train",
)
ds = sr.dataset

msk = ds["t"].to_numpy() >= T_WARMUP
t = ds["t"].to_numpy()[msk]
x = ds["x"].to_numpy()[msk]
xdot = ds["xdot"].to_numpy()[msk]
eta_c = ds["eta"].to_numpy()[msk]
fe = ds["fe_lin"].to_numpy()[msk]

N = len(t)
print(f"  N={N}  t=[{t[0]:.0f}, {t[-1]:.0f}] s")

# %% BUILD FEATURE MATRIX  Θ
#
# columns: [1,  x,  xdot,  eta(t-N_PAST)…eta(t)…eta(t+N_FUT)]

N_skip = max(N_PAST, 0)
N_cut = max(N_FUT, 0)
Nv = N - N_skip - N_cut

# eta window columns + names
eta_lags = list(range(-N_PAST, N_FUT + 1))
eta_cols = np.column_stack(
    [eta_c[N_skip + lag : N_skip + lag + Nv] for lag in eta_lags]
)
eta_names = [f"eta{lag:+d}" for lag in eta_lags]

# state and target, trimmed to valid window
x_v = x[N_skip : N_skip + Nv]
xd_v = xdot[N_skip : N_skip + Nv]
fe_v = fe[N_skip : N_skip + Nv]
t_v = t[N_skip : N_skip + Nv]

# full feature matrix
ones = np.ones((Nv, 1))
Theta = np.column_stack([ones, x_v, xd_v, eta_cols])
feat_names = ["1", "x", "xdot"] + eta_names

# %% STLSQ  (sequential thresholded least squares)


def stlsq(Theta, y, threshold, max_iter=100):
    """
    Sparse regression: find w such that  Theta @ w ≈ y
    with coefficients below `threshold` zeroed out.
    """
    n_feat = Theta.shape[1]
    active = np.ones(n_feat, dtype=bool)

    for _ in range(max_iter):
        w = np.zeros(n_feat)
        if active.sum() == 0:
            break
        w[active], _, _, _ = np.linalg.lstsq(Theta[:, active], y, rcond=None)
        small = np.abs(w) < threshold
        small[~active] = False  # already zeroed — don't reactivate
        if not small.any():
            break
        active[small] = False

    # final fit on active set
    if active.sum() > 0:
        w[active], _, _, _ = np.linalg.lstsq(Theta[:, active], y, rcond=None)
    return w, active


w, active = stlsq(Theta, fe_v, threshold=THRESHOLD)

# %% RESULTS

fe_pred = Theta @ w

err_train = metricError(fe_pred, fe_v, "nrmse_range")
print(f"\nTraining NRMSE fe = {err_train:.4e}")
print(f"Active features ({active.sum()}/{len(feat_names)}):")
for name, coef, act in zip(feat_names, w, active):
    if act:
        print(f"  {name:>10s}  {coef:+.6e}")

# %% COEFFICIENT PLOT — show eta lag coefficients

eta_coefs = w[3:]  # skip [1, x, xdot]

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(eta_lags, eta_coefs, color=["C1" if c != 0 else "lightgrey" for c in eta_coefs])
ax.axvline(0, color="red", ls="--", lw=0.8, label="t=0")
ax.set(
    xlabel="lag (timesteps)",
    ylabel="coefficient [N/m]",
    title=f"eta lag coefficients in fe prediction  (threshold={THRESHOLD})",
)
ax.legend()
ax.grid(True, axis="y")
plt.tight_layout()
Plotter._save(fig, OUT, None, "fe_eta_coefs")

# %% TIME SERIES PLOT

fig, ax = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
fig.suptitle(
    f"fe prediction — N_PAST={N_PAST}  N_FUT={N_FUT}  "
    f"NRMSE={err_train:.3e}  active={active.sum()} features"
)

ax[0].plot(t_v, fe_v, color="black", lw=1.5, label="ref fe_lin")
ax[0].plot(
    t_v, fe_pred, ls="--", lw=1, color="C1", label=f"predicted  NRMSE={err_train:.3e}"
)
ax[0].set(ylabel="fe [N]")
ax[0].legend()
ax[0].grid(True)

ax[1].plot(t_v, fe_v - fe_pred, lw=0.7, color="C3", label="residual")
ax[1].axhline(0, color="k", lw=0.5)
ax[1].set(ylabel="residual [N]", xlabel="t [s]")
ax[1].legend()
ax[1].grid(True)

plt.tight_layout()
plt.show()
