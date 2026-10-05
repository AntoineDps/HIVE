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

N_ETA_PAST = 20  # eta:  eta(t-N·dt) … eta(t)
N_ETA_FUT = 20  # eta:  eta(t+dt) … eta(t+N·dt)
N_X_PAST = 20  # x:    x(t-N·dt) … x(t)
N_XD_PAST = 20  # xdot: xdot(t-N·dt) … xdot(t)

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


# f_tot = fe + fr  (total hydrodynamic force = excitation + radiation)
# from EOM: (m+ma_inf)*xddot = fe + fr - K_hs*x - f_pto
#       →   fe + fr = (m+ma_inf)*xddot + K_hs*x + f_pto
def spectral_diff(y, dt):
    N = len(y)
    yhat = np.fft.rfft(y)
    freq = np.fft.rfftfreq(N, dt)
    return np.fft.irfft(2j * np.pi * freq * yhat, n=N)


m_tot = hs.mass + hs.ma_inf
K_hs = hs.stiffness
f_pto_c = ds["f_pto"].to_numpy()[msk] if "f_pto" in ds.columns else np.zeros(N)
xddot_c = spectral_diff(xdot, DT)
f_tot = m_tot * xddot_c + K_hs * x + f_pto_c

print(f"  f_tot range: [{f_tot.min():.1f}, {f_tot.max():.1f}] N")

# %% BUILD FEATURE MATRIX  Θ
#
# columns: [1,  x,  xdot,  eta(t-N_PAST)…eta(t)…eta(t+N_FUT)]

N_skip = max(N_ETA_PAST, N_X_PAST, N_XD_PAST)
N_cut = max(N_ETA_FUT, 0)
Nv = N - N_skip - N_cut

# trim target to valid window
t_v = t[N_skip : N_skip + Nv]
fe_v = fe[N_skip : N_skip + Nv]
f_tot_v = f_tot[N_skip : N_skip + Nv]

# eta window: past (lag<0) + present + future (lag>0)
eta_lags = list(range(-N_ETA_PAST, N_ETA_FUT + 1))
eta_cols = np.column_stack(
    [eta_c[N_skip + lag : N_skip + lag + Nv] for lag in eta_lags]
)
eta_names = [f"eta{lag:+d}" for lag in eta_lags]

# x history: x(t), x(t-dt), ..., x(t-N_X_PAST*dt)
x_lags = list(range(0, N_X_PAST + 1))
x_cols = np.column_stack([x[N_skip - lag : N_skip - lag + Nv] for lag in x_lags])
x_names = [f"x{-lag:+d}" for lag in x_lags]

# xdot history: xdot(t), ..., xdot(t-N_XD_PAST*dt)
xd_lags = list(range(0, N_XD_PAST + 1))
xd_cols = np.column_stack([xdot[N_skip - lag : N_skip - lag + Nv] for lag in xd_lags])
xd_names = [f"xdot{-lag:+d}" for lag in xd_lags]

# full feature matrix
Theta = np.column_stack([np.ones(Nv), eta_cols, x_cols, xd_cols])
feat_names = ["1"] + eta_names + x_names + xd_names

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


w_fe, active_fe = stlsq(Theta, fe_v, threshold=THRESHOLD)
w_ftot, active_ftot = stlsq(Theta, f_tot_v, threshold=THRESHOLD)

# %% RESULTS

fe_pred = Theta @ w_fe
ftot_pred = Theta @ w_ftot

err_fe = metricError(fe_pred, fe_v, "nrmse_range")
err_ftot = metricError(ftot_pred, f_tot_v, "nrmse_range")
print(f"\nNRMSE  fe={err_fe:.4e}   f_tot={err_ftot:.4e}")

for label, w, active in [("fe", w_fe, active_fe), ("f_tot", w_ftot, active_ftot)]:
    print(f"\nActive features for {label} ({active.sum()}/{len(feat_names)}):")
    for name, coef, act in zip(feat_names, w, active):
        if act:
            print(f"  {name:>12s}  {coef:+.6e}")

# %% COEFFICIENT PLOTS


def plot_coefs(w, active, title, fname):
    n_eta = len(eta_lags)
    n_x = len(x_lags)
    n_xd = len(xd_lags)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(title)

    # eta
    eta_c_ = w[1 : 1 + n_eta]
    axes[0].bar(
        eta_lags, eta_c_, color=["C1" if v != 0 else "lightgrey" for v in eta_c_]
    )
    axes[0].axvline(0, color="red", ls="--", lw=0.8)
    axes[0].set(xlabel="lag (steps)", ylabel="coef [N/m]", title="η lags")
    axes[0].grid(True, axis="y")

    # x
    x_c_ = w[1 + n_eta : 1 + n_eta + n_x]
    axes[1].bar(
        [-k for k in x_lags],
        x_c_,
        color=["C2" if v != 0 else "lightgrey" for v in x_c_],
    )
    axes[1].set(xlabel="lag (steps)", ylabel="coef [N/m]", title="x lags")
    axes[1].grid(True, axis="y")

    # xdot
    xd_c_ = w[1 + n_eta + n_x : 1 + n_eta + n_x + n_xd]
    axes[2].bar(
        [-k for k in xd_lags],
        xd_c_,
        color=["C3" if v != 0 else "lightgrey" for v in xd_c_],
    )
    axes[2].set(xlabel="lag (steps)", ylabel="coef [N·s/m]", title="xdot lags")
    axes[2].grid(True, axis="y")

    plt.tight_layout()
    Plotter._save(fig, OUT, None, fname)


plot_coefs(w_fe, active_fe, f"fe    prediction (threshold={THRESHOLD})", "fe_coefs")
plot_coefs(
    w_ftot, active_ftot, f"f_tot prediction (threshold={THRESHOLD})", "ftot_coefs"
)

# %% TIME SERIES PLOT

fig, ax = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
fig.suptitle(
    f"Force prediction  N_eta=[{N_ETA_PAST},{N_ETA_FUT}]  "
    f"N_x={N_X_PAST}  N_xd={N_XD_PAST}"
)

ax[0].plot(t_v, fe_v, color="black", lw=1.5, label="ref fe")
ax[0].plot(
    t_v, fe_pred, ls="--", lw=1, color="C1", label=f"predicted  NRMSE={err_fe:.3e}"
)
ax[0].set(ylabel="fe [N]")
ax[0].legend()
ax[0].grid(True)

ax[1].plot(t_v, fe_v - fe_pred, lw=0.7, color="C1", label="residual fe")
ax[1].axhline(0, color="k", lw=0.5)
ax[1].set(ylabel="residual [N]")
ax[1].legend()
ax[1].grid(True)

ax[2].plot(t_v, f_tot_v, color="black", lw=1.5, label="ref f_tot")
ax[2].plot(
    t_v, ftot_pred, ls="--", lw=1, color="C2", label=f"predicted  NRMSE={err_ftot:.3e}"
)
ax[2].set(ylabel="f_tot [N]")
ax[2].legend()
ax[2].grid(True)

ax[3].plot(t_v, f_tot_v - ftot_pred, lw=0.7, color="C2", label="residual f_tot")
ax[3].axhline(0, color="k", lw=0.5)
ax[3].set(ylabel="residual [N]", xlabel="t [s]")
ax[3].legend()
ax[3].grid(True)

plt.tight_layout()
plt.show()
