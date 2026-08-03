# %% PACKAGES

import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import lsim
from scipy.signal import StateSpace as ScipySS
from scipy.integrate import cumulative_trapezoid

from source.config import MODEL_DIR, OUT_DIR
from source.function.sea_state_utils import param2S, S2eta
from source.function.error_utils import metricError
from source.classes.SimRun import SimRun, get_solver
from source.classes.Model import Model
from source.classes.Plotter import Plotter

OUT_SI = OUT_DIR / "si"
OUT = OUT_SI
OUT.mkdir(parents=True, exist_ok=True)


def to_dB(x):
    return 20 * np.log10(np.maximum(x, 1e-30))


# %% INPUTS

PKL_NAME = "r5"
REF_MODEL = "linear"
DT = 0.05
T_WAVE = 8.0
H_WAVE = 2.0
T_VAL_DUR = 300.0
T_WARMUP = 60.0  # s — skip transient

# %% LOAD HYDRO SPHERE

with open(MODEL_DIR / "bem" / f"{PKL_NAME}.pkl", "rb") as f:
    hs = pickle.load(f)

# %% LOAD IDENTIFIED STATE-SPACE MODELS

eta2f = np.load(OUT_SI / "model_eta2f.npz")
A_e = eta2f["A"]
B_e = eta2f["B"]
C_e = eta2f["C"]
D_e = eta2f["D"]
tau_e = float(eta2f["tau"][0])
ord_e = int(eta2f["order"][0])

f2v = np.load(OUT_SI / "model_f2v.npz")
A_f = f2v["A"]
B_f = f2v["B"]
C_f = f2v["C"]
D_f = f2v["D"]

print(f"eta2f: order={ord_e}  tau={tau_e:.3f}s  A:{A_e.shape}")
print(f"f2v:   A:{A_f.shape}")

# %% BUILD CASCADE STATE-SPACE  η(t+τ) → ẋ(t)
#
#  System 1 (eta2f):  ẋ_e = A_e x_e + B_e η,   fe  = C_e x_e   (D_e=0)
#  System 2 (f2v):    ẋ_f = A_f x_f + B_f fe,   ẋ   = C_f x_f  (D_f=0)
#
#  Combined z = [x_e; x_f]:
#    dz/dt = [A_e,      0 ] z + [B_e] η
#            [B_f C_e,  A_f]     [0  ]
#    ẋ     = [0,  C_f] z

n_e = A_e.shape[0]
n_f = A_f.shape[0]

A_full = np.block(
    [
        [A_e, np.zeros((n_e, n_f))],
        [B_f @ C_e, A_f],
    ]
)
B_full = np.vstack([B_e, np.zeros((n_f, 1))])
C_full = np.hstack([np.zeros((1, n_e)), C_f])
D_full = np.zeros((1, 1))

sys_full = ScipySS(A_full, B_full, C_full, D_full)
print(f"cascade: A:{A_full.shape} — η(t+τ) → ẋ(t)")

# save fused model
np.savez(
    OUT_SI / "model_eta2f2v.npz",
    A=A_full,
    B=B_full,
    C=C_full,
    D=D_full,
    tau=np.array([tau_e]),
)
print(f"Fused model saved → {OUT_SI / 'model_eta2f2v.npz'}")

# %% VALIDATION WAVE

w_s = np.arange(0.01, 4.0, 0.01)
S_s = param2S(T_WAVE, H_WAVE, w_s, gamma=3.3, type="JONSWAP")
t_v = np.arange(0, T_VAL_DUR, DT)
t_v, eta_v, _, _ = S2eta(w_s, S_s, t_v)

# %% REFERENCE SIMULATION

with open(MODEL_DIR / REF_MODEL / "model.json") as f:
    mdef = json.load(f)

ref = SimRun.simulate(
    Model.from_config(
        mdef, hs, eta_t=t_v, eta_values=eta_v, pto={"damping": 0, "stiffness": 0}
    ),
    get_solver("RK4"),
    t0=t_v[0],
    t_end=t_v[-1],
    dt=DT,
    eta_t=t_v,
    eta_values=eta_v,
    label="ref",
)
x_ref = ref.dataset["x"].to_numpy()
xd_ref = ref.dataset["xdot"].to_numpy()
fe_ref = ref.dataset["fe_lin"].to_numpy()

# %% RUN CASCADE  η(t + τ*) → ẋ(t) → x(t)

tau_samp = int(np.round(tau_e / DT))
eta_adv = eta_v[tau_samp:]  # η(t + τ*) — advanced wave
t_sim = t_v[: len(eta_adv)]
x_ref_sim = x_ref[: len(eta_adv)]
xd_ref_sim = xd_ref[: len(eta_adv)]
fe_ref_sim = fe_ref[: len(eta_adv)]

x0_full = -np.linalg.pinv(C_full) @ (D_full @ eta_adv[0:1])
_, xd_pred, _ = lsim(sys_full, eta_adv, t_sim, X0=x0_full.ravel())
x_pred = cumulative_trapezoid(xd_pred, t_sim, initial=0)
x_pred -= np.mean((x_pred - x_ref_sim)[t_sim >= T_WARMUP])  # remove drift

msk = t_sim >= T_WARMUP

# eta2f alone for diagnostics
sys_e = ScipySS(A_e, B_e, C_e, D_e)
_, fe_pred, _ = lsim(sys_e, eta_adv, t_sim, X0=np.zeros(n_e))

# f2v with TRUE fe — isolates eta2f error contribution
sys_f = ScipySS(A_f, B_f, C_f, D_f)
_, xd_true_fe, _ = lsim(sys_f, fe_ref_sim, t_sim, X0=np.zeros(n_f))
x_true_fe = cumulative_trapezoid(xd_true_fe, t_sim, initial=0)
x_true_fe -= np.mean((x_true_fe - x_ref_sim)[msk])

# %% METRICS

err_fe = metricError(fe_pred[msk], fe_ref_sim[msk], "nrmse_range")
err_xd = metricError(xd_pred[msk], xd_ref_sim[msk], "nrmse_range")
err_x = metricError(x_pred[msk], x_ref_sim[msk], "nrmse_range")
err_xd_truefe = metricError(xd_true_fe[msk], xd_ref_sim[msk], "nrmse_range")
err_x_truefe = metricError(x_true_fe[msk], x_ref_sim[msk], "nrmse_range")
print(f"NRMSE  fe={err_fe:.4f}")
print(f"NRMSE  xdot={err_xd:.4f}   x={err_x:.4f}          (estimated fe)")
print(
    f"NRMSE  xdot={err_xd_truefe:.4f}   x={err_x_truefe:.4f}          (true fe — f2v only)"
)

# %% PLOTS

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
fig.suptitle(
    f"eta2f2v validation — T={T_WAVE}s H={H_WAVE}m  τ={tau_e:.2f}s  "
    f"eta2f order={ord_e}  f2v order={n_f}"
)

axes[0].plot(t_sim[msk], fe_ref_sim[msk], color="black", lw=1.5, label="ref fe_lin")
axes[0].plot(
    t_sim[msk],
    fe_pred[msk],
    color="C0",
    ls="--",
    lw=1,
    label=f"eta2f predicted  NRMSE={err_fe:.3f}",
)
axes[0].set(ylabel="fe [N]")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(t_sim[msk], xd_ref_sim[msk], color="black", lw=1.5, label="ref xdot")
axes[1].plot(
    t_sim[msk],
    xd_true_fe[msk],
    color="C2",
    ls="-.",
    lw=1.2,
    label=f"f2v (true fe)  NRMSE={err_xd_truefe:.3f}",
)
axes[1].plot(
    t_sim[msk],
    xd_pred[msk],
    color="C1",
    ls="--",
    lw=1,
    label=f"cascade (estimated fe)  NRMSE={err_xd:.3f}",
)
axes[1].set(ylabel="xdot [m/s]")
axes[1].legend()
axes[1].grid(True)

axes[2].plot(t_sim[msk], x_ref_sim[msk], color="black", lw=1.5, label="ref x")
axes[2].plot(
    t_sim[msk],
    x_true_fe[msk],
    color="C2",
    ls="-.",
    lw=1.2,
    label=f"f2v (true fe)  NRMSE={err_x_truefe:.3f}",
)
axes[2].plot(
    t_sim[msk],
    x_pred[msk],
    color="C1",
    ls="--",
    lw=1,
    label=f"cascade (estimated fe)  NRMSE={err_x:.3f}",
)
axes[2].set(ylabel="x [m]", xlabel="t [s]")
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
Plotter._save(fig, OUT, None, "validation_eta2f2v")

plt.show()
