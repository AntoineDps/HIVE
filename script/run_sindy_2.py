# %% PACKAGES

import json
import pickle

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import pysindy as ps

from scipy.signal import cont2discrete
from scipy.linalg import expm

from source.config import MODEL_DIR, OUT_DIR
from source.function.sea_state_utils import param2S, S2eta
from source.function.error_utils import metricError
from source.classes.SimRun import SimRun, get_solver
from source.classes.Model import Model
from source.classes.Plotter import Plotter


# %% OUTPUT

OUT = OUT_DIR / "sindy_regularized_pto"
OUT.mkdir(parents=True, exist_ok=True)


# %% INPUTS

PKL_NAME = "r5"
REF_MODEL = "linear"

DT = 0.05  # sampling time [s]


# Training wave
T_TRAIN = 8.0
H_TRAIN = 2.0
T_TRAIN_DUR = 600.0


# Validation wave
T_VAL = 8.0
H_VAL = 2.0
T_VAL_DUR = 300.0


# Remove initial transient
T_WARMUP = 60.0


# PTO damping

B_PTO = 200000  # N s/m

K_PTO = 0.0  # N/m


# SINDy
POLY_DEGREE = 1

THRESHOLD = 1e-5

ALPHA = 1e-3

MAX_ITER = 100

UNBIAS = True


# %% LOAD HYDRODYNAMIC MODEL

print("Loading hydrodynamic model...")

with open(
    MODEL_DIR / "bem" / f"{PKL_NAME}.pkl",
    "rb",
) as f:
    hs = pickle.load(f)


with open(MODEL_DIR / REF_MODEL / "model.json") as f:
    mdef = json.load(f)


# %% EFFECTIVE MASS DIAGNOSTIC

m_body = hs.mass
m_added = hs.ma_inf
m_eff = m_body + m_added

expected_pto_coeff = -B_PTO / m_eff


print("\nHydrodynamic parameters:")

print(f"  Body mass       = {m_body}")
print(f"  Added mass      = {m_added}")
print(f"  Effective mass  = {m_eff}")

print("\nPTO:")

print(f"  B_PTO           = {B_PTO} N s/m")

print(f"  Expected PTO contribution to xdot coefficient = {expected_pto_coeff:.6f} 1/s")


# %% GENERATE TRAINING DATA

print("\nGenerating training data...")

w_s = np.arange(
    0.01,
    4.0,
    0.01,
)


S_s = param2S(
    T_TRAIN,
    H_TRAIN,
    w_s,
    gamma=3.3,
    type="JONSWAP",
)


t_tr = np.arange(
    0,
    T_TRAIN_DUR,
    DT,
)


t_tr, eta_tr, _, _ = S2eta(
    w_s,
    S_s,
    t_tr,
)


# %% SIMULATE TRAINING WEC WITH PTO

print(f"Simulating WEC with PTO damping B_PTO = {B_PTO:.0f} N s/m...")


sr_tr = SimRun.simulate(
    Model.from_config(
        mdef,
        hs,
        eta_t=t_tr,
        eta_values=eta_tr,
        pto={
            "damping": B_PTO,
            "stiffness": K_PTO,
        },
    ),
    get_solver("RK4"),
    t0=t_tr[0],
    t_end=t_tr[-1],
    dt=DT,
    eta_t=t_tr,
    eta_values=eta_tr,
    label="train",
)


# %% EXTRACT TRAINING DATA

t_raw = sr_tr.dataset["t"].to_numpy()

x_raw = sr_tr.dataset["x"].to_numpy()

xdot_raw = sr_tr.dataset["xdot"].to_numpy()

eta_raw = sr_tr.dataset["eta"].to_numpy()


# Remove warmup

msk_tr = t_raw >= T_WARMUP


t_tr_c = t_raw[msk_tr]

x_tr = x_raw[msk_tr]

xdot_tr = xdot_raw[msk_tr]

eta_tr_c = eta_raw[msk_tr]


# State:
#
# X = [x, xdot]

X_tr = np.column_stack(
    [
        x_tr,
        xdot_tr,
    ]
)


# Input:
#
# U = [eta]
#
# PTO force is intentionally NOT included.

U_tr = eta_tr_c[:, None]


print(f"  Training samples: {len(t_tr_c)}")

print(f"  X shape: {X_tr.shape}")

print(f"  U shape: {U_tr.shape}")


# %% COMPUTE STATE DERIVATIVES

# xdot is directly available from the simulator.
#
# Therefore:
#
# d/dt [x   ] = [xdot]
#     [xdot]   [xddot]
#
# Only xddot is estimated numerically.

xddot_tr = np.gradient(
    xdot_tr,
    DT,
)


Xdot_tr = np.column_stack(
    [
        xdot_tr,
        xddot_tr,
    ]
)


# %% PLOT TRAINING DATA

fig, ax = plt.subplots(
    4,
    1,
    figsize=(14, 8),
    sharex=True,
)


fig.suptitle(f"SINDy training data — B_PTO={B_PTO:.0f} N s/m")


ax[0].plot(
    t_tr_c,
    x_tr,
    lw=0.6,
)

ax[0].set_ylabel("x [m]")

ax[0].grid(True)


ax[1].plot(
    t_tr_c,
    xdot_tr,
    lw=0.6,
)

ax[1].set_ylabel("xdot [m/s]")

ax[1].grid(True)


ax[2].plot(
    t_tr_c,
    xddot_tr,
    lw=0.6,
)

ax[2].set_ylabel("xddot [m/s²]")

ax[2].grid(True)


ax[3].plot(
    t_tr_c,
    eta_tr_c,
    lw=0.6,
)

ax[3].set_ylabel("η [m]")

ax[3].set_xlabel("t [s]")

ax[3].grid(True)


plt.tight_layout()


Plotter._save(
    fig,
    OUT,
    None,
    "training_data",
)


# %% FIT REGULARIZED SINDY

print("\nFitting regularized SINDy...")

print(f"  Polynomial degree = {POLY_DEGREE}")

print(f"  STLSQ threshold   = {THRESHOLD}")

print(f"  Regularization    = {ALPHA}")

print(f"  Unbias             = {UNBIAS}")


# With degree=1:
#
# [1, x, xdot, eta]

feature_lib = ps.PolynomialLibrary(
    degree=POLY_DEGREE,
    include_bias=True,
)


optimizer = ps.STLSQ(
    threshold=THRESHOLD,
    alpha=ALPHA,
    max_iter=MAX_ITER,
    unbias=UNBIAS,
)


sindy_model = ps.SINDy(
    optimizer=optimizer,
    feature_library=feature_lib,
)


# Supply derivatives explicitly.

sindy_model.fit(
    X_tr,
    u=U_tr,
    x_dot=Xdot_tr,
    t=DT,
    feature_names=[
        "x",
        "xdot",
        "eta",
    ],
)


# %% PRINT IDENTIFIED MODEL

print("\nIdentified equations:")

sindy_model.print()


# %% COEFFICIENTS

feature_names = sindy_model.get_feature_names()

coefs = sindy_model.coefficients()


df_coef = pd.DataFrame(
    coefs.T,
    index=feature_names,
    columns=[
        "ẋ  (dx/dt)",
        "ẍ  (dxdot/dt)",
    ],
)


df_coef.to_csv(OUT / "sindy_coefficients.csv")


print("\nCoefficients:")

print(df_coef.to_string())


# %% EXTRACT IDENTIFIED COEFFICIENTS

# Feature ordering:
#
# [1, x, xdot, eta]

coef = sindy_model.coefficients()


c_identified = coef[1, 0]

a_x = coef[1, 1]

a_v = coef[1, 2]

a_eta = coef[1, 3]


print("\nIdentified physical coefficients:")

print(f"  x coefficient       = {a_x:.8f} 1/s²")

print(f"  xdot coefficient    = {a_v:.8f} 1/s")

print(f"  eta coefficient     = {a_eta:.8f}")

print(f"  constant            = {c_identified:.8e}")


# %% PTO DIAGNOSTIC

print("\nPTO damping diagnostic:")

print(f"  B_PTO                         = {B_PTO:.6f} N s/m")

print(f"  Effective mass                = {m_eff:.6f} kg")

print(f"  Expected PTO contribution     = {expected_pto_coeff:.6f} 1/s")

print(f"  Identified total xdot coeff   = {a_v:.6f} 1/s")


# %% DERIVATIVE FIT

Xdot_pred = sindy_model.predict(
    X_tr,
    u=U_tr,
)


err_xdot_derivative = metricError(
    Xdot_pred[:, 0],
    Xdot_tr[:, 0],
    "nrmse_range",
)


err_xddot_derivative = metricError(
    Xdot_pred[:, 1],
    Xdot_tr[:, 1],
    "nrmse_range",
)


print("\nDerivative fit:")

print(f"  xdot  NRMSE = {err_xdot_derivative:.6f}")

print(f"  xddot NRMSE = {err_xddot_derivative:.6f}")


# %% TRAINING R2

score_tr = sindy_model.score(
    X_tr,
    u=U_tr,
    x_dot=Xdot_tr,
    t=DT,
)


print(f"\nTraining derivative R² = {score_tr:.6f}")


# %% CONTINUOUS-TIME STATE-SPACE MODEL

# Continuous model:
#
# Xdot = A_c X + B_c eta + c_c

A_c = coef[:, [1, 2]]

B_c = coef[:, [3]]

c_c = coef[:, [0]]


print("\nContinuous-time SINDy model:")

print("\nA_c =")

print(A_c)


print("\nB_c =")

print(B_c)


print("\nc_c =")

print(c_c)


# %% DISCRETIZE LINEAR PART

C_c = np.eye(2)

D_c = np.zeros((2, 1))


A_d, B_d, C_d, D_d, _ = cont2discrete(
    (
        A_c,
        B_c,
        C_c,
        D_c,
    ),
    DT,
    method="zoh",
)


# %% DISCRETIZE AFFINE TERM

# We need to account for:
#
# Xdot = A X + B eta + c
#
# by augmenting the continuous system.

A_aug = np.zeros((3, 3))


A_aug[:2, :2] = A_c

A_aug[:2, 2:] = c_c


A_aug_d = expm(A_aug * DT)


A_d_affine = A_aug_d[
    :2,
    :2,
]


c_d = A_aug_d[
    :2,
    2,
]


print("\nDiscrete-time SINDy model:")

print("\nA_d =")

print(A_d_affine)


print("\nB_d =")

print(B_d)


print("\nc_d =")

print(c_d)


# %% SAVE MATRICES

np.save(
    OUT / "A_c.npy",
    A_c,
)

np.save(
    OUT / "B_c.npy",
    B_c,
)

np.save(
    OUT / "c_c.npy",
    c_c,
)

np.save(
    OUT / "A_d.npy",
    A_d_affine,
)

np.save(
    OUT / "B_d.npy",
    B_d,
)

np.save(
    OUT / "c_d.npy",
    c_d,
)


# %% GENERATE VALIDATION DATA

print("\nGenerating validation data...")


w_v = np.arange(
    0.01,
    4.0,
    0.01,
)


S_v = param2S(
    T_VAL,
    H_VAL,
    w_v,
    gamma=3.3,
    type="JONSWAP",
)


t_v = np.arange(
    0,
    T_VAL_DUR,
    DT,
)


# Different realization

t_v, eta_v, _, _ = S2eta(
    w_v,
    S_v,
    t_v,
)


# %% SIMULATE VALIDATION WEC WITH SAME PTO

print(f"Simulating validation WEC with PTO damping B_PTO = {B_PTO:.0f} N s/m...")


sr_v = SimRun.simulate(
    Model.from_config(
        mdef,
        hs,
        eta_t=t_v,
        eta_values=eta_v,
        pto={
            "damping": B_PTO,
            "stiffness": K_PTO,
        },
    ),
    get_solver("RK4"),
    t0=t_v[0],
    t_end=t_v[-1],
    dt=DT,
    eta_t=t_v,
    eta_values=eta_v,
    label="val",
)


# %% EXTRACT VALIDATION DATA

t_raw = sr_v.dataset["t"].to_numpy()

x_raw = sr_v.dataset["x"].to_numpy()

xdot_raw = sr_v.dataset["xdot"].to_numpy()

eta_raw = sr_v.dataset["eta"].to_numpy()


msk_v = t_raw >= T_WARMUP


t_v_c = t_raw[msk_v]

x_ref = x_raw[msk_v]

xdot_ref = xdot_raw[msk_v]

eta_v_c = eta_raw[msk_v]


# Initial condition

X_v0 = np.array(
    [
        x_ref[0],
        xdot_ref[0],
    ]
)


# %% DISCRETE-TIME VALIDATION

print("\nSimulating discrete SINDy model...")


N = len(t_v_c)


X_sindy = np.zeros((N, 2))


X_sindy[0] = X_v0


# X[k+1] =
#
# A_d X[k]
# + B_d eta[k]
# + c_d

for k in range(N - 1):
    X_sindy[k + 1] = A_d_affine @ X_sindy[k] + B_d[:, 0] * eta_v_c[k] + c_d


x_sindy = X_sindy[:, 0]

xdot_sindy = X_sindy[:, 1]


# %% VALIDATION ERRORS

err_x = metricError(
    x_sindy,
    x_ref,
    "nrmse_range",
)


err_xdot = metricError(
    xdot_sindy,
    xdot_ref,
    "nrmse_range",
)


print("\nValidation NRMSE:")

print(f"  x     = {err_x:.6f}")

print(f"  xdot  = {err_xdot:.6f}")


# %% VALIDATION PLOTS

fig, ax = plt.subplots(
    2,
    1,
    figsize=(14, 7),
    sharex=True,
)


fig.suptitle(
    f"Regularized SINDy WEC validation — B_PTO={B_PTO:.0f} N s/m, alpha={ALPHA:.1e}"
)


# Position

ax[0].plot(
    t_v_c,
    x_ref,
    color="black",
    lw=1.5,
    label="reference WEC",
)


ax[0].plot(
    t_v_c,
    x_sindy,
    "--",
    lw=1.0,
    label=f"SINDy NRMSE={err_x:.3f}",
)


ax[0].set_ylabel("x [m]")

ax[0].legend()

ax[0].grid(True)


# Velocity

ax[1].plot(
    t_v_c,
    xdot_ref,
    color="black",
    lw=1.5,
    label="reference WEC",
)


ax[1].plot(
    t_v_c,
    xdot_sindy,
    "--",
    lw=1.0,
    label=f"SINDy NRMSE={err_xdot:.3f}",
)


ax[1].set_ylabel("xdot [m/s]")

ax[1].set_xlabel("t [s]")

ax[1].legend()

ax[1].grid(True)


plt.tight_layout()


Plotter._save(
    fig,
    OUT,
    None,
    "validation",
)


# %% COEFFICIENT HEATMAP

fig, ax = plt.subplots(
    figsize=(
        max(6, len(feature_names) * 0.6),
        4,
    )
)


im = ax.imshow(
    np.abs(coefs),
    aspect="auto",
    cmap="Blues",
)


ax.set_xticks(range(len(feature_names)))


ax.set_xticklabels(
    feature_names,
    rotation=45,
    ha="right",
    fontsize=8,
)


ax.set_yticks(
    [
        0,
        1,
    ]
)


ax.set_yticklabels(
    [
        "ẋ",
        "ẍ",
    ]
)


plt.colorbar(
    im,
    ax=ax,
    label="|coefficient|",
)


ax.set_title("Regularized SINDy coefficient magnitudes")


plt.tight_layout()


Plotter._save(
    fig,
    OUT,
    None,
    "coefficients",
)


# %% CHECK EIGENVALUES

eig_c = np.linalg.eigvals(A_c)


eig_d = np.linalg.eigvals(A_d_affine)


print("\nContinuous-time eigenvalues:")

for val in eig_c:
    print(f"  {val}")


print("\nDiscrete-time eigenvalues:")

for val in eig_d:
    print(f"  {val}")


print(
    "\nContinuous-time stable:",
    np.all(np.real(eig_c) < 0),
)


print(
    "Discrete-time stable:",
    np.all(np.abs(eig_d) < 1.0),
)


# %% DONE

print(f"\nDone — outputs in {OUT}")


plt.show()
