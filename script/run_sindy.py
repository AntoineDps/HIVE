# %% PACKAGES

import pickle
import numpy as np
import matplotlib.pyplot as plt
import pysindy as ps

from source.config import MODEL_DIR
from source.function.sea_state_utils import param2S, S2eta
from scipy.signal import cont2discrete

# %% PARAMETERS

DT = 0.01
T_END = 100.0
PKL_NAME = "r5"
THRESHOLD = 1e-7

# %% LOAD HYDRO

with open(MODEL_DIR / "bem" / f"{PKL_NAME}.pkl", "rb") as f:
    hs = pickle.load(f)

m_tot = hs.mass + hs.ma_inf
K_hs = hs.stiffness
B_damp = 50_000.0

print(f"m={m_tot:.0f} kg  K={K_hs:.0f} N/m  B={B_damp:.0f} N·s/m")
print(
    f"True: xddot = {-K_hs / m_tot:.4f} x {-B_damp / m_tot:+.4f} xdot + {1 / m_tot:.4e} fe"
)

# %% DATA GENERATION

t = np.arange(0.0, T_END, DT)
N = len(t)

# wave and excitation force
w_s = np.arange(0.01, 4.0, 0.01)
S_s = param2S(8.0, 2.0, w_s, gamma=3.3, type="JONSWAP")
t_w, eta_w, _, _ = S2eta(w_s, S_s, t)
eta = np.interp(t, t_w, eta_w)

Nf = len(t)
f_spec = np.fft.rfftfreq(Nf, DT)
Fe_mod = np.interp(f_spec, hs.w / (2 * np.pi), hs.Fe_mod)
Fe_ang = np.interp(f_spec, hs.w / (2 * np.pi), np.unwrap(hs.Fe_ang))
fe = np.fft.irfft(Fe_mod * np.exp(1j * Fe_ang) * np.fft.rfft(eta), n=Nf)

# RK4 simulation — mass-spring-damper with fe as input
x = np.zeros(N)
x[0] = 0.0
xdot = np.zeros(N)
xdot[0] = 0.0


def dyn(x, xdot, fe_i):
    return xdot, -K_hs / m_tot * x - B_damp / m_tot * xdot + fe_i / m_tot


for i in range(N - 1):
    k1 = dyn(x[i], xdot[i], fe[i])
    k2 = dyn(x[i] + 0.5 * DT * k1[0], xdot[i] + 0.5 * DT * k1[1], fe[i])
    k3 = dyn(x[i] + 0.5 * DT * k2[0], xdot[i] + 0.5 * DT * k2[1], fe[i])
    k4 = dyn(x[i] + DT * k3[0], xdot[i] + DT * k3[1], fe[i])
    x[i + 1] = x[i] + DT / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
    xdot[i + 1] = xdot[i] + DT / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])

# %% FIT SINDY

X = np.column_stack([x, xdot])

lib = ps.PolynomialLibrary(degree=1, include_bias=True)
opt = ps.STLSQ(threshold=THRESHOLD)

# --- case A: u = fe ---
model_fe = ps.SINDy(optimizer=opt, feature_library=lib)
model_fe.fit(X, u=fe.reshape(-1, 1), t=DT, feature_names=["x", "xdot", "fe"])
print("\nSINDy (u=fe):")
model_fe.print()

# --- case B: u = eta ---
model_eta = ps.SINDy(
    optimizer=ps.STLSQ(threshold=THRESHOLD),
    feature_library=ps.PolynomialLibrary(degree=1, include_bias=True),
)
model_eta.fit(X, u=eta.reshape(-1, 1), t=DT, feature_names=["x", "xdot", "eta"])
print("\nSINDy (u=eta):")
model_eta.print()

# %% DISCRETISE


def build_discrete_ss(model, control_col):
    coefs = model.coefficients()
    fnames = list(model.get_feature_names())

    def cv(row, name):
        return coefs[row, fnames.index(name)] if name in fnames else 0.0

    A_c = np.array([[cv(0, "x"), cv(0, "xdot")], [cv(1, "x"), cv(1, "xdot")]])
    B_c = np.array([[cv(0, control_col)], [cv(1, control_col)]])
    bias = np.array([cv(0, "1"), cv(1, "1")])
    B_ext = np.hstack([B_c, bias[:, None]])
    Ad, Bd_ext, _, _, _ = cont2discrete(
        (A_c, B_ext, np.eye(2), np.zeros((2, 3))), dt=DT, method="zoh"
    )
    return Ad, Bd_ext[:, :1], Bd_ext[:, 1]


Ad_fe, Bd_fe, cd_fe = build_discrete_ss(model_fe, "fe")
Ad_eta, Bd_eta, cd_eta = build_discrete_ss(model_eta, "eta")

coefs = model_fe.coefficients()
fnames = list(model_fe.get_feature_names())


# %% VALIDATION


def simulate_discrete(Ad, Bd, cd, X0, U_seq):
    X_s = np.zeros((len(U_seq), 2))
    X_s[0] = X0
    for i in range(len(U_seq) - 1):
        X_s[i + 1] = Ad @ X_s[i] + Bd @ U_seq[i] + cd
    return X_s


X_fe = simulate_discrete(Ad_fe, Bd_fe, cd_fe, X[0], fe.reshape(-1, 1))
X_eta = simulate_discrete(Ad_eta, Bd_eta, cd_eta, X[0], eta.reshape(-1, 1))


def nrmse(pred, true):
    return np.sqrt(np.mean((pred - true) ** 2)) / (np.max(true) - np.min(true))


print(
    f"\nNRMSE (u=fe):   x={nrmse(X_fe[:, 0], x):.4e}   xdot={nrmse(X_fe[:, 1], xdot):.4e}"
)
print(
    f"NRMSE (u=eta):  x={nrmse(X_eta[:, 0], x):.4e}   xdot={nrmse(X_eta[:, 1], xdot):.4e}"
)

# %% PLOT

fig, ax = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
ax[0].plot(t, x, color="black", lw=1.5, label="true")
ax[0].plot(
    t,
    X_fe[:, 0],
    ls="--",
    lw=1,
    color="C1",
    label=f"u=fe   NRMSE={nrmse(X_fe[:, 0], x):.2e}",
)
ax[0].plot(
    t,
    X_eta[:, 0],
    ls=":",
    lw=1,
    color="C2",
    label=f"u=eta  NRMSE={nrmse(X_eta[:, 0], x):.2e}",
)
ax[0].set(ylabel="x [m]")
ax[0].legend()
ax[0].grid(True)

ax[1].plot(t, xdot, color="black", lw=1.5, label="true")
ax[1].plot(
    t,
    X_fe[:, 1],
    ls="--",
    lw=1,
    color="C1",
    label=f"u=fe   NRMSE={nrmse(X_fe[:, 1], xdot):.2e}",
)
ax[1].plot(
    t,
    X_eta[:, 1],
    ls=":",
    lw=1,
    color="C2",
    label=f"u=eta  NRMSE={nrmse(X_eta[:, 1], xdot):.2e}",
)
ax[1].set(ylabel="xdot [m/s]", xlabel="t [s]")
ax[1].legend()
ax[1].grid(True)

fig.suptitle(f"SINDy — mass-spring-damper  B={B_damp:.0f} N·s/m")
plt.tight_layout()
plt.show()
