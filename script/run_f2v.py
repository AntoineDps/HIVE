# %% PACKAGES

import json
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import chirp, welch, csd
# import control
# import vectfit

from source.config import MODEL_DIR, OUT_DIR
from source.function.sea_state_utils import param2S, S2eta
from source.function.error_utils import metricError
from source.classes.SimRun import SimRun, get_solver
from source.classes.Model import Model
from source.classes.Plotter import Plotter

OUT = OUT_DIR / "f2v"
OUT.mkdir(parents=True, exist_ok=True)

# %% FUNCTION


def simulate_f2v(f_ext, t, M, K, Ar, Br, Cr, Dr, n_r):
    dt = float(t[1] - t[0])
    z = np.zeros(2 + n_r)
    x_out = np.empty(len(t))
    xd_out = np.empty(len(t))

    def dz(z_, f_):
        x_, xd_ = z_[0], z_[1]
        qr = z_[2:]
        f_rad = Cr @ qr + Dr * xd_
        xdd = (f_ - K * x_ - f_rad) / M
        return np.concatenate([[xd_, xdd], Ar @ qr + Br * xd_])

    for i, fi in enumerate(f_ext):
        x_out[i], xd_out[i] = z[0], z[1]
        k1 = dz(z, fi)
        k2 = dz(z + dt / 2 * k1, fi)
        k3 = dz(z + dt / 2 * k2, fi)
        k4 = dz(z + dt * k3, fi)
        z = z + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x_out, xd_out


# %% INPUTS

PKL_NAME = "r5"
DT = 0.05  # s
T_CHIRP = 1000.0  # s   — long chirp for good FRF resolution
W_START = 0.22  # rad/s
W_END = 3.14  # rad/s
AMP = 1000000.0  # N

N_POLES = 6  # number of poles for Vector Fitting
T_TRANS = 30.0  # s   — transient to remove

# Validation wave
T_WAVE = 8.0  # s   peak period
H_WAVE = 2.0  # m   Hs
T_VAL_DUR = 300.0  # s

REF_MODEL = "linear_sphere_r5"

# %% LOAD HYDRO SPHERE

with open(MODEL_DIR / "bem" / f"{PKL_NAME}.pkl", "rb") as f:
    hs = pickle.load(f)

m = hs.mass
ma = hs.ma_inf
K = hs.stiffness
rs = hs.rad_ss
Ar = np.array(rs["Ar"])
Br = np.array(rs["Br"])
Cr = np.array(rs["Cr"])
Dr = np.array(rs["Dr"])
n_r = Ar.shape[0]
M = m + ma

# %% MAKE CHIRP SIGNAL

t_ch = np.arange(0, T_CHIRP, DT)
f_ch = AMP * chirp(
    t_ch, f0=W_START / (2 * np.pi), f1=W_END / (2 * np.pi), t1=T_CHIRP, method="linear"
)

# plot chirp
fig, ax = plt.subplots(2, 1, figsize=(12, 5))
ax[0].plot(t_ch, f_ch, lw=0.5)
ax[0].set(ylabel="f [N]", title="Chirp")
ax[0].grid(True)
w_spec_ch = 2 * np.pi * np.fft.rfftfreq(len(t_ch), DT)
ax[1].plot(w_spec_ch, np.abs(np.fft.rfft(f_ch)), lw=0.7)
ax[1].set(xlabel="ω [rad/s]", ylabel="|F|", xlim=[0, W_END * 1.3])
ax[1].grid(True)
plt.tight_layout()
Plotter._save(fig, OUT, None, "chirp")

# %% SIMULATE

x_ch, xd_ch = simulate_f2v(f_ch, t_ch, M, K, Ar, Br, Cr, Dr, n_r)

pd.DataFrame({"t": t_ch, "f": f_ch, "x": x_ch, "xdot": xd_ch}).to_csv(
    OUT / "chirp_response.csv", index=False
)

# plot response
fig, ax = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
ax[0].plot(t_ch, x_ch, lw=0.6)
ax[0].set(ylabel="x [m]")
ax[0].grid(True)
ax[1].plot(t_ch, xd_ch, lw=0.6)
ax[1].set(ylabel="xdot [m/s]", xlabel="t [s]")
ax[1].grid(True)
fig.suptitle("Chirp response (η=0)")
plt.tight_layout()
Plotter._save(fig, OUT, None, "chirp_response")

# %% CLEAN SIGNAL

msk_trans = t_ch <= T_TRANS
msk_clean = t_ch > T_TRANS

t_clean = t_ch[msk_clean]
f_ch_clean = f_ch[msk_clean]
x_ch_clean = x_ch[msk_clean]
xd_ch_clean = xd_ch[msk_clean]

# plot clean signals
fig, ax = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
ax[0].plot(t_ch, f_ch, lw=0.6, label="raw")
ax[0].plot(t_clean, f_ch_clean, lw=0.6, label="cleaned")
ax[0].set(ylabel="fpto [N]")
ax[0].grid(True)
ax[1].plot(t_ch, xd_ch, lw=0.6, label="raw")
ax[1].plot(t_clean, xd_ch_clean, lw=0.6, label="cleaned")
ax[1].set(ylabel="xdot [m/s]", xlabel="t [s]")
ax[1].grid(True)
ax[0].legend(loc="upper right", frameon=True, fontsize=10)
fig.suptitle("Clean Chirp")
plt.tight_layout()
Plotter._save(fig, OUT, None, "clean_chirp")

# %% ETFE

nperseg = min(len(t_clean), 4096)
f, Sf = welch(f_ch_clean, fs=1 / DT, nperseg=nperseg)
f, Sxd = csd(xd_ch_clean, f_ch_clean, fs=1 / DT, nperseg=nperseg)
Y = Sxd / (Sf + 1e-30)  # ?
w = 2 * np.pi * f

# mask for relevant frequency
msk_w = (w >= W_START) & (w <= W_END)
w_fit = w[msk_w]
Y_fit = Y[msk_w]
Sf_fit = Sf[msk_w]
Sxd_fit = Sxd[msk_w]

# Force and response spectra
fig, ax = plt.subplots(4, 1, figsize=(10, 6), sharex=True)
ax[0].semilogy(w_fit, Sf_fit, lw=0.8, label="|F|²")
ax[0].set(ylabel="PSD [N²·s]", title="Force spectrum")
ax[0].legend()
ax[0].grid(True)
ax[0].set_xlim(W_START, W_END)

ax[1].semilogy(w_fit, Sxd_fit, lw=0.8, label="|xd|²")
ax[1].set(
    ylabel="PSD [(m/s)²·s]", xlabel="ω [rad/s]", title="Velocity response spectrum"
)
ax[1].legend()
ax[1].grid(True)
ax[1].set_xlim(W_START, W_END)

ax[2].plot(w_fit, np.abs(Y_fit))
ax[2].set(ylabel="|Y| [m/s/N]", xlabel="ω [rad/s]", title="Velocity admittance")
ax[2].grid(True)
ax[2].set_xlim(W_START, W_END)

ax[3].plot(w_fit, np.degrees(np.angle(Y_fit)))
ax[3].set(ylabel="∠Y [°]", xlabel="ω [rad/s]")
ax[3].grid(True)
ax[3].set_xlim(W_START, W_END)

plt.tight_layout()
Plotter._save(fig, OUT, None, "frequency response")

plt.show()

# %% PARAMETRIZATION

# =============================================================================
# 4. PARAMETRIC FIT  —  Vector Fitting
# =============================================================================

# if HAS_VECTFIT:
#     # vectfit expects complex array s = jω and complex H(s)
#     s = 1j * w_fit
#     # s = jω already in rad/s — vectfit uses rad/s natively
#     poles_init = vectfit.utils.init_poles(s, N_POLES, stable=True)
#     SER, poles, residues, d, h = vectfit.vectfit_auto(
#         Y_fit, s, poles_init, opts={"stable": True}
#     )
#     # build transfer function from pole-residue form
#     #   Y(s) = sum(residues_k / (s - poles_k)) + d + h*s
#     sys_id = control.tf(
#         *control.zpk2tf(
#             [],
#             poles,
#             1.0,  # placeholder — rebuild from residues
#         )
#     )
#     # Build as state-space from poles + residues
#     A_id = np.diag(poles)
#     B_id = np.ones((len(poles), 1))
#     C_id = residues.reshape(1, -1)
#     D_id = np.array([[d]])
#     sys_id = control.ss(A_id.real, B_id, C_id.real, D_id.real)

#     def Y_id(w_arr):
#         s_ = 1j * np.asarray(w_arr)
#         return sum(r / (s_ - p) for r, p in zip(residues, poles)) + d

#     print(f"Vector Fitting done: {N_POLES} poles")
# else:
#     # Fallback: frequency-domain least-squares rational fit
#     print("Using fallback LS rational fit (no vectfit)")
#     n = N_POLES
#     jw = 1j * w_fit  # rad/s

#     def build_poly(w, order):
#         return np.stack([(1j * w) ** k for k in range(order + 1)], axis=1)

#     Vd = build_poly(w_fit, n)
#     Vn = build_poly(w_fit, n - 1)
#     sn = (1j * w_fit) ** n
#     A_ = np.vstack(
#         [(Y_fit[:, None] * Vd[:, :n] - Vn).real, (Y_fit[:, None] * Vd[:, :n] - Vn).imag]
#     )
#     b_ = np.concatenate([(-Y_fit * sn).real, (-Y_fit * sn).imag])
#     p_, *_ = np.linalg.lstsq(A_, b_, rcond=None)
#     d_coef = np.concatenate([[1.0], p_[:n]])
#     n_coef = p_[n:]

#     def Y_id(w_arr):
#         s_ = 1j * np.asarray(w_arr)
#         N_ = sum(c * s_**k for k, c in enumerate(n_coef))
#         D_ = sum(c * s_**k for k, c in enumerate(d_coef))
#         return N_ / D_

#     poles = np.roots(d_coef[::-1])
#     sys_id = control.tf(n_coef[::-1].tolist(), d_coef[::-1].tolist())

# # plot fit
# Y_id_vals = Y_id(w_fit)
# fig, ax = plt.subplots(2, 1, figsize=(10, 6))
# ax[0].plot(w_fit, np.abs(Y_fit), lw=1.5, label="measured")
# ax[0].plot(w_fit, np.abs(Y_id_vals), ls="--", label=f"fit ({N_POLES} poles)")
# ax[0].set(ylabel="|Y| [m/s/N]", xlabel="ω [rad/s]", title="Admittance fit")
# ax[0].legend()
# ax[0].grid(True)
# ax[1].plot(w_fit, np.degrees(np.angle(Y_fit)), lw=1.5)
# ax[1].plot(w_fit, np.degrees(np.angle(Y_id_vals)), ls="--")
# ax[1].set(ylabel="∠Y [°]", xlabel="ω [rad/s]")
# ax[1].grid(True)
# plt.tight_layout()
# Plotter._save(fig, OUT, None, "admittance_fit")

# # =============================================================================
# # 5. STABILITY AND PASSIVITY
# # =============================================================================

# all_stable = np.all(np.real(poles) < 0)
# w_chk = np.linspace(1e-3, W_END * 2, 3000)
# Y_chk = Y_id(w_chk)
# passive = float(np.min(Y_chk.real)) >= 0

# print(f"\nStability: {'✓ STABLE' if all_stable else '✗ UNSTABLE'}")
# print(
#     f"Passivity: {'✓ PASSIVE' if passive else '✗ NOT PASSIVE'}"
#     f"  (min Re[Y]={np.min(Y_chk.real):.4e})"
# )
# for p in poles:
#     print(f"  pole {p.real:+.3f}{p.imag:+.3f}j")

# fig, ax = plt.subplots(1, 2, figsize=(12, 4))
# ax[0].axhline(0, color="k", lw=0.8, ls="--")
# ax[0].plot(w_chk / (2 * np.pi), Y_chk.real)
# ax[0].set(xlabel="ω [rad/s]", ylabel="Re[Y]", title="Passivity  (≥0 required)")
# ax[0].grid(True)
# ax[1].scatter(poles.real, poles.imag, marker="x", s=120, c="red", zorder=5)
# ax[1].axvline(0, color="k", lw=0.8, ls="--")
# ax[1].set(xlabel="Re", ylabel="Im", title="Poles  (stable = left half-plane)")
# ax[1].grid(True)
# plt.tight_layout()
# Plotter._save(fig, OUT, None, "stability_passivity")

# # save model
# with open(OUT / "f2v_model.json", "w") as f:
#     json.dump(
#         {
#             "n_poles": N_POLES,
#             "poles_re": np.real(poles).tolist(),
#             "poles_im": np.imag(poles).tolist(),
#             "stable": bool(all_stable),
#             "passive": bool(passive),
#         },
#         f,
#         indent=2,
#     )

# # =============================================================================
# # 6. VALIDATION  —  synthetic wave, ref vs identified
# # =============================================================================

# print("\nGenerating validation wave…")
# w_s = np.arange(0.01, 4.0, 0.01)
# S_s = param2S(T_WAVE, H_WAVE, w_s, gamma=None, type="JONSWAP")
# t_v = np.arange(0, T_VAL_DUR, DT)
# t_v, eta_v, _, _ = S2eta(w_s, S_s, t_v)

# # ref linear model (no PTO)
# with open(MODEL_DIR / REF_MODEL / "model.json") as f:
#     mdef = json.load(f)

# sr_ref = SimRun.simulate(
#     Model.from_config(
#         mdef, hs, eta_t=t_v, eta_values=eta_v, pto={"damping": 0, "stiffness": 0}
#     ),
#     get_solver("RK4"),
#     t0=t_v[0],
#     t_end=t_v[-1],
#     dt=DT,
#     eta_t=t_v,
#     eta_values=eta_v,
#     label="ref",
# )

# # identified model: frequency-domain convolution
# Nv = len(t_v)
# f_spec = np.fft.rfftfreq(Nv, DT)
# w_spec = 2 * np.pi * f_spec
# Fe_mod = np.interp(f_spec, hs.w / (2 * np.pi), hs.Fe_mod)
# Fe_ang = np.interp(f_spec, hs.w / (2 * np.pi), hs.Fe_ang)
# eta_F = np.fft.rfft(eta_v)
# Fe_F = Fe_mod * np.exp(1j * Fe_ang) * eta_F
# Xd_F = Y_id(w_spec) * Fe_F
# X_F = Xd_F / (1j * w_spec + 1e-30)
# x_id = np.fft.irfft(X_F, n=Nv)
# xd_id = np.fft.irfft(Xd_F, n=Nv)

# msk = t_v >= 60.0
# err_x = metricError(x_id[msk], sr_ref.dataset["x"].to_numpy()[msk], "nrmse_range")
# err_xd = metricError(xd_id[msk], sr_ref.dataset["xdot"].to_numpy()[msk], "nrmse_range")
# print(f"NRMSE  x={err_x:.4f}  xdot={err_xd:.4f}")

# fig, ax = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
# ax[0].plot(
#     t_v[msk],
#     sr_ref.dataset["x"].to_numpy()[msk],
#     color="black",
#     lw=1.5,
#     label="ref (linear)",
# )
# ax[0].plot(t_v[msk], x_id[msk], ls="--", label=f"identified  NRMSE={err_x:.3f}")
# ax[0].set(ylabel="x [m]")
# ax[0].legend()
# ax[0].grid(True)
# ax[0].set_title(f"Validation — T={T_WAVE}s H={H_WAVE}m  no PTO")
# ax[1].plot(
#     t_v[msk], sr_ref.dataset["xdot"].to_numpy()[msk], color="black", lw=1.5, label="ref"
# )
# ax[1].plot(t_v[msk], xd_id[msk], ls="--", label=f"identified  NRMSE={err_xd:.3f}")
# ax[1].set(ylabel="xdot [m/s]", xlabel="t [s]")
# ax[1].legend()
# ax[1].grid(True)
# plt.tight_layout()
# Plotter._save(fig, OUT, None, "validation")

# print(f"\nAll outputs saved to {OUT}")
