# %% PACKAGES

import json
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import chirp

# import control
from source.classes import vectfit

from source.config import MODEL_DIR, OUT_DIR
from source.function.sea_state_utils import param2S, S2eta
from source.function.error_utils import metricError
from source.classes.SimRun import SimRun, get_solver
from source.classes.Model import Model
from source.classes.Plotter import Plotter
from scipy.signal import invres
from scipy.signal import invres, tf2ss, lsim
from scipy.signal import StateSpace as ScipySS
from nfoursid.nfoursid import NFourSID
from scipy.integrate import cumulative_trapezoid
from scipy.signal import zpk2tf, tf2zpk, impulse
from scipy.signal import lti
from scipy.signal import bode
from scipy.linalg import logm

from source.function.passify import check_passivity, passivity_enforcement

OUT = OUT_DIR / "f2v"
OUT.mkdir(parents=True, exist_ok=True)

#! optimization
#! rational model

# %% FUNCTION


# pole-residue evaluation function (no state-space needed)
def G_id_eval(w_arr):
    return vectfit.model(1j * np.asarray(w_arr), poles, residues, d, h=0)


def to_dB(x):
    return 20 * np.log10(np.maximum(x, 1e-30))


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


def compute_fft(signal, dt):
    N = len(signal)
    X = np.fft.rfft(signal) * 2 / N  # one-sided, peak normalisation
    X[0] /= 2  # DC
    if N % 2 == 0:
        X[-1] /= 2  # Nyquist
    w = 2 * np.pi * np.fft.rfftfreq(N, dt)
    return w, X


def smooth_spectrum(amp, window=20):
    kernel = np.ones(window) / window
    return np.convolve(amp, kernel, mode="same")


# %% INPUTS


PKL_NAME = "r5"
DT = 0.05  # s
T_CHIRP = 400.0  # s   — long chirp for good FRF resolution
W_START = 0.08  # rad/s  — broad identification range
W_END = 4  # rad/s
W_REL_START = 1  # rad/s  — relevant (kept) range
W_REL_END = 2  # rad/s
AMP = 1000000.0  # N
freq_swipe = "linear"  # ‘linear’, ‘quadratic’, ‘logarithmic’, ‘hyperbolic’
phi_chirp = -90

SMOOTH_WINDOW = 1  # bins — moving average window for FFT smoothing
T_TRANS = 0  # s   — transient to remove

# Validation wave
T_WAVE = 8.0  # s   peak period
H_WAVE = 2.0  # m   Hs
T_VAL_DUR = 300.0  # s

REF_MODEL = "linear"

# system identification
order = 8  # number of poles for Vector Fitting
rank = order
num_block_rows = 40


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
    t_ch,
    f0=W_START / (2 * np.pi),
    f1=W_END / (2 * np.pi),
    t1=T_CHIRP,
    method=freq_swipe,
    phi=phi_chirp,
)
w_spec_ch = 2 * np.pi * np.fft.rfftfreq(len(t_ch), DT)

# %% SIMULATE

x_ch, xd_ch = simulate_f2v(f_ch, t_ch, M, K, Ar, Br, Cr, Dr, n_r)

pd.DataFrame({"t": t_ch, "f": f_ch, "x": x_ch, "xdot": xd_ch}).to_csv(
    OUT / "chirp_response.csv", index=False
)

# clean transient
msk_trans = t_ch <= T_TRANS
msk_clean = t_ch > T_TRANS

t_clean = t_ch[msk_clean]
f_ch_clean = f_ch[msk_clean]
x_ch_clean = x_ch[msk_clean]
xd_ch_clean = xd_ch[msk_clean]

# plot response
fig, ax = plt.subplots(4, 1, figsize=(12, 5))

ax[0].plot(t_ch, f_ch, label="raw")
ax[0].plot(t_clean, f_ch_clean, label="cleaned")
ax[0].set(ylabel="fpto [N]")
ax[0].grid(True)

ax[1].plot(w_spec_ch, np.abs(np.fft.rfft(f_ch)))
ax[1].set(xlabel="ω [rad/s]", ylabel="|F|")  # xlim=[0, W_END * 1.3]
# ax[1].set_yscale("log")  # FFT magnitude spans many decades
ax[1].set_xscale("log")
ax[1].grid(True)

ax[2].plot(t_ch, x_ch)
ax[2].plot(t_clean, x_ch_clean, label="cleaned")
ax[2].set(ylabel="x [m]")
ax[2].grid(True)

ax[3].plot(t_ch, xd_ch, label="raw")
ax[3].plot(t_clean, xd_ch_clean, label="cleaned")
ax[3].set(ylabel="xdot [m/s]", xlabel="t [s]")
ax[3].grid(True)
fig.suptitle("Input / Output time domain")
plt.tight_layout()
Plotter._save(fig, OUT, None, "chirp_response")

# %% ETFE

# compute FFT
w, F_fft = compute_fft(f_ch_clean, DT)
_, Xd_fft = compute_fft(xd_ch_clean, DT)

# amplitudes and phases
amp_F = np.abs(F_fft)
amp_Xd = np.abs(Xd_fft)
ph_F = np.unwrap(np.angle(F_fft))
ph_Xd = np.unwrap(np.angle(Xd_fft))

# TF
G = Xd_fft / F_fft
G_amp = np.abs(G)
G_phase = np.unwrap(np.angle(G))

# frequency masks
msk_w = (w >= W_START) & (w <= W_END)
msk_rel = (w >= W_REL_START) & (w <= W_REL_END)
w_fit = w[msk_w]
w_rel = w[msk_rel]


fig, ax = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

# Force
ax[0].plot(
    w_fit,
    to_dB(amp_F[msk_w]),
    lw=0.9,
    alpha=0.5,
    color="C0",
    label=f"broad [{W_START:.2f}–{W_END:.2f}])",
)
ax[0].plot(
    w_rel,
    to_dB(amp_F[msk_rel]),
    lw=1.5,
    color="C0",
    label=f"relevant [{W_REL_START:.2f}–{W_REL_END:.2f}])",
)
ax[0].axvline(W_REL_START, color="red", ls="--", lw=0.8)
ax[0].axvline(W_REL_END, color="red", ls="--", lw=0.8)
ax[0].set(ylabel="Magnitude dB", title="Force")
ax[0].legend(fontsize=7)
ax[0].grid(True, which="both")

# velocity
ax[1].plot(
    w_fit,
    to_dB(amp_Xd[msk_w]),
    lw=0.9,
    alpha=0.5,
    color="C1",
    label="broad",
)
ax[1].plot(w_rel, to_dB(amp_Xd[msk_rel]), lw=1.5, color="C1", label="relevant")
ax[1].axvline(W_REL_START, color="red", ls="--", lw=0.8)
ax[1].axvline(W_REL_END, color="red", ls="--", lw=0.8)
ax[1].set(ylabel="Magnitude dB", title="Velocity")
ax[1].legend(fontsize=7)
ax[1].grid(True, which="both")

# G amplitude
ax[2].plot(w_fit, to_dB(G_amp[msk_w]), lw=0.9, alpha=0.5, color="C2", label="broad")
ax[2].plot(w_rel, to_dB(G_amp[msk_rel]), lw=1.5, color="C2", label="relevant")
ax[2].axvline(W_REL_START, color="red", ls="--", lw=0.8)
ax[2].axvline(W_REL_END, color="red", ls="--", lw=0.8)
ax[2].set(ylabel="Magnitude dB", title="Transfert function")
ax[2].legend(fontsize=7)
ax[2].grid(True, which="both")

# G phase
ax[3].plot(
    w_fit, np.degrees(G_phase[msk_w]), lw=0.5, alpha=0.4, color="C3", label="broad"
)
ax[3].plot(w_rel, np.degrees(G_phase[msk_rel]), lw=1.5, color="C3", label="relevant")
ax[3].axvline(W_REL_START, color="red", ls="--", lw=0.8)
ax[3].axvline(W_REL_END, color="red", ls="--", lw=0.8)
ax[3].set(ylabel="Phase [°]", xlabel="ω [rad/s]")
ax[3].legend(fontsize=7)
ax[3].grid(True, which="both")

for a in ax:
    a.set_xscale("log")
    a.set_xlim(W_START, W_END)

plt.tight_layout()
Plotter._save(fig, OUT, None, "frequency_response")

# %% PARAMETRIZATION 1: FREQUENCY DOMAIN

s_id = 1j * w_rel
G_id = G[msk_rel]

# run Vector Fitting
poles, residues, d, h = vectfit.vectfit_auto(
    G_id, s_id, n_poles=int(order / 2), n_iter=50
)

# plot compare ETFE to identified
G_fit_vals = G_id_eval(w)

# get rational form and zero
num_f, den_f = invres(residues, poles, [d], tol=1e-8, rtype="avg")
zeros = np.roots(num_f)

# make state space
A_f, B_f, C_f, D_f = tf2ss(num_f, den_f)
A_f, B_f, C_f, D_f = [np.real(m) for m in tf2ss(num_f, den_f)]
sys_f = ScipySS(A_f, B_f, C_f, D_f)

# passify
passive_f, min_re_f = check_passivity(A_f, B_f, C_f)
print(f"VF   passive={passive_f}   min Re[Y]={min_re_f:.4e}")
if not passive_f:
    C_f_new, Cbar_f = passivity_enforcement(A_f, B_f, C_f, weights=(0.5, 0.5))
    sys_f = ScipySS(A_f, B_f, C_f_new, np.zeros((1, 1)))
    passive_f2, min_re_f2 = check_passivity(A_f, B_f, C_f_new)
    print(f"VF after enforcement:  passive={passive_f2}   min Re[Y]={min_re_f2:.4e}")

# stability and minimum phase check
poles_f = poles
zeros_f = zeros
all_stable_f = np.all(poles.real < 0)
min_phase_f = np.all(zeros.real < 0)

# %% PARAMETRIZATION 2: TIME DOMAIN


# define
df_n4 = pd.DataFrame({"xdot": xd_ch_clean, "f": f_ch_clean})
n4 = NFourSID(
    df_n4,
    output_columns=["xdot"],
    input_columns=["f"],
    num_block_rows=num_block_rows,  #!
)

# fit
n4.subspace_identification()
ss_n4, _ = n4.system_identification(rank=rank)  #!
# fig, ax = plt.subplots(figsize=(8, 4)) # singular values are stored after subspace_identification
# n4.plot_eigenvalues(ax)
# ax.set(title="N4SID eigenvalues — order selection")
# plt.tight_layout()
# plt.show()

# discrete to continuous
A_d = ss_n4.a
B_d = ss_n4.b
n = A_d.shape[0]

A_c = logm(A_d) / DT
B_c = np.linalg.solve(A_d - np.eye(n), A_c @ B_d)
C_c = ss_n4.c
D_c = ss_n4.d
# D_c = np.zeros((1, 1))  # ?
sys_t = ScipySS(A_c.real, B_c.real, C_c, D_c)

# passify
passive_t, min_re_t = check_passivity(A_c, B_c, C_c)
print(f"N4SID passive={passive_t}   min Re[Y]={min_re_t:.4e}")

# if not passive_t:
#     C_c_new, Cbar_t = passivity_enforcement(A_c, B_c, C_c, weights=(0.5, 0.5))
#     sys_t = ScipySS(A_c, B_c, C_c_new, np.zeros((1, 1)))
#     passive_t2, min_re_t2 = check_passivity(A_c, B_c, C_c_new)
#     print(f"N4SID after enforcement: passive={passive_t2}  min Re[Y]={min_re_t2:.4e}")

# stability and minimum phase check
lti_t = lti(A_c.real, B_c.real, C_c, D_c)  #! why A, B can be complex?
poles_t = lti_t.poles
zeros_t = lti_t.zeros
all_stable_t = np.all(poles_t.real < 0)
min_phase_t = np.all(zeros_t.real < 0)

# %% PASSIFICATION CHECK

# # ── Nyquist / Re[Y] plot to visualise ────────────────────────────────────────
# w_chk = np.linspace(1e-2, W_END * 2, 3000)
# I = np.eye(A_f.shape[0])


# def Y_eval_ss(A, B, C, w):
#     return np.array(
#         [(C @ np.linalg.solve(1j * wi * np.eye(A.shape[0]) - A, B))[0, 0] for wi in w]
#     )


# Y_f_before = Y_eval_ss(A_f, B_f, C_f, w_chk)
# Y_f_after = Y_eval_ss(A_f, B_f, C_f_new, w_chk) if not passive_f else Y_f_before
# Y_t_before = Y_eval_ss(A_c, B_c, C_c, w_chk)
# Y_t_after = Y_eval_ss(A_c, B_c, C_c_new, w_chk) if not passive_t else Y_t_before

# fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
# ax[0].axhline(0, color="k", lw=0.8, ls="--")
# ax[0].plot(w_chk, np.real(Y_f_before), lw=1, color="C1", alpha=0.4, label="VF before")
# ax[0].plot(w_chk, np.real(Y_f_after), lw=1.5, color="C1", label="VF passive")
# ax[0].plot(
#     w_chk, np.real(Y_t_before), lw=1, color="C2", alpha=0.4, label="N4SID before"
# )
# ax[0].plot(w_chk, np.real(Y_t_after), lw=1.5, color="C2", label="N4SID passive")
# ax[0].set(ylabel="Re[Y]", title="Passivity check  (Re[Y] ≥ 0 required)")
# ax[0].legend()
# ax[0].grid(True)

# ax[1].plot(w_chk, 20 * np.log10(np.abs(Y_f_before)), lw=1, color="C1", alpha=0.4)
# ax[1].plot(
#     w_chk, 20 * np.log10(np.abs(Y_f_after)), lw=1.5, color="C1", label="VF passive"
# )
# ax[1].plot(w_chk, 20 * np.log10(np.abs(Y_t_before)), lw=1, color="C2", alpha=0.4)
# ax[1].plot(
#     w_chk, 20 * np.log10(np.abs(Y_t_after)), lw=1.5, color="C2", label="N4SID passive"
# )
# ax[1].set(ylabel="|Y| [dB]", xlabel="ω [rad/s]")
# ax[1].legend()
# ax[1].grid(True)
# ax[1].set_xscale("log")
# plt.tight_layout()
# Plotter._save(fig, OUT, None, "passivity")

# %% COMPARISON

# bode
w_bode = np.logspace(np.log10(W_START), np.log10(W_END), 500)
w_f, mag_f, phase_f = bode(sys_f, w=w_bode)
w_t, mag_t, phase_t = bode(sys_t, w=w_bode)

fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

ax[0].plot(w, to_dB(G_amp), color="black", lw=1.5, label="measured FRF")
ax[0].plot(w_bode, mag_f, ls="--", color="C1", label="freq.")
ax[0].plot(w_bode, mag_t, ls=":", color="C2", label="time")
ax[0].set(ylabel="|G| [dB]", title="Bode — measured vs freq vs time")
ax[0].legend()
ax[0].grid(True, which="both")
ax[0].axvline(W_REL_START, color="red", ls="--", lw=0.8)
ax[0].axvline(W_REL_END, color="red", ls="--", lw=0.8)

ax[1].plot(w, np.degrees(G_phase), color="black", lw=1.5)
ax[1].plot(w_bode, phase_f, ls="--", color="C1")
ax[1].plot(w_bode, phase_t, ls=":", color="C2")
ax[1].set(ylabel="Phase [°]", xlabel="ω [rad/s]")
ax[1].grid(True, which="both")
ax[1].axvline(W_REL_START, color="red", ls="--", lw=0.8)
ax[1].axvline(W_REL_END, color="red", ls="--", lw=0.8)
for a in ax:
    a.set_xscale("log")
    a.set_xlim(W_START, W_END)

plt.tight_layout()
Plotter._save(fig, OUT, None, "bode_comparison")

# pole-zero map
fig, ax = plt.subplots(figsize=(7, 6))
ax.axvline(0, color="k", lw=0.8, ls="--")
ax.axhline(0, color="k", lw=0.8, ls="--")
ax.scatter(
    poles_f.real,
    poles_f.imag,
    marker="x",
    s=120,
    color="C0",
    zorder=5,
    label=f"freq: poles ({len(poles)})  {'stable ✓' if all_stable_f else 'UNSTABLE ✗'}",
)
ax.scatter(
    zeros_f.real,
    zeros_f.imag,
    marker="o",
    s=80,
    facecolors="none",
    edgecolors="C0",
    zorder=5,
    label=f"freq: zeros ({len(zeros)})  {'min phase ✓' if min_phase_f else 'non-min phase ✗'}",
)
ax.scatter(
    poles_t.real,
    poles_t.imag,
    marker="x",
    s=120,
    color="C1",
    zorder=5,
    label=f"time: poles ({len(poles)})  {'stable ✓' if all_stable_t else 'UNSTABLE ✗'}",
)
ax.scatter(
    zeros_t.real,
    zeros_t.imag,
    marker="o",
    s=80,
    facecolors="none",
    edgecolors="C1",
    zorder=5,
    label=f"time: zeros ({len(zeros)})  {'min phase ✓' if min_phase_t else 'non-min phase ✗'}",
)
ax.set(xlabel="Re", ylabel="Im", title="Pole-zero map")
ax.legend()
ax.grid(True)
plt.tight_layout()
Plotter._save(fig, OUT, None, "pzmap")

# %% VALIDATION

# synthetic JONSWAP wave
w_s = np.arange(0.01, 4.0, 0.01)
S_s = param2S(T_WAVE, H_WAVE, w_s, gamma=3.3, type="JONSWAP")
t_v = np.arange(0, T_VAL_DUR, DT)
t_v, eta_v, _, _ = S2eta(w_s, S_s, t_v)

# ref sim
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

# excitation force
fe_ref = ref.dataset["fe_lin"].to_numpy()
msk_v = t_v >= 60.0  # transient

# simulate
x0_f = -np.linalg.pinv(C_f) @ (D_f @ fe_ref[0:1])
_, xd_f, _ = lsim(sys_f, fe_ref, t_v, X0=x0_f.ravel())
# xd_f = np.real(xd_f).ravel()  # lsim may return complex due to TF conversion #!
x_f = cumulative_trapezoid(xd_f, t_v, initial=0)

x0_t = -np.linalg.pinv(C_c) @ (D_c @ fe_ref[0:1])
_, xd_t, _ = lsim(sys_t, fe_ref, t_v, X0=x0_t.ravel())
# xd_t = np.real(xd_t).ravel()
x_t = cumulative_trapezoid(xd_t, t_v, initial=0)

# metric
x_ref = ref.dataset["x"].to_numpy()
xd_ref = ref.dataset["xdot"].to_numpy()
err_xd_f = metricError(xd_f[msk_v], xd_ref[msk_v], "nrmse_range")
err_xd_t = metricError(xd_t[msk_v], xd_ref[msk_v], "nrmse_range")
print(f"freq. domain NRMSE: xdot={err_xd_f:.4f}")
print(f"time domain NRMSE: xdot={err_xd_t:.4f}")

# plot
fig, ax = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
fig.suptitle(f"Validation — T={T_WAVE}s H={H_WAVE}m  no PTO")

ax[0].plot(t_v, x_ref, color="black", lw=1.5, label="ref")
ax[0].plot(
    t_v,
    x_f,
    ls="--",
    lw=1,
    color="C1",
    label=f"freq. domain",
)
ax[0].plot(
    t_v,
    x_t,
    ls=":",
    lw=1,
    color="C2",
    label=f"time domain",
)
ax[0].set(ylabel="x [m]")
ax[0].legend()
ax[0].grid(True)

ax[1].plot(t_v, xd_ref, color="black", lw=1.5, label="ref")
ax[1].plot(
    t_v,
    xd_f,
    ls="--",
    lw=1,
    color="C1",
    label=f"freq. domain NRMSE={err_xd_f:.3f}",
)
ax[1].plot(
    t_v,
    xd_t,
    ls=":",
    lw=1,
    color="C2",
    label=f"time NRMSE={err_xd_t:.3f}",
)
ax[1].set(ylabel="xdot [m/s]", xlabel="t [s]")
ax[1].legend()
ax[1].grid(True)

plt.tight_layout()
Plotter._save(fig, OUT, None, "validation")

# %% PLOT

# print("D_f =", D_f)
# print("D_c =", D_c)

# %% SAVE MODEL

OUT_SI = OUT_DIR / "si"
OUT_SI.mkdir(parents=True, exist_ok=True)

np.savez(
    OUT_SI / "model_f2v.npz",
    A=A_f,
    B=B_f,
    C=C_f,
    D=D_f,
)
print(f"f2v model saved → {OUT_SI / 'model_f2v.npz'}")
print(f"  A:{A_f.shape}")

plt.show()
