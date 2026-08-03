# %% PACKAGES

import json
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import chirp
import control
from source.classes import vectfit

from source.config import MODEL_DIR, OUT_DIR
from source.function.sea_state_utils import param2S, S2eta
from source.function.error_utils import metricError
from source.classes.SimRun import SimRun, get_solver
from source.classes.Model import Model
from source.classes.Plotter import Plotter
from scipy.signal import invres
from scipy.signal import tf2ss, lsim
from scipy.signal import StateSpace as ScipySS

OUT = OUT_DIR / "eta2f"
OUT.mkdir(parents=True, exist_ok=True)

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


def schroeder_phases(N):
    k = np.arange(N)
    return -np.pi * k * (k - 1) / N


def multisine(t, freqs_hz, amplitudes, phase="schroeder", seed=None):
    freqs_hz = np.asarray(freqs_hz)
    amplitudes = np.asarray(amplitudes)
    N = len(freqs_hz)

    if phase == "schroeder":
        phases = schroeder_phases(N)
    elif phase == "random":
        rng = np.random.default_rng(seed)
        phases = rng.uniform(-np.pi, np.pi, N)
    elif phase == "zero":
        phases = np.zeros(N)
    else:
        raise ValueError(
            f"Unknown phase option '{phase}'. Choose 'schroeder', 'random', or 'zero'."
        )

    signal = sum(
        a * np.sin(2 * np.pi * f * t + p)
        for f, a, p in zip(freqs_hz, amplitudes, phases)
    )
    return signal, phases


# %% INPUTS

# inputs
dt = 0.05
w_min = 0.3
w_max = 5
n = 100  # number of components
amp = 0.2
phase = "random"
file = OUT / "eta_multisine_signal_1.csv"
PKL_NAME = "r5"

W_REL_START = 0.2  # rad/s  — relevant (kept) range
W_REL_END = 4  # rad/s

order = [2, 4, 6, 8, 10]  # , 16, 18, 20, 22]

# causality inputs
TAU_MIN = 0.0  # s — minimum time advance
TAU_MAX = 8.0  # s — maximum time advance (cover non-causal horizon)
N_TAU = 100  # number of trial advances

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

# %% GENERATE ETA SIGNAL

# compute
dw = (w_max - w_min) / (n - 1)
k_min = int(np.round(w_min / dw))
w = (k_min + np.arange(n)) * dw
f = w / (2 * np.pi)
tper = 2 * np.pi / dw
t = np.arange(0, tper, dt)
amps = np.full(n, amp)

# generate
eta, phase_eta = multisine(t, f, amps, phase=phase)


# %% GENERATE FORCE

# from eta to force
Fe_mod_interp = np.interp(w, hs.w, hs.Fe_mod)
Fe_ang_interp = np.interp(w, hs.w, np.unwrap(hs.Fe_ang))
Fe_interp = Fe_mod_interp * np.exp(1j * Fe_ang_interp)
fe = np.zeros_like(t)

for wk, ak, Fem, Fea, phik in zip(w, amps, Fe_mod_interp, Fe_ang_interp, phase_eta):
    fe += ak * Fem * np.sin(wk * t + phik + Fea)

# frequency domain
w_fft, eta_fft = compute_fft(eta, dt)
_, fe_fft = compute_fft(fe, dt)

# plot response
fig, ax = plt.subplots(4, 1, figsize=(12, 5))

ax[0].plot(t, eta)
ax[0].set(xlabel="t [s]", ylabel="eta [m]")
ax[0].grid(True)

ax[1].plot(w_fft, np.abs(eta_fft))
ax[1].set(xlabel="ω [rad/s]", ylabel="eta [m]")
ax[1].grid(True)
ax[1].set_xscale("log")
# ax[1].set_xlim(w_min, w_max)

ax[2].plot(t, fe)
ax[2].set(xlabel="t [s]", ylabel="fe [N]")
ax[2].grid(True)

ax[3].plot(w_fft, np.abs(fe_fft))
ax[3].set(xlabel="ω [rad/s]", ylabel="fe [N]")
ax[3].grid(True)
ax[3].set_xscale("log")
# ax[3].set_xlim(w_min, w_max)
fig.suptitle("Input / Output time domain")
plt.tight_layout()
Plotter._save(fig, OUT, None, "signals")

# %% ETFE

# amplitudes and phases
amp_eta = np.abs(eta_fft)
amp_fe = np.abs(fe_fft)
ph_eta = np.unwrap(np.angle(eta_fft))
ph_fe = np.unwrap(np.angle(fe_fft))

# TF
G_etfe = fe_fft / eta_fft

# frequency masks
msk_w = (w_fft >= w_min) & (w_fft <= w_max)
msk_rel = (w_fft >= W_REL_START) & (w_fft <= W_REL_END)
w_fit = w_fft[msk_w]
w_rel = w_fft[msk_rel]
G_etfe_fit = G_etfe[msk_w]
G_etfe_rel = G_etfe[msk_rel]

fig, ax = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

# Wave
ax[0].plot(
    w_fit,
    to_dB(amp_eta[msk_w]),
    lw=0.9,
    alpha=0.5,
    color="C0",
    label=f"broad [{w_min:.2f}–{w_max:.2f}])",
)
ax[0].plot(
    w_rel,
    to_dB(amp_eta[msk_rel]),
    lw=1.5,
    color="C0",
    label=f"relevant [{W_REL_START:.2f}–{W_REL_END:.2f}])",
)
ax[0].axvline(W_REL_START, color="red", ls="--", lw=0.8)
ax[0].axvline(W_REL_END, color="red", ls="--", lw=0.8)
ax[0].set(ylabel="Magnitude dB", title="Eta")
ax[0].legend(fontsize=7)
ax[0].grid(True, which="both")

# excitation force
ax[1].plot(
    w_fit,
    to_dB(amp_fe[msk_w]),
    lw=0.9,
    alpha=0.5,
    color="C1",
    label="broad",
)
ax[1].plot(w_rel, to_dB(amp_fe[msk_rel]), lw=1.5, color="C1", label="relevant")
ax[1].axvline(W_REL_START, color="red", ls="--", lw=0.8)
ax[1].axvline(W_REL_END, color="red", ls="--", lw=0.8)
ax[1].set(ylabel="Magnitude dB", title="Excitation force")
ax[1].legend(fontsize=7)
ax[1].grid(True, which="both")

# G amplitude
ax[2].plot(
    w_fit, to_dB(np.abs(G_etfe_fit)), lw=0.9, alpha=0.5, color="C2", label="broad"
)
ax[2].plot(w_rel, to_dB(np.abs(G_etfe_rel)), lw=1.5, color="C2", label="relevant")
ax[2].axvline(W_REL_START, color="red", ls="--", lw=0.8)
ax[2].axvline(W_REL_END, color="red", ls="--", lw=0.8)
ax[2].set(ylabel="Magnitude dB", title="Transfert function")
ax[2].legend(fontsize=7)
ax[2].grid(True, which="both")

# G phase
ax[3].plot(
    w_fit,
    np.degrees(np.unwrap(np.angle(G_etfe_fit))),
    lw=0.5,
    alpha=0.4,
    color="C3",
    label="broad",
)
ax[3].plot(
    w_rel,
    np.degrees(np.unwrap(np.angle(G_etfe_rel))),
    lw=1.5,
    color="C3",
    label="relevant",
)
ax[3].axvline(W_REL_START, color="red", ls="--", lw=0.8)
ax[3].axvline(W_REL_END, color="red", ls="--", lw=0.8)
ax[3].set(ylabel="Phase [°]", xlabel="ω [rad/s]")
ax[3].legend(fontsize=7)
ax[3].grid(True, which="both")

for a in ax:
    a.set_xscale("log")
    a.set_xlim(w_min, w_max)

plt.tight_layout()
Plotter._save(fig, OUT, None, "frequency_response")


# %% PARAMETRIZATION

tau_trials = np.linspace(TAU_MIN, TAU_MAX, N_TAU)
colors_ord = plt.cm.tab10(np.linspace(0, 1, len(order)))

s_bem = 1j * w_rel

results = {}

for ord_i in order:
    errors_i, fits_i = [], []
    for tau in tau_trials:
        G_shifted = G_etfe_rel * np.exp(-1j * w_rel * tau)
        p_i, r_i, d_i, _ = vectfit.vectfit_auto(
            G_shifted, s_bem, n_poles=int(ord_i / 2), n_iter=50
        )
        G_fit_i = vectfit.model(s_bem, p_i, r_i, d_i, h=0) * np.exp(1j * w_rel * tau)
        err_i = float(np.mean(np.abs(G_fit_i - G_etfe_rel) ** 2))
        errors_i.append(err_i)
        fits_i.append((p_i, r_i, d_i, tau))
    best_i = int(np.argmin(errors_i))
    results[ord_i] = dict(
        errors=errors_i, fits=fits_i, best_idx=best_i, best_tau=fits_i[best_i][3]
    )

# pick global best BEFORE plotting so we can highlight it
best_order = min(results, key=lambda o: results[o]["errors"][results[o]["best_idx"]])
r_best = results[best_order]
p_b, r_b, d_b, best_tau_bem = (
    *r_best["fits"][r_best["best_idx"]][:3],
    r_best["best_tau"],
)

# convert best VF model (shifted/causal) to state-space
num_e, den_e = invres(r_b, p_b, [d_b], tol=1e-8, rtype="avg")
A_e, B_e, C_e, D_e = tf2ss(num_e, den_e)
A_e, B_e, C_e, D_e = [np.real(m) for m in tf2ss(num_e, den_e)]
sys_e = ScipySS(A_e, B_e, C_e, D_e)

# error vs delay
fig, ax = plt.subplots(figsize=(10, 5))
for col, ord_i in zip(colors_ord, order):
    r = results[ord_i]
    ax.semilogy(
        tau_trials, r["errors"], "o-", markersize=4, color=col, label=f"order {ord_i}"
    )
    ax.axvline(r["best_tau"], color=col, ls="--", lw=0.8, alpha=0.6)
ax.set(
    xlabel="τ [s]",
    ylabel="MSE",
    title=f"VF on ETFE — error vs delay  |  best: order={best_order}  τ={best_tau_bem:.2f}s",
)
ax.legend(fontsize=8)
ax.grid(True, which="both")
plt.tight_layout()
Plotter._save(fig, OUT, None, "etfe_error_vs_delay")

# Bode comparison
fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
ax[0].plot(hs.w, to_dB(hs.Fe_mod), color="black", lw=3, label="BEM")
ax[0].plot(
    w_fit,
    to_dB(np.abs(G_etfe_fit)),
    color="red",
    lw=3,
    label="ETFE",
    ls="--",
)
ax[1].plot(hs.w, np.degrees(np.unwrap(hs.Fe_ang)), color="black", lw=3, label="BEM")
ax[1].plot(
    w_fit,
    np.degrees(np.unwrap(np.angle(G_etfe_fit))),
    color="red",
    lw=3,
    ls="--",
    label="ETFE",
)

for col, ord_i in zip(colors_ord, order):
    r = results[ord_i]
    p_b_i, r_b_i, d_b_i, tau_b = (*r["fits"][r["best_idx"]][:3], r["best_tau"])
    G_b = vectfit.model(1j * hs.w, p_b_i, r_b_i, d_b_i, h=0) * np.exp(1j * hs.w * tau_b)
    is_best = ord_i == best_order
    lbl = f"VF order {ord_i}  τ={tau_b:.1f}s" + ("  ← BEST" if is_best else "")
    lw = 2.0 if is_best else 1.2
    ax[0].plot(hs.w, to_dB(np.abs(G_b)), color=col, lw=lw, ls="--", label=lbl)
    ax[1].plot(hs.w, np.degrees(np.unwrap(np.angle(G_b))), color=col, lw=lw, ls="--")

for a in ax:
    a.axvline(W_REL_START, color="red", ls="--", lw=0.8)
    a.axvline(W_REL_END, color="red", ls="--", lw=0.8)
    a.set_xscale("log")
    a.set_xlim(w_min, w_max)
    a.grid(True, which="both")
ax[0].set(
    ylabel="|G| [dB]",
    title=f"VF on ETFE — best fit per order  |  BEST: order={best_order}  τ={best_tau_bem:.2f}s",
)
ax[0].legend(fontsize=7, ncol=2)
ax[1].set(ylabel="Phase [°]", xlabel="ω [rad/s]")
ax[1].legend(fontsize=7)
plt.tight_layout()
Plotter._save(fig, OUT, None, "vf_on_etfe")

# pole-zero map
num_best, den_best = invres(r_b, p_b, [d_b], tol=1e-8, rtype="avg")
zeros_best = np.roots(num_best)

all_stable = np.all(p_b.real < 0)
min_phase = np.all(zeros_best.real < 0)

print(
    f"Stability:     {'✓ STABLE' if all_stable else '✗ UNSTABLE'}   "
    f"(max Re(pole) = {p_b.real.max():.4f})"
)
print(
    f"Minimum phase: {'✓ MIN PHASE' if min_phase else '✗ NON-MIN PHASE'}   "
    f"(max Re(zero) = {zeros_best.real.max():.4f})"
)

fig, ax = plt.subplots(figsize=(7, 6))
ax.axvline(0, color="k", lw=0.8, ls="--")
ax.axhline(0, color="k", lw=0.8, ls="--")
ax.scatter(
    p_b.real,
    p_b.imag,
    marker="x",
    s=120,
    color="C1",
    zorder=5,
    label=f"poles ({len(p_b)})  {'stable ✓' if all_stable else 'UNSTABLE ✗'}",
)
ax.scatter(
    zeros_best.real,
    zeros_best.imag,
    marker="o",
    s=80,
    facecolors="none",
    edgecolors="C0",
    zorder=5,
    label=f"zeros ({len(zeros_best)})  {'min phase ✓' if min_phase else 'non-min phase ✗'}",
)
ax.set(
    xlabel="Re",
    ylabel="Im",
    title=f"Pole-zero map — order={best_order}  τ={best_tau_bem:.2f}s",
)
ax.legend()
ax.grid(True)
plt.tight_layout()
Plotter._save(fig, OUT, None, "pzmap")

# %% VALIDATION

# validation inputs
T_WAVE = 8.0  # s — peak period
H_WAVE = 2.0  # m — Hs
T_VAL_DUR = 300.0  # s
REF_MODEL = "linear"
DT = dt

# synthetic JONSWAP wave
w_s = np.arange(0.01, 4.0, 0.01)
S_s = param2S(T_WAVE, H_WAVE, w_s, gamma=3.3, type="JONSWAP")
t_v = np.arange(0, T_VAL_DUR, DT)
t_v, eta_v, _, _ = S2eta(w_s, S_s, t_v)

# ref simulation → ground truth fe_lin
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
fe_ref = ref.dataset["fe_lin"].to_numpy()  # ground truth excitation force

# apply time advance: feed η(t + τ*) to causal model
tau_samp = int(np.round(best_tau_bem / DT))
eta_adv = eta_v[tau_samp:]  # η(t + τ*) — advanced wave
t_sim = t_v[: len(eta_adv)]  # matching time vector
fe_ref_sim = fe_ref[: len(eta_adv)]  # trim reference to same length

_, fe_pred, _ = lsim(sys_e, eta_adv, t_sim, X0=np.zeros(A_e.shape[0]))
fe_pred = np.real(fe_pred).ravel()

# metric
msk_v = t_sim >= 60.0
err_fe = metricError(fe_pred[msk_v], fe_ref_sim[msk_v], "nrmse_range")
print(f"eta2f NRMSE fe = {err_fe:.4f}")

# plot
fig, ax = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
fig.suptitle(
    f"Validation eta2f — T={T_WAVE}s H={H_WAVE}m  "
    f"order={best_order}  τ={best_tau_bem:.2f}s"
)

ax[0].plot(
    t_sim[msk_v], fe_ref_sim[msk_v], color="black", lw=1.5, label="ref (simulation)"
)
ax[0].plot(
    t_sim[msk_v],
    fe_pred[msk_v],
    color="C1",
    ls="--",
    lw=1,
    label=f"identified  NRMSE={err_fe:.3f}",
)
ax[0].set(ylabel="fe [N]")
ax[0].legend()
ax[0].grid(True)

ax[1].plot(
    t_sim[msk_v], eta_v[tau_samp:][msk_v], color="C0", lw=0.8, label="η(t+τ*) — input"
)
ax[1].set(ylabel="η [m]", xlabel="t [s]")
ax[1].legend()
ax[1].grid(True)

plt.tight_layout()
Plotter._save(fig, OUT, None, "validation")

# %% SAVE MODEL

OUT_SI = OUT_DIR / "si"
OUT_SI.mkdir(parents=True, exist_ok=True)

np.savez(
    OUT_SI / "model_eta2f.npz",
    A=A_e,
    B=B_e,
    C=C_e,
    D=D_e,
    tau=np.array([best_tau_bem]),
    order=np.array([best_order]),
)
print(f"eta2f model saved → {OUT_SI / 'model_eta2f.npz'}")
print(f"  order={best_order}  tau={best_tau_bem:.4f}s  A:{A_e.shape}")

# %% PLOT

plt.show()
