# %% PACKAGES

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import butter, filtfilt

from source.config import OUT_DIR

OUT = OUT_DIR / "signals"
OUT.mkdir(parents=True, exist_ok=True)


# %% FUNCTIONS


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

    return sum(
        a * np.sin(2 * np.pi * f * t + p)
        for f, a, p in zip(freqs_hz, amplitudes, phases)
    )


def save_signals(path: Path, t, **signals):
    df = pd.DataFrame({"t [s]": t, **signals})
    df.to_csv(path, index=False)


def compute_fft(signal, dt):
    N = len(signal)
    X = np.fft.rfft(signal) * 2 / N  # one-sided, peak normalisation
    X[0] /= 2  # DC
    if N % 2 == 0:
        X[-1] /= 2  # Nyquist
    w = 2 * np.pi * np.fft.rfftfreq(N, dt)  # rad/s
    return w, X


# %% GENERAL INPUTS

dt = 0.05

# %% SIGNAL 1: wave elevation only (multi-sin)

# inputs
w_min_1 = 0.5
w_max_1 = 2.4
n_1 = 100  # number of components
amp_1 = 0.2
phase_1 = "random"
file_1 = OUT / "eta_multisine_signal_1.csv"

# compute
dw_1 = (w_max_1 - w_min_1) / (n_1 - 1)
k_min_1 = int(np.round(w_min_1 / dw_1))
w_1 = (k_min_1 + np.arange(n_1)) * dw_1
T_1 = 2 * np.pi / w_1
f_1 = w_1 / (2 * np.pi)
tper_1 = 2 * np.pi / dw_1
t_1 = np.arange(0, tper_1, dt)
amps_1 = np.full(n_1, amp_1)

# generate
signal_1 = multisine(t_1, f_1, amps_1, phase=phase_1)

# frequency domain
w_fft_1, fft_1 = compute_fft(signal_1, dt)

# save
save_signals(file_1, t_1, **{"eta": signal_1})

# plot
fig, axes = plt.subplots(2, 1, figsize=(14, 6))
fig.suptitle(
    f"Signal 1: multisin wave elevation only — period = {tper_1:.1f} s - frequencies: {min(w_1):.2f} to {max(w_1):.2f} rad/s - periods: {min(T_1):.2f} to {max(T_1):.2f} s"
)

axes[0].plot(t_1, signal_1, lw=0.7)
axes[0].set(ylabel="eta [m]", xlabel="t [s]")
axes[0].grid(True)

axes[1].plot(w_fft_1, np.abs(fft_1), lw=0.7)
axes[1].set(ylabel="eta [m]", xlabel="ω [rad/s]")
axes[1].grid(True)

plt.tight_layout()

# %% SIGNAL 2: wave elevation (multi-sin) + PTO force (random)

# inputs wave
w_min_2 = 0.5
w_max_2 = 2.4
n_2 = 100  # number of components
amp_2 = 0.2
phase_2 = "random"

# input pto
amp_pto_2 = 900000  # N    — uniform random between -AMP_PTO_2 and +AMP_PTO_2
seed_2 = 42
w_filt_max_2 = 50  # rad/s — low-pass filter cutoff for PTO signal

file_2 = OUT / "eta_multisine_fpto_random_signal_2.csv"

# compute wave
dw_2 = (w_max_2 - w_min_2) / (n_2 - 1)
k_min_2 = int(np.round(w_min_2 / dw_2))
w_2 = (k_min_2 + np.arange(n_2)) * dw_2
T_2 = 2 * np.pi / w_2
f_2 = w_2 / (2 * np.pi)
tper_2 = 2 * np.pi / dw_2
t_2 = np.arange(0, tper_2, dt)
amps_2 = np.full(n_2, amp_2)

# compute pto
b_filt, a_filt = butter(4, w_filt_max_2 / (np.pi / dt), btype="low")

# generate
signal_21 = multisine(t_2, f_2, amps_2, phase=phase_2)

# generate PTO
rng_pto_2 = np.random.default_rng(seed_2)
signal_22 = rng_pto_2.uniform(-amp_pto_2, amp_pto_2, size=len(t_2))
signal_filt_22 = filtfilt(b_filt, a_filt, signal_22)
signal_filt_22 = np.clip(signal_filt_22, -1e6, 1e6)

# frequency domain
w_fft_21, fft_21 = compute_fft(signal_21, dt)
w_fft_22, fft_22 = compute_fft(signal_22, dt)
w_fft_filt_22, fft_filt_22 = compute_fft(signal_filt_22, dt)

# save
save_signals(file_2, t_2, **{"eta": signal_21, "fpto": signal_22})

# plot
fig, axes = plt.subplots(4, 1, figsize=(14, 6))
fig.suptitle(
    f"Signal 2: multisin wave elevation + random PTO — period = {tper_2:.1f} s - frequencies: {min(w_2):.2f} to {max(w_2):.2f} rad/s - periods: {min(T_2):.2f} to {max(T_2):.2f} s"
)

axes[0].plot(t_2, signal_21, lw=0.7)
axes[0].set(ylabel="eta [m]", xlabel="t [s]")
axes[0].grid(True)

axes[1].plot(w_fft_21, np.abs(fft_21), lw=0.7)
axes[1].set(ylabel="eta [m]", xlabel="ω [rad/s]")
axes[1].grid(True)

axes[2].plot(t_2, signal_22, lw=0.7, color="red")
axes[2].plot(t_2, signal_filt_22, lw=0.7, color="orange")
axes[2].set(ylabel="f_pto [N]", xlabel="t [s]")
axes[2].grid(True)

axes[3].plot(w_fft_22, np.abs(fft_22), lw=0.7, color="red")
axes[3].plot(w_fft_filt_22, np.abs(fft_filt_22), lw=0.7, color="orange")
axes[3].set(ylabel="f_pto [N]", xlabel="ω [rad/s]")
axes[3].grid(True)

plt.tight_layout()

# %% SIGNAL 3: wave elevation (multi-sin) + PTO force (multisin)

# inputs wave
w_min_eta_3 = 0.5
w_max_eta_3 = 2.4
n_eta_3 = 100  # number of components
amp_eta_3 = 0.2
phase_eta_3 = "random"

# input pto
w_min_pto_3 = 0.1
w_max_pto_3 = 55
n_pto_3 = 3000  # number of components
amp_pto_3 = 10000
phase_pto_3 = "random"

file_3 = OUT / "eta_multisine_fpto_multisin_signal_3.csv"

# compute wave
dw_eta_3 = (w_max_eta_3 - w_min_eta_3) / (n_eta_3 - 1)
k_min_eta_3 = int(np.round(w_min_eta_3 / dw_eta_3))
w_eta_3 = (k_min_eta_3 + np.arange(n_eta_3)) * dw_eta_3
f_eta_3 = w_eta_3 / (2 * np.pi)
tper_eta_3 = 2 * np.pi / dw_eta_3
t_3 = np.arange(0, tper_eta_3, dt)
amps_eta_3 = np.full(n_eta_3, amp_eta_3)

# compute pto
dw_pto_3 = (w_max_pto_3 - w_min_pto_3) / (n_pto_3 - 1)
k_min_pto_3 = int(np.round(w_min_pto_3 / dw_pto_3))
w_pto_3 = (k_min_pto_3 + np.arange(n_pto_3)) * dw_pto_3
f_pto_3 = w_pto_3 / (2 * np.pi)
tper_pto_3 = 2 * np.pi / dw_pto_3
t_pto_3 = np.arange(0, tper_pto_3, dt)
amps_pto_3 = np.full(n_pto_3, amp_pto_3)

# generate
signal_31 = multisine(t_3, f_eta_3, amps_eta_3, phase=phase_eta_3)
signal_32 = multisine(t_3, f_pto_3, amps_pto_3, phase=phase_pto_3)
signal_32 = np.clip(signal_32, -1e6, 1e6)

# frequency domain
w_fft_31, fft_31 = compute_fft(signal_31, dt)
w_fft_32, fft_32 = compute_fft(signal_32, dt)

# save
save_signals(file_3, t_3, **{"eta": signal_31, "fpto": signal_32})

# plot
fig, axes = plt.subplots(4, 1, figsize=(14, 6))
axes[0].set_title(
    f"Signal 3: multisin wave elevation — period = {tper_eta_3:.1f} s - frequencies: {min(w_eta_3):.2f} to {max(w_eta_3):.2f} rad/s"
)

axes[0].plot(t_3, signal_31, lw=0.7)
axes[0].set(ylabel="eta [m]", xlabel="t [s]")
axes[0].grid(True)

axes[1].plot(w_fft_31, np.abs(fft_31), lw=0.7)
axes[1].set(ylabel="eta [m]", xlabel="ω [rad/s]")
axes[1].grid(True)

axes[2].set_title(
    f"Signal 3: multisin PTO — period = {tper_pto_3:.1f} s - frequencies: {min(w_pto_3):.2f} to {max(w_pto_3):.2f} rad/s"
)
axes[2].plot(t_3, signal_32, lw=0.7, color="red")
axes[2].set(ylabel="f_pto [N]", xlabel="t [s]")
axes[2].grid(True)

axes[3].plot(w_fft_32, np.abs(fft_32), lw=0.7, color="red")
axes[3].set(ylabel="f_pto [N]", xlabel="ω [rad/s]")
axes[3].grid(True)

plt.tight_layout()

# %% PLOT

plt.show()
