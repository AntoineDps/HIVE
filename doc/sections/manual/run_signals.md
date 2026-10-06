# run_signals

Generates broadband multi-sine excitation signals for use with the `linear_si`
identification scheme.

## Usage

```bash
python -m script.run_signals
```

Outputs are saved to `out/signals/`.

## Signal types

### Signal 1 — Wave elevation only

A multi-sine wave elevation $\eta(t)$ with random phases, covering the frequency
range of interest. Used to identify the $\eta \to f_e$ transfer function.

| Parameter | Description |
|---|---|
| `w_min`, `w_max` | Frequency range [rad/s] |
| `n` | Number of sinusoidal components |
| `amp` | Amplitude of each component [m] |
| `phase` | Phase distribution: `"random"`, `"schroeder"`, or `"zero"` |

### Signal 2 — Wave elevation + random PTO force

A multi-sine $\eta(t)$ combined with a broadband random PTO force $f_{pto}(t)$,
low-pass filtered to remove aliasing. Used for combined $f_e + f_{pto} \to \dot{x}$
identification.

Additional parameters:

| Parameter | Description |
|---|---|
| `amp_pto` | PTO force amplitude [N] |
| `seed` | Random seed for reproducibility |
| `w_filt_max` | Low-pass filter cutoff for PTO signal [rad/s] |

### Signal 3 — Wave elevation + multi-sine PTO force

Both $\eta(t)$ and $f_{pto}(t)$ are multi-sine signals, with independently chosen
frequency grids. Gives precise spectral coverage for both inputs.

---

## Signal design

Frequencies are chosen as **integer harmonics** of a fundamental spacing $\Delta\omega$:

$$
\Delta\omega = \frac{\omega_{\max} - \omega_{\min}}{N - 1}, \qquad
\omega_k = (k_{\min} + k)\,\Delta\omega, \quad k = 0, \ldots, N-1
$$

This ensures the signal is **exactly periodic** with period $T = 2\pi / \Delta\omega$,
which is a requirement for unbiased ETFE estimation.

### Phase options

| Option | Description |
|---|---|
| `"schroeder"` | $\phi_k = -\pi k(k-1)/N$ — low crest factor, chirp-like shape |
| `"random"` | Uniform random in $[-\pi, \pi]$ |
| `"zero"` | All phases zero |

Schroeder phases minimise the peak-to-RMS ratio of the signal, which is useful when
actuator saturation is a concern.

---

## Outputs

```
out/signals/
├── eta_multisine_signal_1.csv                  ← columns: t [s], eta
├── eta_multisine_fpto_random_signal_2.csv      ← columns: t [s], eta, fpto
└── eta_multisine_fpto_multisin_signal_3.csv    ← columns: t [s], eta, fpto
```

Each file contains one full period of the signal. The column `fpto` contains the
**unfiltered** PTO signal for signal 2 — apply the filter before use if needed.

---

## Notes

- All scripts use `sin` convention: $s_k(t) = A_k \sin(\omega_k t + \phi_k)$.
- The `multisine` function returns `(signal, phases)` — keep the phases to reconstruct
  the corresponding excitation force analytically from BEM data.
- PTO signals are clipped to $\pm 10^6$ N before saving.
