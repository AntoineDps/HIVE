# Roadmap & Known Gaps

This page tracks features that are planned, in progress, or identified as missing.
It is updated as the codebase evolves.

---

## Pipeline integration gaps

These features exist as standalone scripts but are not yet wired into the main
`run_handle_data` → `run_validate` → `run_identify` pipeline.

### `linear_si` in `run_validate`

The `linear_si` scheme produces a `model.json` with state-space matrices (`eta2f`,
`f2v`, `cascade`), but `run_validate.py` currently only supports physics-based models
via `Model.from_config` + `SimRun.simulate`. A new simulation path is needed that:

- Detects `"type": "linear_si"` in `model.json`
- Loads the cascade SS matrices
- Applies the wave advance $\tau^*$ to the input signal
- Runs the discrete SS loop instead of the ODE integrator
- Returns a `SimRun`-compatible object for metric computation

### Signal generation → `linear_si` identification

`run_signals.py` generates multi-sine excitation signals intended for `linear_si`
identification, but the connection is currently manual. The `linear_si` scheme in
`run_identify.py` loads signals from `DataHandle` (CFD data). A dedicated
signal-based data feed path is needed, so that generated signals can flow directly
into identification without manual CSV loading.

### Multiple ETFE realisations averaging

The theoretical identification procedure (Faedo et al.) recommends averaging the
ETFE over $N$ realisations of the same spectrum with different random phases to
reduce variance. Currently only one realisation is used per case. The `run_eta2f.py`
script should loop over realisations and average $\hat{G}_1(j\omega)$ before fitting.

### Passivity enforcement as pipeline step

`source/function/passify.py` implements LMI-based passivity enforcement (Faedo et al.
2021) and works as a standalone call, but it is not invoked automatically after
`linear_si` identification. It should become an optional post-processing step
controlled by a config flag:

```json
"scheme_params": {
  "passivity_enforcement": true,
  "passivity_weights": [0.5, 0.5]
}
```

The current status is: the VF-identified `eta2f` model is already passive in the
test case; the N4SID-identified `f2v` model is not, and passivity must be enforced
manually.

---

## Wave prediction & free surface

### Wave elevation predictor

The `linear_si` cascade model requires $\eta(t + \tau^*)$ — wave elevation $\tau^*$
seconds in the future. For offline simulation this is trivial (the full signal is
available), but real-time deployment requires a **wave predictor**. Candidates:

- Autoregressive (AR) model fitted to past $\eta$ measurements
- Kalman filter with a linear wave model
- Data-driven predictor using the identified $\hat{G}_1$ impulse response

### Free surface reconstruction

Estimating $\eta(t)$ at the WEC location from the buoy response $(x, \dot{x})$ — the
inverse problem. Useful when no upstream wave gauge is available.

---

## Control

### Model Predictive Control (MPC)

The cascade SS model $\eta \rightarrow \dot{x}$ from `run_eta2f2v` is a natural plant
model for MPC. A receding-horizon controller that maximises absorbed power subject to
displacement and force constraints has not been implemented.

### Closed-loop causal simulation

Running the identified SS model in closed loop with a control law, stepping forward
one sample at a time. The discrete SS loop structure is demonstrated in
`run_sindy_msd.py` but no control law is attached.

---

## Nonlinear identification

### SINDy with nonlinear library

All current SINDy experiments use `PolynomialLibrary(degree=1)` — a linear library.
A degree-2 library would allow identification of:

- Quadratic viscous drag: $c_v |\dot{x}|\dot{x}$ (currently handled separately by
  `run_identify` with scheme `viscous_drag`)
- Nonlinear hydrostatics
- Wave–structure interaction nonlinearities

### SINDy pipeline integration

The SINDy scripts (`run_sindy.py`, `run_sindy_rad.py`, `run_sindy_fe.py`,
`run_sindy_msd.py`) are standalone exploratory scripts. A `Sindy.py` scheme module
following the same contract as `Viscous_drag.py` and `Linear_si.py` would integrate
SINDy into `run_identify` alongside the existing schemes.

---

## Validation & robustness

### Cross-validation across sea states

All identification currently uses the same sea states for training and evaluation.
A proper train/test split — identify on a subset of $(H_s, T_e)$ conditions and
validate on held-out conditions — is not implemented.

### Uncertainty quantification

No confidence intervals exist on identified parameters ($c_v$, PI gains, SS matrices)
or on model predictions. Bootstrap resampling or Monte Carlo over wave realisations
would provide error bars for the metric scatter plots.

### ETFE vs BEM mismatch in `run_eta2f`

The ETFE computed from the multi-sine signal does not match the BEM transfer function
as cleanly as expected. The BEM-direct fitting path works correctly. Root cause not
fully resolved — likely a phase convention difference between `multisine` (uses `sin`)
and the BEM excitation representation (complex exponential), or a normalisation issue
at the FFT level.

---

## Known code issues

| Location | Issue |
|---|---|
| `run_signals.py` — `save_signals` call | Saves unfiltered PTO signal (`signal_22`) instead of filtered (`signal_filt_22`) |
| `run_signals.py` — Signal 3 plot | Uses `axes[0].set_title` on a flat `ax` array; should be `ax[0].set_title` |
| `multisine` in `run_signals.py` | Returns only `signal`, not `(signal, phases)` — callers in `run_eta2f.py` expect the tuple form |
| `run_handle_data.py` line 366 | Leftover `a = 2  #!` debug marker |
| `run_identify.py` | `plot_options` from config is ignored for `linear_si`; all plots always run |

---

## Documentation

The following scripts and features are not yet documented:

- `run_signals.py` — multi-sine signal generation
- `run_eta2f.py`, `run_f2v.py`, `run_eta2f2v.py` — system identification scripts
- SINDy scripts (`run_sindy*.py`)
- Cross-links between pages may break if the `sections/` folder structure changes
