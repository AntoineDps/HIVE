# Roadmap & Known Gaps

This page tracks planned features, missing integrations, and known issues in HIVE.

---

## Pipeline gaps

### 1. `linear_si` models not usable in `run_validate`

Models identified by `linear_si` produce two state-space blocks (`eta2f` and `f2v`). These are not yet wired into `SimRun`, so they cannot be passed to `run_validate` for metric evaluation.

**Planned:** A `LinearSIModel` class that chains the two state-space blocks and exposes the same interface as the existing `SimRun`.

---

### 2. No direct link from `run_signals` to `linear_si`

Signal generation (`run_signals`) and system identification (`linear_si`) are currently separate steps. The user must manually specify `signal_file` in the identify config and ensure it points to a file produced by `run_signals`.

**Planned:** A helper in the identify config or a `run_signals` step embedded in the `linear_si` scheme, so that the signal parameters are versioned alongside the identification result.

---

### 3. ETFE averaging not implemented

Currently a single period of the multi-sine signal is used for ETFE estimation. Averaging over multiple periods (when the experiment runs for $M$ periods) would reduce variance by a factor of $M$.

**Planned:** Support for multi-period signals and averaged ETFE computation.

---

### 4. No passivity enforcement

The identified `f2v` state-space model is not guaranteed to be passive (i.e. it may produce energy from nothing). For physics-consistent simulation, passivity should be enforced as a post-processing step.

**Planned:** LMI-based passivity enforcement or enforcement via balanced truncation.

---

## Missing features

### Wave prediction / forecast

For model-predictive control (MPC), the wave elevation must be known a number of seconds into the future (the non-causal horizon `t_causal`). HIVE currently does not include a wave prediction module.

**Planned:** ARMA/Kalman-based short-term wave predictor, or interface to an external predictor.

---

### Model-predictive control (MPC)

The PI gain scheme provides a simple reactive controller. An MPC layer using the identified linear models and a wave forecast would allow optimal, constrained power extraction.

---

### Nonlinear system identification (SINDy)

For cases where the linear WEC model is insufficient (large amplitude, vortex-induced forces), SINDy (Sparse Identification of Nonlinear Dynamics) could identify a parsimonious nonlinear model from CFD data.

**Planned:** A `sindy` scheme module wrapping PySINDy, with a library of candidate nonlinear terms (drag, Froude–Krylov second-order, etc.).

---

### Cross-validation

Models identified on one set of sea states should be validated on a held-out set. A cross-validation workflow (separate train/test splits in the config) is not yet implemented.

---

### Uncertainty quantification (UQ)

HIVE produces point estimates of model parameters. Bayesian or bootstrap-based UQ would provide confidence intervals on predicted quantities.

---

## Known code issues

| Issue | Severity | Location | Description |
|---|---|---|---|
| Unfiltered PTO saved in signal 2 | Low | `run_signals` | The CSV for signal 2 stores the raw random PTO force without the low-pass filter. Users must filter before use. |
| `set_title` bug in plotter | Low | `src/Plotter.py` | Calling `set_title` on certain plot types raises `AttributeError`. Use `ax.set_title` directly as a workaround. |
| `multisine` return type inconsistency | Low | `src/signals.py` | Returns `(signal, phases)` in some code paths and only `signal` in others. Always unpack with `signal, _ = multisine(...)`. |
| Debug print statements | Low | Various | Several `print()` debug calls remain in `src/`. These do not affect results but clutter logs. |

---

## Undocumented scripts

The following utility scripts exist in `script/` but are not yet documented:

- `run_bem.py` — calls NEMOH/CAPYTAINE and saves the BEM pickle
- `run_plot.py` — standalone plotter for re-rendering saved figures
- `run_export.py` — exports result summaries to Excel

Documentation for these will be added in a future update.
