# Validate run.json

Full reference for `inputs/validate_case/run.json`.

## Complete example

```json
{
  "name": "linear_vs_weakly_nl",
  "wave_sources": [
    { "type": "cfd_case", "case_name": "Hs2p0_Te5p0_D120000_K-280000" },
    { "type": "cfd_case", "case_name": "Hs3p0_Te8p0_D120000_K-280000" }
  ],
  "model_entries": [
    {
      "model": "linear_sphere_r5",
      "label": "linear",
      "solver": {
        "method": "RK4",
        "dt": 0.05,
        "interpolation": {
          "eta":   "linear",
          "fe":    "linear",
          "waves": "cubic"
        }
      },
      "pto": { "damping": 120000.0, "stiffness": 0.0 }
    },
    {
      "model": "weakly_nonlinear_sphere_r5",
      "label": "weakly_nl",
      "solver": { "method": "RK4", "dt": 0.05, "interpolation": {} },
      "pto": { "damping": 120000.0, "stiffness": 0.0 }
    }
  ],
  "plot_options": {
    "states":        true,
    "forces":        false,
    "variable":      true,
    "metric_scalar": true,
    "metric_corr":   true,
    "metric_phase":  false,
    "metric_grid":   { "metrics": ["nrmse_range_P_abs", "time_ratio"] },
    "metric_stat":   true
  }
}
```

## Field reference

### `name`
Output directory: `out/validated_{name}/`

### `wave_sources`

```json
{ "type": "cfd_case", "case_name": "Hs2p0_Te5p0_D120000_K-280000" }
```

`type` is always `"cfd_case"`. `case_name` must exactly match a folder under `data/cfd/` or the label in `data/handled/`.

### `model_entries[].solver`

See [Solvers & Interpolation](../concepts/solvers.md) for full details.

```json
{
  "method": "RK4",
  "dt": 0.05,
  "interpolation": {
    "eta":   "linear",
    "fe":    "cubic",
    "waves": "linear"
  }
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `method` | string | `"RK4"` | ODE integrator: `"RK4"` or `"Euler"` |
| `dt` | float | `0.05` | Time step [s] |
| `interpolation` | object | `{}` | Per-signal interpolation method |

### `model_entries[].pto`

```json
{ "damping": 120000.0, "stiffness": 0.0 }
```

Sets the linear PTO:

$$f_{pto}(t) = -B_{pto}\,\dot{x}(t) - K_{pto}\,x(t)$$

| Field | Unit | Description |
|---|---|---|
| `damping` | N·s/m | $B_{pto}$ — resistive damping |
| `stiffness` | N/m | $K_{pto}$ — reactive stiffness (negative = capacitive) |

### `plot_options`

| Key | Type | Effect |
|---|---|---|
| `states` | bool | Plot x, xdot, eta, f_pto, P_abs (P_abs y-axis starts at 0) |
| `forces` | bool | Plot all force components |
| `variable` | bool | One plot per variable, all models overlaid |
| `metric_scalar` | bool | Subplots per metric, bars per model, x-axis = sea state |
| `metric_corr` | bool | Cross-correlation curves |
| `metric_phase` | bool | Instantaneous phase difference |
| `metric_grid` | object or bool | Te×Hs scatter, coloured by metric. Pass `{"metrics": [...]}` to choose |
| `metric_stat` | bool | Whisker plot: mean line ± std band, min/max caps per model |

#### `metric_grid` metric names

Any combination of:

```
nrmse_range_x      nrmse_range_xdot      nrmse_range_P_abs
nrmse_std_x        nrmse_std_xdot        nrmse_std_P_abs
rmse_x             rmse_xdot             rmse_P_abs
time_ratio
peak_lag_s_x       peak_lag_s_xdot
phase_mean_x_deg   phase_mean_xdot_deg
```
