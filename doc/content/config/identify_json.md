# Identify config JSON

Full reference for `inputs/identify_case/{name}.json`.

## Shared fields (all schemes)

```json
{
  "scheme":      "viscous_drag",
  "waves_train": [ { "type": "cfd_case", "case_name": "..." } ],
  "base_model":  "linear_sphere_r5",
  "output_name": "weakly_nonlinear_sphere_r5",
  "solver": {
    "method": "RK4",
    "dt": 0.05,
    "interpolation": {}
  },
  "scheme_params": { ... }
}
```

| Field | Type | Description |
|---|---|---|
| `scheme` | string | `"viscous_drag"` or `"pi_gain"` |
| `waves_train` | list | Training sea states |
| `base_model` | string | Starting model under `source/models/` |
| `output_name` | string | Output model name |
| `solver` | object | ODE solver config — same as validate |

---

## `viscous_drag` scheme_params

```json
"scheme_params": {
  "methods":          ["global", "response_opt", "response_grid"],
  "c_v_bounds":       [0.0, 5000.0],
  "n_grid":           50,
  "base_force":       "fhyd_lin",
  "metric":           "nrmse_range_x",
  "metric_direction": "minimize"
}
```

| Field | Default | Description |
|---|---|---|
| `methods` | `["global"]` | Which methods to run (all run, best wins model.json) |
| `c_v_bounds` | `[0, 5000]` | Search range for $c_v$ [N·s²/m²] |
| `n_grid` | `50` | Number of $c_v$ samples in `response_grid` |
| `base_force` | `"fhyd_lin"` | Column from DataHandle used as linear baseline |
| `metric` | `"nrmse_range_x"` | Objective metric. Any metric name from [Metrics](../concepts/metrics.md) |
| `metric_direction` | `"minimize"` | `"minimize"` or `"maximize"` |

### Available methods

| Method | How it works |
|---|---|
| `global` | Single $c_v$ minimising mean metric across all cases |
| `response_opt` | Per-case $c_v$ via `scipy.optimize.minimize_scalar` |
| `response_grid` | Per-case $c_v$ via dense grid search; plots the cost curve |

---

## `pi_gain` scheme_params

```json
"scheme_params": {
  "methods":          ["grid", "response_opt"],
  "B_bounds":         [0.0, 500000.0],
  "K_bounds":         [-500000.0, 500000.0],
  "n_B":              20,
  "n_K":              1,
  "metric":           "mean_P_abs",
  "metric_direction": "maximize"
}
```

| Field | Default | Description |
|---|---|---|
| `methods` | `["grid"]` | Which methods to run |
| `B_bounds` | `[0, 500000]` | Damping $B_{pto}$ search range [N·s/m] |
| `K_bounds` | `[-500000, 500000]` | Stiffness $K_{pto}$ search range [N/m] |
| `n_B` | `20` | Grid points in B direction |
| `n_K` | `1` | Grid points in K direction. Set to `1` for pure damping (no reactive) |
| `metric` | `"mean_P_abs"` | Objective metric |
| `metric_direction` | `"maximize"` | `"minimize"` or `"maximize"` |

!!! tip "1D vs 2D search"
    Set `n_K = 1` and `K_bounds = [0, 0]` for a pure damping search (no reactive control).  
    Set `n_K > 1` for full 2D B×K landscape — produces a heatmap plot.

### Available methods

| Method | Description |
|---|---|
| `grid` | 2D B×K grid search. Plots power landscape |
| `response_opt` | Per-case scipy optimisation (faster, no landscape) |

### Available metrics

| Metric | Description |
|---|---|
| `"mean_P_abs"` | Mean absorbed power [W] — use with `"maximize"` |
| `"nrmse_range_x"` | NRMSE of displacement — use with `"minimize"` |
| `"nrmse_range_xdot"` | NRMSE of velocity |
| `"rmse_P_abs"` | RMSE of absorbed power vs CFD |
