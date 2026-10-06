# Input Config Reference

All scripts are driven by JSON config files. This page lists every field for each script.

---

## Handle config

Path: `inputs/handle_case/<name>.json`

```json
{
  "wave_type": "regular",
  "cut_time": [50, 0],
  "duration": null,
  "hydro_sphere_pkl": "r5",
  "dt": 0.05,
  "cases": [
    {
      "T": 8.0,
      "H": 2.0,
      "damping": 120000,
      "stiffness": -280000,
      "plot": true
    }
  ],
  "plot_options": {
    "sea_state": true,
    "power": true,
    "dynamics": true,
    "hydro": false,
    "3d": false,
    "variable": ["fe_lin", "x"]
  }
}
```

| Field | Type | Description |
|---|---|---|
| `wave_type` | string | `"regular"` or `"irregular"` |
| `cut_time` | [float, float] | Seconds to cut from [start, end] |
| `duration` | float \| null | If set, truncates to this duration after cut |
| `hydro_sphere_pkl` | string | BEM pickle filename (without `.pkl`) in `models/bem/` |
| `dt` | float \| null | Resample timestep [s]. Null keeps original |
| `cases` | list | One entry per CFD case |
| `cases[].T` | float | Wave period [s] |
| `cases[].H` | float | Wave height [m] |
| `cases[].damping` | float | PTO damping [N s/m] |
| `cases[].stiffness` | float | PTO stiffness [N/m] |
| `cases[].plot` | bool | Include in plots (default true) |
| `plot_options` | dict | Which plots to generate |

---

## Validate config

Path: `inputs/validate_case/<name>.json`

```json
{
  "waves_validate": ["example"],
  "models": ["linear", "linear_viscous"],
  "solver": {
    "method": "RK4",
    "dt": 0.05,
    "interpolation": {}
  },
  "metrics": ["nrmse_x", "nrmse_xdot", "mean_P_abs"],
  "plot_options": {
    "scatter": true,
    "timeseries": true
  }
}
```

| Field | Type | Description |
|---|---|---|
| `waves_validate` | list[str] | Handle config stems to load from `data/handled/` |
| `models` | list[str] | Model folder names under `models/` |
| `solver.method` | string | `"RK4"`, `"RK2"`, or `"euler"` |
| `solver.dt` | float | Integration timestep [s] |
| `metrics` | list[str] | See [Metrics](../theory/metrics.md) |

---

## Identify config

Path: `inputs/identify_case/<name>.json`

```json
{
  "scheme": "viscous_drag",
  "base_model": "linear",
  "waves_train": ["example"],
  "scheme_params": {
    "c_v_range": [0, 5000],
    "n_points": 50
  },
  "solver": {
    "method": "RK4",
    "dt": 0.05,
    "interpolation": {}
  },
  "plot_options": {
    "grid": true,
    "timeseries": true
  }
}
```

| Field | Type | Description |
|---|---|---|
| `scheme` | string | Identification scheme (`viscous_drag`, `pi_gain`, `linear_si`) |
| `base_model` | string | Folder name under `models/` for the base physics model. Use `"none"` for `linear_si`. |
| `waves_train` | list[str] | Handle config stems to use as training data |
| `scheme_params` | dict | Scheme-specific parameters (see below) |
| `solver` | dict | Same as validate config |

### `viscous_drag` scheme params

| Field | Description |
|---|---|
| `c_v_range` | [min, max] search range for $c_v$ [N s²/m²] |
| `n_points` | Number of grid points |
| `objective` | Objective name (default `"nrmse_x"`) |

### `pi_gain` scheme params

| Field | Description |
|---|---|
| `Kp_range` | [min, max] for proportional gain |
| `Ki_range` | [min, max] for integral gain |
| `n_points` | Grid resolution per dimension |

### `linear_si` scheme params

| Field | Description |
|---|---|
| `signal_file` | Path to signal CSV (relative to `code/`) |
| `input_signals` | List of input column names |
| `output_signals` | List of output column names |
| `identification` | `"frequency"` (VF) or `"time"` (N4SID) |
| `w_start`, `w_end` | Frequency range for fitting [rad/s] |
| `order_list` | List of VF pole orders to sweep |
| `tau_min`, `tau_max`, `n_tau` | Causality delay sweep parameters |
| `smooth_window` | ETFE smoothing window length |
| `num_block_rows` | N4SID block rows (Hankel matrix size) |
