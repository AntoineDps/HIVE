# run_validate

Runs one or more physics-based models on each sea state, compares outputs against the CFD reference, and computes error metrics.

## Run

```bash
python -m script.run_validate
```

## Input

`inputs/validate_case/run.json`

```json
{
  "name": "my_validation",
  "wave_sources": [
    { "type": "cfd_case", "case_name": "Hs2p0_Te5p0_D120000_K-280000" }
  ],
  "model_entries": [
    {
      "model":  "linear_sphere_r5",
      "label":  "linear",
      "solver": { "method": "RK4", "dt": 0.05, "interpolation": {} },
      "pto":    { "damping": 0.0, "stiffness": 0.0 }
    }
  ],
  "plot_options": {
    "states":        true,
    "forces":        true,
    "variable":      true,
    "metric_scalar": true,
    "metric_corr":   true,
    "metric_phase":  true,
    "metric_grid":   { "metrics": ["nrmse_range_P_abs", "time_ratio"] },
    "metric_stat":   true
  }
}
```

### Top-level fields

| Field | Type | Description |
|---|---|---|
| `name` | string | Output folder name → `out/validated_{name}/` |
| `wave_sources` | list | Sea states to simulate (see below) |
| `model_entries` | list | Model configurations to compare (see below) |
| `plot_options` | object | Which plots to generate (see below) |

### `wave_sources`

Each entry is a sea state source:

```json
{ "type": "cfd_case", "case_name": "Hs2p0_Te5p0_D120000_K-280000" }
```

| Field | Type | Description |
|---|---|---|
| `type` | string | `"cfd_case"` — load from handled CFD data |
| `case_name` | string | CFD folder name |

### `model_entries`

| Field | Type | Required | Description |
|---|---|---|---|
| `model` | string | ✓ | Model folder name under `source/models/` |
| `label` | string | ✓ | Display name in plots and metric CSVs |
| `solver` | object | ✓ | Solver configuration (see [Solvers](../concepts/solvers.md)) |
| `pto` | object | ✓ | PTO settings: `{ "damping": float, "stiffness": float }` |

### `plot_options`

| Key | Type | Description |
|---|---|---|
| `states` | bool | Time series of x, xdot, eta, f_pto, P_abs |
| `forces` | bool | Time series of all force components |
| `variable` | bool | Per-variable comparison plots |
| `metric_scalar` | bool | Bar chart of scalar metrics per model and sea state |
| `metric_corr` | bool | Cross-correlation plots |
| `metric_phase` | bool | Instantaneous phase difference plots |
| `metric_grid` | object | Scatter grid Te×Hs coloured by metric value |
| `metric_stat` | bool | Whisker plot: mean±std per model across sea states |

#### `metric_grid` options

```json
"metric_grid": {
  "metrics": ["nrmse_range_P_abs", "nrmse_range_x", "time_ratio"]
}
```

Available metric names: any key produced by `Metric.compute_all()` — see [Metrics](../concepts/metrics.md).

## Options menu

| Option | Effect |
|---|---|
| `[1]` | Run all simulations, overwrite existing results |
| `[2]` | Run only missing cases, keep existing |
| `[3]` | Run all, append to existing metric CSVs |
| `[4]` | Reload existing CSVs, remake plots only — **no simulation** |

!!! tip "Option 4"
    Changing `plot_options` in `run.json` and re-running with option 4 regenerates all plots without re-running any simulation. Metric CSVs are loaded as-is.

## Output structure

```
out/validated_{name}/
├── run.log
├── data/
│   └── {label}_{model}_simrun.csv      # clipped time series per case per model
├── plots/
│   ├── {label}_x.png
│   ├── {label}_P_abs.png
│   └── ...
└── metric/
    ├── {label}.csv                     # scalar metrics per model
    ├── cross_corr_{label}.csv
    ├── inst_phase_{label}.csv
    └── model_stats.csv                 # aggregate stats across sea states
```

### Simulation CSV columns

| Column | Unit | Description |
|---|---|---|
| `t` | s | Time (clipped, warmup removed) |
| `x` | m | Heave displacement |
| `xdot` | m/s | Heave velocity |
| `eta` | m | Wave elevation |
| `f_pto` | N | PTO force |
| `P_abs` | W | Absorbed power |

### Metric CSV columns

| Column | Description |
|---|---|
| `model` | Model label |
| `nrmse_range_x` | NRMSE of heave (range normalised) |
| `nrmse_range_xdot` | NRMSE of velocity |
| `nrmse_range_P_abs` | NRMSE of absorbed power |
| `time_ratio` | Ratio of simulation wall-clock time to signal duration |
| `peak_lag_s_x` | Lag at cross-correlation peak for heave [s] |
| `phase_mean_x_deg` | Mean instantaneous phase error for heave [°] |
