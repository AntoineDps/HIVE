# run_identify

Identifies model parameters from CFD training data using one of the available schemes. Outputs a ready-to-use `model.json` that can be loaded by `run_validate.py`.

## Run

```bash
python -m script.run_identify
```

The script reads the active config from `inputs/identify_case/` (prompts you to pick a file if multiple exist).

## Options menu

| Option | Effect |
|---|---|
| `[1]` | Run identification, overwrite existing |
| `[2]` | Run identification, skip if output exists |
| `[3]` | Append results |

## Input config structure

```json
{
  "scheme":      "viscous_drag",
  "waves_train": [ { "type": "cfd_case", "case_name": "..." } ],
  "base_model":  "linear_sphere_r5",
  "output_name": "weakly_nonlinear_sphere_r5",
  "solver":      { "method": "RK4", "dt": 0.05, "interpolation": {} },
  "scheme_params": { ... }
}
```

| Field | Type | Description |
|---|---|---|
| `scheme` | string | Identification scheme: `"viscous_drag"` or `"pi_gain"` |
| `waves_train` | list | Training sea states (same format as `run_validate.py`) |
| `base_model` | string | Starting model folder under `source/models/` |
| `output_name` | string | Output model folder name |
| `solver` | object | ODE solver config |
| `scheme_params` | object | Scheme-specific parameters (see below) |

---

## Scheme: `viscous_drag`

Identifies a quadratic viscous drag coefficient $c_v$ by fitting:

$$f_{drag}(t) = -c_v \, |\dot{x}| \, \dot{x}$$

The residual force is $f_{res} = f_{hyd,CFD} - f_{hyd,lin}$.

### `scheme_params`

```json
"scheme_params": {
  "methods": ["global", "response_opt", "response_grid"],
  "c_v_bounds": [0.0, 5000.0],
  "n_grid": 50,
  "base_force": "fhyd_lin",
  "metric": "nrmse_range_x",
  "metric_direction": "minimize"
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `methods` | list | `["global"]` | Identification methods to run. See methods below |
| `c_v_bounds` | [float, float] | `[0, 5000]` | Search bounds for $c_v$ [N·s²/m²] |
| `n_grid` | int | `50` | Grid points for `response_grid` method |
| `base_force` | string | `"fhyd_lin"` | Dataset column used as linear force baseline |
| `metric` | string | `"nrmse_range_x"` | Objective metric |
| `metric_direction` | string | `"minimize"` | `"minimize"` or `"maximize"` |

### Methods

| Method | Description |
|---|---|
| `global` | Fits $c_v$ globally across all cases by minimising mean metric |
| `response_opt` | Optimises $c_v$ per sea state independently |
| `response_grid` | Grid search of $c_v$ per sea state, plots power landscape |

### Outputs

- `source/models/{output_name}/model.json` — best method's $c_v$ baked in
- `out/identified_{output_name}/result.json` — all method results
- Plots: `coeff_grid`, `metric_grid`, `coeff_distribution`, `metric_distribution`, timeseries per case

---

## Scheme: `pi_gain`

Identifies optimal PI PTO gains $(B_{pto}, K_{pto})$ that maximise absorbed power.

$$f_{pto}(t) = -B_{pto}\,\dot{x}(t) - K_{pto}\,x(t)$$

### `scheme_params`

```json
"scheme_params": {
  "methods": ["grid", "response_opt"],
  "B_bounds": [0.0, 500000.0],
  "K_bounds": [-500000.0, 500000.0],
  "n_B": 20,
  "n_K": 20,
  "metric": "mean_P_abs",
  "metric_direction": "maximize"
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `methods` | list | `["grid"]` | Identification methods |
| `B_bounds` | [float, float] | `[0, 500000]` | Damping search bounds [N·s/m] |
| `K_bounds` | [float, float] | `[-500000, 500000]` | Stiffness search bounds [N/m] |
| `n_B` | int | `20` | Grid points in B direction |
| `n_K` | int | `20` | Grid points in K direction |
| `metric` | string | `"mean_P_abs"` | Objective metric |
| `metric_direction` | string | `"maximize"` | `"minimize"` or `"maximize"` |

### Methods

| Method | Description |
|---|---|
| `grid` | 2D grid search over $(B, K)$ per sea state |
| `response_opt` | Scipy optimisation per sea state |

### Outputs

- `source/models/{output_name}/model.json` — model with optimal gains
- Plots: `gains_grid` (power landscape), `timeseries` (ref vs identified), `scatter_B`, `scatter_K` (Te×Hs grid)

---

## Output model.json

After identification, a `model.json` is written to `source/models/{output_name}/`:

```json
{
  "pkl_name": "r5",
  "force_terms": {
    "radiation": "state_space",
    "excitation": "linear_conv",
    "drag": "viscous"
  },
  "identified": {
    "c_v": 1234.5,
    "method": "global",
    "metric": "nrmse_range_x",
    "metric_value": 0.042
  }
}
```

This model can immediately be used as a `model_entries` entry in `run_validate.py`.
