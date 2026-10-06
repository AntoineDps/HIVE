# Naming Conventions

HIVE uses consistent naming throughout config files, output folders, and data columns. This page documents those conventions.

---

## Config stems and output folder names

Every script takes a `--case <name>.json` argument. The stem of that filename (without `.json`) becomes the output folder name:

| Script | Config location | Output location |
|---|---|---|
| `run_handle_data` | `inputs/handle_case/<name>.json` | `data/handled/<name>/` |
| `run_validate` | `inputs/validate_case/<name>.json` | `models/<model>/validation/` |
| `run_identify` | `inputs/identify_case/<name>.json` | `models/<name>/` |

Choose stems that are short, lowercase, and use underscores — e.g. `r5_reg` for a regular-wave run using BEM model `r5`.

---

## Case labels

Within a handled dataset, each CFD case is assigned a **label** derived from its wave parameters. The label appears as the CSV filename and as the series label in plots:

```
T8_H2      ← wave period 8 s, wave height 2 m
T10_H3
```

Labels are constructed automatically from `cases[].T` and `cases[].H` in the handle config.

---

## Model folder names

Model folders live under `models/`. Each folder contains a `model.json` describing the model's physics and parameters. The folder name is:

- For base physics models: a short descriptor chosen by the user, e.g. `linear`, `linear_viscous`, `r5_linear`
- For identified models: the stem of the identify config, e.g. `r5_viscous` (output of `run_identify --case r5_viscous.json`)

The field `base_model` in the identify config must match an existing folder name exactly.

---

## BEM pickle files

BEM (boundary element method) results are stored as `.pkl` files under `models/bem/`. The field `hydro_sphere_pkl` in the handle config gives the filename stem (without `.pkl`):

```json
"hydro_sphere_pkl": "r5"
```

This loads `models/bem/r5.pkl`.

---

## Data column names

Processed CSV files (in `data/handled/<name>/data/`) use standardised column names:

| Column | Units | Description |
|---|---|---|
| `t` | s | Time |
| `x` | m | Displacement |
| `xdot` | m/s | Velocity |
| `eta` | m | Wave elevation |
| `fe` | N | Excitation force (CFD-derived) |
| `fe_lin` | N | Linear excitation force (from BEM) |
| `f_rad` | N | Radiation force |
| `f_hs` | N | Hydrostatic (restoring) force |
| `f_pto` | N | PTO force |
| `P_abs` | W | Instantaneous absorbed power |

SimRun output CSVs use the same column names so that both can be passed to the same plotter calls.

---

## Metric names

Metrics passed in the `metrics` list of validate and identify configs follow this pattern:

| Name | Description |
|---|---|
| `nrmse_x` | NRMSE on displacement |
| `nrmse_xdot` | NRMSE on velocity |
| `rmse_x` | RMSE on displacement [m] |
| `rmse_xdot` | RMSE on velocity [m/s] |
| `mean_P_abs` | Mean absorbed power [W] |

See [Metrics](../theory/metrics.md) for definitions.

---

## Signal file naming

Output files from `run_signals` follow this pattern under `out/signals/`:

| File | Contents |
|---|---|
| `eta_multisine_signal_1.csv` | Wave elevation only |
| `eta_multisine_fpto_random_signal_2.csv` | Wave + random PTO force |
| `eta_multisine_fpto_multisine_signal_3.csv` | Wave + multi-sine PTO force |

When using a signal file in an identify config, provide the path relative to `code/`:

```json
"signal_file": "out/signals/eta_multisine_signal_1.csv"
```
