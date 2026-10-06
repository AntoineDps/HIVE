# Pipeline Overview

HIVE is built around three sequential scripts. Each script reads a JSON config file,
performs its task, and writes outputs to a structured folder.

```
CFD data
    │
    ▼
run_handle_data   →  data/handled/<name>/
    │
    ▼
run_validate      →  models/<name>/validation/
    │
    ▼
run_identify      →  models/<name>/identification/
```

Optionally, `run_signals` generates multi-sine excitation signals independently of
CFD data, for use with the `linear_si` identification scheme.

---

## Script roles

| Script | Input | Output |
|---|---|---|
| `run_handle_data` | Raw CFD time series | Processed `data.csv`, wave parameters, plots |
| `run_validate` | Processed data + model JSON | Metric tables, comparison plots |
| `run_identify` | Processed data + base model | Identified parameters, updated model JSON, plots |
| `run_signals` | Config block in script | Multi-sine CSV files for SI |

---

## Rerun options

Every script detects an existing output folder and offers a menu:

```
[1] erase and redo all
[2] stop, do nothing
[3] load saved figures
[4] remake figures only  (run_handle_data) / run with date suffix (run_identify)
```

Option `4` lets you regenerate plots without reprocessing data, which is useful when
tuning plot parameters.

---

## Data flow in detail

### 1. `run_handle_data`

Reads raw SPH/CFD output files, applies cut times and optional resampling,
computes wave parameters, and calls `DataHandle.processData()` to derive
hydrodynamic quantities (radiation force, hydrostatic force, linear excitation force).
One `DataHandle` object is saved per case.

### 2. `run_validate`

Loads the saved `DataHandle` objects and a `model.json`. For each sea state and each
model variant, it runs a simulation via `SimRun.simulate()` and computes metrics
against the CFD reference. Results are aggregated into scatter plots over the
$(H_s, T_e)$ parameter space.

### 3. `run_identify`

Uses a **scheme module** (e.g. `Viscous_drag`, `Pi_gain`, `Linear_si`) to identify
model parameters from the CFD data. The scheme module owns the identification logic;
`Optimizer` provides the generic simulation and saving infrastructure.

---

## Adding a new scheme

1. Create `source/classes/MyScheme.py` implementing `prep()`, `run()`, `save()`,
   `log_result()`, `save_data()`, and `PLOT_DISPATCH`.
2. Register it in `run_identify.py`:
   ```python
   import source.classes.MyScheme as _my_scheme
   SCHEME_MODULES = {
       ...
       "my_scheme": _my_scheme,
   }
   ```
3. Create an `inputs/identify_case/my_scheme_example.json` config.
