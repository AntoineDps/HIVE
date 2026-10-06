# run_validate

Runs one or more physics-based models against processed CFD reference data and
computes performance metrics across the sea state space.

## Usage

```bash
python -m script.run_validate --case <filename>.json
```

Config files live in `inputs/validate_case/`.

## What it does

1. Loads processed `DataHandle` objects from `data/handled/<wave>/`
2. Loads `model.json` definitions from `models/<name>/`
3. For each model and each sea state, runs `SimRun.simulate()` with the chosen solver
4. Clips the simulation to the physically valid window (warmup + causal margins)
5. Computes metrics (NRMSE, RMSE, power) and aggregates results
6. Generates scatter plots over the $(H_s, T_e)$ space and time series comparisons

## Config fields

See [Input Config Reference — validate](config_reference.md#validate-config).

## Rerun options

| Option | Action |
|---|---|
| `[1]` | Erase and redo all simulations and plots |
| `[2]` | Stop |
| `[3]` | Reload saved figures (no recompute) |
| `[4]` | Reload processed CSVs, remake plots only |

## Outputs

```
models/<name>/validation/
├── result.json           ← metric summary per model and sea state
├── run.log
├── data/
│   └── <label>_simrun.csv
└── plots/
    └── *.png / *.pkl
```

## Time margins

HIVE automatically computes two time margins from the base model:

- **`t_warmup`** — radiation state warmup (from radiation SS pole magnitudes)
- **`t_causal`** — non-causal excitation horizon (from BEM IRF tail)

The clipped window `[t_warmup, T − t_causal]` is used for all metric computations.
These values are logged and saved in `result.json`.

## Notes

- Option `[4]` skips simulation entirely and reloads existing CSVs — useful when only plot parameters change.
- Mixed lists of `DataHandle` (CFD reference) and `SimRun` (model) can be passed to the same plotter calls because both expose `.dataset` and `.label`.
