# run_handle_data

Processes raw CFD time series into a standardised format ready for validation and identification.

## Usage

```bash
python -m script.run_handle_data --case <filename>.json
```

Config files live in `inputs/handle_case/`. The filename stem becomes the output folder name under `data/handled/`.

## What it does

1. Reads each SPH/CFD case from `data/cfd/`
2. Instantiates a `DataHandle` object per case
3. Calls `processData()` — computes radiation force, hydrostatic force, linear excitation force (`fe_lin`), and absorbed power
4. Optionally resamples to a target `dt`
5. Saves processed data to `data/handled/<config_stem>/data/`
6. Generates and saves plots to `data/handled/<config_stem>/plots/`

## Config fields

See [Input Config Reference — handle](config_reference.md#handle-config).

## Outputs

```
data/handled/<name>/
├── <name>.json          ← copy of the config used
├── run.log
├── data/
│   ├── <label>.csv      ← one file per case (time series)
│   └── <label>_wave.csv ← wave parameters
└── plots/
    └── *.png / *.pkl
```

## Plot options

| Key | Description |
|---|---|
| `sea_state` | Wave spectrum and elevation |
| `power` | Absorbed power time series |
| `dynamics` | Displacement and velocity |
| `hydro` | Hydrodynamic force breakdown |
| `3d` | 3D scatter over sea state space |
| `variable` | Any named column — one plot per variable |

## Notes

- `DataHandle` satisfies the same `.dataset` / `.label` interface as `SimRun`, so both can be passed to the same `Plotter` calls.
- The `fe_lin` column is the linear excitation force derived from BEM, used as ground truth input by `linear_si`.
- Saved `.pkl` figures can be reopened interactively with option `[3]` at the rerun menu.
