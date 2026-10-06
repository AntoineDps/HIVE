# Output Files & Folder Structure

This page documents every folder and file that HIVE produces, organised by script.

---

## Project layout (before running)

```
code/
├── data/
│   └── cfd/                  ← raw CFD input (provided by user)
├── inputs/
│   ├── handle_case/          ← *.json configs for run_handle_data
│   ├── validate_case/        ← *.json configs for run_validate
│   └── identify_case/        ← *.json configs for run_identify
├── models/
│   └── bem/                  ← *.pkl BEM result files (provided by user)
├── out/
│   └── signals/              ← output of run_signals
├── script/                   ← Python scripts (run_*.py)
└── src/                      ← core library modules
```

---

## run_handle_data

Output root: `data/handled/<config_stem>/`

```
data/handled/<name>/
├── <name>.json               ← copy of the config used to produce this dataset
├── run.log                   ← console output with timing and warnings
├── data/
│   ├── <label>.csv           ← full time series per CFD case (see columns below)
│   └── <label>_wave.csv      ← scalar wave parameters (T, H, Hs, Te, …)
└── plots/
    ├── <label>_sea_state.png / .pkl
    ├── <label>_power.png / .pkl
    ├── <label>_dynamics.png / .pkl
    ├── <label>_hydro.png / .pkl
    └── <label>_<variable>.png / .pkl
```

### Time-series CSV columns

| Column | Units | Description |
|---|---|---|
| `t` | s | Time |
| `x` | m | Displacement |
| `xdot` | m/s | Velocity |
| `eta` | m | Wave elevation |
| `fe` | N | Total excitation force (CFD) |
| `fe_lin` | N | Linear excitation force (BEM) |
| `f_rad` | N | Radiation force |
| `f_hs` | N | Hydrostatic force |
| `f_pto` | N | PTO force |
| `P_abs` | W | Absorbed power |

### Wave parameters CSV columns

| Column | Description |
|---|---|
| `T` | Peak/regular wave period [s] |
| `H` | Wave height [m] |
| `Hs` | Significant wave height [m] (irregular) |
| `Te` | Energy period [s] (irregular) |

---

## run_validate

Output root: `models/<model_name>/validation/`

```
models/<model_name>/validation/
├── result.json               ← metric summary for all models and sea states
├── run.log
├── data/
│   └── <label>_simrun.csv    ← simulated time series per case per model
└── plots/
    ├── scatter_<metric>.png / .pkl
    └── timeseries_<label>.png / .pkl
```

### result.json structure

```json
{
  "t_warmup": 15.0,
  "t_causal": 3.2,
  "models": {
    "linear": {
      "T8_H2": { "nrmse_x": 0.12, "mean_P_abs": 430.5 },
      "T10_H3": { ... }
    }
  }
}
```

### SimRun CSV columns

Same column names as the handled data CSVs (see above), so both can be passed to the same plotter.

---

## run_identify

Output root: `models/<config_stem>/`

```
models/<name>/
├── <name>.json               ← copy of the identify config
├── model.json                ← model definition updated with identified parameters
├── run.log
└── identification/
    ├── result.json           ← scheme output (identified parameters + objective value)
    ├── data/
    │   └── *.csv             ← simulated time series at best parameter values
    └── plots/
        ├── grid_*.png / .pkl     ← viscous_drag / pi_gain: objective over grid
        └── timeseries_*.png / .pkl
```

### result.json structure (viscous_drag example)

```json
{
  "c_v": 1850.0,
  "objective": "nrmse_x",
  "value": 0.09,
  "t_warmup": 15.0
}
```

---

## run_signals

Output root: `out/signals/`

```
out/signals/
├── eta_multisine_signal_1.csv
├── eta_multisine_fpto_random_signal_2.csv
└── eta_multisine_fpto_multisine_signal_3.csv
```

### Signal CSV columns

| File | Columns |
|---|---|
| Signal 1 | `t`, `eta` |
| Signal 2 | `t`, `eta`, `fpto` (unfiltered) |
| Signal 3 | `t`, `eta`, `fpto` |

Each file contains exactly one period of the signal.

---

## Pickle figures (`.pkl`)

Every plot is saved in both `.png` (for quick viewing) and `.pkl` (interactive Matplotlib figure). Reload a `.pkl` interactively with option `[3]` at the rerun menu, or directly:

```python
import pickle, matplotlib.pyplot as plt
fig = pickle.load(open("plots/timeseries_T8_H2.pkl", "rb"))
plt.show()
```

---

## What to commit to git

Large output folders should be excluded from version control. A recommended `.gitignore`:

```
data/handled/
out/
models/*/validation/
models/*/identification/
*.pkl
*.log
__pycache__/
*.pyc
```

Keep in version control: `inputs/`, `models/bem/`, `models/*/model.json`, `script/`, `src/`, configs.
