# run_identify

Identifies model parameters or data-driven surrogates from CFD reference data using
a pluggable scheme system.

## Usage

```bash
python -m script.run_identify --case <filename>.json
```

Config files live in `inputs/identify_case/`. The filename stem becomes the output
folder name under `models/`.

## Available schemes

| Scheme key | Module | What it identifies |
|---|---|---|
| `viscous_drag` | `Viscous_drag.py` | Quadratic drag coefficient $c_v$ via least-squares |
| `pi_gain` | `Pi_gain.py` | PI controller gains $K_p$, $K_i$ |
| `linear_si` | `Linear_si.py` | $\eta \to f_e$ and $f_e \to \dot{x}$ state-space models via VF or N4SID |

## What it does

1. Loads the base model definition from `models/<base_model>/model.json`
2. Creates a `DDFeed` — loads training `DataHandle` objects and runs `scheme.prep()`
3. Creates an `Optimizer` — owns simulation helpers and delegates to `scheme.run()`
4. Saves identified parameters, result JSON, simulation CSVs and plots

## Config fields

See [Input Config Reference — identify](config_reference.md#identify-config).

## Rerun options

| Option | Action |
|---|---|
| `[1]` | Erase and redo |
| `[2]` | Stop |
| `[3]` | Reload saved figures |
| `[4]` | Save results with today's date suffix (keeps existing folder) |

## Outputs

```
models/<name>/
├── <name>.json               ← copy of config
├── model.json                ← updated model with identified parameters
├── run.log
└── identification/
    ├── result.json
    ├── data/
    │   └── *.csv
    └── plots/
        └── *.png / *.pkl
```

## Scheme contract

Every scheme module must implement:

```python
SELF_LOADING = False   # True if scheme loads its own data (e.g. linear_si)
PLOT_DISPATCH = {...}  # {name: fn(data, simruns, result, grids, plots_dir)}

def prep(feed: DDFeed): ...
def run(opt: Optimizer, feed: DDFeed, config) -> tuple: ...
def save(opt: Optimizer, result: dict, output_name: str): ...
def log_result(result: dict): ...
def save_data(result, simruns, grids, feed, data_dir): ...
```

## Notes

- The base model must exist at `models/<base_model>/model.json` before running.
- `t_warmup` and `t_causal` are computed from the base model and applied to all
  simulation clips automatically.
- For `linear_si`, the scheme loads its own signal data (`SELF_LOADING = True`) and
  does not use CFD `DataHandle` objects.
