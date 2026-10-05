# Getting Started

## Requirements

- Python 3.10+
- conda environment `surrogate`

```bash
conda activate surrogate
pip install control nfoursid cvxpy pysindy
```

## Path configuration

All paths are defined in `source/config.py`. The file resolves everything relative to the `source/` directory:

```python
_SRC     = Path(__file__).resolve().parent   # → code/source/
DATA_DIR  = _SRC.parent / "data"             # → code/data/
INPUT_DIR = _SRC.parent / "inputs"           # → code/inputs/
OUT_DIR   = _SRC.parent / "out"              # → code/out/
MODEL_DIR = _SRC / "models"                  # → code/source/models/
```

!!! warning "Running scripts"
    Always run scripts as **modules** from the `code/` directory so imports resolve correctly:
    ```bash
    cd code/
    python -m script.run_validate
    ```

## BEM data

The hydrodynamic coefficients live in `source/models/bem/r5.pkl`.  
Load it with:

```python
import pickle
with open(MODEL_DIR / "bem" / "r5.pkl", "rb") as f:
    hs = pickle.load(f)
```

Attributes available on `hs`:

| Attribute | Description | Unit |
|---|---|---|
| `hs.mass` | Displaced mass | kg |
| `hs.ma_inf` | Infinite-frequency added mass | kg |
| `hs.stiffness` | Hydrostatic restoring coefficient $K_{hs}$ | N/m |
| `hs.r` | Sphere radius | m |
| `hs.w` | BEM frequency grid | rad/s |
| `hs.Fe_mod` | Excitation force magnitude $\|F_e(\omega)\|$ | N/m |
| `hs.Fe_ang` | Excitation force phase $\angle F_e(\omega)$ | rad |
| `hs.rad_ss` | Radiation state-space `{Ar, Br, Cr, Dr}` | — |
| `hs.Kp` | Array of proportional gains | — |
| `hs.Ki` | Array of integral gains | — |

## First run checklist

1. Raw CFD data in `data/cfd/{case_folder}/`
2. `inputs/handle_case/cases.json` lists the cases to process
3. Run `run_handle_data.py` → creates `data/handled/{case}/data/*.csv`
4. Edit `inputs/validate_case/run.json` with desired models and wave sources
5. Run `run_validate.py` → outputs in `out/validated_{name}/`
