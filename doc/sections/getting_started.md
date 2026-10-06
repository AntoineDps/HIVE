# Getting Started

## Requirements

- Python 3.10+
- Conda (recommended)
- A HydroSphere BEM pickle file (`.pkl`) placed under `models/bem/`

## Installation

Clone the repository and create the conda environment:

```bash
git clone https://github.com/AntoineDps/HIVE.git
cd HIVE/code
conda env create -f environment.yml
conda activate surrogate
```

!!! note
    If no `environment.yml` is provided yet, install manually:
    ```bash
    conda create -n surrogate python=3.10
    conda activate surrogate
    pip install numpy scipy pandas matplotlib control vectfit nfoursid cvxpy mkdocs-material
    ```

---

## Quick run

All scripts are run as **modules** from inside the `code/` folder:

```bash
conda activate surrogate
cd HIVE/code

# 1 — process CFD data
python -m script.run_handle_data --case my_cases.json

# 2 — validate models
python -m script.run_validate --case my_validate.json

# 3 — identify parameters
python -m script.run_identify --case my_identify.json
```

Each `--case` argument points to a JSON file inside `inputs/<script>/`.

---

## Test case

A minimal test case is provided under `inputs/` to verify the installation.
Run it with:

```bash
python -m script.run_handle_data --case example.json
python -m script.run_validate --case example.json
python -m script.run_identify --case example_viscous.json
```

Expected outputs appear in `data/handled/example/` and `models/example_viscous/`.

---

## Folder layout before running

```
code/
├── inputs/
│   ├── handle_case/    ← put your handle JSON here
│   ├── validate_case/  ← put your validate JSON here
│   └── identify_case/  ← put your identify JSON here
├── models/
│   └── bem/
│       └── r5.pkl      ← BEM HydroSphere file
└── source/
```

See [Naming Conventions](manual/naming_conventions.md) for label and file naming rules,
and [Input Config Reference](manual/config_reference.md) for all JSON fields.
