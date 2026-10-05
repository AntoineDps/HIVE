# WEC Modeling Benchmark

A Python pipeline for running, identifying, and validating hydrodynamic models of a wave energy converter (WEC) sphere against CFD reference data.

## What it does

```
CFD data  →  handle_data  →  validated/  →  identify  →  validated with identified model
                ↓
          handled/{case}/
            data/*.csv
```

The pipeline has three main scripts, run in order:

| Step | Script | Purpose |
|------|--------|---------|
| 1 | `run_handle_data.py` | Load CFD data, compute derived quantities, save as structured CSVs |
| 2 | `run_validate.py` | Run physics-based models, compare against CFD reference, compute metrics |
| 3 | `run_identify.py` | Identify model parameters (viscous drag, PI gains) from CFD data |

## Project structure

```
code/
├── script/
│   ├── run_handle_data.py
│   ├── run_validate.py
│   └── run_identify.py
├── source/
│   ├── config.py               # paths: DATA_DIR, MODEL_DIR, OUT_DIR, INPUT_DIR
│   ├── classes/
│   │   ├── DataHandle.py       # CFD case container
│   │   ├── HydroSphere.py      # BEM data (added mass, radiation, excitation)
│   │   ├── Model.py            # physics model builder
│   │   ├── SimRun.py           # ODE simulation runner & results
│   │   ├── Metric.py           # error metrics and plots
│   │   ├── Plotter.py          # shared plotting utilities
│   │   ├── Optimizer.py        # identification orchestrator
│   │   ├── Viscous_drag.py     # drag identification scheme
│   │   └── Pi_gain.py          # PI gain identification scheme
│   └── function/
│       └── error_utils.py      # metricError, cross_correlation, …
├── source/models/
│   ├── bem/r5.pkl              # HydroSphere (BEM data)
│   └── {model_name}/model.json # model definitions
├── data/
│   ├── cfd/                    # raw CFD output
│   └── handled/{case}/         # processed data
├── inputs/
│   ├── handle_case/cases.json
│   ├── validate_case/run.json
│   └── identify_case/*.json
└── out/
    └── validated_{name}/       # plots, CSVs, metrics
```

## Quick start

```bash
# 1. Process CFD data
python -m script.run_handle_data

# 2. Validate a model
python -m script.run_validate

# 3. Identify parameters
python -m script.run_identify
```
