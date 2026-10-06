# HIVE

**Hydrodynamic Identification and Validation Environment**

HIVE is a Python framework for wave energy converter (WEC) hydrodynamic modelling,
system identification, and model benchmarking. It is developed as part of a PhD
project at Uppsala University.

---

## What HIVE does

HIVE provides a structured pipeline to:

- **Process** CFD simulation data into a common format
- **Validate** physics-based models (linear, viscous drag, nonlinear FK) against CFD reference data
- **Identify** model parameters and data-driven surrogates from CFD or signal data
- **Benchmark** multiple models against each other across a range of sea states

The framework is built around three main scripts — `run_handle_data`, `run_validate`,
and `run_identify` — each driven by a JSON config file, so no code changes are needed
to run different cases.

---

## Key features

- BEM-based linear hydrodynamic model (radiation, excitation, hydrostatics)
- Viscous drag identification via least-squares regression
- PI gain identification for PTO control
- Frequency-domain system identification: Vector Fitting with causality sweep, N4SID
- Multi-sine signal generation for broadband excitation
- NRMSE, RMSE, and power-based metrics across sea state grids
- MkDocs documentation served at this site

---

## Project structure

```
HIVE/
├── code/                  ← all source code (this repo)
│   ├── source/            ← classes, functions, config
│   │   ├── classes/       ← Model, SimRun, DataHandle, Plotter, …
│   │   └── function/      ← sea state utils, error utils, …
│   ├── script/            ← run_*.py entry points
│   ├── inputs/            ← JSON config files
│   └── doc/               ← this documentation
├── data/                  ← CFD outputs and processed data (not tracked)
└── models/                ← identified model outputs (not tracked)
```

---

## Citing

If you use HIVE in your research, please cite: *(reference to be added)*
