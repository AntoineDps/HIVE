# Pipeline Overview

## Data flow

```
CFD folders                    inputs/handle_case/cases.json
     │                                    │
     └──────────── run_handle_data.py ────┘
                          │
                  data/handled/{case}/
                    data/{label}_data.csv
                          │
          inputs/validate_case/run.json
                    │             │
                    │    source/models/{name}/model.json
                    │             │
                    └── run_validate.py ──┘
                              │
                    out/validated_{name}/
                      run.log
                      data/{label}_*.csv
                      plots/*.png
                      metric/
                        {label}.csv
                        model_stats.csv
                          │
          inputs/identify_case/{scheme}.json
                    │
                    └── run_identify.py
                              │
                    source/models/{output}/
                      model.json
                      result.json
                    out/identified_{name}/
                      plots/*.png
```

## Case naming convention

Every sea state has a unique label derived from its parameters:

```
Hs2p0_Te5p0_d120000_k-280000
│         │         │        │
│         │         │        └── PTO stiffness [N/m], negative allowed
│         │         └── PTO damping [N·s/m]
│         └── Energy period Te = 5.0 s  (decimal → 'p')
└── Significant wave height Hs = 2.0 m
```

This label is used consistently across all output files, plots, and CSVs.

## Options menu

`run_handle_data.py` and `run_validate.py` present an interactive menu when run:

```
[1] run  (overwrite existing)
[2] run  (keep existing, skip)
[3] run  (append)
[4] remake plots only  (reuse existing CSVs)
```

Option `4` reloads previously computed metrics from CSV and regenerates all plots — no recomputation.

## t_warmup and t_causal

Every simulation discards a warmup period and a causal tail before computing metrics:

- **t_warmup** — time for transient to decay. Set by the model's force terms:
    - `radiation: state_space` → 30 s
    - `excitation: linear_conv` or `fk_nonlinear` → IRF length (typically 30 s)
- **t_causal** — acausal tail of the excitation IRF (typically 30 s for `linear_conv`)

These margins are computed automatically by `Model.compute_time_margins()` and applied to all SimRun datasets before metric computation and CSV saving.
