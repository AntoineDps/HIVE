# run_handle_data

Loads raw CFD output, computes derived quantities, and saves structured CSVs that all downstream scripts consume.

## Run

```bash
python -m script.run_handle_data
```

## Input

`inputs/handle_case/cases.json`

```json
{
  "cases": [
    {
      "type": "cfd_case",
      "case_name": "Hs2p0_Te5p0_D120000_K-280000"
    }
  ]
}
```

| Field | Type | Description |
|---|---|---|
| `type` | string | Source type. Currently `"cfd_case"` |
| `case_name` | string | Exact name of the CFD folder under `data/cfd/` |

## Output

For each case, writes `data/handled/{label}/data/{label}_data.csv`.

### CSV columns

| Column | Unit | Description |
|---|---|---|
| `t` | s | Time |
| `x` | m | Heave displacement |
| `xdot` | m/s | Heave velocity |
| `eta` | m | Wave elevation at WEC location |
| `fe_lin` | N | Linear excitation force (BEM convolution) |
| `fhyd_lin` | N | Linear hydrodynamic force |
| `fhyd_cfd` | N | Total CFD hydrodynamic force |
| `f_pto` | N | PTO force |
| `P_abs` | W | Absorbed power |

## Options menu

| Option | Effect |
|---|---|
| `[1]` | Erase existing and reprocess all cases |
| `[2]` | Skip already-processed cases |
| `[3]` | Append new cases only |
| `[4]` | Remake plots only (no reprocessing) |

## Plot output

`out/handled_{case}/plots/`

| File | Content |
|---|---|
| `variable_{label}.png` | Time series of all variables |
| `eta_spectrum_{label}.png` | Wave elevation spectrum |
