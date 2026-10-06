# Metrics

HIVE computes performance metrics by comparing simulated time series to CFD reference data over the clipped window $[t_{warmup},\; T_{end} - t_{causal}]$.

---

## Available metrics

| Key | Name | Units | Definition |
|---|---|---|---|
| `nrmse_x` | NRMSE displacement | — | $\text{NRMSE}(x)$ (see below) |
| `nrmse_xdot` | NRMSE velocity | — | $\text{NRMSE}(\dot{x})$ |
| `rmse_x` | RMSE displacement | m | $\text{RMSE}(x)$ |
| `rmse_xdot` | RMSE velocity | m/s | $\text{RMSE}(\dot{x})$ |
| `mean_P_abs` | Mean absorbed power | W | $\langle P_{abs} \rangle$ |

Pass these keys in the `metrics` list of any validate or identify config.

---

## Definitions

### Root Mean Square Error (RMSE)

$$
\text{RMSE}(y) = \sqrt{\frac{1}{N}\sum_{i=1}^{N}\bigl(y_{sim,i} - y_{ref,i}\bigr)^2}
$$

### Normalised RMSE (NRMSE)

RMSE normalised by the standard deviation of the reference signal, so the metric is dimensionless and comparable across sea states of different amplitude:

$$
\text{NRMSE}(y) = \frac{\text{RMSE}(y)}{\sigma_{ref}(y)}
$$

A value of 0 is a perfect match; a value of 1 means the model error is as large as the signal variability.

### Mean absorbed power

$$
\langle P_{abs} \rangle = \frac{1}{N}\sum_{i=1}^{N} P_{abs,i}
$$

where $P_{abs,i} = -f_{pto,i}\,\dot{x}_i$. For the reference CFD signal, this uses the measured PTO force and velocity; for a simulation it uses the simulated values.

---

## Time clipping

All metrics are computed over the window after startup transients and before the non-causal excitation horizon:

$$
t \in [t_{warmup},\; T_{end} - t_{causal}]
$$

HIVE computes both margins automatically from the model and logs them. Metrics computed outside this window would be contaminated by the initial radiation transient or by the edge artefacts of the excitation convolution.

---

## Objective function for identification

In `run_identify`, one metric acts as the objective function minimised (or maximised) during parameter search. The default is `nrmse_x`. Any metric in the table above can be used by setting `objective` in `scheme_params`:

```json
"scheme_params": {
  "c_v_range": [0, 5000],
  "n_points": 50,
  "objective": "nrmse_xdot"
}
```

For `mean_P_abs`, the objective is **maximised** (best power extraction); for all RMSE/NRMSE metrics it is **minimised**.
