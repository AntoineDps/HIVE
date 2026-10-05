# Metrics

All metrics are computed by `Metric.compute_all()` after clipping simulations to the valid window `[t_warmup, T - t_causal]`.

## Scalar metrics

Computed per model per sea state, saved to `metric/{label}.csv`.

### NRMSE (normalised by range)

$$\text{NRMSE}_{range} = \frac{\sqrt{\frac{1}{N}\sum(y_{sim} - y_{ref})^2}}{\max(y_{ref}) - \min(y_{ref})}$$

Keys: `nrmse_range_x`, `nrmse_range_xdot`, `nrmse_range_P_abs`

### NRMSE (normalised by std)

$$\text{NRMSE}_{std} = \frac{\sqrt{\frac{1}{N}\sum(y_{sim} - y_{ref})^2}}{\sigma(y_{ref})}$$

Keys: `nrmse_std_x`, `nrmse_std_xdot`, `nrmse_std_P_abs`

### RMSE

$$\text{RMSE} = \sqrt{\frac{1}{N}\sum(y_{sim} - y_{ref})^2}$$

Keys: `rmse_x`, `rmse_xdot`, `rmse_P_abs`

### Time ratio

$$\text{time\_ratio} = \frac{t_{wall}}{T_{signal}}$$

Ratio of simulation wall-clock time to signal duration. Values < 1 mean real-time capable.

Key: `time_ratio`

---

## Cross-correlation metrics

Saved to `metric/cross_corr_{label}.csv`.

The cross-correlation between simulated and reference signals is computed at lags from $-T/2$ to $+T/2$:

$$R_{xy}(\tau) = \frac{1}{N}\sum_{t} x_{sim}(t+\tau)\,y_{ref}(t)$$

| Column | Description |
|---|---|
| `lag_s` | Lag vector [s] |
| `ref_x` | Autocorrelation of reference x |
| `ref_xdot` | Autocorrelation of reference xdot |
| `{model}_x` | Cross-correlation for model x |
| `{model}_xdot` | Cross-correlation for model xdot |

Derived scalar metrics:

| Key | Description |
|---|---|
| `peak_lag_s_x` | Lag at correlation peak for x [s] |
| `peak_lag_s_xdot` | Lag at correlation peak for xdot [s] |
| `corr_peak_x` | Peak correlation value for x |
| `corr_peak_xdot` | Peak correlation value for xdot |

---

## Instantaneous phase metrics

Saved to `metric/inst_phase_{label}.csv`.

The instantaneous phase of each signal is extracted via the Hilbert transform. The phase difference reveals systematic phase offsets:

$$\Delta\phi(t) = \angle[H(x_{sim}(t))] - \angle[H(x_{ref}(t))]$$

| Column | Description |
|---|---|
| `t` | Time [s] |
| `{model}_x_deg` | Phase difference for x [°] |
| `{model}_xdot_deg` | Phase difference for xdot [°] |

Derived scalar:

| Key | Description |
|---|---|
| `phase_mean_x_deg` | Mean phase difference x [°] |
| `phase_mean_xdot_deg` | Mean phase difference xdot [°] |

---

## Aggregate statistics

`metric/model_stats.csv` — aggregated across all sea states per model:

| Column | Description |
|---|---|
| `model` | Model label |
| `metric` | Metric name |
| `mean` | Mean across sea states |
| `std` | Standard deviation |
| `min` / `max` | Range |

---

## `metric_stat` whisker plot

The `metric_stat` plot visualises model_stats.csv:
- **Line** at mean
- **Shaded band** = ±std
- **Whiskers** = min / max

One subplot per metric, one whisker per model.

---

## `metric_grid` scatter plot

Plots metric value on the (Te, Hs) grid as a coloured scatter. One subplot per model when multiple models are compared. Useful for identifying which sea states each model struggles with.
