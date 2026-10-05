# model.json

Every model is defined by a `model.json` file in `source/models/{name}/`.

## Minimal example

```json
{
  "pkl_name": "r5",
  "force_terms": {
    "radiation": "state_space",
    "excitation": "linear_conv",
    "drag": "none"
  }
}
```

## Fields

| Field | Type | Required | Description |
|---|---|---|---|
| `pkl_name` | string | ✓ | BEM pickle name under `source/models/bem/` (without `.pkl`) |
| `force_terms` | object | ✓ | Force term configuration (see below) |
| `identified` | object | — | Set automatically by `run_identify.py` |

---

## `force_terms`

### `radiation`

Controls how the radiation force (wave-making damping) is computed.

| Value | Description |
|---|---|
| `"state_space"` | Radiation state-space from BEM (`Ar, Br, Cr, Dr`). Most accurate. Adds ~30 s warmup |
| `"none"` | No radiation force. Fast but inaccurate for transient response |

!!! note
    `"state_space"` is the recommended option. It uses the radiation SS matrices stored in the BEM pickle and adds `t_warmup = 30 s` to the simulation.

---

### `excitation`

Controls how the wave excitation force is computed.

| Value | Description |
|---|---|
| `"linear_conv"` | Linear excitation via IRF convolution: $f_e = h_{ex} * \eta$. Requires BEM IRF. Adds `t_warmup` and `t_causal` margins (~30 s each) |
| `"fk_nonlinear"` | Froude-Krylov nonlinear excitation. Uses instantaneous waterplane area. Slower |
| `"none"` | No excitation force |

---

### `drag`

| Value | Description |
|---|---|
| `"none"` | No drag force |
| `"viscous"` | Quadratic viscous drag: $f_{drag} = -c_v |\dot{x}|\dot{x}$. Requires `identified.c_v` |

---

## Full example with all options

```json
{
  "pkl_name": "r5",
  "force_terms": {
    "radiation": "state_space",
    "excitation": "fk_nonlinear",
    "drag": "viscous"
  },
  "identified": {
    "c_v": 850.0,
    "method": "response_opt",
    "metric": "nrmse_range_x",
    "metric_value": 0.031,
    "identified_from": "out/identified_weakly_nl"
  }
}
```

## Time margins

Different force term combinations produce different time margins that are automatically computed and applied to clip simulation output before metric computation:

| Radiation | Excitation | t_warmup | t_causal |
|---|---|---|---|
| `state_space` | `linear_conv` | 30 s | 30 s |
| `state_space` | `fk_nonlinear` | 30 s | 0 s |
| `none` | `linear_conv` | 30 s | 30 s |
| `none` | `none` | 0 s | 0 s |
