# Models & Force Terms

## Model definition

A model is a `model.json` file stored under `source/models/{name}/`. It specifies which physical force contributions to include and how to compute them.

```
source/models/
├── bem/
│   └── r5.pkl                    # BEM data for all models
├── linear_sphere_r5/
│   └── model.json
└── weakly_nonlinear_sphere_r5/
    └── model.json
```

## Equation of motion

$$(m + m_\infty)\ddot{x} = f_e(t) + f_r(t) - K_{hs}\,x + f_{pto}(t) + f_{drag}(t)$$

Each term is independently configurable:

| Term | Symbol | Controlled by |
|---|---|---|
| Excitation | $f_e$ | `force_terms.excitation` |
| Radiation | $f_r$ | `force_terms.radiation` |
| Hydrostatic | $-K_{hs}x$ | Always included (from BEM) |
| PTO | $f_{pto}$ | `pto` in run.json / identify config |
| Drag | $f_{drag}$ | `force_terms.drag` |

---

## Excitation: `"linear_conv"`

$$f_e(t) = \int_{-\infty}^{+\infty} h_e(\tau)\,\eta(t-\tau)\,d\tau$$

The excitation IRF $h_e(t)$ is obtained by inverse Fourier transform of the BEM excitation transfer function $\hat{F}_e(\omega)$. The IRF spans $[-30\,\text{s},\, +30\,\text{s}]$, making the filter **non-causal** — it uses future wave information.

- `t_warmup = t_{IRF}[-1] = 30\,\text{s}` (IRF end)
- `t_causal = |t_{IRF}[0]| = 30\,\text{s}` (IRF start, acausal horizon)

## Excitation: `"fk_nonlinear"`

Computes the Froude-Krylov pressure integral over the instantaneous wetted surface:

$$f_e(t) = \int_{S_{wet}(t)} p_{FK}(\mathbf{x}, t)\,\mathbf{n}\,dS$$

More accurate for steep waves. Requires accurate wave elevation input and is significantly slower.

---

## Radiation: `"state_space"`

The radiation state-space `{Ar, Br, Cr, Dr}` from the BEM pickle approximates the radiation convolution kernel with a finite-dimensional system. Integrated alongside the main ODE.

| Matrix | Shape | Description |
|---|---|---|
| `Ar` | (n, n) | State transition |
| `Br` | (n, 1) | Input: $\dot{x}$ |
| `Cr` | (1, n) | Output: $f_r$ |
| `Dr` | (1, 1) | Direct feedthrough |

---

## Drag: `"viscous"`

$$f_{drag}(t) = -c_v\,|\dot{x}(t)|\,\dot{x}(t)$$

Requires the `identified.c_v` field to be set in `model.json` (set by `run_identify.py` with scheme `"viscous_drag"`).

---

## PTO

The PTO force is NOT part of `model.json` — it is set per simulation run in the calling script:

```json
"pto": { "damping": 120000.0, "stiffness": 0.0 }
```

$$f_{pto}(t) = -B_{pto}\,\dot{x}(t) - K_{pto}\,x(t)$$

This separation lets you test the same model with different PTO settings without creating new model files.

---

## Creating a new model

1. Create `source/models/{name}/model.json` based on an existing one
2. Set `pkl_name` to match your BEM pickle
3. Set `force_terms` for the physics you want
4. If `drag: "viscous"`, run `run_identify.py` with `scheme: "viscous_drag"` to populate `identified.c_v`
5. Add the model to `run.json` under `model_entries`
