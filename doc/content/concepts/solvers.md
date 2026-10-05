# Solvers & Interpolation

## ODE solvers

The simulation engine integrates the equation of motion:

$$(m + m_\infty)\ddot{x} = f_e(t) + f_r(t) + f_{pto}(t) + f_{drag}(t) - K_{hs}\,x(t)$$

Two integrators are available via the `method` field:

### `"RK4"` — Runge-Kutta 4th order

The default and recommended integrator. 4th-order accurate, fixed time step.

$$x_{n+1} = x_n + \frac{\Delta t}{6}(k_1 + 2k_2 + 2k_3 + k_4)$$

Good balance of accuracy and speed. Use `dt = 0.05 s` as a starting point.

### `"Euler"` — Forward Euler

1st-order, fastest. Only suitable for very small time steps or quick tests.

$$x_{n+1} = x_n + \Delta t \cdot f(x_n, t_n)$$

!!! warning
    Euler can produce unstable results for stiff systems (e.g. high radiation damping). Use RK4 for production runs.

---

## Time step `dt`

| `dt` [s] | Notes |
|---|---|
| `0.1` | Fast, may miss peaks for high-frequency response |
| `0.05` | Recommended — good accuracy up to ~3 rad/s |
| `0.02` | Higher accuracy, 2.5× slower |
| `0.01` | Needed for `fk_nonlinear` with steep waves |

The Nyquist frequency is $\pi / dt$ rad/s:

- `dt=0.05` → up to 62.8 rad/s (more than sufficient for ocean waves)

---

## Interpolation

Wave elevation $\eta(t)$ and excitation force $f_e(t)$ are tabulated at discrete time points from the CFD data. During simulation they must be evaluated at arbitrary ODE sub-steps.

The `interpolation` field controls the method per signal:

```json
"interpolation": {
  "eta":   "linear",
  "fe":    "cubic",
  "waves": "linear"
}
```

| Signal key | What it controls |
|---|---|
| `eta` | Wave elevation interpolation |
| `fe` | Excitation force IRF interpolation |
| `waves` | Wave component interpolation (for nonlinear FK) |

### Methods

| Value | Description |
|---|---|
| `"linear"` | Linear interpolation. Fast, C0 continuous |
| `"cubic"` | Cubic spline. Smoother derivatives, better for `RK4` |
| `""` or omitted | Falls back to linear |

!!! tip
    For `excitation: "fk_nonlinear"`, use `"cubic"` for `waves` to avoid step discontinuities in the wave profile that can make the ODE stiff.

---

## Radiation state-space

When `radiation: "state_space"`, the radiation convolution:

$$f_r(t) = \int_0^\infty k_r(\tau)\,\dot{x}(t-\tau)\,d\tau$$

is replaced by a state-space approximation stored in the BEM pickle:

$$\dot{q}_r = A_r q_r + B_r \dot{x}, \qquad f_r = C_r q_r + D_r \dot{x}$$

The state vector $q_r$ is initialised to zero at $t=0$ and integrated alongside $x$ and $\dot{x}$. This requires a warmup period of `t_warmup = 30 s` for the radiation state to reach steady-state.
