# Hydrodynamics

## HydroSphere

The `HydroSphere` class loads and stores all BEM (Boundary Element Method) hydrodynamic data for the sphere. It is loaded once from `source/models/bem/r5.pkl` and passed to all model builders.

```python
import pickle
with open(MODEL_DIR / "bem" / "r5.pkl", "rb") as f:
    hs = pickle.load(f)
```

### Attributes

| Attribute | Unit | Description |
|---|---|---|
| `hs.r` | m | Sphere radius |
| `hs.mass` | kg | Displaced mass |
| `hs.ma_inf` | kg | Infinite-frequency added mass $m_\infty$ |
| `hs.stiffness` | N/m | Hydrostatic restoring coefficient $K_{hs} = \rho g A_{wp}$ |
| `hs.w` | rad/s | BEM frequency grid |
| `hs.Fe_mod` | N/m | Excitation force magnitude $|F_e(\omega)|$ |
| `hs.Fe_ang` | rad | Excitation force phase $\angle F_e(\omega)$ |
| `hs.rad_ss` | dict | Radiation state-space `{Ar, Br, Cr, Dr}` |
| `hs.Kp` | array | Proportional gain grid |
| `hs.Ki` | array | Integral gain grid |

### Key relationships

Excitation force spectrum from wave elevation:

$$\hat{F}_e(\omega) = F_e(\omega) \cdot e^{j\phi_e(\omega)} \cdot \hat{\eta}(\omega)$$

Natural frequency:

$$\omega_n = \sqrt{\frac{K_{hs}}{m + m_\infty}} \quad [\text{rad/s}]$$

Radiation force (state-space):

$$\dot{q}_r = A_r q_r + B_r \dot{x}, \qquad f_r = C_r q_r + D_r \dot{x}$$

---

## Excitation impulse response function (IRF)

The excitation IRF $h_e(t)$ is the inverse Fourier transform of $F_e(\omega) e^{j\phi_e(\omega)}$:

$$h_e(t) = \frac{1}{2\pi}\int_{-\infty}^{\infty} F_e(\omega) e^{j\phi_e(\omega)} e^{j\omega t} d\omega$$

It spans approximately $t \in [-30\,\text{s},\, +30\,\text{s}]$:
- For $t < 0$: **non-causal** — the force at time $t$ depends on future wave elevation
- For $t > 0$: **causal** part

The non-causal tail requires that the simulation "sees" wave data 30 s into the future when using `excitation: "linear_conv"`. The pipeline handles this by discarding the first 30 s (`t_warmup`) and last 30 s (`t_causal`) of each simulation.

---

## Radiation kernel

The radiation kernel $k_r(t)$ represents the memory effect of the fluid: a past velocity excites waves that exert a force at the present time.

$$f_r(t) = -\int_0^\infty k_r(\tau)\,\dot{x}(t-\tau)\,d\tau$$

This convolution is approximated by the radiation state-space, which uses a finite-dimensional system fitted to the BEM radiation damping curve $B_r(\omega)$:

$$B_r(\omega) = \omega \int_0^\infty k_r(t) \sin(\omega t)\,dt$$
