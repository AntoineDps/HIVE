# Hydrodynamics

HIVE models a single-body wave energy converter (WEC) moving in heave under wave excitation and a power take-off (PTO) force. This page covers the underlying hydrodynamic theory.

---

## Equation of motion

The heave dynamics of the WEC buoy are governed by Newton's second law:

$$
(m + m_\infty)\ddot{x} = f_e(t) + f_{rad}(t) + f_{hs}(t) + f_{pto}(t)
$$

where $m$ is the body mass, $m_\infty$ is the added mass at infinite frequency, and the right-hand side forces are described below.

---

## Boundary element method (BEM)

Potential flow hydrodynamics are solved offline using a BEM solver (e.g. NEMOH or CAPYTAINE). The BEM result provides:

- **Added mass** $A(\omega)$ and **radiation damping** $B(\omega)$ as functions of frequency
- **Excitation force** transfer function $\hat{f}_e(\omega) / \hat{\eta}(\omega)$
- **Impulse response functions** (IRFs) obtained by inverse Fourier transform

Results are stored as a pickle file in `models/bem/` and loaded via `hydro_sphere_pkl` in the handle config.

---

## Cummins equation

The Cummins (1962) equation rewrites the frequency-domain radiation force as a convolution in time:

$$
(m + A(\infty))\ddot{x}(t) = f_e(t) - \int_0^t K_{rad}(\tau)\,\dot{x}(t-\tau)\,d\tau - K_{hs}\,x(t) + f_{pto}(t)
$$

where $K_{rad}(\tau)$ is the radiation IRF and $K_{hs} = \rho g A_{wp}$ is the hydrostatic stiffness.

The convolution integral captures the memory effect of radiated waves — past velocities continue to exert a force on the body.

---

## Radiation force approximation

For time-domain simulation the convolution integral is replaced by a state-space system approximating $K_{rad}$:

$$
\dot{\mathbf{z}}_{rad} = A_{rad}\,\mathbf{z}_{rad} + B_{rad}\,\dot{x}, \qquad
f_{rad} = C_{rad}\,\mathbf{z}_{rad}
$$

The matrices $A_{rad}, B_{rad}, C_{rad}$ are identified from the BEM IRF using Vector Fitting (see [System Identification](identification.md)).

---

## Linear excitation force

Under linear wave theory the excitation force is a convolution of the wave elevation with the excitation IRF $K_e(\tau)$:

$$
f_e^{lin}(t) = \int_{-\infty}^{\infty} K_e(\tau)\,\eta(t - \tau)\,d\tau
$$

This is **non-causal**: the force depends on future wave elevation because the diffraction pressure field reaches the body before the wave crest does. The lead time sets the `t_causal` margin used in HIVE to clip simulation results.

In the CFD data, the total hydrodynamic force is measured directly; `fe` is obtained by subtracting the radiation and hydrostatic contributions. The linear prediction `fe_lin` is computed from BEM and stored alongside the measured signal for comparison.

---

## Wave models

### Regular waves

$$
\eta(t) = \frac{H}{2}\cos(\omega_0 t + \phi_0), \qquad \omega_0 = \frac{2\pi}{T}
$$

### Irregular waves

Irregular seas are modelled as a superposition of sinusoids drawn from a target spectrum $S(\omega)$ (JONSWAP or Pierson–Moskowitz):

$$
\eta(t) = \sum_{k=1}^{N} a_k \cos(\omega_k t + \phi_k), \qquad a_k = \sqrt{2 S(\omega_k)\,\Delta\omega}
$$

The significant wave height $H_s$ and energy period $T_e$ characterise the sea state.

---

## Non-dimensionalisation

HIVE works in SI units throughout. All forces are in Newtons, lengths in metres, and time in seconds.
