# Models & Solvers

---

## Model types

A model in HIVE is a folder under `models/` containing a `model.json` file. The JSON records the model's physics components, identified parameters, and metadata. Currently supported model types:

| Model key | Description |
|---|---|
| `linear` | Cummins equation with radiation state-space, linear PTO |
| `linear_viscous` | As `linear`, plus quadratic drag $c_v|\dot{x}|\dot{x}$ |

Models identified by `run_identify` inherit a base model's `model.json` and add or update the identified parameters.

---

## model.json structure

```json
{
  "type": "linear_viscous",
  "hydro_sphere_pkl": "r5",
  "radiation": {
    "A": [[...], ...],
    "B": [[...]],
    "C": [[...]],
    "D": [[...]]
  },
  "hydrostatic": {
    "K_hs": 142000.0
  },
  "added_mass_inf": 450.0,
  "mass": 1200.0,
  "c_v": 1850.0,
  "t_warmup": 15.0,
  "t_causal": 3.2
}
```

The `radiation` block holds the state-space matrices $(A_{rad}, B_{rad}, C_{rad}, D_{rad})$ fitted from the BEM IRF. These are written by `run_handle_data` during the first call that initialises the BEM data.

---

## SimRun

`SimRun` is the class responsible for running a single forward simulation. It:

1. Loads the model definition from `model.json`
2. Instantiates the radiation state integrator
3. Steps the ODE forward using the chosen solver
4. Returns a `dataset` attribute (a `pandas.DataFrame`) with the same column schema as `DataHandle`

Because `SimRun` and `DataHandle` expose the same `.dataset` and `.label` interface, they can be mixed in plotter calls.

---

## ODE solvers

The `solver` block in validate and identify configs selects the numerical integrator:

```json
"solver": {
  "method": "RK4",
  "dt": 0.05,
  "interpolation": {}
}
```

| Method | Description |
|---|---|
| `RK4` | 4th-order Runge–Kutta (recommended) |
| `RK2` | 2nd-order Runge–Kutta (Heun's method) |
| `euler` | Forward Euler (1st order, fast but inaccurate) |

The integration timestep `dt` is independent of the CFD data's native timestep. HIVE interpolates the forcing signals (wave elevation, excitation force) onto the solver grid using the scheme specified in `interpolation`.

---

## Warmup period

The radiation state-space system starts at rest ($\mathbf{z}_{rad} = 0$). It takes several wave periods for the radiation states to converge to their quasi-steady values. HIVE computes `t_warmup` automatically from the pole magnitudes of $A_{rad}$:

$$
t_{warmup} = \frac{-\log(\epsilon)}{\min_k |\text{Re}(p_k)|}
$$

where $\epsilon$ is a small tolerance (default $10^{-3}$) and $p_k$ are the eigenvalues of $A_{rad}$. The value is logged in `run.log` and saved in `result.json`.

Metrics are computed only over $[t_{warmup},\; T_{end} - t_{causal}]$.

---

## Signal interpolation

When the solver timestep `dt` differs from the CFD data timestep, forcing signals must be interpolated. The `interpolation` dict in the solver config is passed to `scipy.interpolate.interp1d`. Default (empty dict) uses linear interpolation. Example to use cubic splines:

```json
"interpolation": {"kind": "cubic"}
```
