import numpy as np
from scipy.interpolate import interp1d

from source.function.fe_conv import fe_conv

"""
# -------------------------------------------------------------------------
# Name:            Model.py
# Description:     1DOF body dynamics, built from composable
#                  force terms, physics based and data-driven
#
# Author:          Antoine
# Collaborator:    Bona
# Date created:    06/2026
# Project:         surrogate_hydro
# -------------------------------------------------------------------------
"""


# %% FORCE TERMS


class ForceTerm:
    """Base contract. name is used as the output column when reporting forces."""

    name = "force"

    def begin_step(self, t0):
        """Called once per outer solver step. Override to cache frozen values."""
        pass

    def compute(self, t, X, interpolate=None):
        raise NotImplementedError


class RestoringLinear(ForceTerm):
    """F = -k*x. Linear hydrostatic stiffness (fb_lin) or PTO spring (fs_pto)."""

    def __init__(self, k, name="fb_lin"):
        self.k = k
        self.name = name

    def compute(self, t, X, interpolate=None):
        return -self.k * X[0]


class LinearDamping(ForceTerm):
    """F = -b*xdot. PTO damping (fd_pto)."""

    def __init__(self, b, name="fd_pto"):
        self.b = b
        self.name = name

    def compute(self, t, X, interpolate=None):
        return -self.b * X[1]


class ViscousDragForce(ForceTerm):
    """
    Morison-type viscous drag: F = -c_v * |xdot| * xdot.
    Identified from the residual between CFD and BEM forces via least squares.
    """

    def __init__(self, c_v, name="f_visc"):
        self.c_v = c_v
        self.name = name

    def compute(self, t, X, interpolate=None):
        return -self.c_v * abs(X[1]) * X[1]


class RestoringNonlinear(ForceTerm):
    """
    Nonlinear buoyancy net of gravity: Vs(h)*rho*g - m*g.
    Pass eta (a callable eta(t)) for the fb_nl_eta flavor (free-surface
    elevation considered); leave eta=None for the fb_nl flavor.
    """

    def __init__(self, r, rho, g, m, eta=None, interpolate=True, name="fb_nl_eta"):
        self.r, self.rho, self.g, self.m = r, rho, g, m
        self.eta = eta
        self.interpolate = interpolate
        self.name = name
        self._frozen_eta = 0.0

    def begin_step(self, t0):
        if self.eta is not None and not self.interpolate:
            self._frozen_eta = float(self.eta(t0))

    def _eta_at(self, t, interpolate):
        if self.eta is None:
            return 0.0
        use_interp = self.interpolate if interpolate is None else interpolate
        return float(self.eta(t)) if use_interp else self._frozen_eta

    def compute(self, t, X, interpolate=None):
        e = self._eta_at(t, interpolate)
        h = np.clip(self.r + e - X[0], 0, 2 * self.r)
        Vs = np.pi * h**2 * (3 * self.r - h) / 3
        return Vs * self.rho * self.g - self.m * self.g


class RadiationStateSpace(ForceTerm):
    """
    Remainder-form radiation: F = -(Cr@z + Dr*xdot).
    Added mass is folded into (m + ma_inf) on the left-hand side.
    Pure state function -- no exogenous time-series, no begin_step needed.
    """

    def __init__(self, Ar, Br, Cr, Dr, name="fr"):
        self.Ar, self.Br, self.Cr, self.Dr = Ar, Br, Cr, Dr
        self.name = name

    def compute(self, t, X, interpolate=None):
        return -(self.Cr @ X[2:] + self.Dr * X[1])

    def state_derivative(self, t, X):
        return self.Ar @ X[2:] + self.Br * X[1]


class ExcitationLinearConv(ForceTerm):
    """
    Linear excitation force fe = IRF * eta (convolution), precomputed over
    the full wave signal, wrapped in an interpolator.

    NOTE: the convolution is causal -- fe[0:n_irf] is inaccurate because
    the simulation has no wave history before t=0. Apply a t_warmup cutoff
    when computing metrics to exclude this startup transient.
    """

    def __init__(self, fe_series, t_series, interpolate=True, name="fe_lin"):
        self._interp = interp1d(
            t_series,
            fe_series,
            kind="linear",
            bounds_error=False,
            fill_value=(fe_series[0], fe_series[-1]),
        )
        self.interpolate = interpolate
        self.name = name
        self._frozen = 0.0

    def begin_step(self, t0):
        if not self.interpolate:
            self._frozen = float(self._interp(t0))

    def compute(self, t, X, interpolate=None):
        use_interp = self.interpolate if interpolate is None else interpolate
        return float(self._interp(t)) if use_interp else self._frozen


class FKNonlinear(ForceTerm):
    """
    Nonlinear Froude-Krylov force: integrate undisturbed incident wave
    dynamic pressure over the instantaneous wetted sphere surface.

        p(z_lab, t) = rho * g * sum_i[ A_i * exp(k_i * z_lab) * cos(w_i*t + phi_i) ]

    z_lab = panel centroid depth in lab frame = centroid_z_body - x (body displacement)
    Panel is wetted when z_lab < eta(t).

    This covers the dynamic pressure part only -- hydrostatic restoring is
    handled separately by RestoringLinear or RestoringNonlinear.

    NOTE: at each solver sub-stage t the wave phase is evaluated at the
    exact sub-stage time, so higher-order solvers (RK4) genuinely benefit
    from their order for this term. Body position X[0] is also updated at
    each sub-stage (state-dependent wetted surface).
    """

    def __init__(
        self,
        panels,
        wave_components,
        eta_t,
        eta_values,
        rho=1025.0,
        interpolate=True,
        name="f_fk",
    ):
        """
        Parameters
        ----------
        panels          : dict from HydroSphere.panels
                          ('centroids', 'normals', 'areas')
        wave_components : list of (A, w, k, phi) -- from decompose_wave()
        eta_t           : time vector of recorded wave elevation
        eta_values      : wave elevation array
        rho             : water density [kg/m^3]
        interpolate     : if False, eta(t0) is frozen per outer step
        """
        self.centroids_z = panels["centroids"][:, 2]  # body-frame z only
        self.normals_z = panels["normals"][:, 2]
        self.areas = panels["areas"]
        self.wave_components = wave_components
        self._eta_interp = interp1d(
            eta_t,
            eta_values,
            kind="linear",
            bounds_error=False,
            fill_value=(float(eta_values[0]), float(eta_values[-1])),
        )
        self.rho = rho
        self.interpolate = interpolate
        self.name = name
        self._frozen_eta = 0.0

    def begin_step(self, t0):
        if not self.interpolate:
            self._frozen_eta = float(self._eta_interp(t0))

    def _eta_at(self, t):
        return float(self._eta_interp(t)) if self.interpolate else self._frozen_eta

    def compute(self, t, X, interpolate=None):
        use_interp = self.interpolate if interpolate is None else interpolate

        x_body = X[0]  # vertical body displacement [m]
        z_lab = self.centroids_z - x_body  # panel depth in lab frame

        eta = float(self._eta_interp(t)) if use_interp else self._frozen_eta
        wetted = z_lab < eta

        if not np.any(wetted):
            return 0.0

        # dynamic pressure at wetted panel centroids
        p = np.zeros(np.sum(wetted))
        for A, w, k, phi in self.wave_components:
            p += self.rho * 9.81 * A * np.exp(k * z_lab[wetted]) * np.cos(w * t + phi)

        # F_FK = -∫ p n_z dS  (pressure acts inward against outward normal)
        return float(-np.sum(p * self.normals_z[wetted] * self.areas[wetted]))


def decompose_wave(eta, t, n_max=50):
    """
    Decompose a recorded wave elevation into per-component (A, w, k, phi)
    tuples using FFT. Keeps the top n_max components by amplitude to keep
    FK pressure evaluation tractable during simulation.

    Returns list of (A, w, k, phi) sorted by descending amplitude.
    """
    n = len(eta)
    dt = t[1] - t[0]

    spectrum = np.fft.rfft(eta) / n
    freqs = np.fft.rfftfreq(n, d=dt)

    components = []
    for i, f in enumerate(freqs):
        if f <= 0:
            continue
        A = 2.0 * abs(spectrum[i])
        if A < 1e-6:
            continue
        w = 2.0 * np.pi * f
        k = w**2 / 9.81
        phi = float(np.angle(spectrum[i]))
        components.append((A, w, k, phi))

    # keep only the most energetic components
    components.sort(key=lambda c: -c[0])
    return components[:n_max]

    """
    Linear excitation force fe = IRF * eta (convolution), precomputed over
    the full wave signal, wrapped in an interpolator.

    NOTE: the convolution is causal -- fe[0:n_irf] is inaccurate because
    the simulation has no wave history before t=0. Apply a t_warmup cutoff
    when computing metrics to exclude this startup transient.
    """

    def __init__(self, fe_series, t_series, interpolate=True, name="fe_lin"):
        self._interp = interp1d(
            t_series,
            fe_series,
            kind="linear",
            bounds_error=False,
            fill_value=(fe_series[0], fe_series[-1]),
        )
        self.interpolate = interpolate
        self.name = name
        self._frozen = 0.0

    def begin_step(self, t0):
        if not self.interpolate:
            self._frozen = float(self._interp(t0))

    def compute(self, t, X, interpolate=None):
        use_interp = self.interpolate if interpolate is None else interpolate
        return float(self._interp(t)) if use_interp else self._frozen


# %% MODEL


class Model:
    """
    Equation of motion: (m + ma_inf)*xddot = sum(forces).
    Linear vs weakly nonlinear = same class, different restoring term.
    """

    def __init__(
        self, m, forces, ma_inf=0.0, radiation_term=None, n_radiation_states=0
    ):
        self.m = m
        self.ma_inf = ma_inf
        self.forces = forces
        self.radiation_term = radiation_term
        self.n_radiation_states = n_radiation_states

    def begin_step(self, t0):
        for force in self.forces:
            force.begin_step(t0)

    def derivative(self, t, X):
        F = sum(force.compute(t, X) for force in self.forces)
        xdot = X[1]
        xddot = F / (self.m + self.ma_inf)
        if self.radiation_term is not None:
            return np.concatenate(
                ([xdot, xddot], self.radiation_term.state_derivative(t, X))
            )
        return np.array([xdot, xddot])

    def initial_state(self, x0=0.0, xdot0=0.0):
        """
        Return the augmented initial state vector [x, xdot, z1, z2, ...].
        NOTE: x0/xdot0 are scalars covering the physics-model case.
        Data-driven models that require a past-window of inputs as their
        "initial state" need a different entry point -- to be designed when
        a data-driven model is added.
        """
        return np.concatenate(([x0, xdot0], np.zeros(self.n_radiation_states)))

    def force_postprocess(self, t_hist, X_hist):
        """
        Re-evaluate every force term over a recorded trajectory with
        interpolate=True, regardless of how the term was configured during
        integration. Returns {term.name: array} for all output columns.
        """
        out = {}
        for force in self.forces:
            out[force.name] = np.array(
                [force.compute(t, X, interpolate=True) for t, X in zip(t_hist, X_hist)]
            )
        return out

    @classmethod
    def from_config(
        cls,
        model_def,
        hydro_sphere,
        eta_t,
        eta_values,
        wave_components=None,
        pto=None,
        interpolation=None,
    ):
        """
        Build a Model from a model definition dict and a pre-instantiated
        HydroSphere. PTO is a per-run control-law choice sourced from the
        wave source (CFD case config or explicit synthetic entry) -- it is
        NOT part of the model definition.
        """
        interpolation = interpolation or {}
        body = model_def["body"]
        r = hydro_sphere.r
        rho = body.get("rho", 1025)
        g = body.get("g", 9.81)

        m = hydro_sphere.mass
        S = hydro_sphere.stiffness

        ft = model_def["force_terms"]
        forces = []
        ma_inf = 0.0
        radiation_term = None
        n_radiation_states = 0

        # restoring
        restoring = ft.get("restoring", "linear")
        if restoring == "linear":
            forces.append(RestoringLinear(S, name="fb_lin"))
        elif restoring == "nonlinear":
            # nonlinear hydrostatic without free-surface correction (eta=0)
            forces.append(RestoringNonlinear(r, rho, g, m, eta=None, name="fb_nl"))
        elif restoring == "nonlinear_eta":
            # nonlinear hydrostatic with free-surface elevation correction
            eta_interp = interp1d(
                eta_t,
                eta_values,
                kind="linear",
                bounds_error=False,
                fill_value=0.0,
            )
            forces.append(
                RestoringNonlinear(
                    r,
                    rho,
                    g,
                    m,
                    eta=eta_interp,
                    interpolate=interpolation.get("eta_hs", True),
                    name="fb_nl_eta",
                )
            )
        else:
            raise ValueError(
                f"Unknown restoring flavor '{restoring}'. "
                f"Available: 'linear', 'nonlinear', 'nonlinear_eta'"
            )

        # radiation
        if ft.get("radiation") == "state_space":
            rad_ss = hydro_sphere.rad_ss
            radiation_term = RadiationStateSpace(
                rad_ss["Ar"],
                rad_ss["Br"],
                rad_ss["Cr"],
                rad_ss["Dr"],
                name="fr",
            )
            forces.append(radiation_term)
            ma_inf = hydro_sphere.ma_inf
            n_radiation_states = rad_ss["Ar"].shape[0]

        # excitation
        if ft.get("excitation") == "linear_conv":
            fe_series = fe_conv(
                eta_values,
                hydro_sphere.fe_irf,
                eta_t,
                hydro_sphere.t_irf,
                resample="irf",
            )
            forces.append(
                ExcitationLinearConv(
                    fe_series,
                    eta_t,
                    interpolate=interpolation.get("fe", True),
                    name="fe_lin",
                )
            )

        elif ft.get("excitation") == "fk_nonlinear":
            if hydro_sphere.panels is None:
                raise RuntimeError(
                    "Model requires a mesh for nonlinear FK -- "
                    "call hydro_sphere.make_mesh() before building the model."
                )
            if wave_components is None:
                raise ValueError(
                    "fk_nonlinear excitation requires wave_components -- "
                    "pass decompose_wave(eta, t) result to Model.from_config()."
                )
            forces.append(
                FKNonlinear(
                    panels=hydro_sphere.panels,
                    wave_components=wave_components,
                    eta_t=eta_t,
                    eta_values=eta_values,
                    rho=rho,
                    interpolate=interpolation.get("fk", True),
                    name="f_fk",
                )
            )

        # PTO
        if pto is not None:
            forces.append(RestoringLinear(pto["stiffness"], name="fs_pto"))
            forces.append(LinearDamping(pto["damping"], name="fd_pto"))

        # viscous drag -- c_v identified by run_optimize, stored in model_def["identified"]
        if ft.get("drag") == "viscous":
            c_v = model_def.get("identified", {}).get("c_v")
            if c_v is None:
                raise ValueError(
                    "Viscous drag model requires an identified c_v -- "
                    "run run_optimize.py with scheme 'viscous_drag' first."
                )
            forces.append(ViscousDragForce(float(c_v), name="f_visc"))

        return cls(
            m=m,
            forces=forces,
            ma_inf=ma_inf,
            radiation_term=radiation_term,
            n_radiation_states=n_radiation_states,
        )
