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
        cls, model_def, hydro_sphere, eta_t, eta_values, pto=None, interpolation=None
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
            raise ValueError(f"Unknown restoring flavor '{restoring}'")

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

        # PTO
        if pto is not None:
            forces.append(RestoringLinear(pto["stiffness"], name="fs_pto"))
            forces.append(LinearDamping(pto["damping"], name="fd_pto"))

        return cls(
            m=m,
            forces=forces,
            ma_inf=ma_inf,
            radiation_term=radiation_term,
            n_radiation_states=n_radiation_states,
        )
