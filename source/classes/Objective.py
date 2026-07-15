from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from source.function.error_utils import metricError

"""
# -------------------------------------------------------------------------
# Name:            Objective.py
# Description:     Flexible objective class for identification schemes.
#                   An objective is any callable (simrun, ref_obj) -> float,
#                   registered by name in OBJECTIVES. The config references
#                   the name; adding a new objective is one line in the
#                   registry with no class changes.
#
# Author:          Antoine
# Collaborator:    Bona, Bruno, Edoardo
# Date created:    07/2026
# Project:         wec_modeling_benchmark
# -------------------------------------------------------------------------
"""


# %% HELPERS

def _interp(simrun, ref_obj, variable: str) -> np.ndarray:
    """Interpolate simrun variable onto ref_obj time grid."""
    t_ref = ref_obj.dataset["t"].to_numpy()
    t_sim = simrun.dataset["t"].to_numpy()
    return np.interp(t_ref, t_sim, simrun.dataset[variable].to_numpy())


def _is_valid(y: np.ndarray, limit: float = 1e6) -> bool:
    return bool(np.all(np.isfinite(y)) and np.max(np.abs(y)) < limit)


# %% OBJECTIVE REGISTRY

def _rmse(variable: str):
    """RMSE of a simulated state variable vs reference."""
    def fn(simrun, ref_obj):
        y_sim = _interp(simrun, ref_obj, variable)
        if not _is_valid(y_sim):
            return np.inf
        y_ref = ref_obj.dataset[variable].to_numpy()
        return metricError(y_sim, y_ref, "rmse")
    return fn


def _nrmse(variable: str, norm: str = "range"):
    """Normalised RMSE of a simulated state variable vs reference."""
    metric = f"nrmse_{norm}"
    def fn(simrun, ref_obj):
        y_sim = _interp(simrun, ref_obj, variable)
        if not _is_valid(y_sim):
            return np.inf
        y_ref = ref_obj.dataset[variable].to_numpy()
        return metricError(y_sim, y_ref, metric)
    return fn


def _mean_P_abs(simrun, ref_obj):
    """
    Mean absorbed power: P_abs = B_pto * xdot^2.
    Direction should be "maximize" in the Objective.
    """
    xdot_sim = _interp(simrun, ref_obj, "xdot")
    if not _is_valid(xdot_sim):
        return np.inf
    return float(np.mean(ref_obj.damping * xdot_sim ** 2))


def _rmse_P_abs(simrun, ref_obj):
    """RMSE of absorbed power vs reference power."""
    xdot_sim = _interp(simrun, ref_obj, "xdot")
    if not _is_valid(xdot_sim):
        return np.inf
    P_sim = ref_obj.damping * xdot_sim ** 2
    if "P_abs" in ref_obj.dataset.columns:
        P_ref = ref_obj.dataset["P_abs"].to_numpy()
    else:
        P_ref = ref_obj.damping * ref_obj.dataset["xdot"].to_numpy() ** 2
    return metricError(P_sim, P_ref, "rmse")


OBJECTIVES: dict[str, Callable] = {
    "rmse_x":         _rmse("x"),
    "rmse_xdot":      _rmse("xdot"),
    "nrmse_x":        _nrmse("x", "range"),
    "nrmse_xdot":     _nrmse("xdot", "range"),
    "mean_P_abs":     _mean_P_abs,
    "rmse_P_abs":     _rmse_P_abs,
}


# %% OBJECTIVE CLASS

@dataclass
class Objective:
    """
    Wraps an (simrun, ref_obj) -> float callable with a direction and name.

    Parameters
    ----------
    name      : key in OBJECTIVES, used for logging and result keys
    direction : "minimize" | "maximize"
    fn        : the callable — set automatically from name via from_dict()
    """
    name:      str
    direction: str = "minimize"
    fn:        Callable = field(default=None, repr=False)

    def __post_init__(self):
        if self.fn is None:
            if self.name not in OBJECTIVES:
                raise ValueError(
                    f"Unknown objective '{self.name}'. "
                    f"Available: {list(OBJECTIVES)}")
            self.fn = OBJECTIVES[self.name]

    def sign(self) -> float:
        return 1.0 if self.direction == "minimize" else -1.0

    def compute(self, simrun, ref_obj) -> float:
        """
        Evaluate objective. Returns sign * fn(simrun, ref_obj) so scipy
        always minimises regardless of direction.

        Parameters
        ----------
        simrun  : SimRun  — output of SimRun.simulate()
        ref_obj : DataHandle — CFD reference for this sea state.
                  Provides reference time grid, state columns, and damping.
        """
        return self.sign() * self.fn(simrun, ref_obj)

    def label(self) -> str:
        return self.name

    @classmethod
    def from_dict(cls, d) -> "Objective":
        """
        Build from config dict or string shorthand.
        String: "rmse_x"
        Dict:   {"name": "rmse_x", "direction": "minimize"}
        """
        if isinstance(d, str):
            return cls(name=d)
        return cls(
            name      = d.get("name",      "rmse_x"),
            direction = d.get("direction", "minimize"),
        )
