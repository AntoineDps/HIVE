import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from source.function.save_pathing import save_pathing

"""
# -------------------------------------------------------------------------
# Name:            SimRun.py
# Description:     Simulate run, log and save corresponding timeseries and
#                   handle plotting as well.
#
# Author:          Antoine
# Collaborator:    Bona
# Date created:    06/2026
# Project:         surrogate_hydro
# -------------------------------------------------------------------------
"""


# %% SOLVERS


class Euler:
    order = 1

    @staticmethod
    def step(model, t, X, dt):
        model.begin_step(t)
        k1 = model.derivative(t, X)
        return X + dt * k1, t + dt


class RK2:
    """Heun's method (explicit trapezoidal)."""

    order = 2

    @staticmethod
    def step(model, t, X, dt):
        model.begin_step(t)
        k1 = model.derivative(t, X)
        k2 = model.derivative(t + dt, X + dt * k1)
        return X + dt / 2 * (k1 + k2), t + dt


class RK4:
    order = 4

    @staticmethod
    def step(model, t, X, dt):
        model.begin_step(t)
        k1 = model.derivative(t, X)
        k2 = model.derivative(t + dt / 2, X + dt / 2 * k1)
        k3 = model.derivative(t + dt / 2, X + dt / 2 * k2)
        k4 = model.derivative(t + dt, X + dt * k3)
        return X + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4), t + dt


SOLVERS = {"euler": Euler, "rk2": RK2, "rk4": RK4}


def get_solver(method: str):
    key = method.lower()
    if key not in SOLVERS:
        raise ValueError(f"Unknown solver '{method}'. Choose from {list(SOLVERS)}")
    return SOLVERS[key]


# %% RUNNING SIMULATIONS


def _safe_name(text: str) -> str:
    return re.sub(r"[^\w\-]+", "_", text).strip("_")


class SimRun:
    """
    Container for one model's simulated trajectory.
    Exposes .dataset (pd.DataFrame) and .label (str) -- the same interface
    DataHandle uses -- so mixed lists of CFD references and SimRun objects
    feed directly into Plotter without any special-casing.
    """

    def __init__(self, dataset: pd.DataFrame, label: str, sim_time: float = None):
        self.dataset = dataset
        self.label = label
        self.sim_time = sim_time  # wall-clock integration time in seconds

    @classmethod
    def simulate(
        cls,
        model,
        solver,
        t0,
        t_end,
        dt,
        eta_t,
        eta_values,
        x0=0.0,
        xdot0=0.0,
        label="model",
    ):
        """
        Integrate model from t0 to t_end at fixed step dt.

        Time vector is pre-computed from the step count to avoid floating-
        point accumulation (repeated t += dt drifts to values like 1.9999996).
        The solver still receives exact per-step t values from this vector.

        NOTE: x0/xdot0 cover the physics-model initial condition.
        Data-driven models needing a past-window as "initial state"
        require a different entry point -- to be added when needed.
        """
        X = model.initial_state(x0, xdot0)
        n_steps = int(round((t_end - t0) / dt))

        # pre-compute exact times -- avoids floating-point drift
        t_hist = np.round(t0 + np.arange(n_steps + 1) * dt, decimals=10)
        X_hist = np.empty((n_steps + 1, len(X)))
        X_hist[0] = X

        wall_start = time.perf_counter()

        for i in range(1, n_steps + 1):
            # pass the exact pre-computed time; ignore solver's returned t
            X, _ = solver.step(model, t_hist[i - 1], X, dt)
            X_hist[i] = X

        sim_time = time.perf_counter() - wall_start

        forces = model.force_postprocess(t_hist, X_hist)

        dataset = pd.DataFrame(
            {
                "t": t_hist,
                "x": X_hist[:, 0],
                "xdot": X_hist[:, 1],
                "eta": np.interp(t_hist, eta_t, eta_values),
            }
        )
        for col, values in forces.items():
            dataset[col] = values

        return cls(dataset, label, sim_time=sim_time)

    @classmethod
    def load(cls, path, label=None):
        """Reconstruct a SimRun from a saved CSV (remake-plots-only mode)."""
        path = Path(path)
        dataset = pd.read_csv(path)
        return cls(dataset, label or path.stem)

    def save(self, path, name, suffix="simrun"):
        full_path = save_pathing(path, name, suffix, ".csv")
        self.dataset.to_csv(full_path, index=False, encoding="utf-8")
        return full_path

    # plot
    @staticmethod
    def plot_states(objs, save_path=None, name=None):
        """x, xdot, eta -- one figure per variable, all runs overlaid."""
        from source.classes.DataHandle import DataHandle

        safe = _safe_name(name) if name else None
        for var, ylabel in [("x", "x [m]"), ("xdot", "xdot [m/s]"), ("eta", "eta [m]")]:
            DataHandle.plot_line(
                objs,
                y=var,
                ylabel=ylabel,
                title=f"{var}  —  {name}" if name else var,
                save_path=save_path,
                name=safe,
                suffix=var,
            )
