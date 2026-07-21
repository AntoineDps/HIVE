import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from source.function.save_pathing import save_pathing

"""
# -------------------------------------------------------------------------
# Name:            SimRun.py
# Description:     One simulated run: trajectory container (.dataset / .label)
#                   plus the fixed-step solvers and comparison plot methods.
#                   Solvers live here rather than in Model because they
#                   describe HOW you run, not WHAT the model is.
#                   SimRun satisfies the same .dataset / .label interface as
#                   DataHandle, so CFD references and model runs can be passed
#                   to the same Plotter calls without special-casing.
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


# %% SIMRUN


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
        early_stop_fn=None,
    ):
        """
        Integrate model from t0 to t_end at fixed step dt.

        early_stop_fn : optional callable (t, X, eta_val) -> bool
            Called after each step. Return True to stop the simulation early.
            Used for stability checks (e.g. buoy fully submerged or out of water).
            Negligible overhead when provided; None disables the check entirely.
        """
        X = model.initial_state(x0, xdot0)
        n_steps = int(round((t_end - t0) / dt))

        t_hist = np.round(t0 + np.arange(n_steps + 1) * dt, decimals=10)
        X_hist = np.empty((n_steps + 1, len(X)))
        X_hist[0] = X

        wall_start = time.perf_counter()

        for i in range(1, n_steps + 1):
            X, _ = solver.step(model, t_hist[i - 1], X, dt)
            X_hist[i] = X
            if early_stop_fn is not None:
                eta_val = float(np.interp(t_hist[i], eta_t, eta_values))
                if early_stop_fn(t_hist[i], X, eta_val):
                    X_hist = X_hist[: i + 1]
                    t_hist = t_hist[: i + 1]
                    break

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

        # derive f_pto = fs_pto + fd_pto and P_abs = -fd_pto * xdot
        if "fs_pto" in dataset.columns and "fd_pto" in dataset.columns:
            dataset["f_pto"] = dataset["fs_pto"] + dataset["fd_pto"]
            dataset["P_abs"] = -dataset["fd_pto"] * dataset["xdot"]
        elif "fd_pto" in dataset.columns:
            dataset["f_pto"] = dataset["fd_pto"]
            dataset["P_abs"] = -dataset["fd_pto"] * dataset["xdot"]

        return cls(dataset, label, sim_time=sim_time)

    def clip(self, t_warmup: float = 0.0, t_causal: float = 0.0) -> "SimRun":
        """
        Return a new SimRun with the dataset trimmed to the physically valid window:
          - discard first t_warmup seconds (radiation + excitation causal warmup)
          - discard last t_causal seconds (excitation non-causal IRF margin)
        """
        t = self.dataset["t"].to_numpy()
        t_start = t[0] + t_warmup
        t_end = t[-1] - t_causal
        mask = (t >= t_start) & (t <= t_end)
        return SimRun(
            self.dataset[mask].reset_index(drop=True), self.label, self.sim_time
        )

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

    # %% PLOT METHODS
    # Delegate to DataHandle.plot_line so that mixed lists of
    # [DataHandle reference, SimRun, SimRun, ...] plot identically.

    @staticmethod
    def plot_states(objs, save_path=None, name=None):
        """x, xdot, eta, f_pto, P_abs — one figure per variable.
        P_abs y-axis starts at 0 (power is non-negative).
        """
        from source.classes.DataHandle import DataHandle
        import matplotlib.pyplot as plt

        safe = _safe_name(name) if name else None
        for var, ylabel in [
            ("x", "x [m]"),
            ("xdot", "xdot [m/s]"),
            ("eta", "eta [m]"),
            ("f_pto", "f_pto [N]"),
            ("P_abs", "P_abs [W]"),
        ]:
            objs_with = [o for o in objs if var in o.dataset.columns]
            if not objs_with:
                continue
            DataHandle.plot_line(
                objs_with,
                y=var,
                ylabel=ylabel,
                title=f"{var}  —  {name}" if name else var,
                save_path=save_path,
                name=safe,
                suffix=var,
            )
            if var == "P_abs":
                plt.ylim(bottom=0)

    @staticmethod
    def plot_forces(objs, save_path=None, name=None):
        """
        One figure per force term present in SimRun objects.
        CFD reference included only if it shares the column name.
        Excludes t, x, xdot, eta, f_pto, P_abs (handled by plot_states).
        """
        from source.classes.DataHandle import DataHandle

        _skip = {"t", "x", "xdot", "eta", "f_pto", "P_abs"}
        safe = _safe_name(name) if name else None

        force_cols = []
        for obj in objs:
            if isinstance(obj, SimRun):
                for col in obj.dataset.columns:
                    if col not in _skip and col not in force_cols:
                        force_cols.append(col)

        for col in force_cols:
            objs_with = [o for o in objs if col in o.dataset.columns]
            DataHandle.plot_line(
                objs_with,
                y=col,
                ylabel=f"{col} [N]",
                title=f"{col}  —  {name}" if name else col,
                save_path=save_path,
                name=safe,
                suffix=col,
            )
