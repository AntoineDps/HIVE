import json
import logging
import copy

import numpy as np
import matplotlib.pyplot as plt

from source.config import MODEL_DIR
from source.classes.Model import Model, decompose_wave
from source.classes.SimRun import SimRun, get_solver

"""
# -------------------------------------------------------------------------
# Name:            Optimizer.py
# Description:     Generic identification framework. Receives the scheme
#                   module from run_identify and delegates all scheme-specific
#                   logic to it. Does not own the scheme registry.
#
# Author:          Antoine
# Collaborator:    Bona, Bruno, Edoardo
# Date created:    06/2026
# Project:         wec_modeling_benchmark
# -------------------------------------------------------------------------
"""

log = logging.getLogger(__name__)


class Optimizer:
    """
    Generic identification framework.

    Parameters
    ----------
    scheme_module : the scheme module (e.g. Viscous_drag, Pi_gain)
                    passed in from run_identify which owns the registry
    base_model    : model folder name under models/
    solver_cfg    : solver block from config
    """

    def __init__(self, scheme_module, base_model: str, solver_cfg: dict):
        self._mod = scheme_module
        self.scheme = scheme_module.__name__.split(".")[-1]
        self.solver_cfg = solver_cfg
        self.model_def, self.hydro_sphere = self.load_base_model(base_model)

        # time margins — computed once, applied to all simulations
        from source.classes.Model import Model as _Model

        if self.hydro_sphere is not None:
            self.t_warmup, self.t_causal = _Model.compute_time_margins(
                self.model_def, self.hydro_sphere
            )
        else:
            self.t_warmup, self.t_causal = 0.0, 0.0

        ft = self.model_def.get("force_terms", {})
        self.n_wave_components = (
            ft.get("n_wave_components", 50)
            if ft.get("excitation") == "fk_nonlinear"
            else 50
        )

    # %% BASE MODEL LOADING

    @staticmethod
    def load_base_model(name: str):
        """Load model definition and HydroSphere from models/<name>/."""
        import pickle

        path = MODEL_DIR / name / "model.json"
        if not path.exists():
            raise FileNotFoundError(f"Base model not found: {path}")
        with open(path) as f:
            model_def = json.load(f)
        hydro_sphere = None
        pkl_name = model_def.get("body", {}).get("hydro_sphere_pkl")
        if pkl_name:
            pkl_path = MODEL_DIR / "bem" / f"{pkl_name}.pkl"
            with open(pkl_path, "rb") as f:
                hydro_sphere = pickle.load(f)
        return model_def, hydro_sphere

    # %% SIMULATION HELPERS (scheme-agnostic, used by scheme modules)

    def _run_sim(self, c_v, obj, interpolation):
        """Build augmented model with given c_v and simulate."""
        md = copy.deepcopy(self.model_def)
        md["force_terms"]["drag"] = "viscous"
        md.setdefault("identified", {})["c_v"] = float(c_v)

        eta_t = obj.dataset["t"].to_numpy()
        eta_values = obj.dataset["eta"].to_numpy()
        ft = md.get("force_terms", {})
        wc = (
            decompose_wave(eta_values, eta_t, n_max=self.n_wave_components)
            if ft.get("excitation") == "fk_nonlinear"
            else None
        )

        model = Model.from_config(
            md,
            self.hydro_sphere,
            eta_t,
            eta_values,
            wave_components=wc,
            pto={"damping": obj.damping, "stiffness": obj.stiffness},
            interpolation=interpolation,
        )
        result = SimRun.simulate(
            model,
            get_solver(self.solver_cfg["method"]),
            t0=float(eta_t[0]),
            t_end=float(eta_t[-1]),
            dt=self.solver_cfg["dt"],
            eta_t=eta_t,
            eta_values=eta_values,
            label=f"{obj.label} | c_v={c_v:.1f}",
        )
        if self.t_warmup > 0 or self.t_causal > 0:
            result = result.clip(self.t_warmup, self.t_causal)
        return result

    def _run_base_sim(self, obj, interpolation):
        """Run the base model without viscous drag."""
        eta_t = obj.dataset["t"].to_numpy()
        eta_values = obj.dataset["eta"].to_numpy()
        ft = self.model_def.get("force_terms", {})
        wc = (
            decompose_wave(eta_values, eta_t, n_max=self.n_wave_components)
            if ft.get("excitation") == "fk_nonlinear"
            else None
        )
        model = Model.from_config(
            self.model_def,
            self.hydro_sphere,
            eta_t,
            eta_values,
            wave_components=wc,
            pto={"damping": obj.damping, "stiffness": obj.stiffness},
            interpolation=interpolation,
        )
        result = SimRun.simulate(
            model,
            get_solver(self.solver_cfg["method"]),
            t0=float(eta_t[0]),
            t_end=float(eta_t[-1]),
            dt=self.solver_cfg["dt"],
            eta_t=eta_t,
            eta_values=eta_values,
            label=f"{obj.label} | base",
        )
        if self.t_warmup > 0 or self.t_causal > 0:
            result = result.clip(self.t_warmup, self.t_causal)
        return result

    # %% DISPATCH

    def run(self, feed, config):
        return self._mod.run(self, feed, config)

    def save(self, result, output_name):
        self._mod.save(self, result, output_name)

    def save_result(self, result, model_dir):
        model_dir.mkdir(parents=True, exist_ok=True)
        with open(model_dir / "result.json", "w") as f:
            json.dump(result, f, indent=2)
        log.info("Result saved.")
        self._mod.log_result(result)

    def save_data(self, result, simruns_by_method, grids_by_method, feed, model_dir):
        data_dir = model_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        self._mod.save_data(result, simruns_by_method, grids_by_method, feed, data_dir)

    def run_plots(
        self, config, feed, simruns_by_method, result, grids_by_method, plots_dir
    ):
        plots_dir.mkdir(parents=True, exist_ok=True)
        dispatch = self._mod.PLOT_DISPATCH
        # always run all available plots — user config plot_options is ignored here
        for plot_name, fn in dispatch.items():
            log.info("  plot: %s", plot_name)
            try:
                fn(feed.data, simruns_by_method, result, grids_by_method, plots_dir)
            except Exception as e:
                log.error("  plot failed [%s]: %s", plot_name, e)
        plt.show()
