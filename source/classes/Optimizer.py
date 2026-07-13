import copy
import json
import logging
import re
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from tqdm import tqdm

from source.config import MODEL_DIR
from source.classes.DataHandle import DataHandle
from source.classes.Model import Model, decompose_wave
from source.classes.Plotter import Plotter
from source.classes.SimRun import SimRun, get_solver
from source.function.error_utils import metricError

"""
# -------------------------------------------------------------------------
# Name:            Optimizer.py
# Description:     Identification algorithms, associated plot methods, and
#                   model saving. One _run_<scheme>, _save_<scheme>, and
#                   a set of _plot_* methods per scheme. PLOT_DISPATCH is
#                   a property so it stays scheme-specific.
#
# Author:          Antoine
# Collaborator:    Bona, Bruno, Edoardo
# Date created:    06/2026
# Project:         wec_modeling_benchmark
# -------------------------------------------------------------------------
"""

log = logging.getLogger(__name__)


def _safe(text: str) -> str:
    return re.sub(r"[^\w\-]+", "_", text).strip("_")


def _wave_params(obj) -> tuple:
    try:
        wp = obj.wave_params
        return float(wp.Te.iloc[0]), float(wp.Hs.iloc[0]), float(wp.J.iloc[0])
    except Exception:
        return None, None, None


def _methods_from_params(sp: dict) -> list:
    """
    Resolve (type, search) config options into the list of method names to run.

    type   : "force"  | "response" | ["force", "response"]
    search : "opt"    | "grid"     | ["opt", "grid"]

    Resulting method names:
      force    + opt  -> "force"
      force    + grid -> "force_grid"
      response + opt  -> "response"
      response + grid -> "response_grid"
    """
    types = sp.get("type", "force")
    searches = sp.get("search", "opt")
    if isinstance(types, str):
        types = [types]
    if isinstance(searches, str):
        searches = [searches]

    methods = []
    for t in types:
        for s in searches:
            if s == "opt":
                methods.append(t + "_opt")
            elif s == "grid":
                methods.append(f"{t}_grid")
    return methods


class Optimizer:
    """
    Identification algorithms for all schemes.
    Owns: run(), save(), save_result(), save_data(), run_plots(),
          PLOT_DISPATCH, load_base_model(), and simulation helpers.

    Parameters
    ----------
    scheme      : identification scheme name, e.g. "viscous_drag"
    base_model  : name of the base model folder under models/
    solver_cfg  : solver block from config (method, dt, interpolation, ...)
    """

    def __init__(self, scheme: str, base_model: str, solver_cfg: dict):
        self.scheme = scheme
        self.solver_cfg = solver_cfg
        self.model_def, self.hydro_sphere = self.load_base_model(base_model)

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
            pkl_path = MODEL_DIR / f"{pkl_name}.pkl"
            with open(pkl_path, "rb") as f:
                hydro_sphere = pickle.load(f)
        return model_def, hydro_sphere

    # %% ENTRY POINTS

    def run(self, feed, config):
        """Dispatch to _run_<scheme>. Returns (result, grids, simruns_by_method)."""
        method = getattr(self, f"_run_{self.scheme}", None)
        if method is None:
            raise ValueError(f"No optimizer for scheme '{self.scheme}'.")
        return method(feed, config)

    def save(self, result, output_name):
        """Dispatch to _save_<scheme>."""
        method = getattr(self, f"_save_{self.scheme}", None)
        if method is None:
            raise ValueError(f"No save method for scheme '{self.scheme}'.")
        method(result, output_name)

    def save_result(self, result, model_dir):
        """Save result.json and log key metrics."""
        with open(model_dir / "result.json", "w") as f:
            json.dump(result, f, indent=2)
        log.info("Result saved.")
        getattr(self, f"_log_result_{self.scheme}")(result)

    def save_data(self, result, simruns_by_method, feed, model_dir):
        """Save timeseries CSVs. Dispatches to _save_data_<scheme>."""
        fn = getattr(self, f"_save_data_{self.scheme}", None)
        if fn is None:
            return
        data_dir = model_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        fn(result, simruns_by_method, feed, data_dir)

    def run_plots(
        self, config, feed, simruns_by_method, result, grids_by_method, plots_dir
    ):
        """Run all active plots from config.plot_options."""
        plots_dir.mkdir(parents=True, exist_ok=True)
        dispatch = self.PLOT_DISPATCH
        for plot_name, active in config.plot_options.items():
            if not active:
                continue
            if plot_name not in dispatch:
                log.warning(
                    "plot '%s' not in PLOT_DISPATCH for scheme '%s', skipping",
                    plot_name,
                    self.scheme,
                )
                continue
            dispatch[plot_name](
                feed.data, simruns_by_method, result, grids_by_method, plots_dir
            )

    # %% PLOT DISPATCH

    @property
    def PLOT_DISPATCH(self) -> dict:
        dispatches = {
            "viscous_drag": {
                "cost_curves": self._plot_cost_curves,
                "coeff_grid": self._plot_coeff_grid,
                "coeff_distribution": self._plot_coeff_distribution,
                "timeseries": self._plot_timeseries,
            },
        }
        return dispatches.get(self.scheme, {})

    # %% SIMULATION HELPERS

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
        return SimRun.simulate(
            model,
            get_solver(self.solver_cfg["method"]),
            t0=float(eta_t[0]),
            t_end=float(eta_t[-1]),
            dt=self.solver_cfg["dt"],
            eta_t=eta_t,
            eta_values=eta_values,
            label=f"{obj.label} | c_v={c_v:.1f}",
        )

    def _rmse_state(self, simrun, obj, objective_vars):
        t_cfd = obj.dataset["t"].to_numpy()
        total = 0.0
        for var in objective_vars:
            if var not in simrun.dataset.columns or var not in obj.dataset.columns:
                continue
            x_sim = np.interp(
                t_cfd, simrun.dataset["t"].to_numpy(), simrun.dataset[var].to_numpy()
            )
            if not np.all(np.isfinite(x_sim)) or np.max(np.abs(x_sim)) > 1e4:
                return np.inf
            total += metricError(x_sim, obj.dataset[var].to_numpy(), "rmse")
        return total / len(objective_vars)

    # %% VISCOUS DRAG — RUN

    def _run_viscous_drag(self, feed, config):
        """
        Identify c_v in F_visc = -c_v|xdot|xdot.

        Config options (scheme_params):
          type   : "force" | "response" | ["force","response"]
          search : "opt"   | "grid"     | ["opt","grid"]

        Both global and per-case always computed.
        Identification time is measured separately from post-hoc simulations.
        """
        sp = config.scheme_params
        methods = _methods_from_params(sp)
        c_min = max(0.0, sp.get("c_v_min", 0.0))
        c_max = sp.get("c_v_max", 5000.0)
        n_pts = sp.get("n_points", 100)
        obj_vars = sp.get("objective_vars", ["x"])
        interp_cfg = config.solver.get("interpolation", {})

        if not feed.per_case:
            raise RuntimeError("DDFeed has no valid cases.")

        results = {}
        grids_by_method = {}
        simruns_by_method = {}

        for method in methods:
            tqdm.write(f"\n{'=' * 60}")
            tqdm.write(f"  Method: {method}")
            tqdm.write(f"{'=' * 60}")
            per_case = {}
            grids = {}

            # ---- per-case identification (timed separately) ----
            for label, entry in tqdm(
                feed.per_case.items(), desc=f"[{method}] per-case", unit="case"
            ):
                obj = entry["obj"]
                phi = entry["regressor"]
                r = entry["residual"]
                t0 = time.perf_counter()

                if method == "force_opt":
                    c_v = max(0.0, float((phi @ r) / (phi @ phi)))
                    rmse_force = metricError(phi * c_v, r, "rmse")
                    elapsed_id = time.perf_counter() - t0

                elif method == "force_grid":
                    c_grid = np.linspace(c_min, c_max, n_pts)
                    costs = np.array([metricError(phi * c, r, "rmse") for c in c_grid])
                    c_v = float(c_grid[np.argmin(costs)])
                    rmse_force = float(np.min(costs))
                    elapsed_id = time.perf_counter() - t0
                    grids[label] = (c_grid, costs)

                elif method == "response_grid":
                    c_grid = np.linspace(c_min, c_max, n_pts)
                    costs = np.array(
                        [
                            self._rmse_state(
                                self._run_sim(c, obj, interp_cfg), obj, obj_vars
                            )
                            for c in c_grid
                        ]
                    )
                    c_v = float(c_grid[np.argmin(costs)])
                    rmse_force = None
                    elapsed_id = time.perf_counter() - t0
                    grids[label] = (c_grid, costs)

                elif method == "response_opt":
                    call_count = [0]

                    def _obj(c):
                        call_count[0] += 1
                        tqdm.write(
                            f"    optimising... eval {call_count[0]:>3d} c_v={c:.1f}",
                            end="\r",
                        )
                        return self._rmse_state(
                            self._run_sim(c, obj, interp_cfg), obj, obj_vars
                        )

                    res = minimize_scalar(_obj, bounds=(c_min, 1e6), method="bounded")
                    c_v = float(res.x)
                    rmse_force = None
                    elapsed_id = time.perf_counter() - t0
                    tqdm.write(
                        f"    optimising... {call_count[0]} evals  "
                        f"c_v*={c_v:.2f}  ({elapsed_id:.1f}s)"
                    )

                else:
                    raise ValueError(f"Unknown method '{method}'")

                # post-hoc state RMSE (not timed)
                simrun = self._run_sim(c_v, obj, interp_cfg)
                rmse_x = self._rmse_state(simrun, obj, obj_vars)

                per_case[label] = {
                    "c_v": c_v,
                    "rmse_x": rmse_x,
                    "rmse_force": rmse_force,
                    "identification_elapsed_s": elapsed_id,
                }
                tqdm.write(f"  {label}  c_v={c_v:.2f}  rmse_x={rmse_x:.5f}")

            # ---- global identification ----
            tqdm.write(f"\n  Global [{method}]")
            phi_all = feed.global_regressor
            r_all = feed.global_residual
            t0_global = time.perf_counter()

            if method == "force_opt":
                c_v_global = max(0.0, float((phi_all @ r_all) / (phi_all @ phi_all)))
                tqdm.write(f"  analytical  c_v_global={c_v_global:.2f}")

            elif method == "force_grid":
                c_grid = np.linspace(c_min, c_max, n_pts)
                costs = np.array(
                    [metricError(phi_all * c, r_all, "rmse") for c in c_grid]
                )
                c_v_global = float(c_grid[np.argmin(costs)])
                grids["__global__"] = (c_grid, costs)
                tqdm.write(f"  grid ({n_pts} pts)  c_v_global={c_v_global:.2f}")

            elif method == "response_grid":
                c_grid = np.linspace(c_min, c_max, n_pts)
                valid = [e["obj"] for e in feed.per_case.values()]
                costs = np.array(
                    [
                        np.mean(
                            [
                                self._rmse_state(
                                    self._run_sim(c, obj, interp_cfg), obj, obj_vars
                                )
                                for obj in valid
                            ]
                        )
                        for c in tqdm(c_grid, desc=f"[{method}] global grid")
                    ]
                )
                c_v_global = float(c_grid[np.argmin(costs)])
                grids["__global__"] = (c_grid, costs)
                tqdm.write(f"  grid ({n_pts} pts)  c_v_global={c_v_global:.2f}")

            elif method == "response_opt":
                valid = [e["obj"] for e in feed.per_case.values()]
                call_count = [0]

                def _global_obj(c):
                    call_count[0] += 1
                    tqdm.write(
                        f"  optimising... eval {call_count[0]:>3d} c_v={c:.1f}",
                        end="\r",
                    )
                    return np.mean(
                        [
                            self._rmse_state(
                                self._run_sim(c, obj, interp_cfg), obj, obj_vars
                            )
                            for obj in valid
                        ]
                    )

                res = minimize_scalar(
                    _global_obj, bounds=(c_min, 1e6), method="bounded"
                )
                c_v_global = float(res.x)
                tqdm.write(
                    f"  optimising... {call_count[0]} evals  "
                    f"c_v_global*={c_v_global:.2f}  "
                    f"({time.perf_counter() - t0_global:.1f}s)"
                )

            global_id_elapsed = time.perf_counter() - t0_global

            # post-hoc simulations (not timed)
            valid = [e["obj"] for e in feed.per_case.values()]
            simruns_global = {
                obj.label: self._run_sim(c_v_global, obj, interp_cfg) for obj in valid
            }
            simruns_per_case = {
                obj.label: self._run_sim(per_case[obj.label]["c_v"], obj, interp_cfg)
                for obj in valid
                if obj.label in per_case
            }
            rmse_x_global = float(
                np.mean(
                    [
                        self._rmse_state(simruns_global[obj.label], obj, obj_vars)
                        for obj in valid
                    ]
                )
            )

            tqdm.write(f"  c_v_global={c_v_global:.2f}  rmse_x={rmse_x_global:.5f}")

            c_v_values = [per_case[l]["c_v"] for l in per_case]
            results[method] = {
                "c_v_global": c_v_global,
                "rmse_x_global": rmse_x_global,
                "c_v_per_case": {l: per_case[l]["c_v"] for l in per_case},
                "rmse_x_per_case": {l: per_case[l]["rmse_x"] for l in per_case},
                "c_v_mean": float(np.mean(c_v_values)),
                "c_v_std": float(np.std(c_v_values)),
                "identification_elapsed_s": {
                    "per_case": {
                        l: per_case[l]["identification_elapsed_s"] for l in per_case
                    },
                    "global": global_id_elapsed,
                },
            }
            grids_by_method[method] = grids
            simruns_by_method[method] = {
                "global": simruns_global,
                "per_case": simruns_per_case,
            }

        result = {
            "scheme": "viscous_drag",
            "base_model": config.base_model,
            "base_force_col": sp.get("base_force", "fhyd_lin"),
            "objective_vars": obj_vars,
            "n_cases": len(feed.per_case),
            "results": results,
        }
        return result, grids_by_method, simruns_by_method

    # %% VISCOUS DRAG — SAVE MODEL

    def _save_viscous_drag(self, result, output_name):
        """
        Write one model.json per method. If single method: model.json.
        If multiple: model_{method}.json.
        All are validate-ready.
        """
        out_dir = MODEL_DIR / output_name
        out_dir.mkdir(parents=True, exist_ok=True)
        methods = list(result["results"].keys())

        for method, r in result["results"].items():
            md = copy.deepcopy(self.model_def)
            md["force_terms"]["drag"] = "viscous"
            md.setdefault("identified", {}).update(
                {
                    "c_v": r["c_v_global"],
                    "method": method,
                    "identified_from": str(out_dir),
                }
            )
            fname = "model.json" if len(methods) == 1 else f"model_{method}.json"
            with open(out_dir / fname, "w") as f:
                json.dump(md, f, indent=2)
            log.info("Model saved: %s", out_dir / fname)

    # %% VISCOUS DRAG — LOG

    def _log_result_viscous_drag(self, result):
        for m, r in result["results"].items():
            log.info(
                "[%s] c_v_global=%.2f  rmse_x_global=%.5f",
                m,
                r["c_v_global"],
                r["rmse_x_global"],
            )

    # %% VISCOUS DRAG — DATA

    def _save_data_viscous_drag(self, result, simruns_by_method, feed, data_dir):
        """
        Per method, per case: 2 CSVs each (per-case c_v and global c_v).
        Columns: t, x_sim, x_ref, xdot_sim, xdot_ref, f_visc, f_residual, f_base
        Naming: {method}_{safe_label}.csv  (per-case c_v)
                {method}_global_{safe_label}.csv  (global c_v)
        """
        base_col = result["base_force_col"]

        for method, r in result["results"].items():
            simruns_g = simruns_by_method[method]["global"]
            simruns_pc = simruns_by_method[method]["per_case"]

            for obj in feed.data:
                label = obj.label
                if label not in simruns_g:
                    continue

                t_ref = obj.dataset["t"].to_numpy()
                f_base = obj.dataset[base_col].to_numpy()
                f_residual = feed.per_case[label]["residual"]

                def _make_df(sr, c_v):
                    t_sim = sr.dataset["t"].to_numpy()
                    x_sim = np.interp(t_ref, t_sim, sr.dataset["x"].to_numpy())
                    xdot_sim = np.interp(t_ref, t_sim, sr.dataset["xdot"].to_numpy())
                    f_visc = -c_v * np.abs(xdot_sim) * xdot_sim
                    return pd.DataFrame(
                        {
                            "t": t_ref,
                            "x_sim": x_sim,
                            "x_ref": obj.dataset["x"].to_numpy(),
                            "xdot_sim": xdot_sim,
                            "xdot_ref": obj.dataset["xdot"].to_numpy(),
                            "f_visc": f_visc,
                            "f_residual": f_residual,
                            "f_base": f_base,
                        }
                    )

                safe = _safe(label)[:50]
                search = "grid" if "grid" in method else "opt"
                mtype = "force" if "force" in method else "response"

                if label in simruns_pc:
                    df_pc = _make_df(simruns_pc[label], r["c_v_per_case"][label])
                    df_pc.to_csv(
                        data_dir / f"{mtype}_{search}_indiv_{safe}.csv", index=False
                    )
                    log.info("Saved: %s_%s_indiv_%s.csv", mtype, search, safe)

                df_gl = _make_df(simruns_g[label], r["c_v_global"])
                df_gl.to_csv(
                    data_dir / f"{mtype}_{search}_global_{safe}.csv", index=False
                )
                log.info("Saved: %s_%s_global_%s.csv", mtype, search, safe)

    # %% VISCOUS DRAG — PLOTS

    def _plot_cost_curves(
        self, data, simruns_by_method, result, grids_by_method, save_path
    ):
        """One figure per method that has grid data. Per-case + global subplots."""
        for method, grids in grids_by_method.items():
            keys = [k for k in grids if k != "__global__"]
            has_gl = "__global__" in grids
            n = len(keys) + (1 if has_gl else 0)
            if n == 0:
                continue

            r = result["results"][method]
            ncols = min(n, 3)
            nrows = (n + ncols - 1) // ncols
            fig, axes = plt.subplots(
                nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False
            )

            def _one(ax, c_grid, costs, c_v_opt, title):
                unit = "kN" if costs.max() > 1e3 else "N/m"
                scale = 1e3 if costs.max() > 1e3 else 1.0
                ax.plot(c_grid, costs / scale, color="steelblue")
                ax.axvline(
                    c_v_opt, color="coral", ls="--", label=f"c_v* = {c_v_opt:.1f}"
                )
                ax.set_xlabel("c_v  [N·s²/m²]")
                ax.set_ylabel(f"RMSE  [{unit}]")
                ax.set_title(title, fontsize=9)
                ax.legend(fontsize=8)
                ax.grid(True)

            for idx, label in enumerate(keys):
                _one(
                    axes[idx // ncols][idx % ncols],
                    *grids[label],
                    r["c_v_per_case"][label],
                    label[:50],
                )
            if has_gl:
                _one(
                    axes[(n - 1) // ncols][(n - 1) % ncols],
                    *grids["__global__"],
                    r["c_v_global"],
                    "GLOBAL",
                )
            for idx in range(n, nrows * ncols):
                axes[idx // ncols][idx % ncols].set_visible(False)

            fig.suptitle(f"Cost curves [{method}]", y=1.01)
            plt.tight_layout()
            if save_path:
                Plotter._save(fig, save_path, "identify", f"cost_curves_{method}")

    def _plot_coeff_grid(
        self, data, simruns_by_method, result, grids_by_method, save_path
    ):
        """Per-case c_v on (Te, Hs) grid — one figure per method."""
        for method, r in result["results"].items():
            c_v_map = r["c_v_per_case"]
            Te_arr, Hs_arr, J_arr, c_v_arr = [], [], [], []
            for obj in data:
                if obj.label not in c_v_map:
                    continue
                Te, Hs, J = _wave_params(obj)
                if Te is None:
                    continue
                Te_arr.append(Te)
                Hs_arr.append(Hs)
                J_arr.append(J)
                c_v_arr.append(c_v_map[obj.label])

            if not Te_arr:
                log.warning("No wave_params for coeff_grid [%s], skipping", method)
                continue

            Plotter.plot_grid(
                np.array(Te_arr),
                np.array(Hs_arr),
                s=np.array(J_arr),
                c=np.array(c_v_arr),
                xlabel="Te [s]",
                ylabel="Hs [m]",
                clabel="c_v  [N·s²/m²]",
                title=(
                    f"Viscous drag coefficient [{method}]\n"
                    f"global={r['c_v_global']:.1f}  "
                    f"mean±std={r['c_v_mean']:.1f}±{r['c_v_std']:.1f}"
                ),
                save_path=save_path,
                name="identify",
                suffix=f"coeff_grid_{method}",
            )

    def _plot_coeff_distribution(
        self, data, simruns_by_method, result, grids_by_method, save_path
    ):
        """Histogram of per-case c_v — one figure per method, using Plotter."""
        for method, r in result["results"].items():
            vals = list(r["c_v_per_case"].values())
            Plotter.plot_histogram(
                series=[(np.array(vals), "per-case c_v")],
                xlabel="c_v  [N·s²/m²]",
                ylabel="count",
                title=(
                    f"Per-case viscous drag coefficient [{method}]\n"
                    f"mean={r['c_v_mean']:.1f}  "
                    f"std={r['c_v_std']:.1f}  "
                    f"global={r['c_v_global']:.1f}"
                ),
                bins=max(5, len(vals) // 2 + 1),
                density=False,
                save_path=save_path,
                name="identify",
                suffix=f"coeff_distribution_{method}",
            )

    def _plot_timeseries(
        self, data, simruns_by_method, result, grids_by_method, save_path
    ):
        """
        Per method, per case: overlay CFD reference with model prediction.
        Uses per-case c_v simrun so the plot shows the best individual fit.
        CFD reference appears as "ref" in the legend.
        """

        class _Labelled:
            """Thin wrapper to override .label without mutating the original."""

            def __init__(self, obj, label):
                self.dataset = obj.dataset
                self.label = label

        for method, r in result["results"].items():
            simruns_pc = simruns_by_method[method]["per_case"]
            for obj in data:
                label = obj.label
                if label not in simruns_pc:
                    continue
                sr = simruns_pc[label]
                DataHandle.plot_line(
                    [_Labelled(obj, "ref"), sr],
                    y="x",
                    ylabel="x [m]",
                    title=(
                        f"x — {label}  [{method}]\n"
                        f"c_v={r['c_v_per_case'].get(label, 0):.1f}  "
                        f"RMSE={r['rmse_x_per_case'].get(label, 0):.5f} m"
                    ),
                    save_path=save_path,
                    name="identify",
                    suffix=f"ts_{method}_{_safe(label)[:40]}",
                )
