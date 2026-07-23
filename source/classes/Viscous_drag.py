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
from source.classes.Objective import Objective
from source.classes.DataHandle import DataHandle
from source.classes.Model import decompose_wave
from source.classes.Plotter import Plotter

"""
# -------------------------------------------------------------------------
# Name:            viscous_drag.py
# Description:     Scheme: viscous_drag.
#                   All DDFeed preparation and Optimizer identification,
#                   saving, and plotting logic for this scheme.
#                   Functions receive opt (Optimizer instance) when they
#                   need simulation helpers.
#
# Author:          Antoine
# Collaborator:    Bona, Bruno, Edoardo
# Date created:    07/2026
# Project:         wec_modeling_benchmark
# -------------------------------------------------------------------------
"""

log = logging.getLogger(__name__)


# %% HELPERS


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
    Resolve (type, search) into method name list.
      type   : "force" | "response" | list
      search : "opt"   | "grid"     | list
    Results: "force_opt", "force_grid", "response_opt", "response_grid"
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
            methods.append(f"{t}_{s}")
    return methods


# %% DDFEED — DATA PREPARATION


def prep(feed):
    """
    Prepare force regressor and residual for viscous drag identification.

    Stores on feed:
      per_case        : {label: {"regressor": phi, "residual": r, "obj": DataHandle}}
      global_regressor: stacked phi across all cases
      global_residual : stacked r  across all cases
    """
    base_col = feed.params.get("base_force", "fhyd_lin")
    feed.per_case = {}
    all_phi, all_r = [], []

    for obj in feed.data:
        label = obj.label
        if base_col not in obj.dataset.columns or "fhyd_cfd" not in obj.dataset.columns:
            log.warning(
                "Case '%s' missing '%s' or 'fhyd_cfd', skipping", label, base_col
            )
            continue
        phi = -np.abs(obj.dataset["xdot"].values) * obj.dataset["xdot"].values
        r = obj.dataset["fhyd_cfd"].values - obj.dataset[base_col].values
        feed.per_case[label] = {"regressor": phi, "residual": r, "obj": obj}
        all_phi.append(phi)
        all_r.append(r)

    feed.global_regressor = np.concatenate(all_phi) if all_phi else None
    feed.global_residual = np.concatenate(all_r) if all_r else None


# %% OPTIMIZER — RUN


def run(opt, feed, config):
    """
    Identify c_v in F_visc = -c_v|xdot|xdot.

    scheme_params:
      type      : "force" | "response" | list
      search    : "opt"   | "grid"     | list
      base_force: column name in DataHandle for the BEM force (default: fhyd_lin)
      c_v_min   : float (default 0)
      c_v_max   : float used for grid methods (default 5000)
      n_points  : int for grid methods (default 100)
      objective : "rmse_x"  or  {"name": "rmse_x", "direction": "minimize"}

    Both global and per-case always computed.
    Identification time measured separately from post-hoc simulations.
    """
    sp = config.scheme_params
    methods = _methods_from_params(sp)
    c_min = max(0.0, sp.get("c_v_min", 0.0))
    c_max = sp.get("c_v_max", 5000.0)
    n_pts = sp.get("n_points", 100)
    interp_cfg = config.solver.get("interpolation", {})
    objective = Objective.from_dict(sp.get("objective", "rmse_x"))

    if not feed.per_case:
        raise RuntimeError("DDFeed has no valid cases.")

    # base model (no drag) — run once, shared across all methods
    tqdm.write("\n  Running base model (no drag)...")
    log.info("Running base model (no drag) for %d case(s)", len(feed.per_case))
    base_simruns = {
        label: opt._run_base_sim(entry["obj"], interp_cfg)
        for label, entry in feed.per_case.items()
    }

    results = {}
    grids_by_method = {}
    simruns_by_method = {"__base__": base_simruns}

    for method in methods:
        tqdm.write(f"\n{'=' * 60}")
        tqdm.write(f"  Method: {method}")
        tqdm.write(f"{'=' * 60}")
        log.info("--- Method: %s ---", method)
        per_case = {}
        grids = {}

        # ---- per-case ----
        for label, entry in tqdm(
            feed.per_case.items(), desc=f"[{method}] per-case", unit="case"
        ):
            obj = entry["obj"]
            phi = entry["regressor"]
            r = entry["residual"]
            t0 = time.perf_counter()

            if method == "force_opt":
                c_v = max(0.0, float((phi @ r) / (phi @ phi)))
                rmse_force = float(np.sqrt(np.mean((r - phi * c_v) ** 2)))
                elapsed_id = time.perf_counter() - t0

            elif method == "force_grid":
                c_grid = np.linspace(c_min, c_max, n_pts)
                costs = np.array(
                    [float(np.sqrt(np.mean((r - phi * c) ** 2))) for c in c_grid]
                )
                c_v = float(c_grid[np.argmin(costs)])
                rmse_force = float(np.min(costs))
                elapsed_id = time.perf_counter() - t0
                grids[label] = (c_grid, costs)

            elif method == "response_grid":
                c_grid = np.linspace(c_min, c_max, n_pts)
                costs = np.array(
                    [
                        objective.compute(opt._run_sim(c, obj, interp_cfg), obj)
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
                    return objective.compute(opt._run_sim(c, obj, interp_cfg), obj)

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

            # post-hoc metric — always objective metric for consistency
            simrun = opt._run_sim(c_v, obj, interp_cfg)
            metric = objective.compute(simrun, obj)

            per_case[label] = {
                "c_v": c_v,
                "metric": metric,
                "rmse_force": rmse_force,
                "identification_elapsed_s": elapsed_id,
            }
            tqdm.write(f"  {label}  c_v={c_v:.2f}  {objective.label()}={metric:.5f}")
            log.info(
                "    %s  c_v=%.2f  %s=%.5f  elapsed=%.2fs",
                label,
                c_v,
                objective.label(),
                metric,
                elapsed_id,
            )

        # ---- global ----
        tqdm.write(f"\n  Global [{method}]")
        log.info("  Global [%s]", method)
        phi_all = feed.global_regressor
        r_all = feed.global_residual
        t0_global = time.perf_counter()

        if method == "force_opt":
            c_v_global = max(0.0, float((phi_all @ r_all) / (phi_all @ phi_all)))
            tqdm.write(f"  analytical  c_v_global={c_v_global:.2f}")
            log.info("  analytical  c_v_global=%.2f", c_v_global)

        elif method == "force_grid":
            c_grid = np.linspace(c_min, c_max, n_pts)
            costs = np.array(
                [float(np.sqrt(np.mean((r_all - phi_all * c) ** 2))) for c in c_grid]
            )
            c_v_global = float(c_grid[np.argmin(costs)])
            grids["__global__"] = (c_grid, costs)
            tqdm.write(f"  grid ({n_pts} pts)  c_v_global={c_v_global:.2f}")
            log.info("  force_grid (%d pts)  c_v_global=%.2f", n_pts, c_v_global)

        elif method == "response_grid":
            c_grid = np.linspace(c_min, c_max, n_pts)
            valid = [e["obj"] for e in feed.per_case.values()]
            costs = np.array(
                [
                    np.mean(
                        [
                            objective.compute(opt._run_sim(c, obj, interp_cfg), obj)
                            for obj in valid
                        ]
                    )
                    for c in tqdm(c_grid, desc=f"[{method}] global grid")
                ]
            )
            c_v_global = float(c_grid[np.argmin(costs)])
            grids["__global__"] = (c_grid, costs)
            tqdm.write(f"  grid ({n_pts} pts)  c_v_global={c_v_global:.2f}")
            log.info("  response_grid (%d pts)  c_v_global=%.2f", n_pts, c_v_global)

        elif method == "response_opt":
            valid = [e["obj"] for e in feed.per_case.values()]
            call_count = [0]

            def _global_obj(c):
                call_count[0] += 1
                tqdm.write(
                    f"  optimising... eval {call_count[0]:>3d} c_v={c:.1f}", end="\r"
                )
                return np.mean(
                    [
                        objective.compute(opt._run_sim(c, obj, interp_cfg), obj)
                        for obj in valid
                    ]
                )

            res = minimize_scalar(_global_obj, bounds=(c_min, 1e6), method="bounded")
            c_v_global = float(res.x)
            tqdm.write(
                f"  optimising... {call_count[0]} evals  "
                f"c_v_global*={c_v_global:.2f}  "
                f"({time.perf_counter() - t0_global:.1f}s)"
            )
            log.info(
                "  response_opt  %d evals  c_v_global=%.2f  elapsed=%.1fs",
                call_count[0],
                c_v_global,
                time.perf_counter() - t0_global,
            )

        global_id_elapsed = time.perf_counter() - t0_global

        # post-hoc simulations (not timed)
        valid = [e["obj"] for e in feed.per_case.values()]
        simruns_global = {
            obj.label: opt._run_sim(c_v_global, obj, interp_cfg) for obj in valid
        }
        simruns_per_case = {
            obj.label: opt._run_sim(per_case[obj.label]["c_v"], obj, interp_cfg)
            for obj in valid
            if obj.label in per_case
        }
        metric_global = float(
            np.mean(
                [objective.compute(simruns_global[obj.label], obj) for obj in valid]
            )
        )

        tqdm.write(
            f"  c_v_global={c_v_global:.2f}  {objective.label()}={metric_global:.5f}"
        )
        log.info(
            "  Global [%s]  c_v_global=%.2f  %s=%.5f",
            method,
            c_v_global,
            objective.label(),
            metric_global,
        )

        c_v_values = [per_case[l]["c_v"] for l in per_case]
        results[method] = {
            "c_v_global": c_v_global,
            "metric_global": metric_global,
            "metric_label": objective.label(),
            "c_v_per_case": {l: per_case[l]["c_v"] for l in per_case},
            "metric_per_case": {l: per_case[l]["metric"] for l in per_case},
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
        "objective": {"name": objective.name, "direction": objective.direction},
        "n_cases": len(feed.per_case),
        "results": results,
    }
    return result, grids_by_method, simruns_by_method


# %% OPTIMIZER — SAVE MODEL


def save(opt, result, output_name):
    """
    Write model.json only when a single method was run — unambiguous result.
    With multiple methods, result.json already contains everything; the user
    picks the best method and creates model.json manually from result.json.
    """
    out_dir = MODEL_DIR / output_name
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = list(result["results"].keys())

    # pick best method (lowest metric = best for RMSE-type objectives)
    def _metric_val(r):
        return float(r.get("metric_global", float("inf")))

    best_method = min(
        result["results"], key=lambda m: _metric_val(result["results"][m])
    )
    r = result["results"][best_method]
    md = copy.deepcopy(opt.model_def)
    md["force_terms"]["drag"] = "viscous"
    md.setdefault("identified", {}).update(
        {
            "c_v": r["c_v_global"],
            "method": best_method,
            "metric": r["metric_label"],
            "metric_value": r["metric_global"],
            "identified_from": str(out_dir),
        }
    )
    with open(out_dir / "model.json", "w") as f:
        json.dump(md, f, indent=2)
    log.info(
        "Model saved: %s  (best method: %s, c_v=%.2f)",
        out_dir / "model.json",
        best_method,
        r["c_v_global"],
    )
    if len(methods) > 1:
        log.info("All methods: %s — full results in result.json", methods)


# %% OPTIMIZER — LOG


def log_result(result):
    for m, r in result["results"].items():
        log.info(
            "[%s] c_v_global=%.2f  %s=%.5f",
            m,
            r["c_v_global"],
            r["metric_label"],
            r["metric_global"],
        )


# %% OPTIMIZER — SAVE DATA


def save_data(result, simruns_by_method, grids_by_method, feed, data_dir):
    """
    Per method:
    - Per-case timeseries CSVs (indiv and global c_v)
    - Cost curve CSVs per case + global (grid methods only)
    - Sea state grid CSV: Te × Hs × c_v × metric
    """
    base_col = result["base_force_col"]
    base_simruns = simruns_by_method.get("__base__", {})

    for method, r in result["results"].items():
        simruns_g = simruns_by_method[method]["global"]
        simruns_pc = simruns_by_method[method]["per_case"]
        grids = grids_by_method.get(method, {})
        search = "grid" if "grid" in method else "opt"
        mtype = "force" if "force" in method else "response"

        # ---- timeseries per case ----
        for obj in feed.data:
            label = obj.label
            if label not in simruns_g:
                continue

            f_residual = feed.per_case[label]["residual"]

            def _make_df(sr, c_v):
                """Build CSV using the simrun's clipped time as the axis."""
                t_s = sr.dataset["t"].to_numpy()  # already clipped to valid window
                t_ref = obj.dataset["t"].to_numpy()
                # mask reference and force data to simrun window
                mk_ref = (t_ref >= t_s[0]) & (t_ref <= t_s[-1])
                t_out = t_s

                x_sim = sr.dataset["x"].to_numpy()
                xdot_sim = sr.dataset["xdot"].to_numpy()

                x_ref_interp = np.interp(t_out, t_ref, obj.dataset["x"].to_numpy())
                xdot_ref_interp = np.interp(
                    t_out, t_ref, obj.dataset["xdot"].to_numpy()
                )
                f_base_interp = np.interp(
                    t_out, t_ref, obj.dataset[base_col].to_numpy()
                )
                f_res_interp = np.interp(t_out, t_ref, f_residual)

                sr_base = base_simruns.get(label)
                if sr_base is not None:
                    t_b = sr_base.dataset["t"].to_numpy()
                    x_base = np.interp(t_out, t_b, sr_base.dataset["x"].to_numpy())
                    xdot_base = np.interp(
                        t_out, t_b, sr_base.dataset["xdot"].to_numpy()
                    )
                else:
                    x_base = xdot_base = np.full_like(t_out, np.nan)

                return pd.DataFrame(
                    {
                        "t": t_out,
                        "x_ref": x_ref_interp,
                        "x_base": x_base,
                        "x_sim": x_sim,
                        "xdot_ref": xdot_ref_interp,
                        "xdot_base": xdot_base,
                        "xdot_sim": xdot_sim,
                        "f_visc": -c_v * np.abs(xdot_sim) * xdot_sim,
                        "f_residual": f_res_interp,
                        "f_base": f_base_interp,
                    }
                )

            safe = _safe(label)[:50]
            if label in simruns_pc:
                _make_df(simruns_pc[label], r["c_v_per_case"][label]).to_csv(
                    data_dir / f"{mtype}_{search}_indiv_{safe}.csv", index=False
                )
            _make_df(simruns_g[label], r["c_v_global"]).to_csv(
                data_dir / f"{mtype}_{search}_global_{safe}.csv", index=False
            )

        # ---- cost curves (grid methods only) ----
        for label, (c_grid, costs) in grids.items():
            suffix = "global" if label == "__global__" else _safe(label)[:50]
            pd.DataFrame({"c_v": c_grid, r["metric_label"]: costs}).to_csv(
                data_dir / f"{mtype}_{search}_cost_curve_{suffix}.csv", index=False
            )
            log.info("Saved cost curve: %s_%s_cost_curve_%s.csv", mtype, search, suffix)

        # ---- sea state grid ----
        rows = []
        for obj in feed.data:
            label = obj.label
            if label not in r["c_v_per_case"]:
                continue
            try:
                wp = obj.wave_params
                Te_v = float(wp.Te.iloc[0])
                Hs_v = float(wp.Hs.iloc[0])
                J_v = float(wp.J.iloc[0])
            except Exception:
                Te_v = Hs_v = J_v = None
            rows.append(
                {
                    "label": label,
                    "Te": Te_v,
                    "Hs": Hs_v,
                    "J": J_v,
                    "c_v": r["c_v_per_case"][label],
                    "c_v_global": r["c_v_global"],
                    r["metric_label"]: r["metric_per_case"][label],
                    f"{r['metric_label']}_global": r["metric_global"],
                }
            )
        if rows:
            pd.DataFrame(rows).to_csv(
                data_dir / f"{mtype}_{search}_sea_state_grid.csv", index=False
            )
            log.info("Saved sea state grid: %s_%s_sea_state_grid.csv", mtype, search)


# %% OPTIMIZER — PLOTS
# Uniform signature: (data, simruns_by_method, result, grids_by_method, save_path)


def plot_cost_curves(data, simruns_by_method, result, grids_by_method, save_path):
    """One figure per method with grid data: per-case + global cost subplots."""
    metric_label = next(iter(result["results"].values()))["metric_label"]

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
            ax.plot(c_grid, costs, color="steelblue")
            ax.axvline(c_v_opt, color="coral", ls="--", label=f"c_v* = {c_v_opt:.1f}")
            ax.set_xlabel("c_v  [N·s²/m²]")
            ax.set_ylabel(metric_label)
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


def plot_coeff_grid(data, simruns_by_method, result, grids_by_method, save_path):
    """Te×Hs scatter colored by c_v — all methods as subplots in one figure."""
    methods = list(result["results"].keys())
    n = len(methods)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols

    # collect data per method
    per_method = {}
    for method, r in result["results"].items():
        c_v_map = r["c_v_per_case"]
        Te_arr, Hs_arr, J_arr, cv_arr = [], [], [], []
        for obj in data:
            if obj.label not in c_v_map:
                continue
            Te, Hs, J = _wave_params(obj)
            if Te is None:
                continue
            Te_arr.append(Te)
            Hs_arr.append(Hs)
            J_arr.append(J)
            cv_arr.append(c_v_map[obj.label])
        if Te_arr:
            per_method[method] = dict(
                Te=np.array(Te_arr),
                Hs=np.array(Hs_arr),
                J=np.array(J_arr),
                cv=np.array(cv_arr),
            )

    if not per_method:
        return

    all_cv = [v for d in per_method.values() for v in d["cv"]]
    vmin, vmax = np.nanmin(all_cv), np.nanmax(all_cv)
    if vmin == vmax:
        vmin -= 0.5
        vmax += 0.5

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False
    )
    sc = None
    for idx, method in enumerate(per_method):
        ax = axes[idx // ncols][idx % ncols]
        d = per_method[method]
        sc = ax.scatter(
            d["Te"], d["Hs"], c=d["cv"], s=60, cmap="viridis", vmin=vmin, vmax=vmax
        )
        r = result["results"][method]
        ax.set_title(
            f"{method}\nglobal={r['c_v_global']:.1f} mean±std={r['c_v_mean']:.1f}±{r['c_v_std']:.1f}",
            fontsize=7,
        )
        ax.set_xlabel("Te [s]")
        ax.set_ylabel("Hs [m]")
        ax.grid(True)
    for idx in range(len(per_method), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)
    fig.subplots_adjust(right=0.87)
    if sc is not None:
        cax = fig.add_axes([0.89, 0.15, 0.02, 0.7])
        fig.colorbar(sc, cax=cax, label="c_v [N·s²/m²]")
    fig.suptitle("Viscous drag coefficient — Te × Hs")
    plt.tight_layout(rect=[0, 0, 0.87, 1])
    if save_path:
        Plotter._save(fig, save_path, "identify", "coeff_grid")


def plot_metric_grid(data, simruns_by_method, result, grids_by_method, save_path):
    """Te×Hs scatter colored by metric — all methods as subplots in one figure."""
    methods = list(result["results"].keys())
    n = len(methods)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols

    per_method = {}
    for method, r in result["results"].items():
        metric_map = r["metric_per_case"]
        metric_label = r["metric_label"]
        Te_arr, Hs_arr, J_arr, m_arr = [], [], [], []
        for obj in data:
            if obj.label not in metric_map:
                continue
            Te, Hs, J = _wave_params(obj)
            if Te is None:
                continue
            Te_arr.append(Te)
            Hs_arr.append(Hs)
            J_arr.append(J)
            m_arr.append(metric_map[obj.label])
        if Te_arr:
            per_method[method] = dict(
                Te=np.array(Te_arr),
                Hs=np.array(Hs_arr),
                m=np.array(m_arr),
                label=metric_label,
                global_v=r["metric_global"],
            )

    if not per_method:
        return

    all_m = [v for d in per_method.values() for v in d["m"]]
    vmin, vmax = np.nanmin(all_m), np.nanmax(all_m)
    if vmin == vmax:
        vmin -= 0.5
        vmax += 0.5
    metric_label = next(iter(per_method.values()))["label"]

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False
    )
    sc = None
    for idx, method in enumerate(per_method):
        ax = axes[idx // ncols][idx % ncols]
        d = per_method[method]
        sc = ax.scatter(
            d["Te"], d["Hs"], c=d["m"], s=60, cmap="viridis", vmin=vmin, vmax=vmax
        )
        ax.set_title(f"{method}  global={d['global_v']:.5f}", fontsize=7)
        ax.set_xlabel("Te [s]")
        ax.set_ylabel("Hs [m]")
        ax.grid(True)
    for idx in range(len(per_method), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)
    fig.subplots_adjust(right=0.87)
    if sc is not None:
        cax = fig.add_axes([0.89, 0.15, 0.02, 0.7])
        fig.colorbar(sc, cax=cax, label=metric_label)
    fig.suptitle(f"{metric_label} — Te × Hs")
    plt.tight_layout(rect=[0, 0, 0.87, 1])
    if save_path:
        Plotter._save(fig, save_path, "identify", "metric_grid")


def plot_coeff_distribution(
    data, simruns_by_method, result, grids_by_method, save_path
):
    """Histogram of per-case c_v — all methods as subplots in one figure."""
    methods = list(result["results"].keys())
    n = len(methods)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False
    )
    for idx, method in enumerate(methods):
        r = result["results"][method]
        ax = axes[idx // ncols][idx % ncols]
        vals = list(r["c_v_per_case"].values())
        ax.hist(
            np.array(vals), bins=max(5, len(vals) // 2 + 1), density=False, alpha=0.8
        )
        ax.axvline(
            r["c_v_global"], color="red", ls="--", label=f"global={r['c_v_global']:.1f}"
        )
        ax.set_title(
            f"{method}\nmean={r['c_v_mean']:.1f} std={r['c_v_std']:.1f}", fontsize=7
        )
        ax.set_xlabel("c_v [N·s²/m²]")
        ax.set_ylabel("count")
        ax.legend(fontsize=7)
        ax.grid(True)
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)
    fig.suptitle("Viscous drag coefficient distribution")
    plt.tight_layout()
    if save_path:
        Plotter._save(fig, save_path, "identify", "coeff_distribution")


def plot_metric_distribution(
    data, simruns_by_method, result, grids_by_method, save_path
):
    """Histogram of per-case metric — all methods as subplots in one figure."""
    methods = list(result["results"].keys())
    n = len(methods)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False
    )
    for idx, method in enumerate(methods):
        r = result["results"][method]
        ax = axes[idx // ncols][idx % ncols]
        vals = list(r["metric_per_case"].values())
        metric_label = r["metric_label"]
        ax.hist(
            np.array(vals), bins=max(5, len(vals) // 2 + 1), density=False, alpha=0.8
        )
        ax.axvline(
            r["metric_global"],
            color="red",
            ls="--",
            label=f"global={r['metric_global']:.5f}",
        )
        ax.set_title(f"{method}  mean={np.mean(vals):.5f}", fontsize=7)
        ax.set_xlabel(metric_label)
        ax.set_ylabel("count")
        ax.legend(fontsize=7)
        ax.grid(True)
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)
    fig.suptitle("Metric distribution")
    plt.tight_layout()
    if save_path:
        Plotter._save(fig, save_path, "identify", "metric_distribution")


def plot_timeseries(data, simruns_by_method, result, grids_by_method, save_path):
    """One plot per case: ref (black) + base (no drag) + one line per method."""
    base_simruns = simruns_by_method.get("__base__", {})

    for obj in data:
        label = obj.label

        # use simrun time as the valid window (already clipped once in Optimizer)
        sr_any = None
        for method in result["results"]:
            sr_any = simruns_by_method.get(method, {}).get("per_case", {}).get(label)
            if sr_any is not None:
                break
        if sr_any is None:
            continue

        t_lo = float(sr_any.dataset["t"].iloc[0])
        t_hi = float(sr_any.dataset["t"].iloc[-1])
        fig, ax = plt.subplots(figsize=(10, 4))

        # CFD reference — matched to simrun window (no double clip)
        if "x" in obj.dataset.columns:
            t_c = obj.dataset["t"].to_numpy()
            x_c = obj.dataset["x"].to_numpy()
            mk = (t_c >= t_lo) & (t_c <= t_hi)
            ax.plot(t_c[mk], x_c[mk], color="black", lw=1.5, label="CFD ref")

        # base (no drag) — plot as-is (already clipped)
        if label in base_simruns:
            bs = base_simruns[label]
            ax.plot(
                bs.dataset["t"].to_numpy(),
                bs.dataset["x"].to_numpy(),
                ls="--",
                label="base (no drag)",
            )

        # each method — plot as-is (already clipped)
        for method, r in result["results"].items():
            sr = simruns_by_method.get(method, {}).get("per_case", {}).get(label)
            if sr is None or "x" not in sr.dataset.columns:
                continue
            cv = r["c_v_per_case"].get(label, 0)
            mv = r["metric_per_case"].get(label, 0)
            ax.plot(
                sr.dataset["t"].to_numpy(),
                sr.dataset["x"].to_numpy(),
                label=f"{method}  c_v={cv:.1f}  {r['metric_label']}={mv:.5f}",
            )

        ax.set_title(label)
        ax.set_xlabel("t [s]")
        ax.set_ylabel("x [m]")
        ax.legend(fontsize=7)
        ax.grid(True)
        plt.tight_layout()
        if save_path:
            Plotter._save(fig, save_path, None, f"{_safe(label)[:50]}")


# %% PLOT DISPATCH

PLOT_DISPATCH = {
    "cost_curves": plot_cost_curves,
    "coeff_grid": plot_coeff_grid,
    "metric_grid": plot_metric_grid,
    "coeff_distribution": plot_coeff_distribution,
    "metric_distribution": plot_metric_distribution,
    "timeseries": plot_timeseries,
}
