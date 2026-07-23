import copy
import json
import logging
import re
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import minimize
from tqdm import tqdm

from source.config import DATA_DIR, MODEL_DIR
from source.classes.DataHandle import DataHandle
from source.classes.Model import Model, decompose_wave
from source.classes.Objective import Objective, OBJECTIVES
from source.classes.Plotter import Plotter
from source.classes.SimRun import SimRun, get_solver

try:
    from source.classes.Wave import Wave

    _WAVE_AVAILABLE = True
except ImportError:
    _WAVE_AVAILABLE = False

"""
# -------------------------------------------------------------------------
# Name:            Pi_gain.py
# Description:     Scheme: pi_gain.
#                   Identify optimal PI controller gains (B_pto, K_pto)
#                   that maximise an objective for each sea state individually.
#
#                   Three methods:
#                     analytical -- complex conjugate control from BEM data
#                     opt        -- L-BFGS-B per case
#                     grid       -- 2D grid search over (B, K) per case
#
#                   Stability check: stops simulation early if the buoy
#                   displacement relative to the free surface exceeds r.
#
# Author:          Antoine
# Collaborator:    Bona, Bruno, Edoardo
# Date created:    07/2026
# Project:         wec_modeling_benchmark
# -------------------------------------------------------------------------
"""

log = logging.getLogger(__name__)
SELF_LOADING = True


# %% HELPERS


def _safe(text: str) -> str:
    return re.sub(r"[^\w\-]+", "_", text).strip("_")


def _methods_from_params(sp: dict) -> list:
    """
    method: "analytical" | "opt" | "grid" | list of the above
    """
    methods = sp.get("method", "opt")
    if isinstance(methods, str):
        methods = [methods]
    valid = {"analytical", "opt", "grid"}
    invalid = set(methods) - valid
    if invalid:
        raise ValueError(f"Unknown PI gain method(s) {invalid}. Available: {valid}")
    return methods


def _make_early_stop(r: float):
    """Stop if |x - eta| > r (buoy fully submerged or out of water)."""

    def fn(t, X, eta_val):
        return abs(X[0] - eta_val) > r

    return fn


def _compute_power(simrun, B_pto: float) -> float:
    """Mean absorbed power: P_abs = B_pto * xdot^2."""
    xdot = simrun.dataset["xdot"].to_numpy()
    if not (np.all(np.isfinite(xdot)) and np.max(np.abs(xdot)) < 1e4):
        return -np.inf
    return float(np.mean(B_pto * xdot**2))


def _eval_grid_point(
    B: float,
    K: float,
    case: dict,
    model_def: dict,
    hydro_sphere,
    solver_cfg: dict,
    interp_cfg: dict,
    stab_r: float,
    stab_on: bool,
) -> float:
    """
    Module-level worker for parallel grid search. Must be picklable — no
    lambdas, no Optimizer instance. Reconstructs what _run_sim_pto needs.
    Returns P_abs or np.nan if unstable.
    """
    early_stop_fn = _make_early_stop(stab_r) if stab_on else None

    md = copy.deepcopy(model_def)
    eta_t = case["eta_t"]
    eta_values = case["eta_values"]
    ft = md.get("force_terms", {})
    wc = (
        decompose_wave(eta_values, eta_t, n_max=50)
        if ft.get("excitation") == "fk_nonlinear"
        else None
    )

    model = Model.from_config(
        md,
        hydro_sphere,
        eta_t,
        eta_values,
        wave_components=wc,
        pto={"damping": B, "stiffness": K},
        interpolation=interp_cfg,
    )
    sr = SimRun.simulate(
        model,
        get_solver(solver_cfg["method"]),
        t0=float(eta_t[0]),
        t_end=float(eta_t[-1]),
        dt=solver_cfg["dt"],
        eta_t=eta_t,
        eta_values=eta_values,
        label=f"B={B:.0f} K={K:.0f}",
        early_stop_fn=early_stop_fn,
    )
    expected = int(round((eta_t[-1] - eta_t[0]) / solver_cfg["dt"])) + 1
    if len(sr.dataset) < expected * 0.5:
        return np.nan
    return _compute_power(sr, B)


# %% ANALYTICAL GAINS


def _analytical_gains(opt, case: dict) -> tuple:
    """
    Complex conjugate control optimal gains using HydroSphere.pi_gain(T).
    Kp = B_rad(ωe)  →  B_pto
    Ki = K_hs - ωe²·(m + ma(ωe))  →  K_pto
    """
    Te = case.get("Te")
    if Te is None:
        raise ValueError(
            "analytical method requires Te in wave case. "
            "Ensure wave_params are set on DataHandle objects."
        )
    Kp, Ki = opt.hydro_sphere.pi_gain(float(Te))
    return float(Kp), float(Ki)  # Kp → B,  Ki → K


# %% WAVE SOURCE LOADING


def _load_cfd_case(case_name: str) -> list:
    root = DATA_DIR / "handled" / case_name
    objs = DataHandle.load_case_in(root / "data", json_path=root / f"{case_name}.json")
    cases = []
    for obj in objs:
        ds = obj.dataset
        wp = (
            obj.wave_params
            if hasattr(obj, "wave_params") and obj.wave_params is not None
            else None
        )
        cases.append(
            {
                "label": obj.label,
                "eta_t": ds["t"].to_numpy(),
                "eta_values": ds["eta"].to_numpy(),
                "Te": float(wp.Te.iloc[0]) if wp is not None else None,
                "Hs": float(wp.Hs.iloc[0]) if wp is not None else None,
                "J": float(wp.J.iloc[0]) if wp is not None else None,
                "ref_obj": obj,  # DataHandle — provides CFD x for timeseries
            }
        )
    return cases


def _load_synthetic(ws: dict, solver_cfg: dict) -> list:
    if not _WAVE_AVAILABLE:
        raise ImportError("Wave class not available.")
    wave = Wave(
        wave_type=ws.get("wave_type", "reg"),
        T=ws.get("T", 7.0),
        H=ws.get("H", 2.0),
        duration=ws.get("duration", 120.0),
        dt=solver_cfg.get("dt", 0.05),
        d=ws.get("d", 200.0),
    )
    return [
        {
            "label": f"synthetic T={ws.get('T')} H={ws.get('H')}",
            "eta_t": wave.t,
            "eta_values": wave.eta,
            "Te": ws.get("T"),
            "Hs": ws.get("H"),
            "J": None,
        }
    ]


def _load_wave_sources(waves_train: list, solver_cfg: dict) -> list:
    cases = []
    for ws in waves_train:
        src_type = ws.get("type", "cfd_case")
        if src_type == "cfd_case":
            loaded = _load_cfd_case(ws["case_name"])
        elif src_type == "synthetic":
            loaded = _load_synthetic(ws, solver_cfg)
        else:
            raise ValueError(f"Unknown wave source type: '{src_type}'")
        cases.extend(loaded)
        log.info("Loaded %d case(s) from '%s'", len(loaded), src_type)
    return cases


# %% SIMULATION HELPER


def _run_sim_pto(
    opt, B: float, K: float, case: dict, interp_cfg: dict, early_stop_fn=None
):
    """Run physics model with PTO (B, K). Returns None if unstable."""
    md = copy.deepcopy(opt.model_def)
    eta_t = case["eta_t"]
    eta_values = case["eta_values"]
    ft = md.get("force_terms", {})
    wc = (
        decompose_wave(eta_values, eta_t, n_max=opt.n_wave_components)
        if ft.get("excitation") == "fk_nonlinear"
        else None
    )

    model = Model.from_config(
        md,
        opt.hydro_sphere,
        eta_t,
        eta_values,
        wave_components=wc,
        pto={"damping": B, "stiffness": K},
        interpolation=interp_cfg,
    )
    sr = SimRun.simulate(
        model,
        get_solver(opt.solver_cfg["method"]),
        t0=float(eta_t[0]),
        t_end=float(eta_t[-1]),
        dt=opt.solver_cfg["dt"],
        eta_t=eta_t,
        eta_values=eta_values,
        label=f"{case['label']} | B={B:.0f} K={K:.0f}",
        early_stop_fn=early_stop_fn,
    )
    expected = int(round((eta_t[-1] - eta_t[0]) / opt.solver_cfg["dt"])) + 1
    return None if len(sr.dataset) < expected * 0.5 else sr


# %% DDFEED — DATA PREPARATION


def prep(feed):
    feed.cases = _load_wave_sources(
        feed.params["waves_train"], feed.params.get("solver", {})
    )
    feed.data = feed.cases  # align with generic dispatch in Optimizer.run_plots


# %% OPTIMIZER — RUN


def run(opt, feed, config):
    """
    Identify optimal PI gains (B_pto, K_pto) per sea state.
    No global identification — each case has its own optimal gains.

    Methods (scheme_params.method):
      analytical -- complex conjugate control from BEM data (no simulation)
      opt        -- L-BFGS-B per case
      grid       -- 2D grid search B × K per case
    """
    sp = config.scheme_params
    methods = _methods_from_params(sp)
    B_min = sp.get("B_min", 0.0)
    B_max = sp.get("B_max", 1e6)
    K_min = sp.get("K_min", -1e6)
    K_max = sp.get("K_max", 1e6)
    n_B = sp.get("n_B", 20)
    n_K = sp.get("n_K", 20)
    B_init = sp.get("B_init", (B_min + B_max) / 2)
    K_init = sp.get("K_init", (K_min + K_max) / 2)
    interp_cfg = config.solver.get("interpolation", {})
    objective = Objective.from_dict(
        sp.get("objective", {"name": "mean_P_abs", "direction": "maximize"})
    )

    stab_cfg = sp.get("stability", {})
    stab_on = stab_cfg.get("enabled", False)
    stab_r = stab_cfg.get("r", getattr(opt.hydro_sphere, "r", 5.0))
    early_stop_fn = _make_early_stop(stab_r) if stab_on else None

    if not feed.cases:
        raise RuntimeError("No wave cases loaded.")

    results = {}
    grids_by_method = {}
    simruns_by_method = {}

    for method in methods:
        tqdm.write(f"\n{'=' * 60}\n  Method: {method}\n{'=' * 60}")
        log.info("--- Method: %s ---", method)
        per_case = {}
        grids = {}
        simruns = {}

        for case in tqdm(feed.cases, desc=f"[{method}] per-case", unit="case"):
            label = case["label"]
            t0 = time.perf_counter()

            if method == "analytical":
                B_opt, K_opt = _analytical_gains(opt, case)
                elapsed_id = time.perf_counter() - t0
                sr = _run_sim_pto(opt, B_opt, K_opt, case, interp_cfg, early_stop_fn)
                power = _compute_power(sr, B_opt) if sr is not None else np.nan

            elif method == "opt":
                call_count = [0]

                def _obj(params, _case=case):
                    call_count[0] += 1
                    B, K = params
                    sr_ = _run_sim_pto(opt, B, K, _case, interp_cfg, early_stop_fn)
                    return (
                        np.inf
                        if sr_ is None
                        else objective.sign() * _compute_power(sr_, B)
                    )

                res = minimize(
                    _obj,
                    x0=[B_init, K_init],
                    method="L-BFGS-B",
                    bounds=[(B_min, B_max), (K_min, K_max)],
                    options={"maxiter": 100, "ftol": 1e-9},
                )
                B_opt, K_opt = res.x
                elapsed_id = time.perf_counter() - t0
                sr = _run_sim_pto(opt, B_opt, K_opt, case, interp_cfg, early_stop_fn)
                power = _compute_power(sr, B_opt) if sr is not None else np.nan
                tqdm.write(
                    f"  {label}  B={B_opt:.0f} K={K_opt:.0f} "
                    f"P={power:.1f}W  ({call_count[0]} evals)"
                )
                log.info(
                    "    %s  B=%.0f  K=%.0f  P=%.1fW  evals=%d  elapsed=%.1fs",
                    label,
                    B_opt,
                    K_opt,
                    power,
                    call_count[0],
                    elapsed_id,
                )

            elif method == "grid":
                B_grid = np.linspace(B_min, B_max, n_B)
                K_grid = np.linspace(K_min, K_max, n_K)
                n_jobs = sp.get("n_jobs", 1)

                pairs = [(B, K) for B in B_grid for K in K_grid]

                if n_jobs != 1:
                    tqdm.write(f"  grid: {len(pairs)} points, n_jobs={n_jobs}")
                    results_flat = Parallel(n_jobs=n_jobs)(
                        delayed(_eval_grid_point)(
                            B,
                            K,
                            case,
                            opt.model_def,
                            opt.hydro_sphere,
                            opt.solver_cfg,
                            interp_cfg,
                            stab_r,
                            stab_on,
                        )
                        for B, K in pairs
                    )
                else:
                    results_flat = [
                        _eval_grid_point(
                            B,
                            K,
                            case,
                            opt.model_def,
                            opt.hydro_sphere,
                            opt.solver_cfg,
                            interp_cfg,
                            stab_r,
                            stab_on,
                        )
                        for B, K in tqdm(pairs, desc="grid", leave=False)
                    ]

                costs = np.array(results_flat).reshape(n_B, n_K)
                idx = np.unravel_index(np.nanargmax(costs), costs.shape)
                B_opt = float(B_grid[idx[0]])
                K_opt = float(K_grid[idx[1]])
                elapsed_id = time.perf_counter() - t0
                grids[label] = {
                    "B_grid": B_grid,
                    "K_grid": K_grid,
                    "costs": costs,
                    "B_opt": B_opt,
                    "K_opt": K_opt,
                }
                sr = _run_sim_pto(opt, B_opt, K_opt, case, interp_cfg, early_stop_fn)
                power = _compute_power(sr, B_opt) if sr is not None else np.nan

            if sr is not None:
                # clip to valid window before storing and saving
                if opt.t_warmup > 0 or opt.t_causal > 0:
                    sr = sr.clip(opt.t_warmup, opt.t_causal)
                simruns[label] = sr

            per_case[label] = {
                "B": float(B_opt),
                "K": float(K_opt),
                "metric": power,
                "identification_elapsed_s": elapsed_id,
            }
            tqdm.write(f"  {label}  B={B_opt:.0f}  K={K_opt:.0f}  P_abs={power:.1f} W")
            log.info(
                "    %s  B=%.0f  K=%.0f  P_abs=%.1fW  elapsed=%.1fs",
                label,
                B_opt,
                K_opt,
                power,
                elapsed_id,
            )

        results[method] = {
            "B_per_case": {l: per_case[l]["B"] for l in per_case},
            "K_per_case": {l: per_case[l]["K"] for l in per_case},
            "metric_per_case": {l: per_case[l]["metric"] for l in per_case},
            "metric_label": objective.label(),
            "identification_elapsed_s": {
                l: per_case[l]["identification_elapsed_s"] for l in per_case
            },
        }
        grids_by_method[method] = grids
        simruns_by_method[method] = simruns

    result = {
        "scheme": "pi_gain",
        "base_model": config.base_model,
        "objective": {"name": objective.name, "direction": objective.direction},
        "stability": {"enabled": stab_on, "r": stab_r},
        "n_cases": len(feed.cases),
        "results": results,
    }
    return result, grids_by_method, simruns_by_method


# %% OPTIMIZER — SAVE


def save(opt, result, output_name):
    """Create output folder. All results are in result.json — no separate file."""
    out_dir = MODEL_DIR / output_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output folder ready: %s", out_dir)


def log_result(result):
    for m, r in result["results"].items():
        mean_P = np.nanmean(list(r["metric_per_case"].values()))
        log.info("[%s] mean %s = %.2f W", m, r["metric_label"], mean_P)


# %% SAVE DATA


def save_data(result, simruns_by_method, grids_by_method, feed, data_dir):
    """
    Per case: one CSV with all methods in columns.
    Also: B×K grid CSV per case (grid), sea state grid CSV per method.
    """
    # ---- per-case CSV (all methods together) ----
    for case in feed.cases:
        label = case["label"]
        t_ref = None
        for simruns in simruns_by_method.values():
            if label in simruns:
                t_ref = simruns[label].dataset["t"].to_numpy()
                break
        if t_ref is None:
            continue

        eta = np.interp(t_ref, case["eta_t"], case["eta_values"])
        df = pd.DataFrame({"t": t_ref, "eta": eta})

        for method, r in result["results"].items():
            simruns = simruns_by_method[method]
            if label not in simruns:
                continue
            sr = simruns[label]
            t_s = sr.dataset["t"].to_numpy()
            x = np.interp(t_ref, t_s, sr.dataset["x"].to_numpy())
            xdot = np.interp(t_ref, t_s, sr.dataset["xdot"].to_numpy())
            B = r["B_per_case"].get(label, 0.0)
            df[f"x_{method}"] = x
            df[f"xdot_{method}"] = xdot
            df[f"x_minus_eta_{method}"] = x - eta
            df[f"P_abs_{method}"] = B * xdot**2

        df.to_csv(data_dir / f"{_safe(label)[:60]}.csv", index=False)
        log.info("Saved: %s.csv", _safe(label)[:60])

    # ---- B×K grid per case (grid method only) ----
    for label, g in grids_by_method.get("grid", {}).items():
        B_flat, K_flat, P_flat, stable_flat = [], [], [], []
        for i, B in enumerate(g["B_grid"]):
            for j, K in enumerate(g["K_grid"]):
                p = g["costs"][i, j]
                B_flat.append(B)
                K_flat.append(K)
                P_flat.append(p)
                stable_flat.append(not np.isnan(p))
        pd.DataFrame(
            {"B": B_flat, "K": K_flat, "P_abs": P_flat, "stable": stable_flat}
        ).to_csv(data_dir / f"grid_BK_power_{_safe(label)[:50]}.csv", index=False)
        log.info("Saved: grid_BK_power_%s.csv", _safe(label)[:50])

    # ---- sea state grid per method ----
    for method, r in result["results"].items():
        rows = []
        for case in feed.cases:
            label = case["label"]
            if label not in r["B_per_case"]:
                continue
            rows.append(
                {
                    "label": label,
                    "Te": case.get("Te"),
                    "Hs": case.get("Hs"),
                    "J": case.get("J"),
                    "B": r["B_per_case"][label],
                    "K": r["K_per_case"][label],
                    "P_abs": r["metric_per_case"][label],
                }
            )
        if rows:
            pd.DataFrame(rows).to_csv(
                data_dir / f"{method}_sea_state_grid.csv", index=False
            )
            log.info("Saved: %s_sea_state_grid.csv", method)


# %% PLOTS


def plot_timeseries(data, simruns_by_method, result, grids_by_method, save_path):
    """One plot per case — CFD ref (black) + all methods.
    Simruns are already clipped to valid window. Reference is matched to simrun range.
    """
    for case in data:
        label = case["label"]

        # use simrun time as the valid window (already clipped once)
        sr_any = None
        for simruns in simruns_by_method.values():
            if label in simruns:
                sr_any = simruns[label]
                break
        if sr_any is None:
            continue

        t_lo = float(sr_any.dataset["t"].iloc[0])
        t_hi = float(sr_any.dataset["t"].iloc[-1])
        fig, ax = plt.subplots(figsize=(10, 4))

        # CFD reference clipped to simrun window
        ref_obj = case.get("ref_obj")
        if ref_obj is not None and "x" in ref_obj.dataset.columns:
            t_cfd = ref_obj.dataset["t"].to_numpy()
            x_cfd = ref_obj.dataset["x"].to_numpy()
            mk = (t_cfd >= t_lo) & (t_cfd <= t_hi)
            ax.plot(t_cfd[mk], x_cfd[mk], color="black", lw=1.5, label="CFD ref")

        # simulated results — plot as-is (already clipped)
        for method, simruns in simruns_by_method.items():
            if label not in simruns:
                continue
            sr = simruns[label]
            r = result["results"].get(method, {})
            B = r.get("B_per_case", {}).get(label, 0)
            K = r.get("K_per_case", {}).get(label, 0)
            P = r.get("metric_per_case", {}).get(label, 0)
            ax.plot(
                sr.dataset["t"].to_numpy(),
                sr.dataset["x"].to_numpy(),
                label=f"{method}  B={B:.0f} K={K:.0f} P={P:.1f} W",
            )

        ax.set_title(label)
        ax.set_xlabel("t [s]")
        ax.set_ylabel("x [m]")
        ax.legend(fontsize=7)
        ax.grid(True)
        plt.tight_layout()
        if save_path:
            Plotter._save(fig, save_path, None, f"{_safe(label)[:50]}")


def plot_relative_position_dist(
    data, simruns_by_method, result, grids_by_method, save_path
):
    """All cases as subplots — histogram of (x - η) per case, all methods overlaid."""
    n = len(data)
    if n == 0:
        return
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(simruns_by_method), 1)))
    r_val = result.get("stability", {}).get("r")

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False
    )

    for idx, case in enumerate(data):
        ax = axes[idx // ncols][idx % ncols]
        label = case["label"]
        for m_idx, (method, simruns) in enumerate(simruns_by_method.items()):
            if label not in simruns:
                continue
            sr = simruns[label]
            t = sr.dataset["t"].to_numpy()
            eta = np.interp(t, case["eta_t"], case["eta_values"])
            rel = sr.dataset["x"].to_numpy() - eta
            ax.hist(
                rel,
                bins=30,
                density=False,
                alpha=0.5,
                color=colors[m_idx],
                label=method,
            )
        if r_val is not None:
            ax.axvline(-r_val, color="red", ls="--", lw=1, label=f"±r={r_val}")
            ax.axvline(r_val, color="red", ls="--", lw=1)
        ax.set_title(label[:35], fontsize=8)
        ax.set_xlabel("x - η [m]")
        ax.set_ylabel("count")
        ax.legend(fontsize=7)
        ax.grid(True)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("Relative position distribution")
    plt.tight_layout()
    if save_path:
        Plotter._save(fig, save_path, "identify", "rel_pos")


def plot_gains_grid(data, simruns_by_method, result, grids_by_method, save_path):
    """Power landscape — all cases as subplots.
    B_grid and K_grid are 1D linspace arrays; costs has shape (n_B, n_K).
    Handles 2D heatmap, 1D-over-B, and 1D-over-K automatically.
    """
    grids = grids_by_method.get("grid", {})
    if not grids:
        return

    labels = list(grids.keys())
    n = len(labels)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False
    )

    for idx, label in enumerate(labels):
        ax = axes[idx // ncols][idx % ncols]
        g = grids[label]
        B_vec = np.array(g["B_grid"])  # 1D, length n_B
        K_vec = np.array(g["K_grid"])  # 1D, length n_K
        costs = np.array(g["costs"])  # shape (n_B, n_K)
        n_B = len(B_vec)
        n_K = len(K_vec)

        if n_B > 1 and n_K > 1:
            # 2D heatmap — needs meshgrid for pcolormesh
            KK, BB = np.meshgrid(K_vec, B_vec)  # both (n_B, n_K)
            im = ax.pcolormesh(KK, BB, costs, shading="auto", cmap="viridis")
            plt.colorbar(im, ax=ax, label="P_abs [W]")
            ax.scatter(
                [g["K_opt"]],
                [g["B_opt"]],
                color="red",
                marker="*",
                s=150,
                label=f"opt B={g['B_opt']:.0f} K={g['K_opt']:.0f}",
            )
            ax.set_xlabel("K [N/m]")
            ax.set_ylabel("B [N·s/m]")
            ax.legend(fontsize=7)

        elif n_K == 1:
            # only B varies
            ax.plot(B_vec, costs[:, 0])
            ax.axvline(
                g["B_opt"], color="red", ls="--", label=f"B_opt={g['B_opt']:.0f}"
            )
            ax.set_xlabel("B [N·s/m]")
            ax.set_ylabel("P_abs [W]")
            ax.legend(fontsize=7)

        else:
            # only K varies
            ax.plot(K_vec, costs[0, :])
            ax.axvline(
                g["K_opt"], color="red", ls="--", label=f"K_opt={g['K_opt']:.0f}"
            )
            ax.set_xlabel("K [N/m]")
            ax.set_ylabel("P_abs [W]")
            ax.legend(fontsize=7)

        ax.set_title(label[:35], fontsize=8)
        ax.grid(True)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("Power landscape")
    plt.tight_layout()
    if save_path:
        Plotter._save(fig, save_path, "identify", "gains_grid")


def plot_gains_scatter(data, simruns_by_method, result, grids_by_method, save_path):
    """Te×Hs scatter colored by B or K — one figure per variable, subplot per method."""
    methods = [m for m in result["results"] if "B_per_case" in result["results"][m]]
    if not methods:
        return

    # collect per-method data
    per_method = {}
    for method in methods:
        r = result["results"][method]
        Te_arr, Hs_arr, J_arr, B_arr, K_arr = [], [], [], [], []
        for case in data:
            label = case["label"]
            if label not in r.get("B_per_case", {}):
                continue
            Te, Hs, J = case.get("Te"), case.get("Hs"), case.get("J")
            if Te is None or Hs is None:
                continue
            Te_arr.append(Te)
            Hs_arr.append(Hs)
            J_arr.append(J or 1.0)
            B_arr.append(r["B_per_case"][label])
            K_arr.append(r["K_per_case"][label])
        if Te_arr:
            per_method[method] = {
                "Te": np.array(Te_arr),
                "Hs": np.array(Hs_arr),
                "J": np.array(J_arr),
                "B": np.array(B_arr),
                "K": np.array(K_arr),
            }

    if not per_method:
        return

    n = len(per_method)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols

    for var, clabel, suffix in [
        ("B", "B [N·s/m]", "scatter_B"),
        ("K", "K [N/m]", "scatter_K"),
    ]:
        all_vals = [v for m in per_method.values() for v in m[var]]
        if not all_vals:
            continue
        vmin, vmax = np.nanmin(all_vals), np.nanmax(all_vals)
        if vmin == vmax:
            vmin -= 0.5
            vmax += 0.5

        fig, axes = plt.subplots(
            nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False
        )
        sc = None
        for idx, (method, d) in enumerate(per_method.items()):
            ax = axes[idx // ncols][idx % ncols]
            sc = ax.scatter(
                d["Te"], d["Hs"], c=d[var], s=60, cmap="viridis", vmin=vmin, vmax=vmax
            )
            ax.set_xlabel("Te [s]")
            ax.set_ylabel("Hs [m]")
            ax.set_title(method, fontsize=8)
            ax.grid(True)

        for idx in range(n, nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        # place colorbar to the right of the whole figure without stealing subplot space
        fig.subplots_adjust(right=0.87)
        cax = fig.add_axes([0.89, 0.15, 0.02, 0.7])
        if sc is not None:
            fig.colorbar(sc, cax=cax, label=clabel)
        fig.suptitle(f"Optimal {var} — Te × Hs")
        plt.tight_layout()
        if save_path:
            Plotter._save(fig, save_path, "identify", suffix)


# %% PLOT DISPATCH

PLOT_DISPATCH = {
    "timeseries": plot_timeseries,
    "relative_position_dist": plot_relative_position_dist,
    "gains_grid": plot_gains_grid,
    "gains_scatter": plot_gains_scatter,
}
