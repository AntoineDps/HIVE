import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from source.function.error_utils import metricError, cross_correlation, inst_phase_diff
from source.classes.Plotter import Plotter

"""
# -------------------------------------------------------------------------
# Name:            Metric.py
# Description:     Hardcoded metric set for WEC model validation.
#
# Scalar metrics (per case CSV, rows=models, cols=metrics):
#   nrmse_range / nrmse_true for x, xdot, f_pto, P_abs
#   rel_error_mean_P_abs
#   error_min / error_max for f_pto, P_abs
#   peak_error for x, xdot, f_pto, P_abs
#   time_ratio
#   dist_mean_error / dist_mae / dist_std / dist_var for x, xdot, f_pto, P_abs
#   phase_lag_s / corr_peak for x, xdot
#   phase_mean_deg / phase_std_deg for x, xdot
#
# 1D CSVs per case:
#   cross_corr_{group}.csv  — lag_s, ref_x, ref_xdot + {model}_x + {model}_xdot cols
#   inst_phase_{group}.csv  — t + {model}_x_deg + {model}_xdot_deg
#
# Summary CSV: rows=models, cols={metric}_{mean|std|min|max}
#
# Plot options (non-bool values):
#   metric_corr   : true
#   metric_phase  : true
#   metric_scatter: {"metrics": ["nrmse_range_x"]}
#   metric_summary: {"metrics": ["nrmse_range_x"], "stat": "mean"}
#   metric_per_case: {"metrics": ["nrmse_range_x"]}
#
# Author:          Antoine
# Collaborator:    Bona, Bruno, Edoardo
# Date created:    07/2026
# -------------------------------------------------------------------------
"""

log = logging.getLogger(__name__)


def _log_table(rows: list, headers: list, width: int = 14):
    """Log a fixed-width table. rows is a list of lists."""
    fmt = "  ".join(f"{{:<{width}}}" for _ in headers)
    sep = "  ".join("-" * width for _ in headers)
    log.info(fmt.format(*headers))
    log.info(sep)
    for row in rows:
        log.info(fmt.format(*[str(v)[:width] for v in row]))


# %% HELPERS


def _safe(text: str) -> str:
    return re.sub(r"[^\w\-]+", "_", text).strip("_")


def _trim(arr: np.ndarray, t: np.ndarray, t_s: float, t_e: float) -> tuple:
    mask = (t >= t_s) & (t <= t_e)
    return arr[mask], t[mask]


def _to_ref(arr_sim: np.ndarray, t_sim: np.ndarray, t_ref: np.ndarray) -> np.ndarray:
    return np.interp(t_ref, t_sim, arr_sim)


def _get_f_pto(dataset: pd.DataFrame, damping: float, stiffness: float) -> np.ndarray:
    """f_pto from SimRun (f_pto column) or DataHandle (derived)."""
    if "f_pto" in dataset.columns:
        return dataset["f_pto"].to_numpy()
    return -(damping * dataset["xdot"].to_numpy() + stiffness * dataset["x"].to_numpy())


def _get_P_abs(dataset: pd.DataFrame, damping: float) -> np.ndarray:
    """P_abs from SimRun (P_abs column) or DataHandle (derived)."""
    if "P_abs" in dataset.columns:
        return dataset["P_abs"].to_numpy()
    return damping * dataset["xdot"].to_numpy() ** 2


def _peak_error(y_sim: np.ndarray, y_ref: np.ndarray) -> float:
    pk_sim, _ = find_peaks(y_sim)
    pk_ref, _ = find_peaks(y_ref)
    if pk_sim.size == 0 or pk_ref.size == 0:
        return np.nan
    return float(np.mean(y_sim[pk_sim]) - np.mean(y_ref[pk_ref]))


def _corr_arrays(y_sim: np.ndarray, y_ref: np.ndarray, t_ref: np.ndarray) -> tuple:
    dt = float(np.mean(np.diff(t_ref)))
    return cross_correlation(y_sim, y_ref, dt)


def _inst_phase_array(y_sim: np.ndarray, y_ref: np.ndarray) -> np.ndarray:
    return inst_phase_diff(y_sim, y_ref)


# %% SCALAR COMPUTATION


def _compute_scalars(simrun, ref, t_s: float, t_e: float) -> dict:
    t_full = ref.dataset["t"].to_numpy()
    B, K = ref.damping, ref.stiffness

    def _r(arr):
        return _trim(arr, t_full, t_s, t_e)

    x_ref, t_ref = _r(ref.dataset["x"].to_numpy())
    xdot_ref, _ = _r(ref.dataset["xdot"].to_numpy())
    F_ref, _ = _r(_get_f_pto(ref.dataset, B, K))
    P_ref, _ = _r(_get_P_abs(ref.dataset, B))

    t_sim = simrun.dataset["t"].to_numpy()
    x_sim = _to_ref(simrun.dataset["x"].to_numpy(), t_sim, t_ref)
    xdot_sim = _to_ref(simrun.dataset["xdot"].to_numpy(), t_sim, t_ref)
    F_sim = _to_ref(_get_f_pto(simrun.dataset, B, K), t_sim, t_ref)
    P_sim = _to_ref(_get_P_abs(simrun.dataset, B), t_sim, t_ref)

    m = {}

    # NRMSE
    for tag, ys, yr in [
        ("x", x_sim, x_ref),
        ("xdot", xdot_sim, xdot_ref),
        ("f_pto", F_sim, F_ref),
        ("P_abs", P_sim, P_ref),
    ]:
        m[f"nrmse_range_{tag}"] = metricError(ys, yr, "nrmse_range")
        m[f"nrmse_true_{tag}"] = metricError(ys, yr, "nrmse_true")

    # relative error on mean power
    mean_P_ref = float(np.mean(np.abs(P_ref)))
    m["rel_error_mean_P_abs"] = (
        (float(np.mean(np.abs(P_sim))) - mean_P_ref) / mean_P_ref
        if mean_P_ref > 0
        else np.nan
    )

    # min/max errors
    for tag, ys, yr in [("f_pto", F_sim, F_ref), ("P_abs", P_sim, P_ref)]:
        m[f"error_min_{tag}"] = float(np.min(ys) - np.min(yr))
        m[f"error_max_{tag}"] = float(np.max(ys) - np.max(yr))

    # peak errors
    for tag, ys, yr in [
        ("x", x_sim, x_ref),
        ("xdot", xdot_sim, xdot_ref),
        ("f_pto", F_sim, F_ref),
        ("P_abs", P_sim, P_ref),
    ]:
        m[f"peak_error_{tag}"] = _peak_error(ys, yr)

    # time ratio
    # computational time ratio: physical_time / wall_clock_time
    # use the full reference duration (not clipped) for the physical time
    phys_s = float(t_full[-1] - t_full[0])
    if simrun.sim_time and simrun.sim_time > 0:
        m["time_ratio"] = phys_s / simrun.sim_time
    else:
        # sim_time unavailable (e.g. loaded from CSV) — mark as NaN
        m["time_ratio"] = np.nan
        log.debug(
            "time_ratio: sim_time not available for %s", getattr(simrun, "label", "?")
        )

    # error distribution
    for tag, ys, yr in [
        ("x", x_sim, x_ref),
        ("xdot", xdot_sim, xdot_ref),
        ("f_pto", F_sim, F_ref),
        ("P_abs", P_sim, P_ref),
    ]:
        err = ys - yr
        m[f"dist_mean_error_{tag}"] = float(np.mean(err))
        m[f"dist_mae_{tag}"] = float(np.mean(np.abs(err)))
        m[f"dist_std_{tag}"] = float(np.std(err))
        m[f"dist_var_{tag}"] = float(np.var(err))

    # cross-correlation scalars
    for tag, ys, yr in [("x", x_sim, x_ref), ("xdot", xdot_sim, xdot_ref)]:
        lag_s, corr = _corr_arrays(ys, yr, t_ref)
        pk = int(np.argmax(corr))
        m[f"phase_lag_s_{tag}"] = float(lag_s[pk])
        m[f"corr_peak_{tag}"] = float(corr[pk])

    # instantaneous phase scalars
    for tag, ys, yr in [("x", x_sim, x_ref), ("xdot", xdot_sim, xdot_ref)]:
        deg = _inst_phase_array(ys, yr)
        m[f"phase_mean_deg_{tag}"] = float(np.mean(deg))
        m[f"phase_std_deg_{tag}"] = float(np.std(deg))

    return m


# %% MAIN CLASS


class Metric:
    @staticmethod
    def load(metric_dir: Path) -> dict:
        """
        Reconstruct metric_results from previously saved CSVs.
        Used by option 4 (remake plots) — no recomputation.

        Returns the same structure as compute_all().
        """
        metric_dir = Path(metric_dir)
        results = {}

        def _parse_float(s: str) -> float:
            return float(s.replace("p", "."))

        def _wave_params_from_name(name: str) -> dict:
            """Extract Te, Hs from compact label e.g. Hs2p0_Te5p0_d120000_k-280000"""
            wp = {}
            for part in name.split("_"):
                if part.startswith("Hs"):
                    try:
                        wp["Hs"] = _parse_float(part[2:])
                    except ValueError:
                        pass
                elif part.startswith("Te"):
                    try:
                        wp["Te"] = _parse_float(part[2:])
                    except ValueError:
                        pass
            return wp or None

        # find all per-case scalar CSVs (exclude cross_corr, inst_phase, model_stats)
        scalar_files = [
            f
            for f in sorted(metric_dir.glob("*.csv"))
            if not f.stem.startswith(
                ("cross_corr_", "inst_phase_", "model_stats", "model_summary")
            )
        ]

        for scalar_path in scalar_files:
            group = scalar_path.stem  # e.g. Hs2p0_Te5p0_d120000_k-280000

            try:
                df_s = pd.read_csv(scalar_path, index_col="model")
                scalars = {row: df_s.loc[row].dropna().to_dict() for row in df_s.index}
            except Exception as e:
                log.warning("Metric.load: could not read %s: %s", scalar_path, e)
                continue

            # cross-correlation
            corr_data, corr_ref = {}, {}
            corr_path = metric_dir / f"cross_corr_{group}.csv"
            if corr_path.exists():
                try:
                    df_c = pd.read_csv(corr_path)
                    if "lag_s" in df_c.columns:
                        lag_s = df_c["lag_s"].to_numpy()
                        corr_ref = {
                            "lag_s": lag_s,
                            "x": df_c.get("ref_x", pd.Series()).to_numpy(),
                            "xdot": df_c.get("ref_xdot", pd.Series()).to_numpy(),
                        }
                        model_cols_x = [
                            c
                            for c in df_c.columns
                            if c.endswith("_x") and not c.startswith("ref")
                        ]
                        model_cols_xdot = [
                            c
                            for c in df_c.columns
                            if c.endswith("_xdot") and not c.startswith("ref")
                        ]
                        model_names = [c[:-2] for c in model_cols_x]
                        for mname, cx, cxd in zip(
                            model_names, model_cols_x, model_cols_xdot
                        ):
                            corr_data[mname] = {
                                "lag_s": lag_s,
                                "x": df_c[cx].to_numpy(),
                                "xdot": df_c[cxd].to_numpy(),
                            }
                except Exception as e:
                    log.warning("Metric.load: could not read %s: %s", corr_path, e)

            # instantaneous phase
            phase_data = {}
            phase_path = metric_dir / f"inst_phase_{group}.csv"
            if phase_path.exists():
                try:
                    df_p = pd.read_csv(phase_path)
                    t_arr = (
                        df_p["t"].to_numpy() if "t" in df_p.columns else np.array([])
                    )
                    x_cols = [c for c in df_p.columns if c.endswith("_x_deg")]
                    xdot_cols = [c for c in df_p.columns if c.endswith("_xdot_deg")]
                    for xc, xdc in zip(x_cols, xdot_cols):
                        mname = xc[: -len("_x_deg")]
                        phase_data[mname] = {
                            "t": t_arr,
                            "x_deg": df_p[xc].to_numpy(),
                            "xdot_deg": df_p[xdc].to_numpy(),
                        }
                except Exception as e:
                    log.warning("Metric.load: could not read %s: %s", phase_path, e)

            results[group] = {
                "scalars": scalars,
                "corr": corr_data,
                "corr_ref": corr_ref,
                "phase": phase_data,
                "wave_params": _wave_params_from_name(group),
            }
            log.info("Metric.load: loaded %s (%d model(s))", group, len(scalars))

        log.info("Metric.load: %d group(s) loaded from %s", len(results), metric_dir)
        return results

    @staticmethod
    def compute_all(groups: dict, t_warmup: float = 0.0, t_causal: float = 0.0) -> dict:
        results = {}
        for group, info in groups.items():
            ref = info.get("reference")
            if ref is None:
                log.warning("'%s' has no CFD reference — skipping", group)
                continue

            t_full = ref.dataset["t"].to_numpy()
            t_s = t_full[0] + t_warmup
            t_e = t_full[-1] - t_causal
            B = ref.damping

            x_ref, t_ref = _trim(ref.dataset["x"].to_numpy(), t_full, t_s, t_e)
            xdot_ref, _ = _trim(ref.dataset["xdot"].to_numpy(), t_full, t_s, t_e)

            # reference autocorrelations
            lag_x, ac_x = _corr_arrays(x_ref, x_ref, t_ref)
            lag_xd, ac_xd = _corr_arrays(xdot_ref, xdot_ref, t_ref)
            corr_ref = {
                "lag_s": lag_x,
                "x": ac_x,
                "xdot": ac_xd,
            }  # single lag_s for both

            scalars, corr_data, phase_data = {}, {}, {}

            for simrun in info["models"]:
                label = simrun.label
                try:
                    scalars[label] = _compute_scalars(simrun, ref, t_s, t_e)
                except Exception as e:
                    log.error("Metrics failed [%s / %s]: %s", group, label, e)
                    continue

                t_sim = simrun.dataset["t"].to_numpy()
                x_sim = _to_ref(simrun.dataset["x"].to_numpy(), t_sim, t_ref)
                xdot_sim = _to_ref(simrun.dataset["xdot"].to_numpy(), t_sim, t_ref)

                lag_x, cc_x = _corr_arrays(x_sim, x_ref, t_ref)
                lag_xd, cc_xd = _corr_arrays(xdot_sim, xdot_ref, t_ref)
                corr_data[label] = {
                    "lag_s": lag_x,
                    "x": cc_x,
                    "xdot": cc_xd,
                }

                phase_data[label] = {
                    "t": t_ref,
                    "x_deg": _inst_phase_array(x_sim, x_ref),
                    "xdot_deg": _inst_phase_array(xdot_sim, xdot_ref),
                }

            # extract wave params from the reference DataHandle
            wp = None
            if (
                ref is not None
                and hasattr(ref, "wave_params")
                and ref.wave_params is not None
            ):
                try:
                    wp = {
                        "Te": float(ref.wave_params.Te.iloc[0]),
                        "Hs": float(ref.wave_params.Hs.iloc[0]),
                        "J": float(ref.wave_params.J.iloc[0])
                        if "J" in ref.wave_params.columns
                        else None,
                    }
                except Exception:
                    pass

            results[group] = {
                "scalars": scalars,
                "corr": corr_data,
                "corr_ref": corr_ref,
                "phase": phase_data,
                "wave_params": wp,
            }
            log.info("Metrics computed: %s  (%d model(s))", group, len(scalars))

        # log summary table of key metrics across all groups
        if results:
            all_models = sorted({l for d in results.values() for l in d["scalars"]})
            key = [
                "nrmse_range_x",
                "nrmse_range_P_abs",
                "rel_error_mean_P_abs",
                "time_ratio",
            ]
            log.info("")
            log.info("=== METRIC SUMMARY (mean across cases) ===")
            headers = ["model"] + key
            rows = []
            per_m = {m: [] for m in all_models}
            for data in results.values():
                for m in all_models:
                    per_m[m].append(data["scalars"].get(m, {}))
            for m in all_models:
                df_m = pd.DataFrame(per_m[m])
                row = [m[:14]]
                for k in key:
                    col = (
                        df_m[k].dropna()
                        if k in df_m.columns
                        else pd.Series(dtype=float)
                    )
                    row.append(f"{col.mean():.4f}" if not col.empty else "—")
                rows.append(row)
            _log_table(rows, headers)
            log.info("")

        return results

    @staticmethod
    def save(metric_results: dict, metric_dir: Path):
        metric_dir.mkdir(parents=True, exist_ok=True)
        all_scalars = {}

        used_names = {}
        for group, data in metric_results.items():
            # labels are now already compact (Hs2p0_Te5p0_d...) — just safe-truncate
            sea_state = group.split(" - ", 1)[-1] if " - " in group else group
            base = _safe(sea_state)[:55]
            # ensure unique filename even if two groups map to the same safe name
            count = used_names.get(base, 0)
            used_names[base] = count + 1
            sg = base if count == 0 else f"{base}_{count}"

            # per-case scalar CSV
            if data["scalars"]:
                try:
                    df = pd.DataFrame(data["scalars"]).T
                    df.index.name = "model"
                    df.to_csv(metric_dir / f"{sg}.csv")
                    log.info("Saved: %s.csv", sg)
                except Exception as e:
                    log.error("Could not save scalar CSV for %s: %s", group, e)
                for label, vals in data["scalars"].items():
                    all_scalars.setdefault(label, []).append(vals)

            # cross-correlation CSV
            if data["corr"]:
                try:
                    cr = data.get("corr_ref", {})
                    # single lag_s (same length for x and xdot, same signal)
                    df_c = pd.DataFrame(
                        {
                            "lag_s": cr.get("lag_s", []),
                            "ref_x": cr.get("x", []),
                            "ref_xdot": cr.get("xdot", []),
                        }
                    )
                    for label, d in data["corr"].items():
                        sl = _safe(label)[:30]
                        df_c[f"{sl}_x"] = d["x"]
                        df_c[f"{sl}_xdot"] = d["xdot"]
                    df_c.to_csv(metric_dir / f"cross_corr_{sg}.csv", index=False)
                    log.info("Saved: cross_corr_%s.csv", sg)
                except Exception as e:
                    log.error("Could not save cross_corr CSV for %s: %s", group, e)

            # instantaneous phase CSV
            if data["phase"]:
                try:
                    first = next(iter(data["phase"].values()))
                    df_p = pd.DataFrame({"t": first["t"]})
                    for label, d in data["phase"].items():
                        sl = _safe(label)[:30]
                        df_p[f"{sl}_x_deg"] = d["x_deg"]
                        df_p[f"{sl}_xdot_deg"] = d["xdot_deg"]
                    df_p.to_csv(metric_dir / f"inst_phase_{sg}.csv", index=False)
                    log.info("Saved: inst_phase_%s.csv", sg)
                except Exception as e:
                    log.error("Could not save inst_phase CSV for %s: %s", group, e)

        # model_stats CSV: rows=models, cols={metric}_{stat}
        if all_scalars:
            try:
                rows = []
                for model_label, case_list in all_scalars.items():
                    df_m = pd.DataFrame(case_list)
                    row = {"model": model_label}
                    for metric in df_m.columns:
                        col = df_m[metric].dropna()
                        if col.empty:
                            continue
                        row[f"{metric}_mean"] = col.mean()
                        row[f"{metric}_std"] = col.std()
                        row[f"{metric}_min"] = col.min()
                        row[f"{metric}_max"] = col.max()
                    rows.append(row)
                pd.DataFrame(rows).set_index("model").to_csv(
                    metric_dir / "model_stats.csv"
                )
                log.info("Saved: model_stats.csv")
            except Exception as e:
                log.error("Could not save model_stats.csv: %s", e)

    # %% PLOTS

    @staticmethod
    def plot_corr(metric_results: dict, plots_dir: Path):
        """
        Cross-correlation: one figure per case, two subplots (x, xdot).
        All models overlaid, reference autocorr as dashed black baseline.
        All cases in one figure as subplots grid (cases × 2).
        """
        groups = [g for g, d in metric_results.items() if d["corr"]]
        if not groups:
            return

        n = len(groups)
        fig, axes = plt.subplots(n, 2, figsize=(14, max(3, 4 * n)), squeeze=False)

        for row, group in enumerate(groups):
            data = metric_results[group]
            cr = data.get("corr_ref", {})
            for col, (var, lk, rk) in enumerate(
                [
                    ("x", "lag_s", "x"),
                    ("xdot", "lag_s", "xdot"),
                ]
            ):
                ax = axes[row, col]
                if lk in cr:
                    ax.plot(
                        cr[lk],
                        cr[rk],
                        color="k",
                        lw=1.5,
                        ls="--",
                        label="ref (autocorr)",
                    )
                for label, d in data["corr"].items():
                    ax.plot(d[lk], d[var], label=label[:30], alpha=0.8)
                ax.axvline(0, color="gray", lw=0.7, ls=":")
                ax.set_xlabel("Lag [s]")
                ax.set_ylabel("Correlation")
                ax.set_title(f"{var} — {group}", fontsize=9)
                ax.legend(fontsize=7)
                ax.grid(True)

        plt.tight_layout()
        Plotter._save(fig, plots_dir, "metric", "cross_corr")

    @staticmethod
    def plot_phase_rose(metric_results: dict, plots_dir: Path):
        """
        Phase rose: one figure per case, cases as row subplots.
        For each case: columns = models, rows = x / xdot.
        """
        groups = [g for g, d in metric_results.items() if d["phase"]]
        if not groups:
            return

        # determine max models across groups for consistent layout
        max_models = max(
            len(d["phase"]) for _, d in metric_results.items() if d["phase"]
        )
        n_cases = len(groups)

        fig, axes = plt.subplots(
            n_cases * 2,
            max_models,
            subplot_kw={"projection": "polar"},
            figsize=(4 * max_models, 3.5 * n_cases * 2),
        )
        if axes.ndim == 1:
            axes = axes.reshape(-1, max_models)

        for r, group in enumerate(groups):
            data = metric_results[group]
            models = list(data["phase"].keys())
            for c, label in enumerate(models):
                for v, (key, var) in enumerate([("x_deg", "x"), ("xdot_deg", "xdot")]):
                    ax = axes[r * 2 + v, c]
                    deg = data["phase"][label][key]
                    rad = np.radians(deg % 360)
                    ax.hist(rad, bins=np.linspace(0, 2 * np.pi, 37), alpha=0.7)
                    ax.set_title(f"{label[:20]}\n{var} — {group[:15]}", fontsize=7)
            # hide unused columns
            for c in range(len(models), max_models):
                for v in range(2):
                    axes[r * 2 + v, c].set_visible(False)

        plt.tight_layout()
        Plotter._save(fig, plots_dir, "metric", "phase_rose")

    @staticmethod
    def plot_grid(metric_results: dict, plots_dir: Path, cfg: dict):
        """
        Te×Hs grid colored by chosen scalar metric.
        One subplot per model in same figure, unified color scale.
        Uses Plotter.plot_multi_scatter.
        cfg: {"metrics": ["nrmse_range_x", ...]}
        """
        metrics = cfg if isinstance(cfg, list) else cfg.get("metrics", [])
        if isinstance(metrics, str):
            metrics = [metrics]

        all_models = sorted({l for d in metric_results.values() for l in d["scalars"]})
        if not all_models:
            return

        for metric in metrics:
            per_model = {m: {"x": [], "y": [], "s": [], "c": []} for m in all_models}
            all_vals = []

            for data in metric_results.values():
                wp = data.get("wave_params", {}) or {}
                Te = wp.get("Te")
                Hs = wp.get("Hs")
                J = wp.get("J", 1.0)
                if Te is None or Hs is None:
                    continue
                for label in all_models:
                    v = data["scalars"].get(label, {}).get(metric, np.nan)
                    per_model[label]["x"].append(Te)
                    per_model[label]["y"].append(Hs)
                    per_model[label]["s"].append(J or 1.0)  # raw J, Plotter clips
                    per_model[label]["c"].append(v)
                    try:
                        fv = float(v)
                        if np.isfinite(fv):
                            all_vals.append(fv)
                    except (TypeError, ValueError):
                        pass

            if not all_vals:
                log.warning(
                    "metric_grid: no finite data for %s — check that metric is computed (time_ratio requires sim_time to be set)",
                    metric,
                )
                continue

            vmin = float(np.nanmin(all_vals))
            vmax = float(np.nanmax(all_vals))
            if vmin == vmax:
                vmin -= 0.5
                vmax += 0.5  # prevent degenerate colormap

            Plotter.plot_multi_scatter(
                per_model,
                vmin=vmin,
                vmax=vmax,
                xlabel="Te [s]",
                ylabel="Hs [m]",
                clabel=metric,
                title=f"{metric}  —  Te × Hs",
                save_path=plots_dir,
                name="metric",
                suffix=f"grid_{_safe(metric)}",
            )

    @staticmethod
    def plot_stat(metric_results: dict, plots_dir: Path, cfg: dict):
        """
        One subplot per metric. X axis = models.
        Layout: mean horizontal line, ±std bar centered on mean (not from 0),
        min/max as vertical tick marks at the ends of a thin line (like whiskers).
        cfg: {"metrics": ["nrmse_range_x", ...]}
        """
        metrics = cfg if isinstance(cfg, list) else cfg.get("metrics", [])
        if isinstance(metrics, str):
            metrics = [metrics]

        all_scalars = {}
        for data in metric_results.values():
            for label, vals in data["scalars"].items():
                all_scalars.setdefault(label, []).append(vals)
        if not all_scalars or not metrics:
            return

        model_labels = sorted(all_scalars.keys())
        n = len(metrics)
        ncols = min(n, 4)
        nrows = (n + ncols - 1) // ncols
        colors = plt.cm.tab10(np.linspace(0, 1, len(model_labels)))
        width = 0.5

        fig, axes = plt.subplots(
            nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False
        )

        for m_idx, metric in enumerate(metrics):
            ax = axes[m_idx // ncols][m_idx % ncols]
            for mod_idx, label in enumerate(model_labels):
                df_m = pd.DataFrame(all_scalars[label])
                col = (
                    df_m[metric].dropna()
                    if metric in df_m.columns
                    else pd.Series(dtype=float)
                )
                if col.empty:
                    continue
                mean = col.mean()
                std = col.std()
                mn = col.min()
                mx = col.max()
                c = colors[mod_idx]
                x = mod_idx

                # std band: rectangle centered on mean from mean-std to mean+std
                ax.bar(
                    x,
                    2 * std,
                    bottom=mean - std,
                    width=width,
                    color=c,
                    alpha=0.35,
                    label=label if m_idx == 0 else "_",
                )

                # mean: horizontal line across the full width of the band
                ax.hlines(mean, x - width / 2, x + width / 2, color=c, lw=2.5, zorder=5)

                # min/max: thin vertical line with tick caps (whisker style)
                ax.vlines(x, mn, mx, color=c, lw=1.2, zorder=4)
                # caps at min and max
                cap = width * 0.3
                ax.hlines(mn, x - cap, x + cap, color=c, lw=1.5, zorder=4)
                ax.hlines(mx, x - cap, x + cap, color=c, lw=1.5, zorder=4)

            ax.set_xticks(range(len(model_labels)))
            ax.set_xticklabels([l[:15] for l in model_labels], rotation=30, ha="right")
            ax.set_title(metric, fontsize=8)
            ax.set_ylim(bottom=0)
            ax.grid(axis="y", alpha=0.4)

        axes[0, 0].legend(fontsize=7)
        for idx in range(n, nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        fig.suptitle(
            "Metric statistics  —  line=mean  |  band=±std  |  whiskers=min/max"
        )
        plt.tight_layout()
        Plotter._save(fig, plots_dir, "metric", "stat")

    @staticmethod
    def plot_scalar(metric_results: dict, plots_dir: Path, cfg: dict):
        """
        One subplot per metric. X axis = sea states. Bars = models.
        cfg: {"metrics": ["nrmse_range_x", ...]}
        """
        metrics = cfg if isinstance(cfg, list) else cfg.get("metrics", [])
        if isinstance(metrics, str):
            metrics = [metrics]

        groups = list(metric_results.keys())
        all_models = sorted({l for d in metric_results.values() for l in d["scalars"]})

        # keep only metrics that have at least one finite value across all groups/models
        def _has_data(metric):
            for data in metric_results.values():
                for label in all_models:
                    v = data["scalars"].get(label, {}).get(metric)
                    if v is not None and np.isfinite(float(v)):
                        return True
            return False

        valid_metrics = [m for m in metrics if _has_data(m)]
        missing = [m for m in metrics if m not in valid_metrics]
        if missing:
            log.warning("plot_scalar: no finite data for %s — skipping", missing)
        metrics = valid_metrics
        if not metrics:
            log.warning("plot_scalar: no valid data for any requested metric")
            return

        # short labels: Hs+Te only, no PTO params
        def _short_label(g):
            parts = g.split("_")
            hs_te = [p for p in parts if p.startswith("Hs") or p.startswith("Te")]
            return "_".join(hs_te) if hs_te else g[:15]

        sea_labels = [_short_label(g) for g in groups]

        n = len(metrics)
        ncols = min(n, 4)
        nrows = (n + ncols - 1) // ncols
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_models)))
        x = np.arange(len(groups))
        width = 0.8 / max(len(all_models), 1)

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(max(6, 2 * len(groups)) * ncols / 2, 4 * nrows),
            squeeze=False,
        )

        for m_idx, metric in enumerate(metrics):
            ax = axes[m_idx // ncols][m_idx % ncols]
            for mod_idx, model in enumerate(all_models):
                vals = [
                    metric_results[g]["scalars"].get(model, {}).get(metric, np.nan)
                    for g in groups
                ]
                ax.bar(
                    x + mod_idx * width,
                    vals,
                    width=width,
                    color=colors[mod_idx],
                    alpha=0.85,
                    label=model if m_idx == 0 else "_",
                )
            ax.set_xticks(x + width * (len(all_models) - 1) / 2)
            ax.set_xticklabels(sea_labels, rotation=30, ha="right")
            ax.set_title(metric, fontsize=8)
            ax.grid(axis="y")

        axes[0, 0].legend(fontsize=7)
        for idx in range(n, nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        fig.suptitle("Scalar metrics per sea state")
        plt.tight_layout()
        Plotter._save(fig, plots_dir, "metric", "scalar")

    PLOT_DISPATCH = {
        "metric_corr": plot_corr.__func__,
        "metric_phase": plot_phase_rose.__func__,
        "metric_grid": plot_grid.__func__,
        "metric_stat": plot_stat.__func__,
        "metric_scalar": plot_scalar.__func__,
    }
