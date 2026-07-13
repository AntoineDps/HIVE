import argparse
import json
import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from source.config import DATA_DIR, INPUT_DIR, MODEL_DIR, OUT_DIR
from source.classes.DataHandle import DataHandle
from source.classes.HydroSphere import HydroSphere
from source.classes.Model import Model, decompose_wave
from source.classes.Plotter import Plotter
from source.classes.SimRun import SimRun, get_solver
from source.classes.Wave import Wave

"""
# -------------------------------------------------------------------------
# Name:            run_validate.py
# Description:     Run models and validate against CFD reference data 
#                  or compare models under a synthetic wave.
#
# Author:          Antoine
# Collaborator:    Bona
# Date created:    06/2026
# Project:         surrogate_hydro
#
# Inputs:
# - [run.json]: runs config solver, model, cases and plotting parameters
#
# Outputs:
# - [<config_name>.json]: copy of the config used, for traceability.
# - [run.log]: this run's log.
# - [data.csv]: simulation timeseries.
# - [plot.png or plot.pkl]: plot image and pickle for active reload.
#
# Dependencies:
# - [Hydrosphere.py]: Class
# - [HydroSphere.pkl]: object mentionned in the config file
# - [Model.py]: Class
# - [DataHandle.py]: Class
# - [SimRun.py]: Class
# - [Wave.py]: Class
# - [Plotter.py]: Class
# -------------------------------------------------------------------------
"""

# %% PRELIMINARIES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PLOT_DISPATCH = {
    "states": SimRun.plot_states,
}


# %% FUNCTION AND CLASSES


def safe_name(text: str) -> str:
    return re.sub(r"[^\w\-]+", "_", text).strip("_")


def run_label(model_entry: dict, solver_entry: dict) -> str:
    return f"{model_entry['name']} | {solver_entry['method']} dt={solver_entry['dt']}"


@dataclass
class ValidateConfig:
    models: list
    wave_sources: list
    solvers: list
    eta_inputs: list
    simulation: list
    plot_options: dict


def load_config(config_path: Path) -> ValidateConfig:
    with open(config_path) as f:
        raw = json.load(f)
    return ValidateConfig(
        models=raw["models"],
        wave_sources=raw["wave_sources"],
        solvers=raw["solvers"],
        eta_inputs=raw.get("eta_inputs", [{"type": "true"}]),
        simulation=raw.get("simulation", [{"type": "closed_loop"}]),
        plot_options=raw.get("plot_options", {}),
    )


def parse_args():
    p = argparse.ArgumentParser(description="Validate physics models against CFD data.")
    p.add_argument(
        "--case",
        type=Path,
        default="run.json",
        help="Filename inside inputs/validate_case/ (default: %(default)s)",
    )
    return p.parse_args()


def check_v1_support(config: ValidateConfig) -> bool:
    ok_eta = [e for e in config.eta_inputs if e.get("type") == "true"]
    ok_sim = [s for s in config.simulation if s.get("type") == "closed_loop"]
    for e in config.eta_inputs:
        if e.get("type") != "true":
            log.warning("eta_input '%s' not implemented yet, skipping", e.get("type"))
    for s in config.simulation:
        if s.get("type") != "closed_loop":
            log.warning(
                "simulation type '%s' not implemented yet, skipping", s.get("type")
            )
    if not ok_eta or not ok_sim:
        log.error("No supported eta_input/simulation combination.")
        return False
    return True


def load_model_entry(kind: str, name: str):
    """Load model definition JSON and, for physics models, instantiate
    HydroSphere from it. Returns (model_def, hydro_sphere_or_None)."""
    path = MODEL_DIR / kind / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Model definition not found: {path}")
    with open(path) as f:
        model_def = json.load(f)

    if kind == "physics":
        import pickle

        body = model_def["body"]
        pkl_path = MODEL_DIR / "physics" / f"{body['hydro_sphere_pkl']}.pkl"
        with open(pkl_path, "rb") as f:
            hs = pickle.load(f)
    else:
        hs = None

    return model_def, hs


def resolve_wave_sources(wave_sources: list) -> list:
    """Turn each wave_source entry into one or more run dicts:
    {group, eta_t, eta_values, pto, reference (DataHandle or None)}."""
    runs = []
    for ws in wave_sources:
        kind = ws.get("type")

        if kind == "cfd_case":
            case_name = ws["case_name"]
            case_root = DATA_DIR / "handled" / case_name
            try:
                refs = DataHandle.load_case_in(
                    case_root / "data",
                    json_path=case_root / f"{case_name}.json",
                )
            except (FileNotFoundError, ValueError) as e:
                log.error("Could not load cfd_case '%s': %s", case_name, e)
                continue
            for ref in refs:
                eta_t = ref.dataset["t"].to_numpy()
                eta_values = ref.dataset["eta"].to_numpy()
                runs.append(
                    {
                        "group": f"{case_name} - {ref.label}",
                        "eta_t": eta_t,
                        "eta_values": eta_values,
                        "wave_components": decompose_wave(eta_values, eta_t),
                        "pto": {"damping": ref.damping, "stiffness": ref.stiffness},
                        "reference": ref,
                    }
                )

        elif kind == "synthetic":
            pto = ws.get("pto")
            if pto is None:
                log.error("Synthetic wave source missing 'pto', skipping")
                continue
            wave = Wave(
                type=ws["wave_type"],
                T=ws.get("T"),
                H=ws.get("H"),
                gamma=ws.get("gamma"),
                t_end=ws["duration"],
                dt=ws["dt"],
            )
            runs.append(
                {
                    "group": f"synthetic {ws['wave_type']} T={ws.get('T')} H={ws.get('H')}",
                    "eta_t": wave.t,
                    "eta_values": wave.eta,
                    "wave_components": [
                        (float(A), float(w), float(w**2 / 9.81), float(phi))
                        for A, w, phi in zip(wave.A, wave.w, wave.phi)
                        if A > 1e-6
                    ],
                    "pto": pto,
                    "reference": None,
                }
            )

        else:
            log.warning("Unknown wave_source type '%s', skipping", kind)

    return runs


def check_dt_mismatch(runs: list, solvers: list) -> bool:
    """Warn when a solver dt differs from the CFD reference dt.
    Ask to continue for each unique mismatched pair; return False to abort."""
    warned = set()
    for run in runs:
        if run["reference"] is None:
            continue
        t = run["reference"].dataset["t"].to_numpy()
        cfd_dt = float(np.round(np.median(np.diff(t)), 8))
        for s in solvers:
            solver_dt = s["dt"]
            pair = (round(cfd_dt, 8), round(solver_dt, 8))
            if pair not in warned and not np.isclose(cfd_dt, solver_dt, rtol=0.01):
                warned.add(pair)
                ans = (
                    input(
                        f"\nWarning: solver dt={solver_dt}s differs from CFD "
                        f"dt≈{cfd_dt}s. Metrics will need resampling. Continue? [y/N] "
                    )
                    .strip()
                    .lower()
                )
                if ans != "y":
                    return False
    return True


def reopen_saved_figures(plots_dir: Path):
    pkls = sorted(plots_dir.glob("*.pkl")) if plots_dir.is_dir() else []
    if not pkls:
        log.warning("No saved figures found in %s", plots_dir)
        return
    for p in pkls:
        log.info("Reopening %s", p.name)
        Plotter.load_figure(p)
    plt.show()


def load_existing_simruns(data_dir: Path, config: ValidateConfig, runs: list) -> dict:
    """Reload saved SimRun CSVs from a previous run (remake-plots mode)."""
    groups = {}
    for m in config.models:
        for run in runs:
            for s in config.solvers:
                label = run_label(m, s)
                csv = data_dir / f"{safe_name(run['group'])}_{safe_name(label)}.csv"
                if csv.exists():
                    result = SimRun.load(csv, label=label)
                    g = groups.setdefault(
                        run["group"], {"reference": run["reference"], "models": []}
                    )
                    g["models"].append(result)
                else:
                    log.warning("CSV not found, skipping: %s", csv.name)
    return groups


# %% PLOTTING


def run_plots(groups: dict, plots_dir: Path, plot_options: dict):
    if plots_dir.exists():
        shutil.rmtree(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    for group, info in groups.items():
        objs = ([info["reference"]] if info["reference"] is not None else []) + info[
            "models"
        ]
        for plot_name, active in plot_options.items():
            if not active:
                continue
            if plot_name not in PLOT_DISPATCH:
                log.warning("plot_option '%s' not implemented yet, skipping", plot_name)
                continue
            PLOT_DISPATCH[plot_name](objs, save_path=plots_dir, name=group)


# %% MAIN


def main():
    args = parse_args()
    config_path = INPUT_DIR / "validate_case" / args.case
    if not config_path.exists():
        log.error("Config not found: %s", config_path)
        return

    config = load_config(config_path)
    if not check_v1_support(config):
        return

    log.info(
        "Loaded config from %s (%d model(s), %d wave source(s), %d solver(s))",
        config_path,
        len(config.models),
        len(config.wave_sources),
        len(config.solvers),
    )

    run_dir = OUT_DIR / "validated_case" / config_path.stem
    data_dir = run_dir / "data"
    plots_dir = run_dir / "plots"

    # existing output handling
    if run_dir.exists():
        answer = input(
            f"Output folder '{run_dir}' already exists. Choose an option:\n"
            "  [1] erase and redo all\n"
            "  [2] stop, do nothing\n"
            "  [3] load saved figures\n"
            "  [4] remake plots only (reuse existing CSVs)\n"
            "> "
        ).strip()

        if answer == "1":
            for attempt in range(5):
                try:
                    shutil.rmtree(run_dir)
                    log.info("Erased %s", run_dir)
                    break
                except PermissionError as e:
                    if attempt < 4:
                        log.warning(
                            "Folder locked (OneDrive sync?), retrying in 2s... (%d/5)",
                            attempt + 1,
                        )
                        import time

                        time.sleep(2)
                    else:
                        log.error("Could not erase folder after 5 attempts: %s", e)
                        log.error("Try pausing OneDrive sync and re-running.")
                        return
        elif answer == "3":
            reopen_saved_figures(plots_dir)
            return
        elif answer == "4":
            # reload wave sources to get CFD references, then load SimRun CSVs
            runs = resolve_wave_sources(config.wave_sources)
            groups = load_existing_simruns(data_dir, config, runs)
            if not groups:
                log.error("No SimRun CSVs found to reload.")
                return
            run_plots(groups, plots_dir, config.plot_options)
            log.info("Done (remake plots).")
            plt.show()
            return
        else:
            log.info("Stopped, nothing changed.")
            return

    run_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, run_dir / config_path.name)

    log_path = run_dir / "run.log"
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(fh)
    log.info("Logging to %s", log_path)

    # resolve inputs
    runs = resolve_wave_sources(config.wave_sources)
    if not runs:
        log.error("No usable wave sources.")
        return

    if not check_dt_mismatch(runs, config.solvers):
        return

    model_entries = {}  # (kind, name) -> (model_def, hydro_sphere)
    for m in config.models:
        key = (m["kind"], m["name"])
        try:
            model_entries[key] = load_model_entry(*key)
        except (FileNotFoundError, Exception) as e:
            log.error("Could not load model '%s/%s': %s", *key, e)

    if not model_entries:
        log.error("No usable model definitions.")
        return

    # simulate
    combos = [
        (m, run, s)
        for m in config.models
        if (m["kind"], m["name"]) in model_entries
        for run in runs
        for s in config.solvers
    ]

    groups = {}
    n_failed = 0

    pbar = tqdm(combos, unit="run", miniters=1, dynamic_ncols=True)
    for model_entry, run, solver_entry in pbar:
        label = run_label(model_entry, solver_entry)
        pbar.set_description(
            f"{model_entry['name'][:20]} | {solver_entry['method']} dt={solver_entry['dt']}"
        )

        model_def, hs = model_entries[(model_entry["kind"], model_entry["name"])]
        try:
            model = Model.from_config(
                model_def,
                hs,
                run["eta_t"],
                run["eta_values"],
                wave_components=run.get("wave_components"),
                pto=run["pto"],
                interpolation=solver_entry.get("interpolation", {}),
            )
            result = SimRun.simulate(
                model,
                get_solver(solver_entry["method"]),
                t0=float(run["eta_t"][0]),
                t_end=float(run["eta_t"][-1]),
                dt=solver_entry["dt"],
                eta_t=run["eta_t"],
                eta_values=run["eta_values"],
                label=label,
            )
            phys_s = float(run["eta_t"][-1]) - float(run["eta_t"][0])
            tqdm.write(
                f"  {label[:50]} | {run['group'][:40]}"
                f"  [{result.sim_time:.2f}s wall, {phys_s / result.sim_time:.0f}x real-time]"
            )
            result.save(data_dir, name=safe_name(run["group"]), suffix=safe_name(label))
            g = groups.setdefault(
                run["group"], {"reference": run["reference"], "models": []}
            )
            g["models"].append(result)

        except Exception as e:
            n_failed += 1
            tqdm.write(f"  FAILED: {label} -- {e}")
            log.error("Run failed [%s]: %s", label, e)

    log.info("Simulated: %d/%d succeeded", len(combos) - n_failed, len(combos))

    if not groups:
        log.error("No successful simulations.")
        return

    # plot
    run_plots(groups, plots_dir, config.plot_options)
    log.info("Done.")
    plt.show()


if __name__ == "__main__":
    main()
