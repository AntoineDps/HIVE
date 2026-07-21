import argparse
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt

from source.config import DATA_DIR, INPUT_DIR, MODEL_DIR
from source.classes.DataHandle import DataHandle
from source.classes.Plotter import Plotter

"""
# -------------------------------------------------------------------------
# Name:            run_handle_data.py
# Description:     process and visualize raw SPH cases
#
# Author:          Antoine
# Collaborator:    Bona
# Date created:    06/2026
# Project:         surrogate_hydro
#
# Inputs:
# - [cases.json]: SPH cases processing and plotting parameters
#
# Outputs:
# - [<config_name>.json]: copy of the config used, for traceability.
# - [run.log]: this run's log.
# - [data.csv]: case timeseries.
# - [wave.csv]: case wave parameters.
# - [plot.png or plot.pkl]: plot image and pickle for active reload.
#
# Dependencies:
# - [DataHandle.py]: Class
# - [Plotter.py]: Class
# - [HydroSphere.pkl]: object mentionned in the config file
# - [Hydrosphere.py]: Class
# -------------------------------------------------------------------------
"""

# %% PRELIMINARIES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# maps a "plot_options" key to the DataHandle classmethod that handles it.
PLOT_DISPATCH = {
    "sea_state": DataHandle.plot_seastate,
    "power": DataHandle.plot_power,
    "dynamics": DataHandle.plot_dynamics,
    "hydro": DataHandle.plot_hydro,
    "3d": DataHandle.scatter3d,
    "variable": DataHandle.plot_variable,
}

# %% FUNCTIONS AND CLASSES


@dataclass
class CaseConfig:
    """Everything one run needs: processing settings, cases, and what to plot."""

    wave_type: str
    cut_time: tuple
    duration: Optional[float]
    hydro_sphere_pkl: str
    dt: Optional[float]
    cases: list
    plot_options: dict


def load_config(config_path: Path) -> CaseConfig:
    """Read a run config from a JSON (.json) file."""
    with open(config_path, "r") as f:
        raw = json.load(f)

    return CaseConfig(
        wave_type=raw["wave_type"],
        cut_time=tuple(raw["cut_time"]),
        duration=raw.get("duration"),
        hydro_sphere_pkl=raw.get("hydro_sphere_pkl", raw.get("hyd_json")),
        dt=raw.get("dt"),
        cases=raw["cases"],
        plot_options=raw.get("plot_options", {}),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Process and visualize SPH cases.")
    parser.add_argument(
        "--case",
        type=Path,
        default="cases.json",
        help="Filename inside inputs/handle/, e.g. cases.json (default: %(default)s)",
    )
    return parser.parse_args()


def process_case(
    case: dict,
    config: CaseConfig,
    hydro_sphere,
    in_path: Path,
    out_path: Path,
) -> DataHandle:
    """Load, process, and save a single case. Returns it. Raises on failure (caught by caller)."""
    log.info(
        "Processing %s | T=%s H=%s | d=%s k=%s",
        config.wave_type,
        case["T"],
        case["H"],
        case["damping"],
        case["stiffness"],
    )

    data = DataHandle(
        case_path=in_path,
        wave_type=config.wave_type,
        T_case=case["T"],
        H_case=case["H"],
        damping=case["damping"],
        stiffness=case["stiffness"],
        cut_time=config.cut_time,
        duration=config.duration,
    )
    data.processData(r=hydro_sphere.r, hydro_sphere=hydro_sphere)

    if config.dt is not None:
        data.resample(new_dt=config.dt)

    data.save(out_path)
    return data


def filter_cases(data: list, cases: list) -> list:
    """Keep only the cases whose entry has "plot": true (the default when omitted)."""
    wanted = {
        (c["T"], c["H"], c["damping"], c["stiffness"])
        for c in cases
        if c.get("plot", True)
    }
    return [
        obj
        for obj in data
        if (obj.T_case, obj.H_case, obj.damping, obj.stiffness) in wanted
    ]


def reopen_saved_figures(plots_dir: Path):
    """Load and display every figure previously pickled into plots_dir."""
    pkl_files = sorted(plots_dir.glob("*.pkl")) if plots_dir.is_dir() else []
    if not pkl_files:
        log.warning("No saved figures (.pkl) found in %s", plots_dir)
        return

    for pkl_path in pkl_files:
        log.info("Reopening %s", pkl_path.name)
        Plotter.load_figure(pkl_path)

    plt.show()


# %% MAIN


def main():

    # setting up
    args = parse_args()

    config_path = INPUT_DIR / "handle_case" / args.case
    if not config_path.exists():
        log.error("Could not find config file: %s", config_path)
        return

    config = load_config(config_path)
    log.info(
        "Loaded config from %s (%d cases, plot_options=%s)",
        config_path,
        len(config.cases),
        config.plot_options,
    )

    run_dir = DATA_DIR / "handled" / config_path.stem
    data_dir = run_dir / "data"
    plots_dir = run_dir / "plots"

    # set up log file BEFORE option menu so every path is captured
    is_new = not run_dir.exists()
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    # clear any FileHandlers left over from a previous run in the same session
    _root = logging.getLogger()
    for _h in _root.handlers[:]:
        if isinstance(_h, logging.FileHandler):
            _h.close()
            _root.removeHandler(_h)
    _fh = logging.FileHandler(log_path, mode="a")
    _fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    _root.addHandler(_fh)
    log.info("=== run_handle_data ===")
    log.info(
        "Config: %s  (%d case(s))  plot_options=%s",
        config_path.stem,
        len(config.cases),
        [k for k, v in config.plot_options.items() if v],
    )

    do_process = True
    if not is_new:
        answer = input(
            f"Output folder '{run_dir}' already exists. Choose an option:\n"
            "  [1] erase and redo all (reprocess + replot)\n"
            "  [2] stop, do nothing\n"
            "  [3] load saved figures\n"
            "  [4] remake figures only (reuse existing processed data)\n"
            "> "
        ).strip()
        log.info("User option: %s", answer)
        if answer == "1":
            # close the log FileHandler before deleting — it holds run.log open
            _fh.close()
            logging.getLogger().removeHandler(_fh)
            for attempt in range(5):
                try:
                    shutil.rmtree(run_dir)
                    run_dir.mkdir(parents=True, exist_ok=True)
                    break
                except PermissionError as e:
                    if attempt < 4:
                        import time

                        time.sleep(2)
                    else:
                        print(f"ERROR: could not erase after 5 attempts: {e}")
                        return
            # reopen log file in the fresh folder
            _fh = logging.FileHandler(log_path, mode="w")
            _fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            logging.getLogger().addHandler(_fh)
            log.info("=== run_handle_data ===")
            log.info(
                "Config: %s  (%d case(s))  plot_options=%s",
                config_path.stem,
                len(config.cases),
                [k for k, v in config.plot_options.items() if v],
            )
        elif answer == "3":
            reopen_saved_figures(plots_dir)
            return
        elif answer == "4":
            do_process = False
        else:
            log.info("Stopped, nothing changed.")
            return

    # keep a copy of the exact config used
    shutil.copy2(config_path, run_dir / config_path.name)

    # process
    if do_process:
        data_dir.mkdir(parents=True, exist_ok=True)

        if config.hydro_sphere_pkl is None:
            log.error(
                "Config must have 'hydro_sphere_pkl' field pointing to a HydroSphere .pkl file."
            )
            return
        import pickle

        pkl_path = MODEL_DIR / "bem" / f"{config.hydro_sphere_pkl}.pkl"
        if not pkl_path.exists():
            log.error("HydroSphere pkl not found: %s", pkl_path)
            return
        with open(pkl_path, "rb") as f:
            hydro_sphere = pickle.load(f)
        log.info("Loaded HydroSphere from %s (r=%.1f)", pkl_path.name, hydro_sphere.r)
        in_path = DATA_DIR / "cfd"

        data = []
        results = []
        for case in config.cases:
            try:
                log.info(
                    "  Processing T=%.1f H=%.1f d=%d k=%d",
                    case["T"],
                    case["H"],
                    case["damping"],
                    case["stiffness"],
                )
                obj = process_case(case, config, hydro_sphere, in_path, data_dir)
                log.info("    saved: %s", obj.label)
                data.append(obj)
                results.append((case, "ok"))
            except Exception as e:
                log.error("  FAILED T=%.1f H=%.1f: %s", case["T"], case["H"], e)
                results.append((case, f"failed: {e}"))

        n_ok = sum(status == "ok" for _, status in results)
        log.info("Processed: %d/%d cases succeeded", n_ok, len(results))
        for case, status in results:
            if status != "ok":
                log.warning("  %s -> %s", case, status)
    else:
        try:
            data = DataHandle.load_case_in(
                data_dir, json_path=run_dir / config_path.name
            )
        except (FileNotFoundError, ValueError) as e:
            log.error("Could not reuse existing data: %s", e)
            return
        log.info("Reusing %d already-processed case(s) from %s", len(data), data_dir)

    if not data:
        log.error("No cases available to plot.")
        return

    # plot
    data_to_plot = filter_cases(data, config.cases)
    if not data_to_plot:
        log.warning('No cases have "plot": true, nothing to plot.')
        return
    log.info("Plotting %d/%d case(s)", len(data_to_plot), len(data))

    # new plot execution erase previous
    if plots_dir.exists():
        shutil.rmtree(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    log.info("Generating plots: %s", [k for k, v in config.plot_options.items() if v])
    for plot_name, active in config.plot_options.items():
        if isinstance(active, bool) and not active:
            continue
        if plot_name not in PLOT_DISPATCH:
            log.warning(
                "plot_option '%s' is enabled but not implemented yet, skipping",
                plot_name,
            )
            continue
        log.info("  Plotting: %s", plot_name)
        fn = PLOT_DISPATCH[plot_name]

        if plot_name == "variable":
            # one plot per case, named {label}_{variable}
            variables = active if isinstance(active, list) else [active]
            for obj in data_to_plot:
                for var in variables:
                    DataHandle.plot_line(
                        [obj],
                        y=var,
                        ylabel=var,
                        title=f"{obj.label} — {var}",
                        save_path=plots_dir,
                        name=obj.label,
                        suffix=var,
                    )
        elif isinstance(active, bool):
            fn(data_to_plot, save_path=plots_dir, name=None)
        else:
            fn(data_to_plot, active, save_path=plots_dir, name=None)

    log.info("Done.")
    plt.show()


if __name__ == "__main__":
    main()
