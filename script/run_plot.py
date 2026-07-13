import argparse
import json
import logging
import shutil
from pathlib import Path

import matplotlib.pyplot as plt

from source.config import INPUT_DIR, OUT_DIR
from source.classes.Plotter import Plotter
from source.classes.SimRun import SimRun

"""
# -------------------------------------------------------------------------
# Name:            run_plot.py
# Description:     Load saved SimRun CSVs from one or more validate_case
#                   output folders and plot them together. No simulation,
#                   no model loading, no HydroSphere needed.
#
#                   Useful for comparing runs that were produced with
#                   different models, solvers, or interpolation settings
#                   without having to re-run anything.
#
# Author:          Antoine
# Date:            06/2026
# Project:         surrogate_hydro
#
# Inputs:  inputs/read_case/<name>.json
# Outputs: out/read_case/<output_name>/plots/
# -------------------------------------------------------------------------
"""

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PLOT_DISPATCH = {
    "states": SimRun.plot_states,
}


# %% LOADING


def load_runs(run_entries: list) -> list:
    """
    For each entry in 'runs', glob CSVs from
    out/validate_case/<folder>/data/<filter> and reconstruct SimRun objects.

    Entry fields:
      folder   -- subfolder name under out/validate_case/
      filter   -- glob pattern to select CSVs (default: *.csv)
      label    -- optional label override; if multiple CSVs match,
                  appends ' (1)', ' (2)', ... to keep them distinct
    """
    base = OUT_DIR / "validate_case"
    simruns = []

    for entry in run_entries:
        folder = entry["folder"]
        data_dir = base / folder / "data"
        pattern = entry.get("filter", "*.csv")

        if not data_dir.exists():
            log.warning("data dir not found: %s", data_dir)
            continue

        csvs = sorted(data_dir.glob(pattern))
        if not csvs:
            log.warning("No CSVs matching '%s' in %s", pattern, data_dir)
            continue

        label_override = entry.get("label")
        for i, csv in enumerate(csvs):
            if label_override:
                label = (
                    label_override if len(csvs) == 1 else f"{label_override} ({i + 1})"
                )
            else:
                label = csv.stem
            simruns.append(SimRun.load(csv, label=label))
            log.info("  loaded %-60s → '%s'", csv.name, label)

    return simruns


# %% MAIN


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot saved SimRun CSVs together without re-simulating."
    )
    p.add_argument(
        "--case",
        type=Path,
        default="comparison.json",
        help="Filename inside inputs/read_case/ (default: %(default)s)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    config_path = INPUT_DIR / "read_case" / args.case
    if not config_path.exists():
        log.error("Config not found: %s", config_path)
        return

    with open(config_path) as f:
        config = json.load(f)

    output_name = config["output_name"]
    run_dir = OUT_DIR / "read_case" / output_name
    plots_dir = run_dir / "plots"

    # existing folder handling
    if run_dir.exists():
        answer = input(
            f"Output folder '{run_dir}' already exists.\n"
            "  [1] erase and redo\n"
            "  [2] stop\n"
            "  [3] load saved figures\n"
            "> "
        ).strip()
        if answer == "1":
            shutil.rmtree(run_dir)
        elif answer == "3":
            pkls = sorted(plots_dir.glob("*.pkl")) if plots_dir.is_dir() else []
            if not pkls:
                log.warning("No saved figures in %s", plots_dir)
            for p in pkls:
                Plotter.load_figure(p)
            plt.show()
            return
        else:
            log.info("Stopped.")
            return

    run_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, run_dir / config_path.name)

    fh = logging.FileHandler(run_dir / "run.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(fh)

    # load
    log.info("Loading SimRun CSVs:")
    simruns = load_runs(config["runs"])
    if not simruns:
        log.error("No SimRun objects loaded, nothing to plot.")
        return
    log.info("Loaded %d SimRun(s) total", len(simruns))

    # plot
    for plot_name, active in config.get("plot_options", {}).items():
        if not active:
            continue
        if plot_name not in PLOT_DISPATCH:
            log.warning("plot_option '%s' not implemented, skipping", plot_name)
            continue
        PLOT_DISPATCH[plot_name](simruns, save_path=plots_dir, name=output_name)

    log.info("Done.")
    plt.show()


if __name__ == "__main__":
    main()
