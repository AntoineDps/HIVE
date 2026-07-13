import argparse
import json
import logging
import shutil
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt

from source.config import INPUT_DIR, MODEL_DIR
from source.classes.DDFeed import DDFeed
from source.classes.Optimizer import Optimizer
from source.classes.Plotter import Plotter

"""
# -------------------------------------------------------------------------
# Name:            run_identify.py
# Description:     Identify models and models parameters from CFD reference data.
#
# Author:          Antoine
# Collaborator:    Bona, Bruno, Edoardo
# Date created:    06/2026
# Data:            CFD runs
# Project:         wec_modeling_benchmark
#
# Inputs:  [inputs/identify_case/<name>.json]
#            The config filename (without extension) becomes the output
#            folder name under models/. No output_name field needed.
#            e.g. inputs/identify_case/linear_viscous_sphere_r5_v1.json
#                 → models/linear_viscous_sphere_r5_v1/
#
# Outputs: models/<config_stem>/
# -------------------------------------------------------------------------
"""

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SCHEMES = ["viscous_drag"]


# %% CONFIG


@dataclass
class IdentifyConfig:
    scheme: str
    waves_train: list
    base_model: str
    scheme_params: dict
    solver: dict
    plot_options: dict


def load_config(path: Path) -> IdentifyConfig:
    with open(path) as f:
        raw = json.load(f)
    return IdentifyConfig(
        scheme=raw["scheme"],
        waves_train=raw["waves_train"],
        base_model=raw["base_model"],
        scheme_params=raw.get("scheme_params", {}),
        solver=raw.get("solver", {"method": "RK4", "dt": 0.05, "interpolation": {}}),
        plot_options=raw.get("plot_options", {}),
    )


def parse_args():
    p = argparse.ArgumentParser(description="Model identification pipeline.")
    p.add_argument(
        "--case",
        type=Path,
        default="linear_sphere_viscous.json",
        help="Filename inside inputs/identify_case/ — stem becomes output folder name",
    )
    return p.parse_args()


# %% MAIN


def main():
    args = parse_args()
    config_path = INPUT_DIR / "identify_case" / args.case
    if not config_path.exists():
        log.error("Config not found: %s", config_path)
        return

    config = load_config(config_path)
    output_name = config_path.stem  # filename stem IS the output name

    log.info(
        "scheme='%s'  base='%s'  output='%s'",
        config.scheme,
        config.base_model,
        output_name,
    )

    if config.scheme not in SCHEMES:
        log.error("Unknown scheme '%s'. Available: %s", config.scheme, SCHEMES)
        return

    model_dir = MODEL_DIR / output_name
    plots_dir = model_dir / "plots"

    # existing folder handling
    if model_dir.exists():
        answer = input(
            f"Model folder '{model_dir.name}' already exists.\n"
            "  [1] erase and redo\n"
            "  [2] stop\n"
            "  [3] load saved figures\n"
            "  [4] run and save with date suffix  (keeps existing folder)\n"
            "> "
        ).strip()
        if answer == "1":
            shutil.rmtree(model_dir)
        elif answer == "2":
            log.info("Stopped.")
            return
        elif answer == "3":
            for p in sorted(plots_dir.glob("*.pkl")):
                Plotter.load_figure(p)
            plt.show()
            return
        elif answer == "4":
            output_name = f"{output_name}_{date.today().strftime('%Y%m%d')}"
            model_dir = MODEL_DIR / output_name
            plots_dir = model_dir / "plots"
            log.info("New output folder: %s", output_name)
        else:
            log.info("Stopped.")
            return

    model_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, model_dir / config_path.name)

    fh = logging.FileHandler(model_dir / "run.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(fh)

    feed = DDFeed(config.scheme, config.waves_train, config.scheme_params)
    opt = Optimizer(config.scheme, config.base_model, config.solver)

    result, grids, simruns = opt.run(feed, config)

    opt.save(result, output_name)
    opt.save_result(result, model_dir)
    opt.save_data(result, simruns, feed, model_dir)
    opt.run_plots(config, feed, simruns, result, grids, plots_dir)

    log.info("Done.")
    plt.show()


if __name__ == "__main__":
    main()
