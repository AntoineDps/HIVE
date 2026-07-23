import argparse
import json
import logging
import shutil
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt

from source.config import INPUT_DIR, MODEL_DIR
from source.classes.Objective import Objective
from source.classes.DDFeed import DDFeed
from source.classes.Optimizer import Optimizer
from source.classes.Plotter import Plotter
from source.classes.Model import Model
from source.classes.Metric import Metric
import source.classes.Viscous_drag as _viscous_drag
import source.classes.Pi_gain as _pi_gain

SCHEME_MODULES = {
    "viscous_drag": _viscous_drag,
    "pi_gain": _pi_gain,
}

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

    if config.scheme not in SCHEME_MODULES:
        log.error(
            "Unknown scheme '%s'. Available: %s", config.scheme, list(SCHEME_MODULES)
        )
        return

    scheme_module = SCHEME_MODULES[config.scheme]
    model_dir = MODEL_DIR / output_name
    id_dir = model_dir / "identification"
    plots_dir = id_dir / "plots"

    # set up log file BEFORE option menu so every path is captured
    is_new = not model_dir.exists()
    model_dir.mkdir(parents=True, exist_ok=True)
    log_path = model_dir / "run.log"
    # clear any FileHandlers left over from a previous run in the same session
    _root = logging.getLogger()
    for _h in _root.handlers[:]:
        if isinstance(_h, logging.FileHandler):
            _h.close()
            _root.removeHandler(_h)
    _fh = logging.FileHandler(log_path, mode="a")
    _fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    _root.addHandler(_fh)
    log.info("=== run_identify ===")
    log.info(
        "scheme='%s'  base='%s'  output='%s'",
        config.scheme,
        config.base_model,
        output_name,
    )

    if not is_new:
        answer = input(
            f"Model folder '{model_dir.name}' already exists.\n"
            "  [1] erase and redo\n"
            "  [2] stop\n"
            "  [3] load saved figures\n"
            "  [4] run and save with date suffix  (keeps existing folder)\n"
            "> "
        ).strip()
        log.info("User option: %s", answer)
        if answer == "1":
            # close log FileHandler before deleting — it holds run.log open
            _fh.close()
            logging.getLogger().removeHandler(_fh)
            import time as _t

            for attempt in range(5):
                try:
                    shutil.rmtree(model_dir)
                    model_dir.mkdir(parents=True, exist_ok=True)
                    break
                except PermissionError as e:
                    if attempt < 4:
                        _t.sleep(2)
                    else:
                        print(f"ERROR: could not erase after 5 attempts: {e}")
                        return
            # reopen log in fresh folder
            _fh = logging.FileHandler(log_path, mode="w")
            _fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            logging.getLogger().addHandler(_fh)
            log.info("=== run_identify ===")
            log.info(
                "scheme='%s'  base='%s'  output='%s'",
                config.scheme,
                config.base_model,
                output_name,
            )
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
            id_dir = model_dir / "identification"
            plots_dir = id_dir / "plots"
            model_dir.mkdir(parents=True, exist_ok=True)
            # re-setup log in the new folder
            _fh.close()
            logging.getLogger().removeHandler(_fh)
            log_path = model_dir / "run.log"
            _fh = logging.FileHandler(log_path, mode="w")
            _fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            logging.getLogger().addHandler(_fh)
            log.info("=== run_identify ===")
            log.info(
                "scheme='%s'  base='%s'  output='%s'",
                config.scheme,
                config.base_model,
                output_name,
            )
            log.info("New output folder: %s", output_name)
        else:
            log.info("Stopped.")
            return

    shutil.copy2(config_path, model_dir / config_path.name)

    feed = DDFeed(
        scheme_module,
        config.waves_train,
        config.scheme_params,
        solver_cfg=config.solver,
    )
    log.info("DDFeed ready: %d case(s) loaded", len(feed.data))
    opt = Optimizer(scheme_module, config.base_model, config.solver)
    log.info("Optimizer ready: base_model='%s'", config.base_model)

    log.info(
        "Starting identification: scheme='%s'  output='%s'", config.scheme, output_name
    )
    log.info("Wave cases: %d", len(getattr(feed, "cases", feed.data)))
    result, grids_by_method, simruns_by_method = opt.run(feed, config)
    log.info("Identification complete: %d method(s)", len(result.get("results", {})))

    # compute time margins from base model for metric trimming
    t_warmup, t_causal = 0.0, 0.0
    if opt.hydro_sphere is not None:
        t_warmup, t_causal = Model.compute_time_margins(opt.model_def, opt.hydro_sphere)
        log.info("Time margins: t_warmup=%.1fs  t_causal=%.1fs", t_warmup, t_causal)
        result["t_warmup"] = t_warmup
        result["t_causal"] = t_causal

    log.info("Saving results...")
    opt.save(result, output_name)
    opt.save_result(result, id_dir)
    opt.save_data(result, simruns_by_method, grids_by_method, feed, id_dir)
    log.info("Data saved to %s/data/", output_name)
    log.info("Generating plots...")
    log.info("Generating plots in %s", plots_dir)
    opt.run_plots(config, feed, simruns_by_method, result, grids_by_method, plots_dir)

    log.info("Done.")
    plt.show()


if __name__ == "__main__":
    main()
