import logging

from source.config import DATA_DIR
from source.classes.DataHandle import DataHandle

"""
# -------------------------------------------------------------------------
# Name:            DDFeed.py
# Description:     Data loading and preparation dispatcher.
#                   Loads training data then dispatches to the scheme module's
#                   prep() function for scheme-specific preparation.
#
# Author:          Antoine
# Collaborator:    Bona, Bruno, Edoardo
# Date created:    06/2026
# Project:         wec_modeling_benchmark
# -------------------------------------------------------------------------
"""

log = logging.getLogger(__name__)


def load_training_data(waves_train: list) -> list:
    data = []
    for case_name in waves_train:
        root = DATA_DIR / "handled" / case_name
        try:
            objs = DataHandle.load_case_in(
                root / "data", json_path=root / f"{case_name}.json"
            )
            data.extend(objs)
            log.info("Loaded %d case(s) from '%s'", len(objs), case_name)
        except (FileNotFoundError, ValueError) as e:
            log.error("Could not load '%s': %s", case_name, e)
    return data


class DDFeed:
    """
    Generic data loading and preparation framework.
    Receives the scheme module from run_identify and calls its prep() function.

    Attributes populated by prep() are scheme-dependent.
    For Viscous_drag: per_case, global_regressor, global_residual.
    For Pi_gain: cases
    """

    def __init__(
        self,
        scheme_module,
        waves_train: list,
        scheme_params: dict,
        solver_cfg: dict = None,
    ):
        self._mod = scheme_module
        self.scheme = scheme_module.__name__.split(".")[-1]
        self.params = {
            **scheme_params,
            "waves_train": waves_train,
            "solver": solver_cfg or {},
        }

        if not hasattr(scheme_module, "prep"):
            raise ValueError(f"Scheme module '{self.scheme}' has no prep() function.")

        if getattr(scheme_module, "SELF_LOADING", False):
            self.data = []
        else:
            self.data = load_training_data(waves_train)

        scheme_module.prep(self)
