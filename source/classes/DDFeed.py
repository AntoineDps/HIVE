import logging

import numpy as np

from source.config import DATA_DIR
from source.classes.DataHandle import DataHandle

"""
# -------------------------------------------------------------------------
# Name:            DDFeed.py
# Description:     Data loading and preparation for identification schemes.
#                   One _prep_<scheme> method per scheme. Produces the data
#                   payload that Optimizer consumes.
#
# Author:          Antoine
# Collaborator:    Bona, Bruno, Edoardo
# Date created:    06/2026
# Project:         wec_modeling_benchmark
# -------------------------------------------------------------------------
"""

log = logging.getLogger(__name__)


def load_training_data(waves_train: list) -> list:
    """Load DataHandle objects for all training wave cases."""
    data = []
    for case_name in waves_train:
        root = DATA_DIR / "handled" / case_name
        try:
            objs = DataHandle.load_case_in(
                root / "data", json_path=root / f"{case_name}.json")
            data.extend(objs)
            log.info("Loaded %d case(s) from '%s'", len(objs), case_name)
        except (FileNotFoundError, ValueError) as e:
            log.error("Could not load '%s': %s", case_name, e)
    return data


class DDFeed:
    """
    Data loading and preparation for identification schemes.
    Loads training data then dispatches to _prep_<scheme>().
    The fully prepared DDFeed is the only data input Optimizer needs.
    """

    def __init__(self, scheme: str, waves_train: list, scheme_params: dict):
        self.scheme = scheme
        self.params = scheme_params
        self.data   = load_training_data(waves_train)

        prep = getattr(self, f"_prep_{scheme}", None)
        if prep is None:
            raise ValueError(
                f"No data preparation for scheme '{scheme}'. "
                f"Add a _prep_{scheme} method to DDFeed.")
        prep()

    # %% VISCOUS DRAG

    def _prep_viscous_drag(self):
        """
        Extract force regressor and residual from each CFD case.

        regressor: phi = -|xdot| * xdot   (Morison drag signal)
        residual:  r   = fhyd_cfd - fhyd_BEM  (unexplained force)

        Stores
        ------
        self.per_case       : {label: {"regressor": phi, "residual": r, "obj": DataHandle}}
        self.global_regressor: stacked phi across all valid cases
        self.global_residual : stacked r  across all valid cases
        """
        base_col = self.params.get("base_force", "fhyd_lin")
        self.per_case = {}
        all_phi, all_r = [], []

        for obj in self.data:
            label = obj.label
            if base_col not in obj.dataset.columns or \
               "fhyd_cfd" not in obj.dataset.columns:
                log.warning("Case '%s' missing '%s' or 'fhyd_cfd', skipping",
                            label, base_col)
                continue

            phi = -np.abs(obj.dataset["xdot"].values) * obj.dataset["xdot"].values
            r   = obj.dataset["fhyd_cfd"].values - obj.dataset[base_col].values

            self.per_case[label] = {"regressor": phi, "residual": r, "obj": obj}
            all_phi.append(phi)
            all_r.append(r)

        if all_phi:
            self.global_regressor = np.concatenate(all_phi)
            self.global_residual  = np.concatenate(all_r)
        else:
            self.global_regressor = None
            self.global_residual  = None
