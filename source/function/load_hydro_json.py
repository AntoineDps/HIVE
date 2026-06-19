from pathlib import Path
import json
import numpy as np


def load_hydro_json(in_path: Path) -> dict:
    """
    Load hydrodynamic data from a JSON file and return it as a dictionary
    with numpy arrays where appropriate.
    """
    with open(in_path, "r") as f:
        data = json.load(f)

    result = {
        "w": np.array(data["w"]),
        "B": np.array(data["B"]),
        "Fe_mod": np.array(data["Fe_mod"]),
        "Fe_ang": np.array(data["Fe_ang"]),
        "ma": np.array(data["ma"]),
        "ma_inf": np.array(data["ma_inf"]),
        "fe_irf": np.array(data["fe_irf"]),
        "t_irf": np.array(data["t_irf"]),
        "rad_ss": {key: np.array(value) for key, value in data["rad_m"].items()},
    }

    return result
