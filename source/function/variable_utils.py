import numpy as np
import json


def dic2json(d, path):
    out = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, dict):
            out[k] = dic2json(v)
        else:
            out[k] = v

    with open(path, "w") as f:
        json.dump(out, f, indent=4)
