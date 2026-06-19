# %%  PACKAGE

import numpy as np

from source.classes.MCR import MCR

# %% INPUT

# wave
wave_param = {
    "T": [5],
    "H": [2],
    "type": "irr",
    "dt": 0.01,
    "t_end": 500,
    "gamma": 3.3,
}

# buoy
body_param = {
    "r": 5,
    "hyd_file": "xModel.json",
}

# wec simulation
wec_param = {
    "integrator": "nlinear",  # "linear" or "nlinear"
    "dyn": "dyn_nonlinearHs_wRad",  # "dyn_linear", "dyn_nonlinear", "dyn_nonlinearHs", "dyn_nonlinearHs_wRad"
    "t0": 0,
    "dt": 0.01,
    "X0": np.array([0.0, 0.0]),
    "B_pto": np.arange(100000, 830000, 10000),
    "S_pto": [0],  # np.arange(0, 0, 0),
    "eta_nl_hs_bool": True,
}

# log
store = ["wec.mean_power"]

# %% MCR

mcr = MCR(wave_param, body_param, wec_param, store=store)
mcr.run()
mcr.save_log_csv("out/PI_num_irr_NlHsEta_log.csv")

mcr.PI_map(output="mean_power", save_path="out/PI_num_irr_NlHsEta")

# %%
