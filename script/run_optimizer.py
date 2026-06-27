import pickle
from pathlib import Path

from source.classes.Optimizer import Optimizer

# TODO : batch run with several train_wave, eval_wave combination
# TODO : when problem happen in one simulation, NaN value arise and this is canceling the metric return... need to handle this case properly

# %% INPUT

# exp name
run_name = "test_debug"

# buoy
hyd_object = "r5"

# waves list
train_waves = [
    "irr_T5.5_H2.0_B120000_S-280000",
    "irr_T5.5_H2.9_B130000_S-270000",
    "irr_T6.6_H2.0_B120000_S-430000",
    # "irr_T6.6_H2.9_B140000_S-400000",
    # "irr_T8.0_H1.9_B140000_S-480000",
    # "irr_T8.1_H2.9_B230000_S-460000",
]
# eval_waves = train_waves
eval_waves = [
    # "irr_T5.5_H2.0_B120000_S-280000",
    # "irr_T5.5_H2.9_B130000_S-270000",
    # "irr_T6.6_H2.0_B120000_S-430000",
    # "irr_T6.6_H2.9_B140000_S-400000",
    # "irr_T8.0_H1.9_B140000_S-480000",
    # "irr_T8.1_H2.9_B230000_S-460000",
]

# optimizer param
wave_param = {"train_waves": train_waves, "eval_waves": eval_waves}

# optimizer param
opt_param = {
    "n_trials": 1,
    "n_stochastic": 1,
    "objective": "known_mean_dyn",
    "metric": "nrmse_range",
}

# dnn param
dnn_param = {
    "dropout": 0.001,
    "lr": 1e-3,
    "batch_size": 128,
    "epochs": 50,
    "patience": 25,
    "activation": "relu",
    "mid_neurons": 100,
    "min_neurons": 100,
    "layers": 1,
}

# feed param
feed_param = {
    "train_p": 0.7,
    "val_p": 0.15,
    "test_p": 0.15,
    "eta_p": 100,
    "eta_f": 50,
    "x_p": 100,
    "xdot_p": 100,
    "norm_from": "train",
    # "norm_bond": [-1, 1],
    "X_data": ["eta", "x", "xdot"],
    "y_data": ["fhd_nl_r"],
}

# search space
search_space = {
    "dropout": {
        "type": "float",
        "low": 0.01,
        "high": 0.10,
    },
    "lr": {
        "type": "float",
        "low": 1e-4,
        "high": 1e-2,
        "log": True,
    },
    "batch_size": {
        "type": "categorical",
        "choices": [32, 64, 128, 256, 516],
    },
    "activation": {
        "type": "categorical",
        "choices": ["relu", "tanh"],
    },
    "layers": {
        "type": "int",
        "low": 1,
        "high": 50,
    },
    "mid_neurons": {
        "type": "int",
        "low": 1,
        "high": 200,
    },
    "min_neurons": {
        "type": "int",
        "low": 1,
        "high": 200,
    },
    # "norm_bond": {
    #     "type": "categorical",
    #     "choices": ["relu", "tanh", "gelu"],
    # },
    "eta_p": {
        "type": "int",
        "low": 0,
        "high": 160,
    },
    "eta_f": {
        "type": "int",
        "low": 0,
        "high": 160,
    },
    "x_p": {
        "type": "int",
        "low": 0,
        "high": 160,
    },
    "xdot_p": {
        "type": "int",
        "low": 0,
        "high": 160,
    },
}

# %% PREP RUN

# load hydrobody
file_path = Path("data/hydro") / f"{hyd_object}.pkl"
with open(file_path, "rb") as f:
    body = pickle.load(f)
r = body.r
m = body.mass

# combine all parameters
all_params = dnn_param | feed_param | opt_param | wave_param

opt = Optimizer(
    search_space,
    all_params,
    log_experiment=run_name,
    r=body.r,
    m=body.mass,
)

# %% OPTIMIZE

opt.run()
