import numpy as np
import torch
import json
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import mlflow
import joblib
import re

from source.classes.DataHandle_old import DataHandle
from source.classes.DataBasedFeed import DataBasedFeed
from source.classes.HydroSphere import HydroSphere
from source.classes.Estimator import Estimator
from source.classes.IntegratorNLinear1DOF import IntegratorNLinear1DOF
from source.classes.IntegratorPhyLinear1DOF import IntegratorPhyLinear1DOF
from source.classes.Wave import Wave
from source.classes.TikzDataWriter import TikzDataWriter

# TODO : improve logging system


def name2params(s: str) -> dict:
    pattern = (
        r"T(?P<T>[-+]?\d*\.?\d+)_"
        r"H(?P<H>[-+]?\d*\.?\d+)_"
        r"B(?P<B>-?\d+)_"
        r"S(?P<S>-?\d+)"
    )

    match = re.search(pattern, s)
    if not match:
        raise ValueError(f"String does not match expected format: {s}")

    return {
        "T": float(match.group("T")),
        "H": float(match.group("H")),
        "B": int(match.group("B")),
        "S": int(match.group("S")),
    }


# %% INPUT

# buoy
hyd_object = "r5"

# path
path = "data/handled_data/"

# model
# run_id = "5b34a1cebaa24001b97b9e8aafaa1a91"  # test
# run_id = "99113daca801418d920c25f69b4b0902"  # all
# run_id = "b02f711119b640129eda648fb959152e"  # interp
# run_id = "e3582a71aa314eb69f18a8bdbe36a385"  # extrap
# run_id = "e96ed33074d34426bbb5d83fb792f4f8"  # excelleent estimator all
run_id = "0bb665c0bc934f7fac09e1ae7a2b6fa6"

# cases
cases = [
    "irr_T5.5_H2.0_B120000_S-280000",
    "irr_T5.5_H2.9_B130000_S-270000",
    "irr_T6.6_H2.0_B120000_S-430000",
    "irr_T6.6_H2.9_B140000_S-400000",
    "irr_T8.0_H1.9_B140000_S-480000",
    "irr_T8.1_H2.9_B230000_S-460000",
]

# flagS
plot = 1
log_dyn = 0
log_fit = 0

# name
log_dyn_name = "series_inter"
log_fit_name = "grid_metric_inter"

# %% LOAD

# hydro body
file_path = Path("data/hydro") / f"{hyd_object}.pkl"
with open(file_path, "rb") as f:
    body = pickle.load(f)
r = body.r
m = body.mass

# handled data
directory = Path(path)

# for each case
data = {}
for case in cases:
    # load handled data
    file_path = directory / f"{case}.pkl"

    with open(file_path, "rb") as f:
        data[case] = pickle.load(f)

# estimator
model_uri = f"runs:/{run_id}/model"
model = mlflow.pytorch.load_model(model_uri)
params_path = mlflow.artifacts.download_artifacts(
    run_id=run_id, artifact_path="param.json"
)
with open(params_path) as f:
    params = json.load(f)

normalizer_path = mlflow.artifacts.download_artifacts(
    run_id=run_id, artifact_path="normalizer.pkl"
)
normalizer = joblib.load(normalizer_path)

run = mlflow.get_run(run_id)
metrics = run.data.metrics

# %% INITAILIZE

# fit log dict
grid_log = {
    "T": [],
    "H": [],
    "nn_fit_force": [],
    "nn_fit_x": [],
    "lin_fit_x": [],
    "t_sim_nn": [],
    "t_rel_nn": [],
    "t_sim_lin": [],
    "t_rel_lin": [],
}

# prep data
dataset_train = {k: data[k].dataset for k in params["train_waves"]}
feed = DataBasedFeed(
    dataset=dataset_train,
    params=params,
)

dataset_all = {k: data[k].dataset for k in cases}
feed.prep_eval(dataset_all)
norm_y = normalizer[params["y_data"][0]]
t_sim = feed.t_eval
dt = round(t_sim[1] - t_sim[0], 4)
n_steps = len(t_sim)
for case in feed.eval_data_per_wave.keys():
    df_wave = dataset_all[case]

    # get parameters
    case_params = name2params(case)
    damping = case_params["B"]
    stiffness = case_params["S"]

    # NN
    t_start = t_sim[0]
    t_end = t_sim[-1]
    t_total = t_end - t_start
    dataset_cut = DataHandle.cut_time(
        df_wave,
        t_start=t_sim[0],
        t_end=t_sim[-1],
    )

    feed.init_simulation(df_wave, t_sim, device="cpu")
    X0 = feed.x
    eta = dataset_cut["eta"]
    x_true = dataset_cut["x"]
    X_nn = feed.X_nn

    wec_nn = IntegratorNLinear1DOF(
        dt,
        body.mass,
        t0=t_sim[0],
        X0=X0,
        eta=eta,
        r=body.r,
        B_pto=damping,
        S_pto=stiffness,
    )

    # linear
    wec_lin = IntegratorPhyLinear1DOF(
        dt,
        body.mass,
        body.ma_inf,
        body.rad_ss,
        body.stiffness,
        t0=t_sim[0],
        X0=X0,
        B_pto=damping,
        S_pto=stiffness,
    )

    wec_lin.build_fe(eta, body.t_irf, body.fe_irf)

    # %% RUN

    # estimator
    estimator = Estimator(
        model_type="dnn",
        params=params,
        train_data=feed.eval_data_per_wave[case]["X"],
        val_data=feed.eval_data_per_wave[case]["y"],
    )
    estimator.model = model
    y = estimator.evaluate(
        feed.eval_data_per_wave[case]["X"],
        feed.eval_data_per_wave[case]["y"],
        plot=False,
        metric_fn=params["metric"],
        time=t_sim,
        norm=norm_y,
    )

    n_steps = len(t_sim)

    # run lin integrator
    t_start_lin = time.perf_counter()
    for i, t in enumerate(tqdm(t_sim, desc="Linear Simulation Progress")):
        # step simuation
        state_lin, t_i = wec_lin.RK4_step(wec_lin.dyn_linear, wec_lin.fe_hist[i])

    t_end_lin = time.perf_counter()

    lin_total_time = t_end_lin - t_start_lin
    lin_rel_time = lin_total_time / t_total

    # run NN integrator
    model.eval()
    with torch.inference_mode():
        t_start_nn = time.perf_counter()
        for i, t in enumerate(tqdm(t_sim, desc="Simulation Progress")):
            # predict NN
            y_nn = Estimator.predict_norm(model, X_nn, norm_y)

            # step simuation
            state_nn, t_i = wec_nn.RK4_step(wec_nn.dyn_nonlinearHs, y_nn, log="fhd")

            # update input vector except for last step
            if i < n_steps - 1:
                X_nn = feed.step(state_nn)
        t_end_nn = time.perf_counter()

    nn_total_time = t_end_nn - t_start_nn
    nn_rel_time = nn_total_time / t_total

    # post process
    wec_nn.post_process(True)
    wec_lin.post_process()

    # %% FIT

    nn_fit_x = wec_nn.fit(x_true, "x", t=t_sim)
    lin_fit_x = wec_lin.fit(x_true, "x", t=t_sim)

    # %% VISUALIZE

    if plot:
        # repsonse
        plt.figure(figsize=(10, 6))
        plt.plot(t_sim, x_true, label="cfd - ref")
        plt.plot(t_sim, eta, label="eta")
        plt.plot(t_sim, wec_nn.X_hist[:, 0], label="nn", linestyle="--")
        plt.plot(t_sim, wec_lin.X_hist[:, 0], label="lin", linestyle="--")
        plt.grid(True)
        plt.title(
            f"Response T={case_params['T']}s H={case_params['H']}m / Fitting NN = {nn_fit_x}, lin = {lin_fit_x} / Time NN = {nn_rel_time:.3f}, lin = {lin_rel_time:.3f}"
        )
        plt.legend()

    # %% LOG

    # dynamics (toward txt)
    if log_dyn:
        data_log = {
            "t": t_sim,
            "x_ref": x_true,
            "x_nn": wec_nn.X_hist[:, 0],
            "x_lin": wec_lin.X_hist[:, 0],
            "f_ref": norm_y.inverse_transform(
                feed.eval_data_per_wave[case]["y"]
            ).flatten(),
            "f_pred": y.flatten(),
        }
        writer_dyn = TikzDataWriter("plot/val_" + case + "_" + log_dyn_name + ".txt")
        writer_dyn.write(data_log)

    # fit (toward excel)
    if log_fit:
        grid_log["T"].append(case_params["T"])
        grid_log["H"].append(case_params["H"])
        grid_log["nn_fit_force"].append(run.data.metrics.get(case + "_estimator"))
        grid_log["nn_fit_x"].append(run.data.metrics.get(case + "_dyn"))
        grid_log["lin_fit_x"].append(lin_fit_x)
        grid_log["t_sim_nn"].append(nn_total_time)
        grid_log["t_rel_nn"].append(nn_rel_time)
        grid_log["t_sim_lin"].append(lin_total_time)
        grid_log["t_rel_lin"].append(lin_rel_time)


plt.show()

# log save
if log_fit:
    writer_grid = TikzDataWriter("plot/fit_" + log_fit_name + ".txt")
    writer_grid.write(grid_log)
