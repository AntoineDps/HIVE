# %%  PACKAGE

import numpy as np
import torch
import time
from contextlib import contextmanager
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

from source.classes import HydroSphere
from source.classes.DataBasedFeed import DataBasedFeed
from source.classes.DataHandle_old import DataHandle
from source.classes.Estimator import Estimator
from source.classes.HydroSphere import HydroSphere
from source.classes.IntegratorNLinear1DOF import IntegratorNLinear1DOF
from source.classes.IntegratorPhyLinear1DOF import IntegratorPhyLinear1DOF
from source.classes.Wave import Wave

# title: train
# load data
# process data
# train estimator
# inferdynamic
# compare against true
# save estimator -> compare with physical WEC in validation_script


@contextmanager
def timer(name):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"[{name}] {end - start:.4f} s")


# %% INPUT

# wave
# wave_list = ["reg_T4.4_H1.2", "reg_T5.5_H1.1"]
wave_list = ["irr_T5.4_H2.1"]

# buoy
r = 5
hyd_file = "xModel.json"

# pto
damping = 120000
stiffness = -300000

# feed
feed_param = {
    "train_p": 0.7,
    "val_p": 0.15,
    "test_p": 0.15,
    "eta_p": 100,
    "eta_f": 50,
    "x_p": 100,
    "xdot_p": 100,
    # "fhd_w": [0, 0],
    "norm_from": "train",
    "X_data": ["eta", "x", "xdot"],
    "y_data": ["fhd"],
}


# dnn
dnn_param = {
    "dropout": 0.001,
    "lr": 1e-3,
    "batch_size": 128,
    "epochs": 10,
    "patience": 300,
    "activation": "relu",
    "mid_neurons": 100,
    "min_neurons": 100,
    "layers": 1,
}

# %% FEED DATA BASED

with timer("FEED DATA BASED"):
    # get data
    datasets = DataBasedFeed.dataHandle2dict(wave_list)

    # check
    plt.figure(figsize=(10, 6))
    plt.plot(datasets["irr_T5.4_H2.1"]["fhd"], label="fhd")
    plt.grid(True)
    plt.legend()
    plt.show()

    # prepare data
    feed = DataBasedFeed(datasets, feed_param)

# %% ESTIMATOR

# create DNN
dnn = Estimator(
    "dnn",
    dnn_param,
    [feed.train_X, feed.train_y],
    [feed.val_X, feed.val_y],
    t_split=feed.t_split,
    wave=None,
)

# train DNN
dnn.build()

# save
# dnn.save_model("models/dnn_best")
# dnn.save_estimator("models/estimator")

# %% SIMULATION SETUP

# simulation sime
t_sim = feed.t_split[0]

# estimator feed initialization
feed.init_simulation(datasets["irr_T5.4_H2.1"], t_sim, device="cpu")
state_nn = feed.x

# %% LINEAR WEC SETUP

# input
T = 5
H = 2

# load handleData class
directory = Path("data/handled_data/")

file_path = directory / "irr_T5.4_H2.1.pkl"

with open(file_path, "rb") as f:
    exp = pickle.load(f)

# initialize body
body = HydroSphere(r, hyd_file)
body.load_hydro_data()
body.resample_irf(dt_new=0.05)
# body.save()

state_lin = state_nn
eta_lin = datasets["irr_T5.4_H2.1"][datasets["irr_T5.4_H2.1"]["t"].isin(t_sim)]["eta"]
wave = Wave(T=T, H=H, type="import", dt=exp.dt, eta=eta_lin, t=t_sim)
wec_lin = IntegratorPhyLinear1DOF(
    exp.dt,
    body.mass,
    body.ma_inf,
    body.rad_ss,
    body.stiffness,
    t0=t_sim[0],
    X0=state_lin,
    B_pto=damping,
    S_pto=stiffness,
)
wec_lin.build_fe(wave.eta, body.t_irf, body.fe_irf)


# %% CFD WEC SETUP

wec_cfd = IntegratorNLinear1DOF(
    exp.dt,
    body.mass,
    X0=state_lin,
    t0=t_sim[0],
    eta=wave.eta,
    r=r,
    B_pto=damping,
    S_pto=stiffness,
)

wec_cfd.load_f(
    "fhd_hist",
    datasets["irr_T5.4_H2.1"]
    .loc[datasets["irr_T5.4_H2.1"]["t"].isin(t_sim), "fhd"]
    .reset_index(drop=True),
)

# %% ESTIMATOR WEC SETUP

# initialize integrator
wec_nn = IntegratorNLinear1DOF(
    exp.dt,
    body.mass,
    t0=t_sim[0],
    X0=state_nn,
    eta=wave.eta,
    r=r,
    B_pto=damping,
    S_pto=stiffness,
)

# %% SIMULATION [NN + LIN + CFD]

norm_y = feed.get_normalizer(feed_param["y_data"][0])  #! way to delete this line?
X_nn = feed.X_nn

n_steps = len(t_sim)

with timer("SIMULATION [NN + LIN + CFD]"):
    # run integrator
    dnn.model.eval()
    with torch.no_grad():
        for i, t in enumerate(tqdm(t_sim, desc="Simulation Progress")):
            # predict NN
            y_nn = dnn.predict_norm(X_nn, norm_y)

            # step simuation
            state_nn, t_i = wec_nn.RK4_step(wec_nn.dyn_nonlinearHs, y_nn, log="fhd")
            state_lin, t_i = wec_lin.RK4_step(wec_lin.dyn_linear, wec_lin.fe_hist[i])
            state_cfd, t_i = wec_cfd.RK4_step(
                wec_cfd.dyn_nonlinearHs, wec_cfd.fhd_hist[i]
            )

            # update input vectorcexcept for last step
            if i < n_steps - 1:
                X_nn = feed.step(state_nn)

# %% POST PROCESS & VISUALIZE

# post process
wec_lin.post_process()  #! pbl fe with inconstant time step
wec_nn.post_process(True)
wec_cfd.post_process(True)

# full wave
feed.prep_eval(datasets)

# %% FITTING

# NN against cfd response
x_true = datasets["irr_T5.4_H2.1"].loc[datasets["irr_T5.4_H2.1"]["t"].isin(t_sim), "x"]
# wec_cfd.fit(x_true, "x", t=t_sim, plot=True)
# wec_nn.fit(x_true, "x", t=t_sim, plot=True)
# wec_lin.fit(x_true, "x", t=t_sim, plot=True)

# check forces
# wec_nn.fit(wec_lin.fhs_hist, "fhs", t=t_sim, plot=True)
# wec_nn.fit(wec_lin.fhyd_hist, "fhyd", t=t_sim, plot=True)
# wec_nn.fit(wec_lin.fe_hist, "fhd", t=t_sim, plot=True)

# check X data
# plt.figure(figsize=(10, 6))
# plt.plot(
#     feed.data_per_wave["irr_T5.4_H2.1"]["X"]["train"].ravel(),
#     label="X data per wave train",
# )
# plt.plot(feed.eval_data_per_wave["irr_T5.4_H2.1"]["X"].ravel(), label="X eval 1 wave")
# plt.plot(feed.train_X.ravel(), label="X train")
# plt.grid(True)
# plt.legend()
# plt.show()

# %% PLOT ARTICLE

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 1.2,
        "pdf.fonttype": 42,  # editable text in PDF
        "ps.fonttype": 42,
    }
)

cmap = plt.cm.Blues  # or plt.cm.Greys
colors = cmap(np.linspace(0.4, 0.9, 3))  # train, val, test

# force prediction
fnn_train = dnn.evaluate(
    feed.train_X,
    feed.train_y,
    metric_fn="nrmse_range",
    # plot=True,
    # log="test",
    # time=dnn.t_split[0],
    norm=feed.normalizer[feed_param["y_data"][0]],
)
fnn_val = dnn.evaluate(
    feed.val_X,
    feed.val_y,
    metric_fn="nrmse_range",
    # plot=True,
    # log="test",
    # time=dnn.t_split[0],
    norm=feed.normalizer[feed_param["y_data"][0]],
)
fnn_test = dnn.evaluate(
    feed.test_X,
    feed.test_y,
    metric_fn="nrmse_range",
    # plot=True,
    # log="test",
    # time=dnn.t_split[0],
    norm=feed.normalizer[feed_param["y_data"][0]],
)

fig, ax = plt.subplots(figsize=(7, 3.4))

# Train
ax.plot(feed.t_split[0], feed.train_y, color="k", label="True")
ax.plot(feed.t_split[0], fnn_train, "--", color=colors[0], label="Train – predicted")

# Validation
ax.plot(feed.t_split[1], feed.val_y, color="k")
ax.plot(feed.t_split[1], fnn_val, "--", color=colors[1], label="Validation – predicted")

# Test
ax.plot(feed.t_split[2], feed.test_y, color="k")
ax.plot(feed.t_split[2], fnn_test, "--", color=colors[2], label="Test – predicted")

ax.set_xlabel(r"$t\;[\mathrm{s}]$")
ax.set_ylabel(r"$F_{\mathrm{NL}}\;[\mathrm{N}]$")
ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), frameon=False, ncol=4)
fig.tight_layout()

fig.savefig("nonlinear_force_single_column.pdf", format="pdf")
plt.close(fig)


# dynamic
fig, ax = plt.subplots(figsize=(7, 3.4))

ax.plot(t_sim, x_true, linewidth=2, color="k", label="Reference - SPH")
ax.plot(t_sim, wec_cfd.X_hist[:, 0], alpha=1, color=colors[2], label="Reconstructed")
ax.plot(t_sim, wec_lin.X_hist[:, 0], alpha=0.8, color=colors[0], label="Linear")
ax.plot(t_sim, wec_nn.X_hist[:, 0], alpha=0.8, color="r", label="Data-driven")

ax.set_xlabel(r"$t\;[\mathrm{s}]$")
ax.set_ylabel(r"$\xi\;[\mathrm{m}]$")
ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), frameon=False, ncol=4)

fig.tight_layout()
fig.savefig("dynamic_response_single_column.pdf", format="pdf")
plt.close(fig)


a = None
