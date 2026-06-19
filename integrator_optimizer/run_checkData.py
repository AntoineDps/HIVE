# %%  PACKAGE

import numpy as np
import json
from tqdm import tqdm

# from source.classes import HydroSphere
from source.classes.IntegratorNLinear1DOF import IntegratorNLinear1DOF
from source.classes.DataHandle_old import DataHandle
from source.classes.HydroSphere import HydroSphere
from source.classes.Wave import Wave
from source.classes.IntegratorPhyLinear1DOF import IntegratorPhyLinear1DOF

# %% INPUT

# exp
T = 5.0
H = 2.0
wave_type = "irr"

# buoy
r = 5
hyd_file = "xModel.json"

# pto
damping = 120000
stiffness = -300000

# time
cut_time = [60, 0]
duration = None

# %% INITIALIZE

# initialize dataHandle
exp = DataHandle(
    wave_type,
    T,
    H,
    damping=damping,
    stiffness=stiffness,
    cut_time=cut_time,
)

# %% VISUALIZE

# downsample
# spectrum #?

# process data
exp.waveParam()
exp.processData(r=r, m=-4 / 3 * np.pi * r**3 * exp.rho / 2)

# plot
# exp.plot_timeseries(["fhyd", "fhd", "fhs", "fg"])
# exp.plot_timeseries(["eta", "x"])
# exp.distri_data("Vs")
# exp.wetted_view(r=r)

# %% VALIDATE INTEGRATOR

# initialize body
body = HydroSphere(r, hyd_file)
body.load_hydro_data()
body.resample_irf(dt_new=0.05)

# time
t_sim = exp.dataset["t"]

# wave
eta = exp.dataset["eta"]

# states
X0 = np.array([exp.dataset["x"][0], exp.dataset["xdot"][0]])

# initialize integrator
wec_cfd = IntegratorNLinear1DOF(
    exp.dt,
    body.mass,
    X0=X0,
    t0=t_sim[0],
    eta=eta,
    r=r,
    B_pto=damping,
    S_pto=stiffness,
    eta_nl_hs_bool=True,
)

# get fhd_r
wec_cfd.load_f("fhd_hist", exp.dataset["fhd"])

# run
for i, t in enumerate(tqdm(t_sim, desc="Simulation Progress")):
    # step simuation
    state_cfd, t_i = wec_cfd.RK4_step(wec_cfd.dyn_nonlinearHs, wec_cfd.fhd_hist[i])

# %% VALIDATE LINEAR

wec_lin = IntegratorPhyLinear1DOF(
    exp.dt,
    body.mass,
    body.ma_inf,
    body.rad_ss,
    body.stiffness,
    t0=t_sim[0],
    X0=X0,
    B_pto=damping,
    S_pto=stiffness,
)
wec_lin.build_fe(
    eta,
    body.t_irf,
    body.fe_irf,
)

# run
for i, t in enumerate(tqdm(t_sim, desc="Simulation Progress")):
    # step simuation
    state_lin, t_i = wec_lin.RK4_step(wec_lin.dyn_linear, wec_lin.fe_hist[i])

# %% VALIDATE WEAK NLINEAR

wec_nl = IntegratorNLinear1DOF(
    exp.dt,
    body.mass,
    rad_ss=body.rad_ss,
    ma_inf=body.ma_inf,
    t0=t_sim[0],
    X0=X0,
    r=r,
    B_pto=damping,
    S_pto=stiffness,
    eta=eta,
    eta_nl_hs_bool=True,
)

# get fhd
wec_nl.load_f("fhd_hist", wec_lin.fe_hist)

# run
for i, t in enumerate(tqdm(t_sim, desc="Simulation Progress")):
    # step simuation
    state_nl, t_i = wec_nl.RK4_step(wec_nl.dyn_nonlinearHs_wRad, wec_nl.fhd_hist[i])

# %% SIMULATION RESULT

# simulation against true
wec_cfd.post_process(True)
wec_cfd.fit(exp.dataset["x"], "x", t=t_sim, plot=True)
wec_cfd.fit(exp.dataset["fhyd"], "fhyd", t=t_sim, plot=True)
wec_cfd.fit(exp.dataset["fhd"], "fhd", t=t_sim, plot=True)
wec_cfd.fit(exp.dataset["fhs"], "fhs", t=t_sim, plot=True)
wec_lin.post_process()
wec_lin.fit(exp.dataset["x"], "x", t=t_sim, plot=True)
wec_nl.post_process(True)
# wec_nl.fit(exp.dataset["x"], "x", t=t_sim, plot=True)
# wec_nl.plot_hist(["fhyd", "fhs", "fr", "fhd"])

# weakly nl against linear
# wec_nl.compare(wec_cfd, "fhd", t=t_sim, plot=True)
# wec_nl.compare(wec_lin, "x", t=t_sim, plot=True)
# wec_nl.compare(wec_lin, "eta", t=t_sim, plot=True)
# wec_nl.compare(wec_lin, "fhyd", t=t_sim, plot=True)
# wec_nl.compare(wec_lin, "fhs", t=t_sim, plot=True)

a = None
