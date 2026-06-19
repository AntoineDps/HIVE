# %%  PACKAGE

import numpy as np
from tqdm import tqdm

# from source.classes import HydroSphere
from source.classes.IntegratorNLinear1DOF import IntegratorNLinear1DOF
from source.classes.DataHandle_old import DataHandle
from source.classes.HydroSphere import HydroSphere
from source.classes.Wave import Wave
from source.classes.IntegratorPhyLinear1DOF import IntegratorPhyLinear1DOF

# %% INPUT

# time
dt = 0.01
t_end = 500

# wave elevation
T = 7
H = 3
gamma = 3.3
wave_type = "reg"

# buoy
r = 5
hyd_file = "xModel.json"

# pto
damping = 91000
stiffness = -540000

# initial state
X0 = np.array([0.0, 0.0])
t0 = 0.0

# %% INITIALIZE LINEAR

# initialize wave
wave = Wave(T=T, H=H, type=wave_type, dt=dt, t_end=t_end, gamma=gamma)

# time
t_sim = wave.t

# initialize buoy
body = HydroSphere(r, hyd_file)
body.load_hydro_data()

# initialize solver
wec_lin = IntegratorPhyLinear1DOF(
    dt,
    body.mass,
    body.ma_inf,
    body.rad_ss,
    body.stiffness,
    t0=t0,
    X0=X0,
    B_pto=damping,
    S_pto=stiffness,
)

# excitation force
wec_lin.build_fe(wave.eta, body.t_irf, body.fe_irf)

# %% INITIALIZE NONLINEAR

# initialize integrator
body = HydroSphere(r, hyd_file)

wec_nl = IntegratorNLinear1DOF(
    dt,
    body.mass,
    rad_ss=body.rad_ss,
    ma_inf=body.ma_inf,
    r=r,
    X0=X0,
    eta=wave.eta,
    B_pto=damping,
    S_pto=stiffness,
    eta_nl_hs_bool=False,
)

# get fhyd
wec_nl.load_f("fhd_hist", wec_lin.fe_hist)

# %% RUN

for i, t in enumerate(tqdm(t_sim, desc="Simulation Progress")):
    # step simuation
    X_nl, t_i = wec_nl.RK4_step(wec_nl.dyn_nonlinearHs_wRad, wec_nl.fhd_hist[i])
    # X_lin, t_ii = wec_lin.RK4_step(wec_lin.dyn_linear, wec_lin.fe_hist[i])

wec_nl.post_process(True)
# wec_lin.post_process()

# %% RESULTS

# wave.plot_eta()
# wave.plot_spectrum()

wec_nl.plot_hist(["eta", "x"])
# wec_nl.plot_states()
# wec_nl.plot_forces()

# wec_nl.compare(wec_lin, "x", t=t_sim, plot=True)

# wec_nl.fit(wec_lin.fe_hist, "fhd", t=t_sim, plot=True)
# wec_nl.fit(wec_lin.fhs_hist - wec_lin.fg_hist, "fhs", t=t_sim, plot=True)
# wec_nl.fit(wec_lin.fr_hist, "fr", t=t_sim, plot=True)

# wec_nl.plot_hist(["fhs", "x", "fhd", "fr"], normalize=True)
