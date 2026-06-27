# %%  PACKAGE

import numpy as np

# from source.classes import HydroSphere
from source.classes.IntegratorPhyLinear1DOF import IntegratorPhyLinear1DOF
from source.classes.HydroSphere import HydroSphere
from source.classes.Wave import Wave

"""
# -------------------------------------------------------------------------
# Name:            run_sphere.py
# Description:     Run sphere for different model (NN, linear, nonlinear), and validate against true data if requested
#
# Author:          Antoine
# Collaborator:    Bona
# Date created:    01/2026
# Data:            hydroSphere object
# Project:         surrogate_hydro
#
# Inputs: 
# - [Case]: Simulation input (wave, Hydrobody, PTO, etc).
#
# Outputs:
# - [plot]: timeseries
# - [scalar]: fitting metric
# - [file]: print data as txt/csv/json ?
#
# Dependencies:
# - [Integrator]: wec system
# - [model]: estimator
# - [Feed]: data feeder
# - [DataHandle]: true dataset
#
"""

# %% INPUT

# time
dt = 0.01
t_end = 65

# wave elevation
T = 7
H = 9
gamma = 3.3
wave_type = "reg"

# buoy
r = 5
hyd_file = "xModel.json"

# pto
damping = 120000
stiffness = -300000

# initial state
X0 = np.array([0.0, 0.0])
t0 = 0.0

# %% INITIALIZE

# initialize wave
wave = Wave(T=T, H=H, type=wave_type, dt=dt, t_end=t_end, gamma=gamma)

# initialize buoy
body = HydroSphere(r, hyd_file)
body.load_hydro_data()

# initialize solver
wec = IntegratorPhyLinear1DOF(
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
fe = wec.build_fe(wave.eta, body.t_irf, body.fe_irf)

# %% RUN

for f in wec.fe_hist:
    # step simuation
    X, t_i = wec.RK4_step(wec.dyn_linear, f)

wec.post_process()

# %% RESULTS

wave.plot_eta()
wave.plot_spectrum()

wec.plot_states()
wec.plot_forces()

wec.plot_hist(["x", "eta"])
