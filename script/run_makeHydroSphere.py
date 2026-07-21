# %%  PACKAGE

from source.classes.HydroSphere import HydroSphere
from source.config import DATA_DIR, MODEL_DIR


"""
# -------------------------------------------------------------------------
# Name:            run_makeHydroSphere.py
# Description:     create HydroSphere object and save it for later use
#
# Author:          Antoine
# Collaborator:    
# Date created:    04/2026
# Data:            [Any relevant data sources or descriptions]
# Project:         surrogate_hydro
#
# Inputs: 
# - [r]: radius and json hyd_file, object name and path
# - [mesh, n_phi, n_theta]: mesh flag and mesh parameters
#
# Outputs:
# - [HydroSphere.pkl]: HydroSphere object
#
# Dependencies:
# - [HydroSphere.py]: HydroSphere class
#
"""

# %% INPUT

# buoy
r = 5
full_hyd_file = DATA_DIR / "bem" / "wamit" / "sphere" / "xModel.json"

# mesh
mesh = True
n_phi = 20
n_eta = 20

# output saving
out_path = MODEL_DIR / "bem"

# %% INITIALIZE HYDROSPHERE

HS = HydroSphere(r, full_hyd_file)

if mesh:
    HS.make_mesh(n_eta, n_phi)

    HS.plot_mesh()

# %% SAVE OBJECT

HS.save(path=out_path)
