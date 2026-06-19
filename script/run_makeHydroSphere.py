# %%  PACKAGE

from source.classes.HydroSphere import HydroSphere
from source.config import DATA_DIR, SOURCE_DIR


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
# - [-]: radius and json hyd_file, object name and path
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
full_hyd_file = DATA_DIR / "handled_data" / "xModel.json"

# output saving
out_path = SOURCE_DIR / "models"

# %% INITIALIZE HYDROSPHERE

HS = HydroSphere(r, full_hyd_file)

# %% SAVE OBJECT

HS.save(path=out_path)
