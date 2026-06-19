# %% PACKAGE

import numpy as np
import matplotlib.pyplot as plt

"""
# -------------------------------------------------------------------------
# Name:            run_buoyancyType.py
# Description:     Plot several type for hydrostatic force as a function of the displacement of the body.
#
# Author:          Antoine
# Collaborator:    Bona
# Date created:    01/2026
# Data:            hydroSphere object
# Project:         surrogate_hydro
#
# Inputs: 
# - 
#
# Outputs:
# - [plot]: F vs x
#
# Dependencies:
# - 
#
"""

# %% INPUT

rho = 1025
g = 9.81
m = 268344

eta = 0

r = 5
x = np.arange(-9, 9, 0.01)

h = r + (eta - x)
h = np.clip(h, 0, 2 * r)
s_hs_nl = ((np.pi * h**2 * (3 * r - h)) / 3 * rho * g) - m * g
s_hs_l = -(789737 * x)
s_pto = -(-290000 * x)

f_total_l = s_hs_l + s_pto
f_total_nl = s_hs_nl + s_pto

# %% VISUALIZE

plt.figure(figsize=(10, 6))

plt.plot(x, s_hs_l, label="linear hs")
plt.plot(x, s_hs_nl, label="nonlinear hs")
plt.plot(x, s_pto, label="pto")
plt.plot(x, f_total_l, label="total force linear")
plt.plot(x, f_total_nl, label="total force nonlinear")
plt.xlabel("Displacement [m]")
plt.ylabel("Force [N]")
plt.title("Stiffness associated force comparison")
plt.grid(True)
plt.legend()
plt.show()
