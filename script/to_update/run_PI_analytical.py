# %%  PACKAGE

from source.classes.HydroSphere import HydroSphere

# %% INPUT

# buoy
r = 5
hyd_file = "xModel.json"

# wave grid
T = [5 / 0.903, 7.5 / 0.903, 10 / 0.903]
# T = [5, 7.5, 10]
H = [2, 3, 4]

# %% run

# initialize buoy
body = HydroSphere(r, hyd_file)
body.PI_map(T, H, save_path="out/PI_analytical_Tp")
# body.PI_map(T, H, save_path="out/PI_analytical")
