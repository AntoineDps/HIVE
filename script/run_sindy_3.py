# %% PACKAGES

import numpy as np
import matplotlib.pyplot as plt
import pysindy as ps


# %% PARAMETERS

DT = 0.01
T_END = 100.0

m = 10.0
c = 20.0
k = 20.0

# Forcing amplitude
F_AMP = 10.0

# SINDy
THRESHOLD = 1e-7


# %% TIME

t = np.arange(
    0.0,
    T_END,
    DT,
)

N = len(t)


# %% GENERATE EXCITATION

# Multisine forcing
rng = np.random.default_rng(42)

freqs = np.linspace(
    0.05,
    2.0,
    30,
)

phases = rng.uniform(
    0.0,
    2.0 * np.pi,
    len(freqs),
)

u = np.zeros_like(t)

for f, phi in zip(freqs, phases):
    u += F_AMP * np.sin(2.0 * np.pi * f * t + phi) / len(freqs)


# %% SIMULATE TRUE OSCILLATOR

x = np.zeros(N)
xdot = np.zeros(N)


# Initial condition
x[0] = 0.1
xdot[0] = 0.0


def dynamics(x, xdot, u):

    xddot = -k / m * x - c / m * xdot + 1.0 / m * u

    return xdot, xddot


# RK4 integration

for i in range(N - 1):
    xi = x[i]
    vi = xdot[i]
    ui = u[i]

    k1_x, k1_v = dynamics(
        xi,
        vi,
        ui,
    )

    k2_x, k2_v = dynamics(
        xi + 0.5 * DT * k1_x,
        vi + 0.5 * DT * k1_v,
        ui,
    )

    k3_x, k3_v = dynamics(
        xi + 0.5 * DT * k2_x,
        vi + 0.5 * DT * k2_v,
        ui,
    )

    k4_x, k4_v = dynamics(
        xi + DT * k3_x,
        vi + DT * k3_v,
        ui,
    )

    x[i + 1] = xi + DT / 6.0 * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)

    xdot[i + 1] = vi + DT / 6.0 * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)


# %% BUILD SINDY DATA

X = np.column_stack(
    [
        x,
        xdot,
    ]
)

U = u.reshape(
    -1,
    1,
)


# %% FIT SINDY

print("Fitting SINDy...")

feature_library = ps.PolynomialLibrary(
    degree=1,
    include_bias=True,
)

optimizer = ps.STLSQ(
    threshold=THRESHOLD,
)

model = ps.SINDy(
    optimizer=optimizer,
    feature_library=feature_library,
)

model.fit(
    X,
    u=U,
    t=DT,
    feature_names=[
        "x",
        "xdot",
        "u",
    ],
)


# %% PRINT IDENTIFIED MODEL

print("\nTrue model:")

print(f"xdot  = xdot")

print(f"xddot = {-k / m:.6f} x {-c / m:+.6f} xdot {1 / m:+.6f} u")


print("\nSINDy model:")

model.print()


# %% DERIVATIVE FIT

Xdot_pred = model.predict(
    X,
    u=U,
)

xddot_true = -k / m * x - c / m * xdot + 1.0 / m * u

err_xddot = np.sqrt(np.mean((Xdot_pred[:, 1] - xddot_true) ** 2)) / (
    np.max(xddot_true) - np.min(xddot_true)
)


print(f"\nAcceleration NRMSE = {err_xddot:.6e}")


# %% RECURSIVE SINDY SIMULATION

print("\nRunning recursive SINDy simulation...")

X_sindy = np.zeros_like(X)

X_sindy[0] = X[0]


for i in range(N - 1):
    Xi = X_sindy[i].reshape(
        1,
        -1,
    )

    Ui = U[i].reshape(
        1,
        -1,
    )

    dX = model.predict(
        Xi,
        u=Ui,
    )[0]

    # Euler integration
    X_sindy[i + 1] = X_sindy[i] + DT * dX


# %% ERRORS

x_sindy = X_sindy[:, 0]
xdot_sindy = X_sindy[:, 1]


nrmse_x = np.sqrt(np.mean((x_sindy - x) ** 2)) / (np.max(x) - np.min(x))


nrmse_xdot = np.sqrt(np.mean((xdot_sindy - xdot) ** 2)) / (np.max(xdot) - np.min(xdot))


print(f"\nRecursive training NRMSE:")

print(f"x     = {nrmse_x:.6e}")

print(f"xdot  = {nrmse_xdot:.6e}")


# %% PLOTS

fig, ax = plt.subplots(
    3,
    1,
    figsize=(12, 8),
    sharex=True,
)


ax[0].plot(
    t,
    u,
    lw=0.7,
)

ax[0].set_ylabel("u")

ax[0].grid(True)


ax[1].plot(
    t,
    x,
    label="true",
    lw=1.5,
)

ax[1].plot(
    t,
    x_sindy,
    "--",
    label="SINDy",
    lw=1.0,
)

ax[1].set_ylabel("x")

ax[1].legend()
ax[1].grid(True)


ax[2].plot(
    t,
    xdot,
    label="true",
    lw=1.5,
)

ax[2].plot(
    t,
    xdot_sindy,
    "--",
    label="SINDy",
    lw=1.0,
)

ax[2].set_ylabel("xdot")

ax[2].set_xlabel("Time [s]")

ax[2].legend()
ax[2].grid(True)


plt.tight_layout()
plt.show()
