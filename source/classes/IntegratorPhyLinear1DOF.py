import numpy as np
import matplotlib.pyplot as plt
from source.function.fe_conv import fe_conv

from source.function.error_utils import metricError

# TODO : post processing avoid list to np array
# TODO : smoothing function at the beginning of simulation
# TODO : check stability


class IntegratorPhyLinear1DOF:
    """
    Class for linear physical 1DOF system dynamics solver
    """

    def __init__(
        self,
        dt,
        m,
        ma_inf,
        rad_ss,
        S,
        t0=0,
        X0=np.array([0.0, 0.0]),
        inertia=0,
        B_pto=0,
        S_pto=0,
        lever=0,
        beta=0,
        stability=True,
        g=9.81,
        rho=1025,
    ):
        """
        Constructor for IntegratorPhyLinear1DOF
        """
        self.dt = dt
        self.t_i = t0
        self.X_hist = []
        self.t_hist = []
        self.fe_hist = []
        self.fr_hist = []
        self.eta_hist = []
        self.m = m
        self.ma_inf = ma_inf
        self.rad_ss = rad_ss
        self.S = S
        self.inertia = inertia
        self.B_pto = B_pto
        self.S_pto = S_pto
        self.l = lever
        self.beta = beta  # angle with horizontal plan at equilibrium, in radians
        self.g = g
        self.rho = rho
        self.eta_for_hs = []  #! just meanwhile fusion of integrator

        # radiation state
        self.Ar = rad_ss["Ar"]
        self.Br = rad_ss["Br"]
        self.Cr = rad_ss["Cr"]
        self.Dr = rad_ss["Dr"]

        nr = self.Ar.shape[0]
        if not isinstance(X0, np.ndarray) or X0.dtype != float:
            X0 = np.asarray(X0, dtype=float)
        self.X = np.concatenate([X0.copy(), np.zeros(nr)])

        # append initial conditions
        self.X_hist.append(self.X.copy())
        self.t_hist.append(self.t_i)

        # stability #!
        self.stability = stability
        # self.B_pto_stab =
        # self.S_pto_stab =
        if self.stability is True:
            if self.B_pto < 0:
                self.B_pto = 0
                print("PTO PI gains rectified for stability")
            if self.S_pto < -self.S:
                self.S_pto = -self.S
                print("PTO PI gains rectified for stability")

    def RK4_step(self, dyn, fe):
        """
        Perform a single RK4 step
        """

        self.eta_for_hs.append(0)

        k1 = dyn(self.t_i, self.X, fe)
        k2 = dyn(self.t_i + 0.5 * self.dt, self.X + 0.5 * k1 * self.dt, fe)
        k3 = dyn(self.t_i + 0.5 * self.dt, self.X + 0.5 * k2 * self.dt, fe)
        k4 = dyn(self.t_i + self.dt, self.X + k3 * self.dt, fe)

        self.X += (k1 + 2 * k2 + 2 * k3 + k4) * (self.dt / 6)
        self.t_i += self.dt

        # append
        self.X_hist.append(self.X.copy())
        self.t_hist.append(self.t_i)

        return self.X, self.t_i

    def dyn_linear(self, t, X, fe):
        # radiation force
        fr = self.Cr @ X[2:] + self.Dr * X[1]

        # physical state
        xdot = X[1]
        xddot = (fe - X[0] * (self.S + self.S_pto) - self.B_pto * X[1] - fr) / (
            self.m + self.ma_inf
        )

        # radiation state equation
        zdot = self.Ar @ X[2:] + self.Br * X[1]

        return np.concatenate(([xdot, xddot], zdot))

        # def dyn_rotation(self, t, X, T_exc):
        #     xdot = X[1]
        #     xddot = (
        #         1
        #         / self.inertia
        #         * (
        #             T_exc
        #             - self.mass * 9.81 * self.l * np.cos(self.beta - X[0])
        #             - (X[0] * self.stiffness + self.damping * X[1])
        #         )
        #     )

        # return np.array([xdot, xddot])

    def post_process(
        self,
    ):
        """
        Compute additional forces based on historical state
        """
        # convert lists to arrays
        self.X_hist = np.array(self.X_hist[:-1])
        self.t_hist = np.array(self.t_hist[:-1])

        # gravity
        self.fg_hist = -(self.m * self.g) * np.ones(len(self.X_hist))

        # PTO force
        self.fs_pto_hist = -self.S_pto * self.X_hist[:, 0]
        self.fd_pto_hist = -self.B_pto * self.X_hist[:, 1]
        self.fpto_hist = self.fs_pto_hist + self.fd_pto_hist

        # hydro forces
        self.fr_hist = -((self.X_hist[:, 2:] @ self.Cr.T) + self.Dr * self.X_hist[:, 1])
        self.fhs_hist = -self.S * self.X_hist[:, 0] - self.fg_hist

        # acceleration
        self.xdotdot_hist = (
            self.fe_hist + self.fhs_hist + self.fr_hist + self.fpto_hist + self.fg_hist
        ) / (self.m + self.ma_inf)

        # added mass
        self.fma_hist = -self.xdotdot_hist * self.ma_inf

        # total hydro
        self.fhyd_hist = self.fr_hist + self.fhs_hist + self.fe_hist + self.fma_hist

        # power
        self.power_hist = self.B_pto * self.X_hist[:, 1] ** 2
        self.mean_power = np.mean(self.power_hist)

    def build_fe(self, eta, t_irf, fe_irf):
        dt_irf = t_irf[1] - t_irf[0]
        if not np.isclose(self.dt, dt_irf, rtol=1e-6, atol=0):
            raise ValueError("dt irf and wave inconsistent")

        self.eta_hist = eta
        self.fe_hist = fe_conv(fe_irf, eta, dt_irf)

    def FK(self, x, eta):  #: for nonlinear integrator
        return None

    def plot_hist(self, variables=None, normalize=False):
        """
        Plot historical data

        variables: list of strings to plot. Possible values:
            'x', 'xdot', 'xddot', 'fe', 'fr', 'fs', 'fd', 'fpto', 'eta'
        normalize: bool
            If True, normalize each signal by its maximum absolute value
        """

        var_map = {
            "x": self.X_hist[:, 0],
            "xdot": self.X_hist[:, 1],
            "xddot": self.xdotdot_hist,
            "fma": self.fma_hist,
            "fe": self.fe_hist,
            "fr": self.fr_hist,
            "fhs": self.fhs_hist,
            "fhyd": self.fhyd_hist,
            "fs_pto": self.fs_pto_hist,
            "fd_pto": self.fd_pto_hist,
            "fpto": self.fpto_hist,
            "eta": self.eta_hist,
        }

        if variables is None:
            variables = ["x", "xdot", "eta"]

        plt.figure(figsize=(10, 6))

        for var in variables:
            if var in var_map:
                data = var_map[var]

                if normalize:
                    max_val = np.max(np.abs(data))
                    if max_val > 0:
                        data = data / max_val

                plt.plot(self.t_hist, data, label=var)
            else:
                print(f"Warning: '{var}' not recognized")

        plt.xlabel("Time [s]")
        plt.ylabel("Normalized value" if normalize else "Value")
        plt.title("Historical Data")
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot_states(self):
        self.plot_hist(["x", "xdot", "xddot"])

    def plot_forces(self):
        self.plot_hist(["fe", "fr", "fs_hyd", "fpto", "fi_ma", "fi"])

    def fit(self, x_true, var, t=None, metric="nrmse_range", plot=False):
        var_map = {
            "x": self.X_hist[:, 0],
            "xdot": self.X_hist[:, 1],
            "xddot": self.xdotdot_hist,
            "fhyd": self.fhyd_hist,
            "fs_pto": self.fs_pto_hist,
            "fd_pto": self.fd_pto_hist,
            "fpto": self.fpto_hist,
            "eta": self.eta_hist,
        }

        if hasattr(self, "fhd_hist") is True:
            var_map["fhd"] = self.fhd_hist
            var_map["fhs"] = self.fhs_hist

        if var not in var_map:
            raise ValueError(f"Unknown property '{var}'. ")

        # extract predicted data
        x_pred = var_map[var]

        # time vector
        if t is not None:
            index = np.isclose(self.t_hist, t)
            if not np.any(index):
                raise ValueError(f"No entries found for t={t}")
            x_pred = x_pred[index]
            x_true = x_true[index]
            x_axis = self.t_hist[index]
            x_label = "t"
        else:
            x_axis = np.arange(len(x_pred))
            x_label = "Sample index"

        # metric
        err = metricError(x_pred, x_true, error_type=metric)

        # plot
        if plot:
            plt.figure()
            plt.plot(x_axis, x_true, label="True", linewidth=2)
            plt.plot(x_axis, x_pred, "--", label=f"Predicted {var}")
            plt.xlabel(x_label)
            plt.ylabel(var)
            plt.title(f"Fit {var} ({metric} = {err:.4f})")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        return err

    def compare(self, integrator, var, t=None, metric="nrmse_range", plot=False):
        def get_var(obj, var):
            var_map = {
                "x": obj.X_hist[:, 0],
                "xdot": obj.X_hist[:, 1],
                "xddot": obj.xdotdot_hist,
                "fhyd": obj.fhyd_hist,
                "fs_pto": obj.fs_pto_hist,
                "fd_pto": obj.fd_pto_hist,
                "fpto": obj.fpto_hist,
                "eta": obj.eta_hist,
            }

            if hasattr(obj, "fhd_hist") is True:
                var_map["fhd"] = obj.fhd_hist
                var_map["fhs"] = obj.fhs_hist

            if var not in var_map:
                raise ValueError(f"Unknown property '{var}'. ")

            return var_map[var]

        x_pred = get_var(self, var)
        x_true = get_var(integrator, var)

        # time vector
        if t is not None:
            index = np.isclose(self.t_hist, t)
            if not np.any(index):
                raise ValueError(f"No entries found for t={t}")

            x_pred = x_pred[index]
            x_true = x_true[index]
            x_axis = self.t_hist[index]
            x_label = "t"
        else:
            x_axis = np.arange(len(x_pred))
            x_label = "Sample index"

        # metric
        err = metricError(x_pred, x_true, error_type=metric)

        # plot
        if plot:
            plt.figure()
            plt.plot(x_axis, x_true, label="Reference", linewidth=2)
            plt.plot(x_axis, x_pred, "--", label=f"{var} (self)")
            plt.xlabel(x_label)
            plt.ylabel(var)
            plt.title(f"Fit {var} ({metric} = {err:.4f})")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        return err
