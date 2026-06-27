import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from pathlib import Path

from source.function.get_ccPIgains import get_ccPIgains
from source.function.load_hydro_json import load_hydro_json

"""
# -------------------------------------------------------------------------
# Name:            HydroSphere.py
# Description:     Class for hydrodynamic body, including methods to load hydrodynamic data,
#                    compute PI controller gains, and plot hydrodynamic coefficients and impulse response function.
#
# Author:          Antoine
# Collaborator:    Bona
# Date created:    06/2026
# Project:         surrogate_hydro
# -------------------------------------------------------------------------
"""


class HydroSphere:
    """
    Class for hydrodynamic body
    """

    def __init__(self, r, full_hyd_file_path, mesh=None):
        """
        Constructor for hydrodynamic body
        """
        self.r = r
        self.hyd_file = full_hyd_file_path

        self.load_hydro_data()

        self.rho = 1025
        self.mesh = mesh

        self.mass = 0.5 * 4 / 3 * np.pi * self.r**3 * self.rho
        self.stiffness = np.pi * self.r**2 * self.rho * 9.81

        self.wn = np.sqrt(self.stiffness / (self.mass + self.ma_inf))
        self.fn = self.wn / (2 * np.pi)
        self.Tn = 1 / self.fn

        # compute PI controller gains
        self.Kp, self.Ki = get_ccPIgains(
            self.w, self.mass, self.ma, self.B, self.stiffness
        )

    def load_hydro_data(self):
        """
        Load hydrodynamic data unsing function.
        """
        data = load_hydro_json(self.hyd_file)

        self.w = data["w"]
        self.B = data["B"]
        self.Fe_mod = data["Fe_mod"]
        self.Fe_ang = data["Fe_ang"]
        self.ma = data["ma"]
        self.ma_inf = data["ma_inf"]
        self.fe_irf = data["fe_irf"]
        self.t_irf = data["t_irf"]
        self.rad_ss = data["rad_ss"]

    def resample_irf(self, dt_new):
        t_old = self.t_irf
        fe_old = self.fe_irf

        # original time span
        t0 = t_old[0]
        tf = t_old[-1]

        # build new time vector
        n_new = int(np.floor((tf - t0) / dt_new)) + 1
        t_new = t0 + dt_new * np.arange(n_new)

        # interpolate IRF
        fe_new = np.interp(t_new, t_old, fe_old)

        # overwrite stored IRF
        self.t_irf = t_new
        self.fe_irf = fe_new

    def PI_map(self, T, H, save_path="PI_map"):
        # initialize
        PI_map_Kp = pd.DataFrame(index=H, columns=T)
        PI_map_Ki = pd.DataFrame(index=H, columns=T)
        rows_PI = []

        # get Ki, Kp
        for h in H:
            for t in T:
                Kp, Ki = self.pi_gain(t)

                # store Ki, Kp
                PI_map_Kp.loc[h, t] = Kp
                PI_map_Ki.loc[h, t] = Ki

                rows_PI.append(
                    {
                        "T": t,
                        "H": h,
                        "Kp": Kp,
                        "Ki": Ki,
                    }
                )

        # plot
        df_plot = pd.DataFrame(rows_PI)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Kp subplot
        sc1 = axes[0].scatter(
            df_plot["T"],
            df_plot["H"],
            c=df_plot["Kp"],
            cmap="viridis",
            s=100,
        )

        axes[0].set_xlabel("T")
        axes[0].set_ylabel("H")
        axes[0].set_title("Optimal Kp")

        for _, row in df_plot.iterrows():
            axes[0].text(
                row["T"],
                row["H"],
                f"{row['Kp']:.2g}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        cbar1 = fig.colorbar(sc1, ax=axes[0])
        cbar1.set_label("Kp")

        # Ki subplot
        sc2 = axes[1].scatter(
            df_plot["T"],
            df_plot["H"],
            c=df_plot["Ki"],
            cmap="viridis",
            s=100,
        )

        axes[1].set_xlabel("T")
        axes[1].set_ylabel("H")
        axes[1].set_title("Optimal Ki")

        for _, row in df_plot.iterrows():
            axes[1].text(
                row["T"],
                row["H"],
                f"{row['Ki']:.2g}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        cbar2 = fig.colorbar(sc2, ax=axes[1])
        cbar2.set_label("Ki")

        # finalize & save
        plt.tight_layout()
        plt.savefig(save_path + ".png", dpi=300)
        plt.show()
        plt.close(fig)

        PI_map_Kp.to_csv(save_path + "_Kp.csv")
        PI_map_Ki.to_csv(save_path + "_Ki.csv")

    def pi_gain(self, T):
        """
        Calculate the complex conjugate control associated PI controller gain for period T
        """
        wi = 2 * np.pi / T
        Ki = np.interp(wi, self.w, self.Ki)
        Kp = np.interp(wi, self.w, self.Kp)

        return Kp, Ki

    def plot_irf(self):
        """Plot impulse response function."""
        plt.figure()
        plt.plot(self.t_irf, self.fe_irf)
        plt.xlabel("Time [s]")
        plt.ylabel("IRF")
        plt.title("Impulse Response Function")
        plt.grid(True)
        plt.show()

    def plot_hydro(self, coef=None):
        """
        Plot hydrodynamic coefficients.

        Parameters
        ----------
        coef : str or list of str, optional
            Options: 'B', 'ma', 'Fe_mod', 'Fe_ang'
            If None, all coefficients are plotted.
        """

        coef_map = {
            "B": self.B,
            "ma": self.ma,
            "Fe_mod": self.Fe_mod,
            "Fe_ang": self.Fe_ang,
        }

        if coef is None:
            coef = list(coef_map.keys())
        elif isinstance(coef, str):
            coef = [coef]

        plt.figure()
        for c in coef:
            if c not in coef_map:
                raise ValueError(f"Unknown coefficient '{c}'")
            plt.plot(self.w, coef_map[c], label=c)

        plt.xlabel("Frequency [rad/s]")
        plt.ylabel("Value")
        plt.title("Hydrodynamic Coefficients")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_all(self):
        """Plot IRF and all hydrodynamic coefficients."""

        fig, axes = plt.subplots(2, 1, figsize=(8, 8))

        # IRF
        axes[0].plot(self.t_irf, self.fe_irf)
        axes[0].set_xlabel("Time [s]")
        axes[0].set_ylabel("IRF")
        axes[0].set_title("Impulse Response Function")
        axes[0].grid(True)

        # Hydro coefficients
        axes[1].plot(self.w, self.B, label="B")
        axes[1].plot(self.w, self.ma, label="ma")
        axes[1].plot(self.w, self.Fe_mod, label="Fe_mod")
        axes[1].plot(self.w, self.Fe_ang, label="Fe_ang")

        axes[1].set_xlabel("Frequency [rad/s]")
        axes[1].set_ylabel("Value")
        axes[1].set_title("Hydrodynamic Coefficients")
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()

    def plot_pi_gains(self):
        """Plot PI controller gains."""

        fig, ax = plt.subplots(figsize=(7, 4))

        ax.plot(self.w, self.Kp, label="Kp")
        ax.plot(self.w, self.Ki, label="Ki")

        ax.set_xlabel("Frequency [rad/s]")
        ax.set_ylabel("Gain value")
        ax.set_title("PI Controller Gains")
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        plt.show()

    def save(self, path, suffix=None):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)  # make sure directory exists

        # build filename
        name_parts = [
            f"r{self.r}",
        ]

        if suffix:
            name_parts.append(str(suffix).replace(" ", "_"))

        filename = "_".join(name_parts) + ".pkl"

        # save in the correct directory
        file_path = path / filename
        with open(file_path, "wb") as f:
            pickle.dump(self, f)
