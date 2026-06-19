import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from source.function.sea_state_utils import param2S, eta2S, S2eta, S2param
# from source.function.waveJ import waveJ
# from source.function.eta2S import eta2S
# from source.function.S2eta import S2eta
# from source.function.S2param import S2param


class Wave:
    def __init__(
        self,
        type,
        T=None,
        H=None,
        gamma=None,
        w_end=8,
        dw=0.01,
        t_end=100,
        dt=0.01,
        eta=None,
        t=None,
    ):
        """
        Constructor for Wave class
        """
        self.T = T
        self.H = H
        self.type = type  # 'irr' for irregular, 'regular' for regular waves
        self.gamma = gamma  # peak enhancement factor for JONSWAP spectrum
        self.t_end = t_end
        self.w_end = w_end
        self.dt = dt
        self.dw = dw
        self.w = np.arange(dw, w_end, dw)

        if self.type == "irr":
            # time
            self.t = np.arange(0, t_end, dt)

            # spectrum
            self.S = param2S(self.T, self.H, self.w, self.gamma, type="JONSWAP")

            # wave elevation
            self.t, self.eta, self.phi, self.A = S2eta(self.w, self.S, self.t)

            # energy flux
            self.J = waveJ(self.T, self.H, self.type)

        elif self.type == "reg":
            # time
            self.t = np.arange(0, t_end, dt)

            # wave elevation
            self.eta = self.H / 2 * np.sin(1 / self.T * 2 * np.pi * self.t)

            # spectrum
            self.w, self.S = eta2S(self.t, self.eta, freq="w")

            # parameter
            self.T, self.H = S2param(self.w, self.S)

            # energy flux
            self.J = waveJ(self.T, self.H, self.type)

        elif self.type == "import":
            # time
            self.t = t
            self.dt = self.t[1] - self.t[0]

            # wave elevation
            if isinstance(eta, (pd.DataFrame, pd.Series)):
                self.eta = eta.reset_index(drop=True)
            else:
                self.eta = eta

            # spectrum
            self.w, self.S = eta2S(self.t, self.eta)

            # energy flux
            self.J = waveJ(self.T, self.H, "irr")

        else:
            raise ValueError("Wave type not recognized. Use 'irr' or 'reg'.")

    def plot_eta(self):
        """Plot wave elevation η(t)."""
        plt.figure(figsize=(8, 4))
        plt.plot(self.t, self.eta)
        plt.xlabel("Time [s]")
        plt.ylabel("Wave elevation η [m]")
        plt.title("Wave Elevation")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_spectrum(self):
        """Plot wave spectrum S(ω) for irregular waves."""
        if not hasattr(self, "S"):
            raise ValueError("Spectrum not available for regular waves.")
        plt.figure(figsize=(8, 4))
        plt.plot(self.w, self.S)
        plt.xlabel("Angular frequency ω [rad/s]")
        plt.ylabel("Spectral density S(ω) [m²s]")
        plt.title("Wave Spectrum")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
